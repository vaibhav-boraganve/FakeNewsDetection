"""Newspaper / article text detector.

Hybrid ensemble of
  1. TF-IDF + Logistic Regression baseline (scikit-learn, always available),
  2. a pretrained BERT/RoBERTa sequence classifier (Hugging Face, optional),
  3. stylometric cues (sensationalism, clickbait, attribution).

Weights for (1) are loaded from models/weights/text_tfidf.joblib; if missing, the
baseline is trained on training/data/news_seed.csv at startup so the pipeline is
runnable out of the box. Fine-tuned transformer weights in
models/weights/text_transformer/ take precedence over the Hugging Face hub model.
"""
from __future__ import annotations

import csv
import json
import logging
import os
import threading
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from backend.config import settings
from models.base import DetectorOutput, clamp, get_device, get_torch, sigmoid, weighted_ensemble
from preprocessing.text_preprocessing import linguistic_features, tokenize

log = logging.getLogger("models.text")

TFIDF_PATH = settings.WEIGHTS_DIR / "text_tfidf.joblib"
TFIDF_METRICS_PATH = settings.WEIGHTS_DIR / "text_tfidf.metrics.json"
LOCAL_TRANSFORMER_DIR = settings.WEIGHTS_DIR / "text_transformer"
SEED_DATASET = settings.TRAINING_DATA_DIR / "news_seed.csv"

FAKE_LABEL_HINTS = ("fake", "false", "unreliable", "generated", "machine", "spam")
REAL_LABEL_HINTS = ("real", "true", "reliable", "human")


def build_tfidf_pipeline() -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    tokenizer=tokenize,
                    lowercase=False,
                    token_pattern=None,
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                    min_df=1,
                    max_features=50000,
                ),
            ),
            ("clf", LogisticRegression(C=4.0, class_weight="balanced", max_iter=2000)),
        ]
    )


def load_csv_dataset(path: Path) -> tuple[list[str], list[int]]:
    texts, labels = [], []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            text = f"{row.get('title', '')}. {row.get('text', '')}".strip(". ")
            texts.append(text)
            labels.append(int(row["label"]))
    return texts, labels


def evaluate_and_fit(texts: list[str], labels: list[int]) -> tuple[Pipeline, dict]:
    """Cross-validated metrics, then fit on the full dataset."""
    y = np.array(labels)
    n_splits = max(2, min(5, int(np.bincount(y).min())))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    y_pred = cross_val_predict(build_tfidf_pipeline(), texts, y, cv=cv)
    metrics = {
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y, y_pred, zero_division=0)),
        "num_samples": len(texts),
    }
    pipe = build_tfidf_pipeline().fit(texts, y)
    return pipe, metrics


class TfidfBaseline:
    name = "TF-IDF + LogisticRegression"

    def __init__(self) -> None:
        self.pipeline: Pipeline | None = None
        self.bootstrap_metrics: dict | None = None
        self.source = "none"

    def load(self) -> None:
        if TFIDF_PATH.exists():
            self.pipeline = joblib.load(TFIDF_PATH)
            self.source = "weights"
            if TFIDF_METRICS_PATH.exists():
                self.bootstrap_metrics = json.loads(TFIDF_METRICS_PATH.read_text())
            return
        log.info("No TF-IDF weights found; training baseline on bundled seed dataset")
        texts, labels = load_csv_dataset(SEED_DATASET)
        self.pipeline, self.bootstrap_metrics = evaluate_and_fit(texts, labels)
        self.bootstrap_metrics["dataset"] = "news_seed.csv (bundled, 5-fold CV)"
        joblib.dump(self.pipeline, TFIDF_PATH)
        TFIDF_METRICS_PATH.write_text(json.dumps(self.bootstrap_metrics, indent=2))
        self.source = "seed-trained"

    def predict(self, text: str) -> tuple[float, dict]:
        assert self.pipeline is not None
        prob = float(self.pipeline.predict_proba([text])[0][1])
        return prob, {"top_terms": self._explain(text)}

    def _explain(self, text: str, k: int = 6) -> dict[str, list[str]]:
        vec: TfidfVectorizer = self.pipeline.named_steps["tfidf"]
        clf: LogisticRegression = self.pipeline.named_steps["clf"]
        x = vec.transform([text])
        if x.nnz == 0:
            return {"fake_indicators": [], "real_indicators": []}
        names = vec.get_feature_names_out()
        contrib = x.multiply(clf.coef_[0]).tocsr()
        idx, vals = contrib.indices, contrib.data
        order = np.argsort(vals)
        fake = [names[idx[i]] for i in order[::-1][:k] if vals[i] > 0]
        real = [names[idx[i]] for i in order[:k] if vals[i] < 0]
        return {"fake_indicators": fake, "real_indicators": real}


class TransformerTextClassifier:
    """Wrapper around any Hugging Face sequence-classification checkpoint."""

    def __init__(self, display_name: str, model_ref: str, template: str = "{title}{sep}{text}") -> None:
        self.name = display_name
        self.model_ref = model_ref
        self.template = template
        self.status = "not-loaded"
        self.error: str | None = None
        self._model = None
        self._tokenizer = None
        self._fake_idx = 1
        self._lock = threading.Lock()

    @property
    def ready(self) -> bool:
        return self.status == "ready"

    def load(self) -> None:
        torch = get_torch()
        if torch is None:
            self.status, self.error = "unavailable", "PyTorch not installed"
            return
        self.status = "loading"
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_ref)
            model = AutoModelForSequenceClassification.from_pretrained(self.model_ref)
            model.eval().to(get_device())
            self._model = model
            self._fake_idx = self._resolve_fake_index(model.config.id2label)
            self.status = "ready"
            log.info("Loaded transformer %s (fake label index %d)", self.model_ref, self._fake_idx)
        except Exception as exc:
            self.status, self.error = "failed", str(exc)[:300]
            log.warning("Transformer %s unavailable: %s", self.model_ref, exc)

    @staticmethod
    def _resolve_fake_index(id2label: dict) -> int:
        override = os.getenv("TEXT_TRANSFORMER_FAKE_INDEX")
        if override is not None:
            return int(override)
        labels = {int(k): str(v).lower() for k, v in id2label.items()}
        for i, lab in labels.items():
            if any(h in lab for h in FAKE_LABEL_HINTS):
                return i
        if len(labels) == 2:
            for i, lab in labels.items():
                if any(h in lab for h in REAL_LABEL_HINTS):
                    return 1 - i
        return 1

    def predict(self, text: str, title: str = "") -> float:
        torch = get_torch()
        payload = self.template.format(title=title or "", text=text, sep=". " if title else "")
        with self._lock, torch.inference_mode():
            enc = self._tokenizer(payload, truncation=True, max_length=512, return_tensors="pt").to(get_device())
            logits = self._model(**enc).logits[0]
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        return float(probs[self._fake_idx])


def style_score(features: dict[str, float]) -> tuple[float, list[str]]:
    """Map stylometric features to a fake-probability with human-readable signals."""
    z = -1.0
    z += 1.2 * min(features["exclamation_ratio"], 2.0)
    z += 8.0 * min(features["all_caps_ratio"], 0.3)
    z += 25.0 * min(features["sensational_ratio"], 0.1)
    z += 0.9 * min(features["clickbait_hits"], 3.0)
    z -= 0.6 * min(features["attribution_hits"], 4.0)
    z -= 0.4 if features["quote_count"] >= 1 else 0.0

    signals = []
    if features["exclamation_ratio"] > 0.5:
        signals.append("Excessive exclamation marks")
    if features["all_caps_ratio"] > 0.05:
        signals.append("Frequent ALL-CAPS words")
    if features["sensational_ratio"] > 0.02:
        signals.append("Sensational / emotionally loaded vocabulary")
    if features["clickbait_hits"] >= 1:
        signals.append("Clickbait or share-bait phrasing")
    if features["attribution_hits"] >= 2:
        signals.append("Cites sources, officials or data (credibility cue)")
    if features["word_count"] < 40:
        signals.append("Very short text - lower reliability")
    return clamp(sigmoid(z), 0.02, 0.98), signals


class TextDetector:
    modality = "text"

    def __init__(self) -> None:
        self.baseline = TfidfBaseline()
        self.transformer: TransformerTextClassifier | None = None
        self.ai_detector: TransformerTextClassifier | None = None
        if LOCAL_TRANSFORMER_DIR.exists() and any(LOCAL_TRANSFORMER_DIR.iterdir()):
            self.transformer = TransformerTextClassifier("Fine-tuned BERT (local)", str(LOCAL_TRANSFORMER_DIR))
        elif settings.ENABLE_TRANSFORMER and settings.TEXT_TRANSFORMER_MODEL:
            template = os.getenv("TEXT_TRANSFORMER_TEMPLATE", "<title>{title}<content>{text}<end>")
            self.transformer = TransformerTextClassifier(
                f"RoBERTa ({settings.TEXT_TRANSFORMER_MODEL})", settings.TEXT_TRANSFORMER_MODEL, template
            )
        ai_model = os.getenv("AI_TEXT_DETECTOR_MODEL", "").strip()
        if settings.ENABLE_TRANSFORMER and ai_model:
            self.ai_detector = TransformerTextClassifier(f"AI-text detector ({ai_model})", ai_model, "{text}")

    def load(self) -> None:
        self.baseline.load()

    def load_deep(self) -> None:
        for m in (self.transformer, self.ai_detector):
            if m is not None:
                m.load()

    def status(self) -> dict:
        return {
            "baseline": self.baseline.source,
            "transformer": self.transformer.status if self.transformer else "disabled",
            "transformer_model": self.transformer.model_ref if self.transformer else None,
            "ai_text_detector": self.ai_detector.status if self.ai_detector else "disabled",
        }

    def predict(self, text: str, title: str | None = None) -> DetectorOutput:
        full_text = f"{title}. {text}" if title else text
        feats = linguistic_features(full_text)
        s_prob, signals = style_score(feats)
        b_prob, b_details = self.baseline.predict(full_text)

        scores: dict[str, tuple[float, float]] = {}
        component_probs: dict[str, float] = {}
        used = [self.baseline.name, "Stylometric analysis (NLTK)"]

        if self.transformer and self.transformer.ready:
            t_prob = self.transformer.predict(text, title or "")
            scores["transformer"] = (t_prob, 0.55)
            component_probs[self.transformer.name] = round(t_prob * 100, 2)
            used.insert(0, self.transformer.name)
            scores["tfidf"] = (b_prob, 0.30)
            scores["style"] = (s_prob, 0.15)
        else:
            scores["tfidf"] = (b_prob, 0.65)
            scores["style"] = (s_prob, 0.35)

        if self.ai_detector and self.ai_detector.ready:
            a_prob = self.ai_detector.predict(text)
            scores["ai_text"] = (a_prob, 0.20)
            component_probs[self.ai_detector.name] = round(a_prob * 100, 2)
            used.append(self.ai_detector.name)

        component_probs[self.baseline.name] = round(b_prob * 100, 2)
        component_probs["Stylometric analysis (NLTK)"] = round(s_prob * 100, 2)
        prob = weighted_ensemble(scores)

        return DetectorOutput(
            prob_fake=prob,
            models_used=used,
            details={
                "component_fake_probabilities": component_probs,
                "signals": signals,
                "top_terms": b_details["top_terms"],
                "features": {k: round(v, 4) for k, v in feats.items()},
                "transformer_status": self.transformer.status if self.transformer else "disabled",
            },
        )
