"""Train / evaluate the text (newspaper) models.

Examples
--------
# Generic CSV with columns text,label[,title]  (label: 1/FAKE = fake, 0/REAL = real)
python -m training.train_text --csv data/news.csv

# ISOT "Fake and Real News" dataset (Kaggle: Fake.csv / True.csv)
python -m training.train_text --fake-csv data/Fake.csv --true-csv data/True.csv

# WELFake (label 1 = real) - flip the mapping
python -m training.train_text --csv data/WELFake_Dataset.csv --fake-value 0

# Also fine-tune a transformer (saved to models/weights/text_transformer/)
python -m training.train_text --csv data/news.csv --transformer distilbert-base-uncased --epochs 2
"""
from __future__ import annotations

import argparse
import csv
import json
import sys

import joblib
from sklearn.model_selection import train_test_split

from backend.metrics_store import record_metrics
from models.text_detector import LOCAL_TRANSFORMER_DIR, TFIDF_METRICS_PATH, TFIDF_PATH, build_tfidf_pipeline
from training.common import compute_metrics, seed_everything

csv.field_size_limit(sys.maxsize)
FAKE_STRINGS = {"1", "fake", "false", "pants-fire", "barely-true", "unreliable"}


def read_rows(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8", errors="ignore") as f:
        return list(csv.DictReader(f))


def row_text(r: dict) -> str:
    return f"{r.get('title') or ''}. {r.get('text') or r.get('content') or r.get('statement') or ''}".strip(". ")


def load_data(args) -> tuple[list[str], list[int]]:
    texts, labels = [], []
    if args.csv:
        for r in read_rows(args.csv):
            raw = str(r.get(args.label_col, "")).strip().lower()
            if raw == "":
                continue
            is_fake = raw == str(args.fake_value).lower() if args.fake_value is not None else raw in FAKE_STRINGS
            t = row_text(r)
            if len(t) > 20:
                texts.append(t)
                labels.append(int(is_fake))
    if args.fake_csv:
        for r in read_rows(args.fake_csv):
            texts.append(row_text(r)); labels.append(1)
    if args.true_csv:
        for r in read_rows(args.true_csv):
            texts.append(row_text(r)); labels.append(0)
    if args.max_samples and len(texts) > args.max_samples:
        texts, _, labels, _ = train_test_split(
            texts, labels, train_size=args.max_samples, stratify=labels, random_state=42
        )
    if not texts:
        sys.exit("No training data loaded - check --csv / --fake-csv / --true-csv")
    print(f"Loaded {len(texts)} samples ({sum(labels)} fake, {len(labels) - sum(labels)} real)")
    return texts, labels


def train_tfidf(x_tr, x_te, y_tr, y_te, dataset: str) -> None:
    print("\n=== TF-IDF + Logistic Regression ===")
    pipe = build_tfidf_pipeline().fit(x_tr, y_tr)
    m = compute_metrics(y_te, pipe.predict(x_te))
    joblib.dump(pipe, TFIDF_PATH)
    TFIDF_METRICS_PATH.write_text(json.dumps({**m, "dataset": dataset}, indent=2))
    record_metrics("TF-IDF + LogisticRegression", "text", dataset, m)
    print(f"Saved -> {TFIDF_PATH}")


def train_transformer(model_name, x_tr, x_te, y_tr, y_te, dataset, epochs, batch_size, lr, max_len) -> None:
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    print(f"\n=== Fine-tuning {model_name} ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2, id2label={0: "REAL", 1: "FAKE"}, label2id={"REAL": 0, "FAKE": 1}
    ).to(device)

    def collate(batch):
        enc = tok([b[0] for b in batch], truncation=True, max_length=max_len, padding=True, return_tensors="pt")
        enc["labels"] = torch.tensor([b[1] for b in batch])
        return enc

    train_dl = DataLoader(list(zip(x_tr, y_tr)), batch_size=batch_size, shuffle=True, collate_fn=collate)
    test_dl = DataLoader(list(zip(x_te, y_te)), batch_size=batch_size, collate_fn=collate)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * len(train_dl) * epochs), len(train_dl) * epochs)

    for ep in range(epochs):
        model.train()
        total = 0.0
        for i, batch in enumerate(train_dl):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            total += loss.item()
            if i % 50 == 0:
                print(f"epoch {ep + 1} step {i}/{len(train_dl)} loss {loss.item():.4f}")
        print(f"epoch {ep + 1} mean loss {total / max(len(train_dl), 1):.4f}")

    model.eval()
    preds = []
    with torch.inference_mode():
        for batch in test_dl:
            batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            preds.extend(model(**batch).logits.argmax(-1).cpu().tolist())
    m = compute_metrics(y_te, preds)
    LOCAL_TRANSFORMER_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(LOCAL_TRANSFORMER_DIR)
    tok.save_pretrained(LOCAL_TRANSFORMER_DIR)
    record_metrics(f"Fine-tuned {model_name}", "text", dataset, m)
    print(f"Saved -> {LOCAL_TRANSFORMER_DIR}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv")
    p.add_argument("--label-col", default="label")
    p.add_argument("--fake-value", default=None, help="Label value that means FAKE (default: 1/fake/false)")
    p.add_argument("--fake-csv")
    p.add_argument("--true-csv")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--transformer", help="HF model to fine-tune, e.g. bert-base-uncased")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max-len", type=int, default=256)
    p.add_argument("--dataset-name", default=None)
    args = p.parse_args()

    seed_everything()
    texts, labels = load_data(args)
    x_tr, x_te, y_tr, y_te = train_test_split(texts, labels, test_size=args.test_size, stratify=labels, random_state=42)
    name = args.dataset_name or (args.csv or f"{args.fake_csv}+{args.true_csv}")
    dataset = f"{name} (hold-out {int(args.test_size * 100)}%)"

    train_tfidf(x_tr, x_te, y_tr, y_te, dataset)
    if args.transformer:
        train_transformer(args.transformer, x_tr, x_te, y_tr, y_te, dataset, args.epochs, args.batch_size, args.lr, args.max_len)


if __name__ == "__main__":
    main()
