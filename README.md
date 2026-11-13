# FakeGuard – Fake Detection in Newspaper and Images

A full-stack web application that analyzes **newspaper / article text**, **images** and **videos**
and classifies each input as **REAL** or **FAKE** with a percentage confidence score.

*B.Tech Project – Dept. of CSE, Sanjay Ghodawat University (2026–27).*

---

## 1. Architecture (three layers)

```
┌──────────────────────────── Presentation layer (frontend/) ────────────────────────────┐
│  Login / Register (JWT)  ·  Dashboard: [Text] [Image] [Video] tabs  ·  Result card      │
│  (verdict badge, confidence gauge, modality, inference time)  ·  History  ·  Metrics    │
└───────────────────────────────────────────┬─────────────────────────────────────────────┘
                                 REST / JSON │ (FastAPI, backend/)
┌──────────────────────────── Application layer ─────────────────────────────────────────┐
│ TEXT : NLTK clean+tokenize → TF-IDF + LogisticRegression ┐                              │
│                              RoBERTa / BERT transformer  ├─ weighted ensemble           │
│                              stylometric cues (NLTK)     ┘                              │
│ IMAGE: OpenCV decode/resize/normalize → EfficientNet-B0 CNN ┐                           │
│        Haar face detection → XceptionNet on face crops      ├─ ensemble                 │
│        ELA · noise residual · resampling spectrum · EXIF    ┘                           │
│ VIDEO: OpenCV key-frame extraction → face track → EfficientNet-B0 features → BiLSTM     │
│        + temporal consistency (CNN-embedding jitter, flicker, blending) + XceptionNet   │
│  → {"prediction": "REAL|FAKE", "confidence_score": 0-100, "modality": ...}              │
└───────────────────────────────────────────┬─────────────────────────────────────────────┘
┌──────────────────────────── Database & model layer ────────────────────────────────────┐
│ SQLite (SQLAlchemy): users · detection_history · model_metrics (Acc/Prec/Rec/F1)        │
│ models/ inference wrappers + models/weights/ (trained checkpoints)                       │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

### Works out of the box, gets better with trained weights

| Pipeline | With no extra weights (first run) | After training / adding weights |
|---|---|---|
| Text | TF-IDF baseline auto-trained on the bundled seed set **+** pretrained RoBERTa fake-news model `hamzab/roberta-fake-news-classification` (downloaded once, ~500 MB) **+** stylometry | `train_text.py` on ISOT / WELFake / LIAR; optional BERT fine-tune |
| Image | Forensic analysis: Error Level Analysis, noise-residual consistency, resampling-periodicity spectrum, AI-generator / editor metadata signatures, face blending check | `image_efficientnet_b0.pt` (CIFAKE, CASIA) and `xception_faces.pt` (FaceForensics++) |
| Video | Key frames + face track, ImageNet EfficientNet-B0 embedding stability, face flicker / blending, spectral cues | `video_cnn_lstm.pt` (FaceForensics++, Celeb-DF, DFDC) |

The result card always shows which models produced the verdict and the **analysis mode**
(e.g. *"forensic heuristics (no trained CNN weights found)"*). The heuristic fallbacks reliably
flag resampled or upscaled images and obvious temporal inconsistencies, but **detecting
splicing and face swaps reliably requires the trained CNNs**. Train them before you evaluate
those use cases.

---

## 2. Project structure

```
FakeNewsDetection/
├── run.py                      # python run.py  → http://127.0.0.1:8000
├── requirements.txt
├── .env.example                # copy to .env
├── backend/
│   ├── main.py                 # FastAPI app, startup (DB init, model load/warm-up), static frontend
│   ├── config.py               # settings from .env
│   ├── database.py             # SQLAlchemy engine/session (SQLite)
│   ├── db_models.py            # User, DetectionHistory, ModelMetrics
│   ├── schemas.py              # Pydantic request/response models
│   ├── auth.py                 # bcrypt hashing, JWT issue/verify, current-user dependency
│   ├── metrics_store.py        # persist evaluation metrics
│   └── routes/
│       ├── auth_routes.py      # /api/auth/register | login | me
│       ├── detect_routes.py    # /api/detect/text | image | video
│       ├── history_routes.py   # /api/history (list, stats, detail, delete)
│       └── system_routes.py    # /api/health, /api/system/status, /api/metrics
├── preprocessing/
│   ├── text_preprocessing.py   # NLTK cleaning, tokenizing, stop-words, stemming, stylometry
│   ├── image_preprocessing.py  # OpenCV decode/resize/normalize, face detection, EXIF
│   └── video_preprocessing.py  # OpenCV key-frame extraction (+ adjacent frames)
├── models/
│   ├── base.py                 # DetectorOutput, torch helpers, ensembling
│   ├── architectures.py        # EfficientNet-B0, XceptionNet, CNN+BiLSTM (PyTorch)
│   ├── text_detector.py        # TF-IDF+LR, HF transformer wrapper, stylometry ensemble
│   ├── image_detector.py       # forensic analyzer + CNN/Xception wrappers
│   ├── video_detector.py       # CNN+LSTM + temporal consistency analysis
│   ├── registry.py             # singletons + background warm-up
│   └── weights/                # trained checkpoints (see weights/README.md)
├── training/
│   ├── data/news_seed.csv      # small bundled seed set (bootstraps the TF-IDF model)
│   ├── train_text.py           # TF-IDF (+ optional BERT fine-tune) → metrics to DB
│   ├── train_image.py          # EfficientNet-B0 / XceptionNet fine-tune → metrics to DB
│   ├── train_video.py          # CNN (frozen) + BiLSTM training → metrics to DB
│   └── common.py
├── frontend/
│   ├── index.html  register.html  dashboard.html  history.html
│   ├── css/styles.css
│   ├── js/api.js auth.js result.js dashboard.js history.js
│   └── img/logo.svg
├── tests/test_api.py           # end-to-end API tests (pytest)
├── data/                       # SQLite DB (created at runtime)
└── uploads/                    # temp uploads (videos are deleted after analysis)
```

---

## 3. Setup and run locally

Requirements: Python **3.10–3.12**, about 3 GB free disk for PyTorch and models, and 8 GB RAM.

```bash
# 1. create a virtual environment
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. install dependencies (CPU PyTorch is fine)
pip install --upgrade pip
pip install -r requirements.txt
#   Windows/Linux CPU-only (smaller download):
#   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 3. configure
cp .env.example .env               # then set SECRET_KEY to a long random string

# 4. run
python run.py                      # or: uvicorn backend.main:app --reload
```

Open **http://127.0.0.1:8000**, create an account, and start analyzing. If port 8000 is
already in use, run `python run.py --port 8010`.

On first start:
* the TF-IDF baseline is trained on the seed set (about 1 s) and its cross-validated metrics are stored;
* the RoBERTa model (~500 MB) and the EfficientNet ImageNet weights (~20 MB) download **in the
  background**. Until they are ready, the lighter models answer requests, and the dashboard's
  *Model Status* panel shows progress.

Fully offline / low-RAM mode: set `ENABLE_TRANSFORMER=false` and `ENABLE_PRETRAINED_DOWNLOAD=false` in `.env`.

Run tests: `pytest -q`

Interactive API docs: **http://127.0.0.1:8000/docs**

---

## 4. REST API

All endpoints except auth and health require `Authorization: Bearer <token>`.

| Method | Endpoint | Body | Description |
|---|---|---|---|
| POST | `/api/auth/register` | `{full_name, email, password}` | Create account → JWT |
| POST | `/api/auth/login` | `{email, password}` | Login → JWT |
| GET | `/api/auth/me` | – | Current user |
| POST | `/api/detect/text` | `{text, title?}` | Newspaper/text analysis |
| POST | `/api/detect/image` | multipart `file` (jpg/png/bmp/webp ≤ 10 MB) | Image forensics |
| POST | `/api/detect/video` | multipart `file` (mp4/avi/mov/mkv/webm ≤ 100 MB) | Deepfake video |
| GET | `/api/history?modality=&prediction=&limit=&offset=` | – | Paginated history |
| GET | `/api/history/stats` | – | Totals by verdict/modality |
| GET / DELETE | `/api/history/{id}` | – | Detail / delete |
| GET | `/api/metrics` | – | Latest Accuracy/Precision/Recall/F1 per model |
| GET | `/api/system/status` | – | Which models are loaded |

Example response:

```json
{
  "id": 12,
  "prediction": "FAKE",
  "confidence_score": 93.4,
  "modality": "text",
  "inference_time_ms": 18.6,
  "models_used": ["RoBERTa (hamzab/roberta-fake-news-classification)", "TF-IDF + LogisticRegression", "Stylometric analysis (NLTK)"],
  "details": {
    "component_fake_probabilities": {"RoBERTa (...)": 99.9, "TF-IDF + LogisticRegression": 79.55, "Stylometric analysis (NLTK)": 97.25},
    "signals": ["Excessive exclamation marks", "Clickbait or share-bait phrasing"],
    "fake_probability": 93.4
  },
  "created_at": "2026-11-12T07:35:10"
}
```

`confidence_score` is the probability of the predicted class (always ≥ 50%).

---

## 5. Training and evaluation (populates `ModelMetrics`)

Each script holds out a validation/test split, prints a classification report, saves weights
to `models/weights/`, and writes Accuracy / Precision / Recall / F1 to the database. The
metrics then appear on the dashboard.

```bash
# Text – ISOT Fake and Real News (Kaggle: Fake.csv, True.csv)
python -m training.train_text --fake-csv datasets/Fake.csv --true-csv datasets/True.csv
# Text – generic CSV (text,label[,title]); WELFake uses 1 = real, so flip it:
python -m training.train_text --csv datasets/WELFake_Dataset.csv --fake-value 0 --max-samples 30000
# Text – also fine-tune BERT (GPU recommended; on CPU use distilbert + --max-samples)
python -m training.train_text --csv datasets/news.csv --transformer distilbert-base-uncased --epochs 2

# Image – folders real/ and fake/ (e.g. CIFAKE, CASIA v2)
python -m training.train_image --data datasets/cifake --arch efficientnet --epochs 5
# Faces – FaceForensics++ / Celeb-DF face crops
python -m training.train_image --data datasets/ff_faces --arch xception --faces --epochs 5

# Video – real/*.mp4 and fake/*.mp4 (FaceForensics++, Celeb-DF v2, DFDC sample)
python -m training.train_video --data datasets/celebdf --frames 16 --epochs 15
```

Restart the server after training so the new weights load.

8 GB RAM tips: keep `--batch-size` small (4–8). The video trainer freezes the CNN and caches
frame features, so only the LSTM trains (CPU-friendly). Set `TORCH_THREADS` in `.env`.

---

## 6. Security notes

* Passwords are hashed with bcrypt; sessions use HS256 JWTs (`SECRET_KEY` from `.env`).
* Uploads are checked for extension and size limit and decoded/validated by OpenCV. Video
  temp files are deleted right after analysis, and images are processed in memory.
* History is scoped per user; users can only read or delete their own records.

## 7. Limitations

Predictions come from the trained models and the available datasets. They will not catch
every novel or highly sophisticated fake. The system supports human judgement and should not
replace expert verification or trusted fact-checking.

## References

Rössler et al., *FaceForensics++* (ICCV 2019) · Dolhansky et al., *DFDC* (2020) · Li et al.,
*Celeb-DF* (CVPRW 2020) · Devlin et al., *BERT* (NAACL 2019) · Vaswani et al., *Attention Is
All You Need* (NeurIPS 2017) · He et al., *ResNet* (CVPR 2016) · Goodfellow et al., *GANs*
(NeurIPS 2014) · Popescu & Farid, *Exposing digital forgeries by detecting traces of resampling*
(IEEE TSP 2005).
