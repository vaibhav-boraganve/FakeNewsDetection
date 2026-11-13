"""Train the CNN + LSTM deepfake video classifier.

Layout:
    DATA/real/*.mp4
    DATA/fake/*.mp4
(e.g. FaceForensics++, Celeb-DF v2, or a DFDC sample.)

To stay within 8 GB RAM / CPU-only hardware, the EfficientNet-B0 spatial encoder is
frozen (ImageNet weights) and per-video frame features are cached once; only the
BiLSTM temporal model + head are trained. The full CNN+LSTM state dict is saved to
models/weights/video_cnn_lstm.pt.

Example
-------
python -m training.train_video --data datasets/celebdf_small --frames 16 --epochs 15
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from backend.metrics_store import record_metrics
from models.architectures import CNNLSTM
from models.video_detector import CNN_LSTM_PATH, face_or_frame
from preprocessing.image_preprocessing import detect_faces, resize_and_normalize
from preprocessing.video_preprocessing import InvalidVideoError, extract_key_frames
from training.common import compute_metrics, seed_everything

EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


@torch.inference_mode()
def video_features(model: CNNLSTM, path: Path, n_frames: int, device) -> torch.Tensor | None:
    try:
        vf = extract_key_frames(str(path), n_frames)
    except InvalidVideoError as exc:
        print(f"  skip {path.name}: {exc}")
        return None
    crops = []
    for fr in vf.frames:
        faces = detect_faces(fr)
        crops.append(face_or_frame(fr, faces[0] if faces else None))
    while len(crops) < n_frames:
        crops.append(crops[-1])
    x = torch.from_numpy(np.stack([resize_and_normalize(c, 224) for c in crops[:n_frames]])).to(device)
    return model.encoder(x).cpu()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True)
    p.add_argument("--frames", type=int, default=16)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--val-split", type=float, default=0.2)
    args = p.parse_args()

    seed_everything()
    root = Path(args.data)
    samples = [(v, y) for cls, y in (("real", 0), ("fake", 1)) for v in (root / cls).rglob("*") if v.suffix.lower() in EXTS]
    if len(samples) < 4:
        raise SystemExit(f"Need at least a few videos under {root}/real and {root}/fake")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNLSTM(pretrained_backbone=True, freeze_backbone=True).to(device)
    model.encoder.eval()

    feats, labels = [], []
    for i, (path, y) in enumerate(samples):
        print(f"[{i + 1}/{len(samples)}] extracting {path.name}")
        f = video_features(model, path, args.frames, device)
        if f is not None:
            feats.append(f); labels.append(y)
    X, Y = torch.stack(feats), torch.tensor(labels)

    idx = np.random.permutation(len(Y))
    k = int(len(idx) * (1 - args.val_split))
    tr, va = idx[:k], idx[k:]
    params = list(model.lstm.parameters()) + list(model.head.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    best_f1, best_m = -1.0, None
    for ep in range(args.epochs):
        model.lstm.train(); model.head.train()
        np.random.shuffle(tr)
        for b in range(0, len(tr), args.batch_size):
            bi = tr[b : b + args.batch_size]
            logits = model.classify_features(X[bi].to(device))
            loss = loss_fn(logits, Y[bi].to(device))
            opt.zero_grad(); loss.backward(); opt.step()
        model.lstm.eval(); model.head.eval()
        with torch.inference_mode():
            preds = model.classify_features(X[va].to(device)).argmax(-1).cpu().tolist()
        print(f"epoch {ep + 1}: loss {loss.item():.4f}")
        m = compute_metrics(Y[va].tolist(), preds)
        if m["f1_score"] > best_f1:
            best_f1, best_m = m["f1_score"], m
            torch.save(model.state_dict(), CNN_LSTM_PATH)
            print(f"  new best F1 {best_f1:.4f} -> saved {CNN_LSTM_PATH}")

    record_metrics("CNN + LSTM (EfficientNet-B0 + BiLSTM)", "video", f"{root.name} (validation)", best_m)


if __name__ == "__main__":
    main()
