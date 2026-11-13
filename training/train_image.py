"""Fine-tune the image CNNs (EfficientNet-B0 or XceptionNet) for REAL/FAKE classification.

Expected layout (ImageFolder):
    DATA/real/*.jpg
    DATA/fake/*.jpg
or with explicit splits: DATA/train/{real,fake}, DATA/val/{real,fake}

Suitable datasets: CIFAKE, CASIA v2 (splicing), "140k Real and Fake Faces",
FaceForensics++ / Celeb-DF face crops (use --arch xception --faces).

Examples
--------
python -m training.train_image --data datasets/cifake --arch efficientnet --epochs 5
python -m training.train_image --data datasets/ff_faces --arch xception --faces --epochs 5
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from backend.metrics_store import record_metrics
from models.architectures import build_efficientnet_b0, build_xception
from models.image_detector import EFFNET_PATH, XCEPTION_PATH
from preprocessing.image_preprocessing import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    XCEPTION_MEAN,
    XCEPTION_STD,
    crop_face,
    detect_faces,
    resize_and_normalize,
)
from training.common import compute_metrics, seed_everything

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASSES = {"real": 0, "fake": 1}


def list_samples(root: Path) -> list[tuple[Path, int]]:
    out = []
    for cls, y in CLASSES.items():
        d = root / cls
        if d.is_dir():
            out += [(p, y) for p in d.rglob("*") if p.suffix.lower() in EXTS]
    return out


class ImageSet(Dataset):
    def __init__(self, samples, size, mean, std, faces: bool, augment: bool):
        self.samples, self.size, self.mean, self.std = samples, size, mean, std
        self.faces, self.augment = faces, augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, y = self.samples[i]
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            img = np.zeros((self.size, self.size, 3), np.uint8)
        if self.faces:
            boxes = detect_faces(img)
            if boxes:
                img = crop_face(img, boxes[0])
        if self.augment:
            if np.random.rand() < 0.5:
                img = cv2.flip(img, 1)
            if np.random.rand() < 0.3:
                q = int(np.random.randint(60, 95))
                img = cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])[1], cv2.IMREAD_COLOR)
        return torch.from_numpy(resize_and_normalize(img, self.size, self.mean, self.std)), y


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True)
    p.add_argument("--arch", choices=["efficientnet", "xception"], default="efficientnet")
    p.add_argument("--faces", action="store_true", help="Crop the largest face before training")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--val-split", type=float, default=0.2)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--no-pretrained", action="store_true")
    args = p.parse_args()

    seed_everything()
    root = Path(args.data)
    if (root / "train").is_dir():
        train_s, val_s = list_samples(root / "train"), list_samples(root / ("val" if (root / "val").is_dir() else "test"))
    else:
        alls = list_samples(root)
        np.random.shuffle(alls)
        k = int(len(alls) * (1 - args.val_split))
        train_s, val_s = alls[:k], alls[k:]
    if not train_s or not val_s:
        raise SystemExit(f"No images found under {root} (expected real/ and fake/ sub-folders)")
    print(f"train={len(train_s)} val={len(val_s)}")

    if args.arch == "efficientnet":
        model, size, mean, std, out_path = build_efficientnet_b0(pretrained=not args.no_pretrained), 224, IMAGENET_MEAN, IMAGENET_STD, EFFNET_PATH
        name = "EfficientNet-B0 CNN"
    else:
        model, size, mean, std, out_path = build_xception(pretrained=not args.no_pretrained), 299, XCEPTION_MEAN, XCEPTION_STD, XCEPTION_PATH
        name = "XceptionNet (faces)"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    train_dl = DataLoader(ImageSet(train_s, size, mean, std, args.faces, True), args.batch_size, shuffle=True, num_workers=args.workers)
    val_dl = DataLoader(ImageSet(val_s, size, mean, std, args.faces, False), args.batch_size, num_workers=args.workers)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss()

    best_f1, best_metrics = -1.0, None
    for ep in range(args.epochs):
        model.train()
        for i, (x, y) in enumerate(train_dl):
            x, y = x.to(device), y.to(device)
            loss = loss_fn(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
            if i % 20 == 0:
                print(f"epoch {ep + 1} step {i}/{len(train_dl)} loss {loss.item():.4f}")
        sched.step()

        model.eval()
        ys, ps = [], []
        with torch.inference_mode():
            for x, y in val_dl:
                ps += model(x.to(device)).argmax(-1).cpu().tolist()
                ys += y.tolist()
        m = compute_metrics(ys, ps)
        if m["f1_score"] > best_f1:
            best_f1, best_metrics = m["f1_score"], m
            torch.save(model.state_dict(), out_path)
            print(f"  new best F1 {best_f1:.4f} -> saved {out_path}")

    record_metrics(name, "image", f"{root.name} (validation)", best_metrics)


if __name__ == "__main__":
    main()
