"""Image preprocessing with OpenCV: decode, validate, resize, normalise, face detection."""
from __future__ import annotations

import io
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
XCEPTION_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
XCEPTION_STD = np.array([0.5, 0.5, 0.5], dtype=np.float32)

MAX_SIDE = 2048


class InvalidImageError(ValueError):
    pass


def decode_image(data: bytes) -> np.ndarray:
    """Decode bytes into a BGR uint8 image, downscaling very large inputs."""
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise InvalidImageError("File is not a valid image")
    h, w = img.shape[:2]
    if min(h, w) < 32:
        raise InvalidImageError("Image is too small to analyse (min 32x32)")
    scale = MAX_SIDE / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def resize_and_normalize(
    img_bgr: np.ndarray,
    size: int = 224,
    mean: np.ndarray = IMAGENET_MEAN,
    std: np.ndarray = IMAGENET_STD,
) -> np.ndarray:
    """Return a CHW float32 array ready for a CNN."""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    x = rgb.astype(np.float32) / 255.0
    x = (x - mean) / std
    return np.transpose(x, (2, 0, 1)).copy()


@lru_cache(maxsize=1)
def _face_cascade() -> cv2.CascadeClassifier:
    return cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def detect_faces(img_bgr: np.ndarray, min_size: int = 40) -> list[tuple[int, int, int, int]]:
    """Detect frontal faces; returns (x, y, w, h) boxes sorted by area (largest first)."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = _face_cascade().detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(min_size, min_size))
    boxes = [tuple(int(v) for v in f) for f in faces] if len(faces) else []
    return sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)


def crop_face(img_bgr: np.ndarray, box: tuple[int, int, int, int], margin: float = 0.3) -> np.ndarray:
    x, y, w, h = box
    H, W = img_bgr.shape[:2]
    mx, my = int(w * margin), int(h * margin)
    x0, y0 = max(x - mx, 0), max(y - my, 0)
    x1, y1 = min(x + w + mx, W), min(y + h + my, H)
    return img_bgr[y0:y1, x0:x1]


def read_metadata(data: bytes) -> dict:
    """Extract EXIF / PNG text metadata that may reveal editing or generation software."""
    meta: dict = {"format": None, "exif": {}, "text_chunks": {}}
    try:
        with Image.open(io.BytesIO(data)) as im:
            meta["format"] = im.format
            meta["mode"] = im.mode
            exif = im.getexif()
            if exif:
                from PIL import ExifTags

                for tag_id, value in exif.items():
                    name = ExifTags.TAGS.get(tag_id, str(tag_id))
                    if name in {"Make", "Model", "Software", "DateTime", "Artist", "ImageDescription"}:
                        meta["exif"][name] = str(value)[:200]
            for k, v in (im.info or {}).items():
                if isinstance(v, (str, bytes)) and k.lower() not in {"icc_profile", "exif"}:
                    meta["text_chunks"][k] = (v.decode(errors="ignore") if isinstance(v, bytes) else v)[:300]
    except Exception:
        pass
    return meta
