import logging
import os
import tempfile
import time
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from backend.auth import get_current_user
from backend.config import settings
from backend.database import get_db
from backend.db_models import DetectionHistory, User
from backend.schemas import DetectionResult, TextDetectRequest
from models.base import DetectorOutput
from models.registry import registry
from preprocessing.image_preprocessing import InvalidImageError
from preprocessing.video_preprocessing import InvalidVideoError

log = logging.getLogger("api.detect")
router = APIRouter(prefix="/api/detect", tags=["detection"])

CHUNK = 1024 * 1024


def _save_result(
    db: Session, user: User, modality: str, summary: str, out: DetectorOutput, elapsed_ms: float
) -> DetectionResult:
    row = DetectionHistory(
        user_id=user.id,
        modality=modality,
        input_summary=summary[:300],
        prediction=out.prediction,
        confidence_score=out.confidence_score,
        inference_time_ms=round(elapsed_ms, 1),
        models_used=out.models_used,
        details=out.details,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return DetectionResult(
        id=row.id,
        prediction=row.prediction,
        confidence_score=row.confidence_score,
        modality=modality,
        inference_time_ms=row.inference_time_ms,
        models_used=row.models_used,
        details={**row.details, "fake_probability": round(out.prob_fake * 100, 2)},
        created_at=row.created_at,
    )


def _validate_ext(upload: UploadFile, allowed: set[str]) -> str:
    ext = Path(upload.filename or "").suffix.lower()
    if ext not in allowed:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(allowed))}"
        )
    return ext


def _read_limited(upload: UploadFile, max_mb: int) -> bytes:
    limit = max_mb * CHUNK
    data = bytearray()
    while chunk := upload.file.read(CHUNK):
        data.extend(chunk)
        if len(data) > limit:
            raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, f"File exceeds {max_mb} MB limit")
    if not data:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Uploaded file is empty")
    return bytes(data)


@router.post("/text", response_model=DetectionResult)
def detect_text(body: TextDetectRequest, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    t0 = time.perf_counter()
    out = registry.text.predict(body.text, body.title)
    elapsed = (time.perf_counter() - t0) * 1000
    summary = (body.title + " - " if body.title else "") + " ".join(body.text.split())[:200]
    return _save_result(db, user, "text", summary, out, elapsed)


@router.post("/image", response_model=DetectionResult)
def detect_image(
    file: UploadFile = File(...), user: User = Depends(get_current_user), db: Session = Depends(get_db)
):
    _validate_ext(file, settings.IMAGE_EXTENSIONS)
    data = _read_limited(file, settings.MAX_IMAGE_MB)
    t0 = time.perf_counter()
    try:
        out = registry.image.predict(data)
    except InvalidImageError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc))
    elapsed = (time.perf_counter() - t0) * 1000
    return _save_result(db, user, "image", file.filename or "image", out, elapsed)


@router.post("/video", response_model=DetectionResult)
def detect_video(
    file: UploadFile = File(...), user: User = Depends(get_current_user), db: Session = Depends(get_db)
):
    ext = _validate_ext(file, settings.VIDEO_EXTENSIONS)
    data = _read_limited(file, settings.MAX_VIDEO_MB)
    fd, tmp_path = tempfile.mkstemp(suffix=ext, dir=settings.UPLOAD_DIR)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        t0 = time.perf_counter()
        try:
            out = registry.video.predict(tmp_path)
        except InvalidVideoError as exc:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc))
        elapsed = (time.perf_counter() - t0) * 1000
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            log.warning("Could not delete temp upload %s", tmp_path)
    return _save_result(db, user, "video", file.filename or "video", out, elapsed)
