from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from backend.auth import get_current_user
from backend.database import get_db
from backend.db_models import DetectionHistory, User
from backend.schemas import HistoryPage, HistoryStats

router = APIRouter(prefix="/api/history", tags=["history"])


@router.get("", response_model=HistoryPage)
def list_history(
    modality: Literal["text", "image", "video"] | None = None,
    prediction: Literal["REAL", "FAKE"] | None = None,
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    q = select(DetectionHistory).where(DetectionHistory.user_id == user.id)
    if modality:
        q = q.where(DetectionHistory.modality == modality)
    if prediction:
        q = q.where(DetectionHistory.prediction == prediction)
    total = db.scalar(select(func.count()).select_from(q.subquery()))
    items = db.scalars(q.order_by(DetectionHistory.created_at.desc()).limit(limit).offset(offset)).all()
    return HistoryPage(total=total or 0, items=items)


@router.get("/stats", response_model=HistoryStats)
def history_stats(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    rows = db.execute(
        select(DetectionHistory.modality, DetectionHistory.prediction, func.count())
        .where(DetectionHistory.user_id == user.id)
        .group_by(DetectionHistory.modality, DetectionHistory.prediction)
    ).all()
    by_modality = {"text": 0, "image": 0, "video": 0}
    fake = real = 0
    for modality, prediction, n in rows:
        by_modality[modality] = by_modality.get(modality, 0) + n
        if prediction == "FAKE":
            fake += n
        else:
            real += n
    return HistoryStats(total=fake + real, fake=fake, real=real, by_modality=by_modality)


@router.get("/{item_id}")
def get_history_item(item_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    row = db.get(DetectionHistory, item_id)
    if not row or row.user_id != user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Not found")
    return {
        "id": row.id,
        "modality": row.modality,
        "input_summary": row.input_summary,
        "prediction": row.prediction,
        "confidence_score": row.confidence_score,
        "inference_time_ms": row.inference_time_ms,
        "models_used": row.models_used,
        "details": row.details,
        "created_at": row.created_at,
    }


@router.delete("/{item_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_history_item(item_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    res = db.execute(
        delete(DetectionHistory).where(DetectionHistory.id == item_id, DetectionHistory.user_id == user.id)
    )
    db.commit()
    if res.rowcount == 0:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Not found")
