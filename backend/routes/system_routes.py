from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.auth import get_current_user
from backend.database import get_db
from backend.db_models import ModelMetrics, User
from backend.schemas import MetricOut
from models.registry import registry

router = APIRouter(prefix="/api", tags=["system"])


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/system/status")
def system_status(_: User = Depends(get_current_user)):
    return registry.status()


@router.get("/metrics", response_model=list[MetricOut])
def list_metrics(_: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Latest evaluation per model."""
    rows = db.scalars(select(ModelMetrics).order_by(ModelMetrics.evaluated_at.desc())).all()
    latest: dict[str, ModelMetrics] = {}
    for r in rows:
        latest.setdefault(r.model_name, r)
    return sorted(latest.values(), key=lambda r: (r.modality, r.model_name))
