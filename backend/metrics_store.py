from sqlalchemy import select

from backend.database import SessionLocal, init_db
from backend.db_models import ModelMetrics


def ensure_metrics(model_name: str, modality: str, metrics: dict) -> None:
    """Record metrics for a model only if this evaluation is not stored yet."""
    with SessionLocal() as db:
        exists = db.scalar(
            select(ModelMetrics.id).where(
                ModelMetrics.model_name == model_name, ModelMetrics.dataset == metrics["dataset"]
            )
        )
    if not exists:
        record_metrics(model_name, modality, metrics["dataset"], metrics)


def record_metrics(model_name: str, modality: str, dataset: str, metrics: dict) -> None:
    """Persist an evaluation run (accuracy / precision / recall / F1) to ModelMetrics."""
    init_db()
    with SessionLocal() as db:
        db.add(
            ModelMetrics(
                model_name=model_name,
                modality=modality,
                dataset=dataset,
                accuracy=round(float(metrics["accuracy"]), 4),
                precision=round(float(metrics["precision"]), 4),
                recall=round(float(metrics["recall"]), 4),
                f1_score=round(float(metrics["f1_score"]), 4),
                num_samples=int(metrics.get("num_samples", 0)),
            )
        )
        db.commit()
