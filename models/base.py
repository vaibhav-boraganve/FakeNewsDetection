"""Shared types and helpers for the inference wrappers."""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from backend.config import settings

log = logging.getLogger("models")


@dataclass
class DetectorOutput:
    prob_fake: float
    models_used: list[str]
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def prediction(self) -> str:
        return "FAKE" if self.prob_fake >= 0.5 else "REAL"

    @property
    def confidence_score(self) -> float:
        p = self.prob_fake if self.prediction == "FAKE" else 1.0 - self.prob_fake
        return round(p * 100.0, 2)


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


@lru_cache(maxsize=1)
def get_torch():
    """Import torch lazily so the app still starts (heuristic mode) if it is not installed."""
    try:
        import torch

        torch.set_num_threads(max(1, settings.TORCH_THREADS))
        return torch
    except Exception as exc:  # pragma: no cover - environment dependent
        log.warning("PyTorch unavailable (%s); deep models disabled", exc)
        return None


def get_device():
    torch = get_torch()
    if torch is None:
        return None
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def weighted_ensemble(scores: dict[str, tuple[float, float]]) -> float:
    """Combine {name: (prob_fake, weight)} into a single probability."""
    total_w = sum(w for _, w in scores.values())
    if total_w <= 0:
        return 0.5
    return sum(p * w for p, w in scores.values()) / total_w
