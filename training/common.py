import random

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def compute_metrics(y_true, y_pred) -> dict:
    """Binary metrics with FAKE (=1) as the positive class."""
    m = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "num_samples": int(len(y_true)),
    }
    print(classification_report(y_true, y_pred, target_names=["REAL", "FAKE"], zero_division=0))
    print({k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()})
    return m
