from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

from .utils import LABELS

def multilabel_f1(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    # y_* shape: [N,3]
    out = {}
    for i, l in enumerate(LABELS):
        out[f"f1_{l}"] = float(f1_score(y_true[:, i], y_pred[:, i], zero_division=0))
    out["f1_macro"] = float(np.mean([out[f"f1_{l}"] for l in LABELS]))
    return out

def multilabel_auc(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    out = {}
    for i, l in enumerate(LABELS):
        # handle degenerate case
        if len(np.unique(y_true[:, i])) < 2:
            out[f"auc_{l}"] = float("nan")
        else:
            out[f"auc_{l}"] = float(roc_auc_score(y_true[:, i], y_prob[:, i]))
    return out

def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))
