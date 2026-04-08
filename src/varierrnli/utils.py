from __future__ import annotations
import json
import os
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


LABELS = ["C", "E", "N"]
LABEL_TO_IDX = {l: i for i, l in enumerate(LABELS)}
RAW_TO_CANON = {
    "contradiction": "C",
    "entailment": "E",
    "neutral": "N",
}

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def safe_load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def safe_load_annotator_meta_json(path: str) -> Dict[str, Dict[str, str]]:
    """The uploaded annotator meta JSON may include trailing commas.
    We repair the JSON by removing commas before closing braces/brackets.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    fixed = re.sub(r",(\s*[}\]])", r"\1", raw)
    return json.loads(fixed)

def soft_label_to_vec(soft_label: Dict[str, Dict[str, float]]) -> np.ndarray:
    # y_soft = [y_C, y_E, y_N] using soft_label[L]['1']
    y = np.zeros(3, dtype=np.float32)
    y[LABEL_TO_IDX["C"]] = float(soft_label["contradiction"]["1"])
    y[LABEL_TO_IDX["E"]] = float(soft_label["entailment"]["1"])
    y[LABEL_TO_IDX["N"]] = float(soft_label["neutral"]["1"])
    return y

def labelset_str_to_multihot(labelset_str: str) -> np.ndarray:
    # e.g. "contradiction,neutral" -> [1,0,1] in C,E,N order
    mh = np.zeros(3, dtype=np.float32)
    parts = [p.strip() for p in labelset_str.split(",") if p.strip()]
    for p in parts:
        canon = RAW_TO_CANON.get(p)
        if canon is None:
            raise ValueError(f"Unknown label token: {p}")
        mh[LABEL_TO_IDX[canon]] = 1.0
    return mh

def probs_to_labelset(probs: np.ndarray, thresholds: np.ndarray) -> List[str]:
    """Convert probs [3] to label set using per-label thresholds.
    If nothing passes thresholds, fall back to argmax (never empty).
    """
    keep = [LABELS[i] for i in range(3) if probs[i] >= thresholds[i]]
    if not keep:
        keep = [LABELS[int(np.argmax(probs))]]
    return keep

def format_labelset(label_set: List[str]) -> str:
    return ",".join(label_set)

def clip_text(s: str, max_chars: int = 220) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 1].rstrip() + "…"

def confidence_tag(p_soft: np.ndarray) -> str:
    # very small heuristic tag; feel free to replace with calibration
    mx = float(np.max(p_soft))
    second = float(np.partition(p_soft, -2)[-2])
    margin = mx - second
    if mx >= 0.75 and margin >= 0.20:
        return "high"
    if mx >= 0.60 and margin >= 0.10:
        return "medium"
    return "low"
