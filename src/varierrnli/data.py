from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .utils import (
    LABELS, LABEL_TO_IDX, RAW_TO_CANON,
    labelset_str_to_multihot,
    soft_label_to_vec,
    safe_load_json,
    safe_load_annotator_meta_json,
)

@dataclass
class Example:
    ex_id: str
    context: str
    statement: str
    y_soft: np.ndarray              # [3] float
    y_by_annotator: Dict[str, np.ndarray]  # annotator -> [3] float (0/1)
    # For LLM training (optional)
    explanations_by_annotator: Optional[Dict[str, str]] = None

def _assign_explanations_to_labels(
    annotator_list: List[str],
    annotations_dict: Dict[str, str],
    explanations: List[str],
) -> Dict[str, List[Tuple[str, str]]]:
    """Map explanation strings to annotator labels.

    Dataset behavior we observe:
    - `annotations[AnnX]` can be multi-label like "contradiction,neutral"
    - `annotators` string can repeat AnnX once per label selected
    - `other_info['explanations']` has the same length as `annotators` list

    We assign occurrences of an annotator to the labels in their label-set in order.
    Returns: annotator -> list[(label_canon, explanation_text)]
    """
    # Prepare label queues per annotator
    queues: Dict[str, List[str]] = {}
    for a, labelset in annotations_dict.items():
        parts = [p.strip() for p in labelset.split(",") if p.strip()]
        queues[a] = [RAW_TO_CANON[p] for p in parts]

    out: Dict[str, List[Tuple[str, str]]] = {a: [] for a in annotations_dict.keys()}
    if len(annotator_list) != len(explanations):
        # fallback: cannot align reliably
        for a, labelset in annotations_dict.items():
            out[a].append((RAW_TO_CANON[labelset.split(',')[0].strip()], " ".join(explanations)))
        return out

    for a, expl in zip(annotator_list, explanations):
        if a not in queues or len(queues[a]) == 0:
            # unknown or already consumed
            continue
        lab = queues[a].pop(0)
        out.setdefault(a, []).append((lab, expl))
    return out

def load_varierrnli_json(
    path: str,
    include_explanations: bool = True,
) -> List[Example]:
    data = safe_load_json(path)  # dict keyed by id
    examples: List[Example] = []
    for ex_id, ex in data.items():
        text = ex["text"]
        context, statement = text["context"], text["statement"]
        y_soft = soft_label_to_vec(ex["soft_label"])

        y_by_ann: Dict[str, np.ndarray] = {}
        for ann_id, lab_str in ex["annotations"].items():
            y_by_ann[ann_id] = labelset_str_to_multihot(lab_str)

        exp_by_ann = None
        if include_explanations:
            expl_list = ex.get("other_info", {}).get("explanations", None)
            ann_str = ex.get("annotators", "")
            ann_list = [a.strip() for a in ann_str.split(",") if a.strip()]
            if isinstance(expl_list, list) and len(expl_list) > 0:
                mapped = _assign_explanations_to_labels(ann_list, ex["annotations"], expl_list)
                # Join per annotator explanations in a stable order
                exp_by_ann = {}
                for a, pairs in mapped.items():
                    # Keep original order; join with newlines if multiple
                    exp_by_ann[a] = "\n".join([p[1].strip() for p in pairs if p[1].strip()])
        examples.append(Example(
            ex_id=str(ex_id),
            context=context,
            statement=statement,
            y_soft=y_soft,
            y_by_annotator=y_by_ann,
            explanations_by_annotator=exp_by_ann
        ))
    return examples

class VariErrNLIDataset(Dataset):
    """Torch dataset for CLF training."""
    def __init__(
        self,
        examples: List[Example],
        tokenizer,
        annotators: List[str],
        max_length: int = 256,
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.annotators = annotators
        self.max_length = max_length
        self.ann_to_idx = {a: i for i, a in enumerate(annotators)}

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self.examples[idx]
        enc = self.tokenizer(
            ex.context,
            ex.statement,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors=None,
        )
        # Build y_ann and mask: [A,3], [A]
        A = len(self.annotators)
        y_ann = np.zeros((A, 3), dtype=np.float32)
        mask = np.zeros((A,), dtype=np.float32)
        for a, y in ex.y_by_annotator.items():
            if a in self.ann_to_idx:
                j = self.ann_to_idx[a]
                y_ann[j] = y
                mask[j] = 1.0
        item = {
            "id": ex.ex_id,
            "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
            "y_soft": torch.tensor(ex.y_soft, dtype=torch.float32),
            "y_ann": torch.tensor(y_ann, dtype=torch.float32),
            "ann_mask": torch.tensor(mask, dtype=torch.float32),
        }
        return item

def build_annotator_list(meta_json_path: str, train_examples: List[Example]) -> List[str]:
    # Prefer meta file keys; else union from train
    try:
        meta = safe_load_annotator_meta_json(meta_json_path)
        ann = sorted(meta.keys())
        if ann:
            return ann
    except Exception:
        pass
    # fallback
    s = set()
    for ex in train_examples:
        for a in ex.y_by_annotator.keys():
            s.add(a)
    return sorted(s)

def build_meta_feature_matrix(meta_json_path: str, annotators: List[str]) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Returns:
    - X_meta: [A, D] float32 (one-hot + age)
    - info: dictionaries with category vocabularies (for reproducibility)
    """
    meta = safe_load_annotator_meta_json(meta_json_path)
    genders = sorted({meta[a].get("Gender","Unknown") for a in meta})
    nats = sorted({meta[a].get("Nationality","Unknown") for a in meta})
    edus = sorted({meta[a].get("Education","Unknown") for a in meta})
    g2i = {g:i for i,g in enumerate(genders)}
    n2i = {n:i for i,n in enumerate(nats)}
    e2i = {e:i for i,e in enumerate(edus)}

    D = len(genders) + len(nats) + len(edus) + 1  # + normalized age
    X = np.zeros((len(annotators), D), dtype=np.float32)
    ages = []
    for a in annotators:
        ages.append(float(meta.get(a,{}).get("Age","0") or "0"))
    max_age = max(ages) if ages else 1.0
    min_age = min(ages) if ages else 0.0
    denom = max(max_age - min_age, 1.0)

    for i,a in enumerate(annotators):
        m = meta.get(a, {})
        g = m.get("Gender","Unknown")
        n = m.get("Nationality","Unknown")
        e = m.get("Education","Unknown")
        age = float(m.get("Age","0") or "0")
        X[i, g2i.get(g, 0)] = 1.0
        X[i, len(genders) + n2i.get(n, 0)] = 1.0
        X[i, len(genders) + len(nats) + e2i.get(e, 0)] = 1.0
        X[i, -1] = (age - min_age) / denom
    info = {"genders": genders, "nationalities": nats, "educations": edus}
    return X, info
