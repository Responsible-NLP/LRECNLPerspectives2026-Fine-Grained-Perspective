from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import json
import os
import numpy as np

from .utils import clip_text

@dataclass
class Retrieved:
    ex_id: str
    score: float
    snippet: str
    context: str
    statement: str

class TrainOnlyRetriever:
    """FAISS-based retriever over the training set only."""
    def __init__(self, index, meta: List[Dict]):
        self.index = index
        self.meta = meta

    @classmethod
    def load(cls, retrieval_dir: str):
        import faiss  # local import
        index = faiss.read_index(os.path.join(retrieval_dir, "index.faiss"))
        meta_path = os.path.join(retrieval_dir, "meta.jsonl")
        meta = []
        with open(meta_path, "r", encoding="utf-8") as f:
            for line in f:
                meta.append(json.loads(line))
        return cls(index=index, meta=meta)

    def search(self, query_emb: np.ndarray, top_k: int = 5) -> List[Retrieved]:
        if query_emb.ndim == 1:
            query_emb = query_emb[None, :]
        D, I = self.index.search(query_emb.astype(np.float32), top_k)
        out = []
        for dist, idx in zip(D[0], I[0]):
            if idx < 0 or idx >= len(self.meta):
                continue
            m = self.meta[idx]
            out.append(Retrieved(
                ex_id=m["id"],
                score=float(dist),
                snippet=m["snippet"],
                context=m["context"],
                statement=m["statement"],
            ))
        return out

def build_retrieval_snippet(explanations_by_annotator: Dict[str, str], max_chars: int = 240) -> str:
    # join a few snippets from different annotators
    parts = []
    for a, txt in explanations_by_annotator.items():
        if not txt:
            continue
        # take first line / sentence-like chunk
        first = txt.strip().splitlines()[0]
        parts.append(clip_text(first, max_chars=max_chars))
        if len(parts) >= 3:
            break
    return " | ".join(parts) if parts else ""

