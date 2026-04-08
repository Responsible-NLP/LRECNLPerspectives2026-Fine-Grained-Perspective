from __future__ import annotations
import sys
from pathlib import Path as _Path
sys.path.append(str((_Path(__file__).resolve().parents[1] / "src")))
import argparse
import json
import os
from pathlib import Path

import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from varierrnli.data import load_varierrnli_json
from varierrnli.retrieval import build_retrieval_snippet
from varierrnli.utils import clip_text

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_json", required=True)
    ap.add_argument("--out_dir", default="runs/retrieval")
    ap.add_argument("--embed_model", default="sentence-transformers/all-MiniLM-L6-v2")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exs = load_varierrnli_json(args.train_json, include_explanations=True)
    model = SentenceTransformer(args.embed_model)

    texts = [f"{e.context} [SEP] {e.statement}" for e in exs]
    embs = model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    embs = np.asarray(embs, dtype=np.float32)

    import faiss
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)

    faiss.write_index(index, str(out_dir / "index.faiss"))

    with open(out_dir / "meta.jsonl", "w", encoding="utf-8") as f:
        for e in exs:
            snippet = ""
            if e.explanations_by_annotator:
                snippet = build_retrieval_snippet(e.explanations_by_annotator)
            rec = {
                "id": e.ex_id,
                "context": clip_text(e.context, 600),
                "statement": clip_text(e.statement, 300),
                "snippet": snippet,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    with open(out_dir / "retrieval_info.json", "w", encoding="utf-8") as f:
        json.dump({"embed_model": args.embed_model, "count": len(exs)}, f, indent=2)

    print(f"Saved retrieval index to: {out_dir}")

if __name__ == "__main__":
    main()
