from __future__ import annotations
import sys
from pathlib import Path as _Path
sys.path.append(str((_Path(__file__).resolve().parents[1] / "src")))
import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from varierrnli.data import load_varierrnli_json, VariErrNLIDataset, build_meta_feature_matrix
from varierrnli.model_clf import AnnotatorAwareNLI
from varierrnli.utils import LABELS, probs_to_labelset, labelset_str_to_multihot, safe_load_annotator_meta_json

def jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if len(sa | sb) == 0:
        return 1.0
    return len(sa & sb) / len(sa | sb)

def multihot_to_labelset(y: np.ndarray) -> List[str]:
    out = []
    for i,l in enumerate(LABELS):
        if y[i] >= 0.5:
            out.append(l)
    if not out:
        out = [LABELS[int(np.argmax(y))]]
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", required=True, help="Directory with best.pt and tokenizer files")
    ap.add_argument("--dev_json", required=True)
    ap.add_argument("--annotator_meta_json", required=True)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--grid", type=int, default=31, help="grid points between 0.1 and 0.9")
    args = ap.parse_args()

    ckpt_path = Path(args.ckpt_dir) / "best.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu")
    annotators: List[str] = ckpt["annotators"]
    model_name = ckpt["model_name"]
    X_meta = ckpt.get("meta_features", None)
    meta_feat_dim = 0 if X_meta is None else len(X_meta[0])
    meta_features = None
    if X_meta is not None:
        meta_features = torch.tensor(np.array(X_meta, dtype=np.float32))

    tok = AutoTokenizer.from_pretrained(args.ckpt_dir, use_fast=False)
    dev_ex = load_varierrnli_json(args.dev_json, include_explanations=False)
    dev_ds = VariErrNLIDataset(dev_ex, tok, annotators, max_length=args.max_length)
    loader = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AnnotatorAwareNLI(
        model_name=model_name,
        num_annotators=len(annotators),
        meta_feat_dim=meta_feat_dim,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Collect predictions and gold
    probs_all = []     # [N,A,3]
    gold_all = []      # [N,A,3]
    mask_all = []      # [N,A]
    with torch.inference_mode():
        for batch in tqdm(loader, desc="predict dev"):
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            y_ann = batch["y_ann"].cpu().numpy()
            mask = batch["ann_mask"].cpu().numpy()

            out = model(input_ids=input_ids, attention_mask=attn, meta_features=meta_features)
            probs = out.p_by_annotator.detach().cpu().numpy()
            probs_all.append(probs)
            gold_all.append(y_ann)
            mask_all.append(mask)
    probs_all = np.concatenate(probs_all, axis=0)
    gold_all = np.concatenate(gold_all, axis=0)
    mask_all = np.concatenate(mask_all, axis=0)

    # grid search per-label thresholds to maximize mean Jaccard across labeled annotators
    grid = np.linspace(0.1, 0.9, args.grid, dtype=np.float32)
    best_tau = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    best_score = -1.0

    # Small brute-force (grid^3). With 31^3=29791 it's fine.
    for t0 in grid:
        for t1 in grid:
            for t2 in grid:
                tau = np.array([t0, t1, t2], dtype=np.float32)
                s = 0.0
                n = 0
                # iterate examples and annotators with mask==1
                pred_sets = (probs_all >= tau[None,None,:]).astype(np.float32)
                # fix empty sets by argmax
                for i in range(pred_sets.shape[0]):
                    for a in range(pred_sets.shape[1]):
                        if mask_all[i,a] < 0.5:
                            continue
                        y_pred = pred_sets[i,a]
                        if y_pred.sum() == 0:
                            y_pred[np.argmax(probs_all[i,a])] = 1.0
                        pred_ls = multihot_to_labelset(y_pred)
                        gold_ls = multihot_to_labelset(gold_all[i,a])
                        s += jaccard(pred_ls, gold_ls)
                        n += 1
                score = s / max(n,1)
                if score > best_score:
                    best_score = score
                    best_tau = tau.copy()

    out = {"tau_C": float(best_tau[0]), "tau_E": float(best_tau[1]), "tau_N": float(best_tau[2]), "dev_jaccard": float(best_score)}
    out_path = Path(args.ckpt_dir) / "thresholds.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("Best thresholds:", out)

if __name__ == "__main__":
    main()
