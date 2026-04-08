from __future__ import annotations
import sys
from pathlib import Path as _Path
sys.path.append(str((_Path(__file__).resolve().parents[1] / "src")))
import argparse
import json
import os
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from varierrnli.data import load_varierrnli_json, VariErrNLIDataset, build_annotator_list, build_meta_feature_matrix
from varierrnli.model_clf import AnnotatorAwareNLI, masked_bce_with_logits, masked_focal_bce_with_logits
from varierrnli.metrics import mse
from varierrnli.utils import seed_everything


def _compute_pos_weight(train_ex) -> torch.Tensor:
    """Compute pos_weight per class across all available annotator labels.

    pos_weight[c] = #neg / #pos
    """
    pos = np.zeros((3,), dtype=np.float64)
    neg = np.zeros((3,), dtype=np.float64)
    for e in train_ex:
        for _, y in e.y_by_annotator.items():
            y = np.asarray(y, dtype=np.float64)
            pos += y
            neg += (1.0 - y)
    eps = 1e-6
    pw = neg / (pos + eps)
    return torch.tensor(pw, dtype=torch.float32)

def eval_epoch(
    model,
    loader,
    meta_features,
    lambda_soft: float,
    device: torch.device,
    *,
    use_focal: bool,
    focal_gamma: float,
    pos_weight: torch.Tensor,
):
    model.eval()
    losses = []
    soft_mses = []
    with torch.inference_mode():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            y_soft = batch["y_soft"].to(device)
            y_ann = batch["y_ann"].to(device)
            mask = batch["ann_mask"].to(device)

            out = model(input_ids=input_ids, attention_mask=attn, meta_features=meta_features)
            if use_focal:
                l_ann = masked_focal_bce_with_logits(
                    out.logits_by_annotator,
                    y_ann,
                    mask,
                    pos_weight=pos_weight,
                    gamma=float(focal_gamma),
                )
            else:
                l_ann = masked_bce_with_logits(out.logits_by_annotator, y_ann, mask)
            l_soft = torch.nn.functional.binary_cross_entropy(out.p_soft, y_soft)
            loss = l_ann + lambda_soft * l_soft

            losses.append(float(loss.item()))
            soft_mses.append(mse(out.p_soft.detach().cpu().numpy(), y_soft.detach().cpu().numpy()))
    return float(np.mean(losses)), float(np.mean(soft_mses))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_json", required=True)
    ap.add_argument("--dev_json", required=True)
    ap.add_argument("--annotator_meta_json", required=True)
    ap.add_argument("--model_name", default="microsoft/deberta-v3-base")
    ap.add_argument("--out_dir", default="runs/clf")
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.06)
    ap.add_argument("--lambda_soft", type=float, default=1.0)
    ap.add_argument("--annotator_emb_dim", type=int, default=64)
    ap.add_argument("--use_meta", type=int, default=1)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--freeze_encoder", type=int, default=0)
    ap.add_argument("--use_focal", type=int, default=1, help="Use focal BCE (with pos_weight) for annotator loss")
    ap.add_argument("--focal_gamma", type=float, default=2.0)
    args = ap.parse_args()

    seed_everything(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ex = load_varierrnli_json(args.train_json, include_explanations=False)
    dev_ex = load_varierrnli_json(args.dev_json, include_explanations=False)

    # Imbalance handling: compute per-label pos_weight on train annotations
    pos_weight = _compute_pos_weight(train_ex)
    print("pos_weight (C,E,N):", [round(float(x), 4) for x in pos_weight.tolist()])

    annotators = build_annotator_list(args.annotator_meta_json, train_ex)

    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)

    train_ds = VariErrNLIDataset(train_ex, tok, annotators, max_length=args.max_length)
    dev_ds = VariErrNLIDataset(dev_ex, tok, annotators, max_length=args.max_length)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    dev_loader = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    meta_features = None
    meta_info = None
    X_meta = None
    if args.use_meta:
        X_meta, meta_info = build_meta_feature_matrix(args.annotator_meta_json, annotators)
        meta_features = torch.tensor(X_meta, dtype=torch.float32)
        meta_feat_dim = X_meta.shape[1]
    else:
        meta_feat_dim = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AnnotatorAwareNLI(
        model_name=args.model_name,
        num_annotators=len(annotators),
        annotator_emb_dim=args.annotator_emb_dim,
        meta_feat_dim=meta_feat_dim,
        freeze_encoder=bool(args.freeze_encoder),
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    total_steps = args.epochs * len(train_loader)
    warmup_steps = int(args.warmup_ratio * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best = {"dev_loss": 1e9, "epoch": -1}
    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"train epoch {epoch}")
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            y_soft = batch["y_soft"].to(device)
            y_ann = batch["y_ann"].to(device)
            mask = batch["ann_mask"].to(device)

            out = model(input_ids=input_ids, attention_mask=attn, meta_features=meta_features)
            if args.use_focal:
                l_ann = masked_focal_bce_with_logits(
                    out.logits_by_annotator,
                    y_ann,
                    mask,
                    pos_weight=pos_weight.to(device),
                    gamma=float(args.focal_gamma),
                )
            else:
                l_ann = masked_bce_with_logits(out.logits_by_annotator, y_ann, mask)
            l_soft = torch.nn.functional.binary_cross_entropy(out.p_soft, y_soft)
            loss = l_ann + args.lambda_soft * l_soft

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            pbar.set_postfix({"loss": float(loss.item()), "l_ann": float(l_ann.item()), "l_soft": float(l_soft.item())})

        dev_loss, dev_soft_mse = eval_epoch(
            model,
            dev_loader,
            meta_features,
            args.lambda_soft,
            device,
            use_focal=bool(args.use_focal),
            focal_gamma=float(args.focal_gamma),
            pos_weight=pos_weight.to(device),
        )
        print(f"[epoch {epoch}] dev_loss={dev_loss:.4f} dev_soft_mse={dev_soft_mse:.4f}")

        ckpt = {
            "model_state": model.state_dict(),
            "model_name": args.model_name,
            "annotators": annotators,
            "meta_info": meta_info,
            "meta_features": X_meta.tolist() if (meta_features is not None) else None,
            "pos_weight": pos_weight.tolist(),
            "config": vars(args),
        }
        torch.save(ckpt, out_dir / "last.pt")

        if dev_loss < best["dev_loss"]:
            best = {"dev_loss": dev_loss, "epoch": epoch}
            torch.save(ckpt, out_dir / "best.pt")

    print("Best:", best)
    # Save tokenizer
    tok.save_pretrained(out_dir)
    # Save best summary
    with open(out_dir / "train_summary.json", "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)

if __name__ == "__main__":
    main()
