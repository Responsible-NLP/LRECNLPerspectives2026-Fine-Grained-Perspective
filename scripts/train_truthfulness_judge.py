from __future__ import annotations

import sys
from pathlib import Path as _Path
sys.path.append(str((_Path(__file__).resolve().parents[1] / "src")))

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from varierrnli.data import load_varierrnli_json
from varierrnli.model_clf import AnnotatorAwareNLI, masked_bce_with_logits
from varierrnli.metrics import multilabel_f1
from varierrnli.utils import seed_everything


def _safe_load_json_allow_trailing_commas(path: str) -> Dict:
    txt = Path(path).read_text(encoding="utf-8")
    txt = txt.replace(",}", "}").replace(",]", "]")
    return json.loads(txt)


@dataclass
class JudgeRecord:
    context: str
    statement_plus_expl: str
    ann_idx: int
    y: np.ndarray  # [3]


class JudgeDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        json_path: str,
        annotators: List[str],
        max_length: int,
        tok,
    ):
        self.tok = tok
        self.max_length = int(max_length)
        ann_to_idx = {a: i for i, a in enumerate(annotators)}

        exs = load_varierrnli_json(json_path, include_explanations=True)
        recs: List[JudgeRecord] = []
        for e in exs:
            if not e.explanations_by_annotator:
                continue
            for ann_id, y in e.y_by_annotator.items():
                if ann_id not in ann_to_idx:
                    continue
                expl = (e.explanations_by_annotator.get(ann_id) or "").strip()
                if not expl:
                    continue
                recs.append(
                    JudgeRecord(
                        context=e.context,
                        statement_plus_expl=e.statement + "\nExplanation: " + expl,
                        ann_idx=ann_to_idx[ann_id],
                        y=y.astype(np.float32),
                    )
                )
        self.recs = recs
        self.num_annotators = len(annotators)

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, idx: int):
        r = self.recs[idx]
        enc = self.tok(
            r.context,
            r.statement_plus_expl,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors=None,
        )

        A = self.num_annotators
        y_ann = np.zeros((A, 3), dtype=np.float32)
        mask = np.zeros((A,), dtype=np.float32)
        y_ann[r.ann_idx] = r.y
        mask[r.ann_idx] = 1.0

        return {
            "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
            "y_ann": torch.tensor(y_ann, dtype=torch.float32),
            "ann_mask": torch.tensor(mask, dtype=torch.float32),
        }


def eval_epoch(model, loader, meta_features: Optional[torch.Tensor], device: torch.device):
    model.eval()
    losses = []
    y_true = []
    y_prob = []
    with torch.inference_mode():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            y_ann = batch["y_ann"].to(device)
            mask = batch["ann_mask"].to(device)

            out = model(input_ids=input_ids, attention_mask=attn, meta_features=meta_features)
            loss = masked_bce_with_logits(out.logits_by_annotator, y_ann, mask)
            losses.append(float(loss.item()))

            # collect the active annotator row per sample
            # (since mask is 1 for exactly one annotator per sample)
            idx = mask.argmax(dim=1)  # [B]
            b_idx = torch.arange(input_ids.size(0), device=device)
            y_true.append(y_ann[b_idx, idx].detach().cpu().numpy())
            y_prob.append(out.p_by_annotator[b_idx, idx].detach().cpu().numpy())

    y_true = np.concatenate(y_true, axis=0) if y_true else np.zeros((0, 3), dtype=np.float32)
    y_prob = np.concatenate(y_prob, axis=0) if y_prob else np.zeros((0, 3), dtype=np.float32)
    y_pred = (y_prob >= 0.5).astype(np.int32)
    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        **multilabel_f1(y_true.astype(np.int32), y_pred.astype(np.int32)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_json", required=True)
    ap.add_argument("--dev_json", required=True)
    ap.add_argument("--annotator_meta_json", required=True)
    ap.add_argument("--model_name", default="microsoft/deberta-v3-base")
    ap.add_argument("--out_dir", default="runs/judge")
    ap.add_argument("--max_length", type=int, default=320)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.06)
    ap.add_argument("--annotator_emb_dim", type=int, default=64)
    ap.add_argument("--use_meta", type=int, default=1)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    seed_everything(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # annotators list from meta (prefer) else from data
    meta = _safe_load_json_allow_trailing_commas(args.annotator_meta_json)
    annotators = sorted(meta.keys())

    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    train_ds = JudgeDataset(args.train_json, annotators, args.max_length, tok)
    dev_ds = JudgeDataset(args.dev_json, annotators, args.max_length, tok)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False)

    # meta features (same encoding as main CLF)
    meta_features = None
    meta_info = None
    if args.use_meta:
        # reuse helper from varierrnli.data to build meta matrix
        from varierrnli.data import build_meta_feature_matrix

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
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * len(train_loader)
    warmup_steps = int(args.warmup_ratio * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best = {"dev_loss": 1e9, "epoch": -1}
    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"train judge epoch {epoch}")
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            y_ann = batch["y_ann"].to(device)
            mask = batch["ann_mask"].to(device)

            out = model(input_ids=input_ids, attention_mask=attn, meta_features=meta_features)
            loss = masked_bce_with_logits(out.logits_by_annotator, y_ann, mask)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            pbar.set_postfix({"loss": float(loss.item())})

        dev_metrics = eval_epoch(model, dev_loader, meta_features.to(device) if meta_features is not None else None, device)
        print(f"[epoch {epoch}] {json.dumps(dev_metrics, indent=2)}")

        ckpt = {
            "model_state": model.state_dict(),
            "model_name": args.model_name,
            "annotators": annotators,
            "meta_info": meta_info,
            "meta_features": meta_features.cpu().numpy().tolist() if meta_features is not None else None,
            "config": vars(args),
        }
        torch.save(ckpt, out_dir / "last.pt")
        if dev_metrics["loss"] < best["dev_loss"]:
            best = {"dev_loss": dev_metrics["loss"], "epoch": epoch}
            torch.save(ckpt, out_dir / "best.pt")

    tok.save_pretrained(out_dir)
    (out_dir / "train_summary.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
    print("Best:", best)


if __name__ == "__main__":
    main()
