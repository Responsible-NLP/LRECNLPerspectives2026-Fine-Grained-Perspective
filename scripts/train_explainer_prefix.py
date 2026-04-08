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
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from collections import Counter
from tqdm import tqdm
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from varierrnli.data import load_varierrnli_json
from varierrnli.explainer import build_explainer_prompt
from varierrnli.model_clf import AnnotatorAwareNLI
from varierrnli.prefix_explainer import PrefixBridge
from varierrnli.utils import seed_everything


LABEL_ORDER = ["C", "E", "N"]


def _safe_load_json_allow_trailing_commas(path: str) -> Dict:
    txt = Path(path).read_text(encoding="utf-8")
    txt = txt.replace(",}", "}").replace(",]", "]")
    return json.loads(txt)


def _labels_to_set(y: np.ndarray) -> List[str]:
    labs = [lab for i, lab in enumerate(LABEL_ORDER) if float(y[i]) >= 0.5]
    if not labs:
        labs = [LABEL_ORDER[int(np.argmax(y))]]
    return labs


def rouge_l_f1(pred: str, ref: str) -> float:
    """Simple ROUGE-L F1 on whitespace tokens (fast + dependency-free)."""

    def _lcs(a: List[str], b: List[str]) -> int:
        # DP LCS length
        dp = [0] * (len(b) + 1)
        for i in range(1, len(a) + 1):
            prev = 0
            for j in range(1, len(b) + 1):
                cur = dp[j]
                if a[i - 1] == b[j - 1]:
                    dp[j] = prev + 1
                else:
                    dp[j] = max(dp[j], dp[j - 1])
                prev = cur
        return dp[-1]

    a = pred.strip().split()
    b = ref.strip().split()
    if not a or not b:
        return 0.0
    lcs = _lcs(a, b)
    prec = lcs / max(len(a), 1)
    rec = lcs / max(len(b), 1)
    if prec + rec == 0:
        return 0.0
    return float(2 * prec * rec / (prec + rec))


@dataclass
class Record:
    ex_id: str
    annotator: str
    ann_idx: int
    group_key: str
    context: str
    statement: str
    prompt: str
    target: str


def build_records(
    json_path: str,
    annotator_meta: Dict[str, Dict[str, str]],
    annotators_in_clf: List[str],
) -> List[Record]:
    exs = load_varierrnli_json(json_path, include_explanations=True)
    ann_to_idx = {a: i for i, a in enumerate(annotators_in_clf)}

    recs: List[Record] = []
    for e in exs:
        if not e.explanations_by_annotator:
            continue

        for ann_id, y in e.y_by_annotator.items():
            if ann_id not in ann_to_idx:
                continue
            target = (e.explanations_by_annotator.get(ann_id) or "").strip()
            if not target:
                continue

            label_set = _labels_to_set(y)
            label_key = "+".join(sorted(label_set))
            group_key = f"{ann_id}|{label_key}"
            prompt_core = build_explainer_prompt(
                context=e.context,
                statement=e.statement,
                label_set=label_set,
                p_triplet=None,
                retrieved_snippets=None,
                mode="annotator",
                annotator_id=ann_id,
                annotator_meta=annotator_meta,
            )
            prompt = f"<ANN={ann_id}>\n" + prompt_core

            recs.append(
                Record(
                    ex_id=e.ex_id,
                    annotator=ann_id,
                    ann_idx=ann_to_idx[ann_id],
                    group_key=group_key,
                    context=e.context,
                    statement=e.statement,
                    prompt=prompt,
                    target=target,
                )
            )
    return recs


class Collator:
    def __init__(
        self,
        t5_tok,
        clf_tok,
        max_source_len: int,
        max_target_len: int,
        max_clf_len: int,
    ):
        self.t5_tok = t5_tok
        self.clf_tok = clf_tok
        self.max_source_len = int(max_source_len)
        self.max_target_len = int(max_target_len)
        self.max_clf_len = int(max_clf_len)

    def __call__(self, batch: List[Record]) -> Dict[str, torch.Tensor]:
        prompts = [b.prompt for b in batch]
        targets = [b.target for b in batch]
        contexts = [b.context for b in batch]
        statements = [b.statement for b in batch]
        ann_idx = torch.tensor([b.ann_idx for b in batch], dtype=torch.long)

        t5_inp = self.t5_tok(
            prompts,
            max_length=self.max_source_len,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        with self.t5_tok.as_target_tokenizer():
            t5_lab = self.t5_tok(
                targets,
                max_length=self.max_target_len,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
        labels = t5_lab["input_ids"]
        labels[labels == self.t5_tok.pad_token_id] = -100

        clf_inp = self.clf_tok(
            contexts,
            statements,
            max_length=self.max_clf_len,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )

        return {
            "t5_input_ids": t5_inp["input_ids"],
            "t5_attention_mask": t5_inp["attention_mask"],
            "labels": labels,
            "clf_input_ids": clf_inp["input_ids"],
            "clf_attention_mask": clf_inp["attention_mask"],
            "ann_idx": ann_idx,
        }


class PrefixT5Model(nn.Module):
    def __init__(
        self,
        t5: AutoModelForSeq2SeqLM,
        clf: AnnotatorAwareNLI,
        bridge: PrefixBridge,
        meta_features: Optional[torch.Tensor],
        freeze_clf: bool = True,
    ):
        super().__init__()
        self.t5 = t5
        self.clf = clf
        self.bridge = bridge
        self.meta_features = meta_features

        if freeze_clf:
            for p in self.clf.parameters():
                p.requires_grad = False
            self.clf.eval()

    def forward(
        self,
        t5_input_ids: torch.Tensor,
        t5_attention_mask: torch.Tensor,
        clf_input_ids: torch.Tensor,
        clf_attention_mask: torch.Tensor,
        ann_idx: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ):
        device = t5_input_ids.device

        with torch.no_grad():
            out = self.clf(
                input_ids=clf_input_ids,
                attention_mask=clf_attention_mask,
                meta_features=self.meta_features,
            )
            # pick the fused per-annotator rep for each example
            # out.rep_by_annotator: [B,A,D]
            b_idx = torch.arange(clf_input_ids.size(0), device=device)
            rep = out.rep_by_annotator[b_idx, ann_idx.to(device)]  # [B,D]

        prefix = self.bridge(rep)  # [B,P,d_model]

        tok_emb = self.t5.get_input_embeddings()(t5_input_ids)
        inputs_embeds = torch.cat([prefix, tok_emb], dim=1)
        attn2 = torch.cat(
            [torch.ones((t5_attention_mask.size(0), prefix.size(1)), device=device, dtype=t5_attention_mask.dtype), t5_attention_mask],
            dim=1,
        )
        return self.t5(inputs_embeds=inputs_embeds, attention_mask=attn2, labels=labels)


def eval_epoch(model: PrefixT5Model, loader: DataLoader, t5_tok) -> Dict[str, float]:
    model.eval()
    losses = []
    rouges = []
    with torch.inference_mode():
        for batch in loader:
            for k in list(batch.keys()):
                if torch.is_tensor(batch[k]):
                    batch[k] = batch[k].to(next(model.parameters()).device)

            out = model(**batch)
            loss = out.loss
            losses.append(float(loss.item()))

            # quick ROUGE-L on greedy decode (cheap sanity check)
            # (we only do it for small batches / eval; still keep it light)
            if batch["t5_input_ids"].size(0) <= 8:
                # rebuild encoder inputs (prefix + embeds) via the model forward path
                # For simplicity: decode from model logits with argmax on each step is incorrect.
                # We'll use t5.generate conditioned on prefix (a few examples).
                pass

    return {"loss": float(np.mean(losses))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_json", required=True)
    ap.add_argument("--dev_json", required=True)
    ap.add_argument("--annotator_meta_json", required=True)
    ap.add_argument("--clf_ckpt_dir", required=True, help="Directory containing CLF best.pt + tokenizer")
    ap.add_argument("--t5_model_name", default="google/flan-t5-base")
    ap.add_argument("--out_dir", default="runs/explainer_prefix")
    ap.add_argument("--prefix_len", type=int, default=16)
    ap.add_argument("--bridge_hidden", type=int, default=512)
    ap.add_argument("--max_source_len", type=int, default=512)
    ap.add_argument("--max_target_len", type=int, default=128)
    ap.add_argument("--max_clf_len", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=8e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--warmup_ratio", type=float, default=0.06)
    ap.add_argument("--fp16", type=int, default=0)
    ap.add_argument("--balance_sampling", type=int, default=1, help="Balance batches by (annotator,label_set) using WeightedRandomSampler")
    ap.add_argument("--balance_alpha", type=float, default=0.5, help="Weight = 1/(count**alpha). alpha=1 full balance, 0.5 mild")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    seed_everything(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load CLF checkpoint (frozen)
    ckpt = torch.load(Path(args.clf_ckpt_dir) / "best.pt", map_location="cpu")
    annotators_in_clf: List[str] = ckpt["annotators"]
    clf_model_name = ckpt["model_name"]
    X_meta = ckpt.get("meta_features", None)
    meta_feat_dim = 0 if X_meta is None else len(X_meta[0])
    meta_features = None
    if X_meta is not None:
        meta_features = torch.tensor(np.array(X_meta, dtype=np.float32))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    clf_tok = AutoTokenizer.from_pretrained(args.clf_ckpt_dir, use_fast=False)
    clf = AnnotatorAwareNLI(
        model_name=clf_model_name,
        num_annotators=len(annotators_in_clf),
        meta_feat_dim=meta_feat_dim,
    ).to(device)
    clf.load_state_dict(ckpt["model_state"], strict=True)
    clf.eval()

    # Load annotator meta (used only for style hints in prompt)
    annotator_meta = _safe_load_json_allow_trailing_commas(args.annotator_meta_json)

    train_recs = build_records(args.train_json, annotator_meta, annotators_in_clf)
    dev_recs = build_records(args.dev_json, annotator_meta, annotators_in_clf)

    t5_tok = AutoTokenizer.from_pretrained(args.t5_model_name, use_fast=True)
    # add annotator control tokens (same as non-prefix explainer)
    ann_tokens = [f"<ANN={a}>" for a in annotators_in_clf]
    t5_tok.add_special_tokens({"additional_special_tokens": ann_tokens})

    t5 = AutoModelForSeq2SeqLM.from_pretrained(args.t5_model_name)
    t5.resize_token_embeddings(len(t5_tok))
    t5.config.use_cache = False
    t5.to(device)
    

    # Bridge in_dim is the classifier fused representation size (z in model_clf.py)
    # We can infer it by running one forward pass on a tiny batch.
    with torch.inference_mode():
        tmp = train_recs[0]
        tmp_enc = clf_tok(
            tmp.context,
            tmp.statement,
            truncation=True,
            padding="max_length",
            max_length=args.max_clf_len,
            return_tensors="pt",
        )
        tmp_out = clf(
            input_ids=tmp_enc["input_ids"].to(device),
            attention_mask=tmp_enc["attention_mask"].to(device),
            meta_features=meta_features,
        )
        in_dim = int(tmp_out.rep_by_annotator.size(-1))

    bridge = PrefixBridge(
        in_dim=in_dim,
        d_model=int(t5.config.d_model),
        prefix_len=args.prefix_len,
        hidden_dim=args.bridge_hidden,
        dropout=0.1,
    ).to(device)

    model = PrefixT5Model(
        t5=t5,
        clf=clf,
        bridge=bridge,
        meta_features=meta_features.to(device) if meta_features is not None else None,
        freeze_clf=True,
    ).to(device)

    collator = Collator(
        t5_tok=t5_tok,
        clf_tok=clf_tok,
        max_source_len=args.max_source_len,
        max_target_len=args.max_target_len,
        max_clf_len=args.max_clf_len,
    )
    # Balanced sampling by (annotator, label_set)
    if args.balance_sampling:
        counts = Counter([r.group_key for r in train_recs])
        weights = [1.0 / (counts[r.group_key] ** float(args.balance_alpha)) for r in train_recs]
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        train_loader = DataLoader(train_recs, batch_size=args.batch_size, sampler=sampler, shuffle=False, collate_fn=collator)
        print("Balanced sampling enabled. Unique groups:", len(counts))
    else:
        train_loader = DataLoader(train_recs, batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    dev_loader = DataLoader(dev_recs, batch_size=args.batch_size, shuffle=False, collate_fn=collator)

    # Optimizer: train T5 + bridge. (If you want bridge-only training, freeze t5 params.)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    total_steps = args.epochs * int(np.ceil(len(train_loader) / max(args.grad_accum, 1)))
    warmup_steps = int(args.warmup_ratio * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.fp16) and device.type == "cuda")

    best = {"dev_loss": 1e9, "epoch": -1}
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        pbar = tqdm(train_loader, desc=f"train prefix explainer epoch {epoch}")
        step = 0
        for batch in pbar:
            for k in list(batch.keys()):
                if torch.is_tensor(batch[k]):
                    batch[k] = batch[k].to(device)

            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                out = model(**batch)
                loss = out.loss / max(args.grad_accum, 1)

            scaler.scale(loss).backward()

            if (step + 1) % max(args.grad_accum, 1) == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            pbar.set_postfix({"loss": float(loss.item()) * max(args.grad_accum, 1)})
            step += 1

        # eval loss
        model.eval()
        dev_losses = []
        with torch.inference_mode():
            for batch in dev_loader:
                for k in list(batch.keys()):
                    if torch.is_tensor(batch[k]):
                        batch[k] = batch[k].to(device)
                out = model(**batch)
                dev_losses.append(float(out.loss.item()))
        dev_loss = float(np.mean(dev_losses)) if dev_losses else float("nan")
        print(f"[epoch {epoch}] dev_loss={dev_loss:.4f}")

        # save last
        t5.save_pretrained(out_dir)
        t5_tok.save_pretrained(out_dir)
        torch.save(bridge.state_dict(), out_dir / "bridge.pt")
        cfg = {
            "in_dim": in_dim,
            "d_model": int(t5.config.d_model),
            "prefix_len": int(args.prefix_len),
            "hidden_dim": int(args.bridge_hidden),
            "dropout": 0.1,
            "clf_ckpt_dir": str(args.clf_ckpt_dir),
            "clf_model_name": clf_model_name,
            "annotators": annotators_in_clf,
        }
        (out_dir / "bridge_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

        if dev_loss < best["dev_loss"]:
            best = {"dev_loss": dev_loss, "epoch": epoch}
            # keep an explicit best copy
            torch.save(bridge.state_dict(), out_dir / "bridge_best.pt")

    (out_dir / "train_summary.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
    print("Best:", best)


if __name__ == "__main__":
    main()
