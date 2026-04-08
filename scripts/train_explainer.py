from __future__ import annotations
import sys
from pathlib import Path as _Path
sys.path.append(str((_Path(__file__).resolve().parents[1] / "src")))

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

from collections import Counter

import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer, Seq2SeqTrainingArguments,
)

from varierrnli.data import load_varierrnli_json
from varierrnli.explainer import build_explainer_prompt

LABEL_ORDER = ["C", "E", "N"]

def _safe_load_json_allow_trailing_commas(path: str) -> Dict:
    """Your annotator meta JSON may contain trailing commas; this makes it robust."""
    txt = Path(path).read_text(encoding="utf-8")
    # remove trailing commas before } or ]
    txt = txt.replace(",}", "}").replace(",]", "]")
    return json.loads(txt)

def build_records(
    json_path: str,
    annotator_meta: Dict[str, Dict[str, str]],
    include_group: bool = False,
) -> List[Dict]:
    exs = load_varierrnli_json(json_path, include_explanations=True)
    recs: List[Dict] = []

    for e in exs:
        if not e.explanations_by_annotator:
            continue

        for ann_id, y in e.y_by_annotator.items():
            # multi-label set from multi-hot y
            label_set: List[str] = [lab for i, lab in enumerate(LABEL_ORDER) if y[i] >= 0.5]
            if not label_set:
                label_set = [LABEL_ORDER[int(np.argmax(y))]]  # FIX: must be list

            label_key = "+".join(sorted(label_set))

            target = (e.explanations_by_annotator.get(ann_id) or "").strip()
            if not target:
                continue

            # Important: add annotator control token
            ann_token = f"<ANN={ann_id}>"

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

            prompt = ann_token + "\n" + prompt_core

            recs.append({
                "id": e.ex_id,
                "annotator": ann_id,
                "group_key": f"{ann_id}|{label_key}",
                "prompt": prompt,
                "target": target,
            })

        # Optional: group-level training records (only if you have group explanations in your data)
        if include_group and getattr(e, "group_explanation", None):
            group_target = (e.group_explanation or "").strip()
            if group_target:
                # group labels from soft label (or compute from y_soft)
                y_soft = np.array(e.y_soft, dtype=np.float32)
                group_label_set = [lab for i, lab in enumerate(LABEL_ORDER) if y_soft[i] >= 0.5]
                if not group_label_set:
                    group_label_set = [LABEL_ORDER[int(np.argmax(y_soft))]]

                prompt = build_explainer_prompt(
                    context=e.context,
                    statement=e.statement,
                    label_set=group_label_set,
                    p_triplet=None,
                    retrieved_snippets=None,
                    mode="group",
                    annotator_id=None,
                    annotator_meta=None,
                )
                recs.append({
                    "id": e.ex_id,
                    "annotator": "GROUP",
                    "prompt": prompt,
                    "target": group_target,
                })

    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_json", required=True)
    ap.add_argument("--dev_json", required=True)
    ap.add_argument("--annotator_meta_json", required=True)
    ap.add_argument("--model_name", default="google/flan-t5-base")
    ap.add_argument("--out_dir", default="runs/explainer")
    ap.add_argument("--max_source_len", type=int, default=512)
    ap.add_argument("--max_target_len", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--fp16", type=int, default=0)
    ap.add_argument("--include_group", type=int, default=0)
    ap.add_argument("--balance_sampling", type=int, default=1, help="Balance train records by (annotator,label_set) via upsampling")
    ap.add_argument("--balance_alpha", type=float, default=0.5, help="Weight = 1/(count**alpha). alpha=1 full, 0.5 mild")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    annotator_meta = _safe_load_json_allow_trailing_commas(args.annotator_meta_json)
    annotator_ids = sorted([k for k in annotator_meta.keys() if k.startswith("Ann")])

    train_recs = build_records(args.train_json, annotator_meta, include_group=bool(args.include_group))
    dev_recs   = build_records(args.dev_json,   annotator_meta, include_group=bool(args.include_group))

    if args.balance_sampling and train_recs:
        counts = Counter([r.get("group_key", "") for r in train_recs])
        w = np.array([1.0 / (counts[r.get("group_key", "")] ** float(args.balance_alpha)) for r in train_recs], dtype=np.float64)
        p = w / w.sum()
        rng = np.random.default_rng(int(args.seed))
        idx = rng.choice(len(train_recs), size=len(train_recs), replace=True, p=p)
        train_recs = [train_recs[i] for i in idx]
        print("Balanced upsampling enabled. Unique groups:", len(counts))

    ds_train = Dataset.from_list(train_recs)
    ds_dev   = Dataset.from_list(dev_recs)

    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    # Add annotator control tokens so the model can learn them
    ann_tokens = [f"<ANN={a}>" for a in annotator_ids]
    tok.add_special_tokens({"additional_special_tokens": ann_tokens})

    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    model.resize_token_embeddings(len(tok))
    model.config.use_cache = False

    def preprocess(batch):
        model_inputs = tok(batch["prompt"], max_length=args.max_source_len, truncation=True)

        # Modern target tokenization
        labels = tok(text_target=batch["target"], max_length=args.max_target_len, truncation=True)

        # Replace padding token id with -100 so it is ignored in loss
        label_ids = labels["input_ids"]
        pad_id = tok.pad_token_id
        label_ids = [
            [(tid if tid != pad_id else -100) for tid in seq]
            for seq in label_ids
        ]
        model_inputs["labels"] = label_ids
        return model_inputs

    ds_train = ds_train.map(preprocess, batched=True, remove_columns=ds_train.column_names)
    ds_dev   = ds_dev.map(preprocess, batched=True, remove_columns=ds_dev.column_names)

    collator = DataCollatorForSeq2Seq(tok, model=model)

    targs = Seq2SeqTrainingArguments(
        output_dir=str(out_dir),
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        predict_with_generate=True,
        logging_steps=25,
        fp16=bool(args.fp16),
        report_to=[],
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=targs,
        train_dataset=ds_train,
        eval_dataset=ds_dev,
        tokenizer=tok,
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))
    print(f"Saved explainer to: {out_dir}")

if __name__ == "__main__":
    main()
