from __future__ import annotations
import sys
from pathlib import Path as _Path
sys.path.append(str((_Path(__file__).resolve().parents[1] / "src")))
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from transformers import AutoTokenizer

from varierrnli.model_clf import AnnotatorAwareNLI
from varierrnli.explainer import Explainer, build_explainer_prompt
from varierrnli.prefix_explainer import PrefixExplainer
from varierrnli.retrieval import TrainOnlyRetriever
from varierrnli.utils import LABELS, probs_to_labelset, confidence_tag

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", required=True, help="Directory containing best.pt + tokenizer")
    ap.add_argument("--thresholds_json", required=True)
    ap.add_argument("--explainer_dir", required=True)
    ap.add_argument("--use_prefix_bridge", type=int, default=1, help="If 1 and explainer_dir contains bridge_config.json, condition explainer with CLF→prefix bridge")
    ap.add_argument("--retrieval_dir", default=None)
    ap.add_argument("--use_retrieval", type=int, default=0)
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--context", required=True)
    ap.add_argument("--statement", required=True)
    ap.add_argument("--id", default="ex_custom")
    args = ap.parse_args()

    ckpt = torch.load(Path(args.ckpt_dir) / "best.pt", map_location="cpu")
    annotators: List[str] = ckpt["annotators"]
    model_name = ckpt["model_name"]
    X_meta = ckpt.get("meta_features", None)
    meta_feat_dim = 0 if X_meta is None else len(X_meta[0])
    meta_features = None
    if X_meta is not None:
        meta_features = torch.tensor(np.array(X_meta, dtype=np.float32))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tok = AutoTokenizer.from_pretrained(args.ckpt_dir, use_fast=False)

    clf = AnnotatorAwareNLI(
        model_name=model_name,
        num_annotators=len(annotators),
        meta_feat_dim=meta_feat_dim,
    ).to(device)
    clf.load_state_dict(ckpt["model_state"])
    clf.eval()

    thr = json.load(open(args.thresholds_json, "r", encoding="utf-8"))
    thresholds = np.array([thr["tau_C"], thr["tau_E"], thr["tau_N"]], dtype=np.float32)

    # optional retrieval
    retriever = None
    retrieved_snips = []
    if args.use_retrieval and args.retrieval_dir:
        retriever = TrainOnlyRetriever.load(args.retrieval_dir)

    # encode
    enc = tok(
        args.context,
        args.statement,
        truncation=True,
        padding="max_length",
        max_length=args.max_length,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(device)
    attn = enc["attention_mask"].to(device)

    with torch.inference_mode():
        out = clf(input_ids=input_ids, attention_mask=attn, meta_features=meta_features)
        p_by = out.p_by_annotator[0].detach().cpu().numpy()   # [A,3]
        p_soft = out.p_soft[0].detach().cpu().numpy()         # [3]

    label_set_by_ann = {}
    p_by_ann = {}
    for i,a in enumerate(annotators):
        probs = p_by[i]
        label_set = probs_to_labelset(probs, thresholds)
        label_set_by_ann[a] = label_set
        p_by_ann[a] = {"C": float(probs[0]), "E": float(probs[1]), "N": float(probs[2])}

    # retrieval snippets
    if retriever is not None:
        # embed with sentence-transformers model used during indexing; stored in retrieval_info.json
        from sentence_transformers import SentenceTransformer
        info = json.load(open(Path(args.retrieval_dir) / "retrieval_info.json", "r", encoding="utf-8"))
        embed_model = SentenceTransformer(info["embed_model"])
        q = f"{args.context} [SEP] {args.statement}"
        q_emb = embed_model.encode([q], normalize_embeddings=True)[0]
        hits = retriever.search(q_emb, top_k=args.top_k)
        retrieved_snips = [h.snippet for h in hits if h.snippet]

    # If the explainer was trained with a prefix bridge, use it automatically (unless disabled).
    use_prefix = False
    if args.use_prefix_bridge:
        if (Path(args.explainer_dir) / "bridge_config.json").exists():
            use_prefix = True

    if use_prefix:
        explainer_prefix = PrefixExplainer.load(args.explainer_dir)
        explainer_plain = None
    else:
        explainer_prefix = None
        explainer_plain = Explainer.load(args.explainer_dir)

    explanations_by_ann = {}
    for i,a in enumerate(annotators):
        prompt_core = build_explainer_prompt(
            context=args.context,
            statement=args.statement,
            label_set=label_set_by_ann[a],
            p_triplet=np.array([p_by[i][0], p_by[i][1], p_by[i][2]], dtype=np.float32),
            retrieved_snippets=retrieved_snips if args.use_retrieval else None,
            mode="annotator",
            annotator_id=a,
            annotator_meta=None,
        )
        prompt = f"<ANN={a}>\n" + prompt_core

        if use_prefix:
            rep = out.rep_by_annotator[0, i].detach()  # [D]
            expl_text = explainer_prefix.generate_from_rep(prompt, rep)
        else:
            expl_text = explainer_plain.generate(prompt)
        explanations_by_ann[a] = {
            "label_set": label_set_by_ann[a],
            "explanation": expl_text,
        }

    group_label_set = probs_to_labelset(p_soft, thresholds)
    group_prompt_core = build_explainer_prompt(
        context=args.context,
        statement=args.statement,
        label_set=group_label_set,
        p_triplet=p_soft.astype(np.float32),
        retrieved_snippets=retrieved_snips if args.use_retrieval else None,
        mode="group",
    )
    group_prompt = "<ANN=GROUP>\n" + group_prompt_core
    if use_prefix:
        rep_g = out.rep_by_annotator[0].mean(dim=0).detach()
        explanation_group = explainer_prefix.generate_from_rep(group_prompt, rep_g)
    else:
        explanation_group = explainer_plain.generate(group_prompt)

    output = {
        "id": args.id,
        "p_soft": {"C": float(p_soft[0]), "E": float(p_soft[1]), "N": float(p_soft[2])},
        "p_by_annotator": p_by_ann,
        "label_set_by_annotator": label_set_by_ann,
        "explanations_by_annotator": explanations_by_ann,
        "explanation_group": explanation_group,
        "confidence": confidence_tag(p_soft),
    }

    print(json.dumps(output, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
