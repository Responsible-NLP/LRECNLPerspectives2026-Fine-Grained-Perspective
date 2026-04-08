from __future__ import annotations

import sys
from pathlib import Path as _Path
sys.path.append(str((_Path(__file__).resolve().parents[1] / "src")))

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

from varierrnli.data import load_varierrnli_json
from varierrnli.explainer import Explainer, build_explainer_prompt
from varierrnli.model_clf import AnnotatorAwareNLI
from varierrnli.metrics import multilabel_f1
from varierrnli.prefix_explainer import PrefixExplainer
from varierrnli.utils import probs_to_labelset, seed_everything


LABELS = ["C", "E", "N"]


def _labels_to_multihot(label_set: List[str]) -> np.ndarray:
    y = np.zeros((3,), dtype=np.int32)
    for i, l in enumerate(LABELS):
        if l in label_set:
            y[i] = 1
    return y


def rouge_l_f1(pred: str, ref: str) -> float:
    """Token LCS-based ROUGE-L F1 (fast, no external deps)."""
    def _lcs(a: List[str], b: List[str]) -> int:
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


class SentenceEmbedder:
    """
    Lightweight sentence embedding (mean pooling over last hidden states).
    Default model works well for semantic similarity without requiring sentence-transformers.
    """
    def __init__(self, model_name: str, device: torch.device):
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()

    @torch.inference_mode()
    def encode(self, texts: List[str], max_length: int = 256) -> torch.Tensor:
        enc = self.tok(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(self.device)

        out = self.model(**enc)
        hid = out.last_hidden_state  # [B,T,H]
        mask = enc["attention_mask"].unsqueeze(-1)  # [B,T,1]
        hid = hid * mask
        denom = mask.sum(dim=1).clamp(min=1)
        emb = hid.sum(dim=1) / denom  # [B,H]
        emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
        return emb

    def cosine(self, a: str, b: str, max_length: int = 256) -> float:
        if not a.strip() or not b.strip():
            return float("nan")
        e = self.encode([a, b], max_length=max_length)
        return float((e[0] * e[1]).sum().item())


class NLIEntailmentScorer:
    """
    Contextual faithfulness via NLI: P(entailment | premise=context+statement, hypothesis=explanation)
    """
    def __init__(self, model_name: str, device: torch.device):
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
        self.model.eval()

        # map common MNLI label layouts to indices
        id2label = {int(k): v for k, v in getattr(self.model.config, "id2label", {}).items()} if getattr(self.model.config, "id2label", None) else {}
        # normalize strings
        norm = {i: str(v).lower() for i, v in id2label.items()}
        self.entail_idx = None
        self.contra_idx = None
        for i, v in norm.items():
            if "entail" in v:
                self.entail_idx = i
            if "contrad" in v:
                self.contra_idx = i

        # fallback to the common (contradiction, neutral, entailment)
        if self.entail_idx is None and self.model.config.num_labels == 3:
            self.entail_idx = 2
        if self.contra_idx is None and self.model.config.num_labels == 3:
            self.contra_idx = 0

    @torch.inference_mode()
    def score(self, premise: str, hypothesis: str, max_length: int = 384) -> Dict[str, float]:
        if not premise.strip() or not hypothesis.strip():
            return {"entail_prob": float("nan"), "contra_prob": float("nan")}
        enc = self.tok(
            premise,
            hypothesis,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        ).to(self.device)
        logits = self.model(**enc).logits[0]
        probs = torch.softmax(logits, dim=-1)
        ent = float(probs[self.entail_idx].item()) if self.entail_idx is not None else float("nan")
        con = float(probs[self.contra_idx].item()) if self.contra_idx is not None else float("nan")
        return {"entail_prob": ent, "contra_prob": con}


def _best_same_label_diff_expl_example(
    context: str,
    statement: str,
    ann_ids: List[str],
    label_sets: Dict[str, Tuple[str, ...]],
    expls: Dict[str, str],
    embedder: Optional[SentenceEmbedder],
    sim_threshold: float,
) -> Optional[Dict]:
    """
    Find a pair of annotators with the same label-set but very different explanations.
    Returns the lowest-similarity pair under sim_threshold (if found).
    """
    # group annotators by label-set
    groups: Dict[Tuple[str, ...], List[str]] = {}
    for a in ann_ids:
        ls = label_sets.get(a)
        if not ls:
            continue
        groups.setdefault(ls, []).append(a)

    best = None
    best_sim = 1e9
    for ls, members in groups.items():
        if len(members) < 2:
            continue
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                a1, a2 = members[i], members[j]
                e1, e2 = (expls.get(a1) or "").strip(), (expls.get(a2) or "").strip()
                if not e1 or not e2:
                    continue
                if embedder is not None:
                    sim = embedder.cosine(e1, e2)
                else:
                    sim = rouge_l_f1(e1, e2)
                if np.isnan(sim):
                    continue
                if sim < best_sim:
                    best_sim = sim
                    best = {
                        "context": context,
                        "statement": statement,
                        "label_set": list(ls),
                        "ann1": a1,
                        "ann2": a2,
                        "sim": float(sim),
                        "expl1": e1,
                        "expl2": e2,
                    }

    if best is not None and best_sim <= sim_threshold:
        return best
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev_json", required=True)
    ap.add_argument("--annotator_meta_json", required=True)
    ap.add_argument("--clf_ckpt_dir", required=True)
    ap.add_argument("--thresholds_json", required=True)
    ap.add_argument("--explainer_dir", required=True)
    ap.add_argument("--judge_ckpt_dir", required=True)

    ap.add_argument("--truth_target", choices=["gold", "clf"], default="gold",
                    help="What label-set the explanation should align with")
    ap.add_argument("--max_clf_len", type=int, default=256)
    ap.add_argument("--max_judge_len", type=int, default=320)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--num_beams", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=2026)

    # contextual / semantic metrics
    ap.add_argument("--embed_model", default="sentence-transformers/all-MiniLM-L6-v2",
                    help="Model for semantic similarity (set to '' to disable)")
    ap.add_argument("--embed_max_len", type=int, default=256)
    ap.add_argument("--nli_model", default="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
                    help="MNLI-style model for entailment-based faithfulness (set to '' to disable)")
    ap.add_argument("--nli_max_len", type=int, default=384)

    # analysis printing
    ap.add_argument("--print_example", type=int, default=1,
                    help="Print one 'same label, different explanation' example (1=yes, 0=no)")
    ap.add_argument("--example_sim_threshold", type=float, default=0.35,
                    help="Similarity threshold for selecting the printed example (lower = more different)")

    # Optional: dump per-(example,annotator) rows for notebook plots/paper figures.
    ap.add_argument(
        "--save_detailed_json",
        default="",
        help="If set, writes a JSON with an additional 'rows' field containing per-(example,annotator) metrics.",
    )
    ap.add_argument(
        "--save_text",
        type=int,
        default=0,
        help="If 1 and --save_detailed_json is set, include statement/context/generated/gold explanations in each row.",
    )
    ap.add_argument(
        "--max_saved_rows",
        type=int,
        default=0,
        help="If >0, cap number of saved rows (useful to keep files small).",
    )

    args = ap.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------------
    # Load dev data + meta
    # ---------------------
    dev_ex = load_varierrnli_json(args.dev_json, include_explanations=True)
    meta_txt = Path(args.annotator_meta_json).read_text(encoding="utf-8")
    meta_txt = meta_txt.replace(",}", "}").replace(",]", "]")
    annotator_meta = json.loads(meta_txt)

    # ---------------------
    # Optional metric models
    # ---------------------
    embedder = None
    if args.embed_model and args.embed_model.strip():
        embedder = SentenceEmbedder(args.embed_model.strip(), device)

    nli = None
    if args.nli_model and args.nli_model.strip():
        nli = NLIEntailmentScorer(args.nli_model.strip(), device)

    # ---------------------
    # Load CLF
    # ---------------------
    ckpt = torch.load(Path(args.clf_ckpt_dir) / "best.pt", map_location="cpu")
    annotators: List[str] = ckpt["annotators"]
    model_name = ckpt["model_name"]
    X_meta = ckpt.get("meta_features", None)
    meta_feat_dim = 0 if X_meta is None else len(X_meta[0])
    meta_features = None
    if X_meta is not None:
        meta_features = torch.tensor(np.array(X_meta, dtype=np.float32)).to(device)

    clf_tok = AutoTokenizer.from_pretrained(args.clf_ckpt_dir, use_fast=False)
    clf = AnnotatorAwareNLI(model_name=model_name, num_annotators=len(annotators), meta_feat_dim=meta_feat_dim).to(device)
    clf.load_state_dict(ckpt["model_state"], strict=True)
    clf.eval()

    thr = json.loads(Path(args.thresholds_json).read_text(encoding="utf-8"))
    thresholds = np.array([thr["tau_C"], thr["tau_E"], thr["tau_N"]], dtype=np.float32)

    # ---------------------
    # Load explainer
    # ---------------------
    use_prefix = (Path(args.explainer_dir) / "bridge_config.json").exists()
    if use_prefix:
        explainer_prefix = PrefixExplainer.load(args.explainer_dir)
        explainer_plain = None
    else:
        explainer_plain = Explainer.load(args.explainer_dir)
        explainer_prefix = None

    # ---------------------
    # Load judge
    # ---------------------
    j_ckpt = torch.load(Path(args.judge_ckpt_dir) / "best.pt", map_location="cpu")
    j_annotators: List[str] = j_ckpt["annotators"]
    j_model_name = j_ckpt["model_name"]
    X_meta_j = j_ckpt.get("meta_features", None)
    meta_feat_dim_j = 0 if X_meta_j is None else len(X_meta_j[0])
    meta_features_j = None
    if X_meta_j is not None:
        meta_features_j = torch.tensor(np.array(X_meta_j, dtype=np.float32)).to(device)

    judge_tok = AutoTokenizer.from_pretrained(args.judge_ckpt_dir, use_fast=True)
    judge = AnnotatorAwareNLI(model_name=j_model_name, num_annotators=len(j_annotators), meta_feat_dim=meta_feat_dim_j).to(device)
    judge.load_state_dict(j_ckpt["model_state"], strict=True)
    judge.eval()

    ann_to_idx_clf = {a: i for i, a in enumerate(annotators)}
    ann_to_idx_j = {a: i for i, a in enumerate(j_annotators)}

    # ---------------------
    # Evaluate (aggregate + per-annotator)
    # ---------------------
    agg_true, agg_pred = [], []
    agg_exact = []
    agg_rouge, agg_sem = [], []
    agg_entail, agg_contra = [], []

    rows: List[Dict] = []

    per = {
        ann: {
            "y_true": [],
            "y_pred": [],
            "exact": [],
            "rouge": [],
            "sem": [],
            "entail": [],
            "contra": [],
        }
        for ann in annotators
    }

    printed_example = None

    it = dev_ex[: args.limit] if args.limit and args.limit > 0 else dev_ex
    for ex in tqdm(it, desc="eval truthfulness"):
        # one CLF forward per example
        enc = clf_tok(
            ex.context,
            ex.statement,
            truncation=True,
            padding="max_length",
            max_length=args.max_clf_len,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)
        attn = enc["attention_mask"].to(device)

        with torch.inference_mode():
            out = clf(input_ids=input_ids, attention_mask=attn, meta_features=meta_features)
            p_by = out.p_by_annotator[0].detach().cpu().numpy()  # [A,3]

        # try to capture the requested analysis example (gold-only)
        if args.print_example and printed_example is None and ex.explanations_by_annotator:
            # label-set per annotator (gold)
            label_sets_gold = {}
            for ann_id, y_gold in ex.y_by_annotator.items():
                if ann_id not in ann_to_idx_clf:
                    continue
                ls = [l for i, l in enumerate(LABELS) if float(y_gold[i]) >= 0.5]
                if not ls:
                    ls = [LABELS[int(np.argmax(y_gold))]]
                label_sets_gold[ann_id] = tuple(sorted(ls))
            printed_example = _best_same_label_diff_expl_example(
                ex.context,
                ex.statement,
                list(ex.y_by_annotator.keys()),
                label_sets_gold,
                ex.explanations_by_annotator,
                embedder=embedder,
                sim_threshold=args.example_sim_threshold,
            )

        for ann_id, y_gold in ex.y_by_annotator.items():
            if ann_id not in ann_to_idx_clf or ann_id not in ann_to_idx_j:
                continue
            ann_idx = ann_to_idx_clf[ann_id]

            label_set_clf = probs_to_labelset(p_by[ann_idx], thresholds)
            label_set_gold = [l for i, l in enumerate(LABELS) if float(y_gold[i]) >= 0.5]
            if not label_set_gold:
                label_set_gold = [LABELS[int(np.argmax(y_gold))]]

            label_set_target = label_set_gold if args.truth_target == "gold" else label_set_clf

            prompt_core = build_explainer_prompt(
                context=ex.context,
                statement=ex.statement,
                label_set=label_set_target,
                p_triplet=p_by[ann_idx].astype(np.float32),
                retrieved_snippets=None,
                mode="annotator",
                annotator_id=ann_id,
                annotator_meta=annotator_meta,
            )
            prompt = f"<ANN={ann_id}>\n" + prompt_core

            if use_prefix:
                rep = out.rep_by_annotator[0, ann_idx].detach()
                gen = explainer_prefix.generate_from_rep(prompt, rep, max_new_tokens=args.max_new_tokens, num_beams=args.num_beams)
            else:
                gen = explainer_plain.generate(prompt, max_new_tokens=args.max_new_tokens, num_beams=args.num_beams)
            gen = (gen or "").strip()

            # compare to gold explanation (lexical + semantic)
            gold_expl = ""
            if ex.explanations_by_annotator:
                gold_expl = (ex.explanations_by_annotator.get(ann_id) or "").strip()

            rouge = float("nan")
            sem = float("nan")
            if gold_expl:
                rouge = rouge_l_f1(gen, gold_expl)
                if embedder is not None:
                    sem = embedder.cosine(gen, gold_expl, max_length=args.embed_max_len)

            # contextual faithfulness via entailment
            entail = float("nan")
            contra = float("nan")
            if nli is not None and gen:
                premise = ex.context.strip() + "\n\n" + ex.statement.strip()
                sc = nli.score(premise, gen, max_length=args.nli_max_len)
                entail, contra = sc["entail_prob"], sc["contra_prob"]

            # judge: predict label-set from (context, statement + generated explanation)
            st2 = ex.statement + "\nExplanation: " + gen
            j_enc = judge_tok(
                ex.context,
                st2,
                truncation=True,
                padding="max_length",
                max_length=args.max_judge_len,
                return_tensors="pt",
            )
            with torch.inference_mode():
                j_out = judge(
                    input_ids=j_enc["input_ids"].to(device),
                    attention_mask=j_enc["attention_mask"].to(device),
                    meta_features=meta_features_j,
                )
                j_probs = j_out.p_by_annotator[0, ann_to_idx_j[ann_id]].detach().cpu().numpy()
            j_pred = (j_probs >= 0.5).astype(np.int32)
            y_true = _labels_to_multihot(label_set_target)
            exm = int(np.all(j_pred == y_true))

            agg_true.append(y_true)
            agg_pred.append(j_pred)
            agg_exact.append(exm)
            if not np.isnan(rouge):
                agg_rouge.append(float(rouge))
            if not np.isnan(sem):
                agg_sem.append(float(sem))
            if not np.isnan(entail):
                agg_entail.append(float(entail))
            if not np.isnan(contra):
                agg_contra.append(float(contra))

            per[ann_id]["y_true"].append(y_true)
            per[ann_id]["y_pred"].append(j_pred)
            per[ann_id]["exact"].append(exm)
            if not np.isnan(rouge):
                per[ann_id]["rouge"].append(float(rouge))
            if not np.isnan(sem):
                per[ann_id]["sem"].append(float(sem))
            if not np.isnan(entail):
                per[ann_id]["entail"].append(float(entail))
            if not np.isnan(contra):
                per[ann_id]["contra"].append(float(contra))

            # Optional detailed rows for notebook analysis / paper plots.
            if args.save_detailed_json:
                if args.max_saved_rows <= 0 or len(rows) < args.max_saved_rows:
                    row = {
                        "example_id": getattr(ex, "example_id", None) or getattr(ex, "id", None) or None,
                        "annotator": ann_id,
                        "label_set_true": list(label_set_target),
                        "label_set_pred": [LABELS[i] for i in range(3) if int(j_pred[i]) == 1],
                        "exact_match": int(exm),
                        "rougeL_f1": float(rouge) if not np.isnan(rouge) else None,
                        "semantic_sim": float(sem) if not np.isnan(sem) else None,
                        "nli_entail_prob": float(entail) if not np.isnan(entail) else None,
                        "nli_contra_prob": float(contra) if not np.isnan(contra) else None,
                    }
                    if args.save_text:
                        row.update(
                            {
                                "statement": ex.statement,
                                "context": ex.context,
                                "gen_expl": gen,
                                "gold_expl": gold_expl,
                            }
                        )
                    rows.append(row)

    # ---------------------
    # Summaries
    # ---------------------
    y_true_all = np.stack(agg_true, axis=0) if agg_true else np.zeros((0, 3), dtype=np.int32)
    y_pred_all = np.stack(agg_pred, axis=0) if agg_pred else np.zeros((0, 3), dtype=np.int32)
    metrics = multilabel_f1(y_true_all, y_pred_all)
    metrics["exact_match"] = float(np.mean(agg_exact)) if agg_exact else float("nan")
    metrics["rougeL_f1"] = float(np.mean(agg_rouge)) if agg_rouge else float("nan")
    metrics["semantic_sim"] = float(np.mean(agg_sem)) if agg_sem else float("nan")
    metrics["nli_entail_prob"] = float(np.mean(agg_entail)) if agg_entail else float("nan")
    metrics["nli_contra_prob"] = float(np.mean(agg_contra)) if agg_contra else float("nan")

    per_metrics = {}
    for ann, d in per.items():
        if not d["y_true"]:
            continue
        yt = np.stack(d["y_true"], axis=0).astype(np.int32)
        yp = np.stack(d["y_pred"], axis=0).astype(np.int32)
        m = multilabel_f1(yt, yp)
        m["exact_match"] = float(np.mean(d["exact"])) if d["exact"] else float("nan")
        m["rougeL_f1"] = float(np.mean(d["rouge"])) if d["rouge"] else float("nan")
        m["semantic_sim"] = float(np.mean(d["sem"])) if d["sem"] else float("nan")
        m["nli_entail_prob"] = float(np.mean(d["entail"])) if d["entail"] else float("nan")
        m["nli_contra_prob"] = float(np.mean(d["contra"])) if d["contra"] else float("nan")
        m["n_samples"] = int(len(d["exact"]))
        per_metrics[ann] = m

    out = {
        "aggregate": metrics,
        "per_annotator": per_metrics,
        "rows": rows if args.save_detailed_json else None,
        "meta": {
            "truth_target": args.truth_target,
            "embed_model": args.embed_model if args.embed_model else None,
            "nli_model": args.nli_model if args.nli_model else None,
        },
    }

    if args.save_detailed_json:
        p = Path(args.save_detailed_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False)

    if args.print_example and printed_example is not None:
        print("\n" + "=" * 80)
        print("Example: two annotators with SAME label-set but DIFFERENT explanations")
        print("=" * 80)
        print(f"Label-set: {printed_example['label_set']}")
        print(f"Annotator 1: {printed_example['ann1']}")
        print(f"Annotator 2: {printed_example['ann2']}")
        print(f"Explanation similarity: {printed_example['sim']:.4f}  "
              f"({'cosine' if embedder is not None else 'rougeL_f1'})")
        print("\n[STATEMENT]\n" + printed_example["statement"])
        print("\n[CONTEXT]\n" + printed_example["context"])
        print("\n[EXPL 1]\n" + printed_example["expl1"])
        print("\n[EXPL 2]\n" + printed_example["expl2"])
        print("=" * 80 + "\n")

    # Always print a compact summary JSON (rows omitted unless not saving).
    out_print = dict(out)
    if args.save_detailed_json:
        out_print["rows"] = f"<saved to {args.save_detailed_json} ({len(rows)} rows)>"
    print(json.dumps(out_print, indent=2))


if __name__ == "__main__":
    main()