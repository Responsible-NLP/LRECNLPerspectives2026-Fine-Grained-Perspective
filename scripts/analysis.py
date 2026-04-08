"""analysis.py

Paper-ready analysis plots for VARI truthfulness/faithfulness evaluation.

Expected input:
- JSON produced by `scripts/eval_truthfulness.py --save_detailed_json ...`
  It contains:
    - aggregate (dict)
    - per_annotator (dict)
    - rows: list[dict]  # per (example, annotator)

This script saves figures as PNG into --out_dir.

Example:
python /scripts/eval_truthfulness.py ... \
  --save_detailed_json runs/eval_detailed.json --save_text 0

python scripts/analysis.py \
  --eval_json runs/eval_detailed.json --out_dir runs/figs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


def _safe_array(rows: List[Dict], key: str) -> np.ndarray:
    vals = []
    for r in rows:
        v = r.get(key, None)
        if v is None:
            continue
        try:
            vals.append(float(v))
        except Exception:
            continue
    return np.asarray(vals, dtype=float)


def _savefig(out_dir: Path, name: str, dpi: int = 220) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / name
    plt.savefig(p, dpi=dpi, bbox_inches="tight")


def plot_metric_distributions(rows: List[Dict], out_dir: Path) -> None:
    rouge = _safe_array(rows, "rougeL_f1")
    sem = _safe_array(rows, "semantic_sim")
    ent = _safe_array(rows, "nli_entail_prob")

    data = []
    labels = []
    if len(rouge):
        data.append(rouge)
        labels.append("ROUGE-L F1")
    if len(sem):
        data.append(sem)
        labels.append("Semantic sim")
    if len(ent):
        data.append(ent)
        labels.append("NLI entail")

    if not data:
        print("[analysis] No per-row metric data found; skipping distributions plot.")
        return

    plt.figure(figsize=(8, 4))
    plt.violinplot(data, showmeans=True, showextrema=True)
    plt.xticks(range(1, len(labels) + 1), labels, rotation=12, ha="right")
    plt.ylabel("Score")
    plt.title("Distribution of Faithfulness Metrics (per row)")
    _savefig(out_dir, "fig_metric_distributions.png")
    plt.close()


def plot_quadrant_scatter(rows: List[Dict], out_dir: Path, x_key: str, y_key: str,
                          x_thr: float = 0.30, y_thr: float = 0.75,
                          fname: str = "fig_quadrant_scatter.png") -> None:
    xs, ys = [], []
    for r in rows:
        x = r.get(x_key, None)
        y = r.get(y_key, None)
        if x is None or y is None:
            continue
        try:
            xs.append(float(x)); ys.append(float(y))
        except Exception:
            continue

    if not xs:
        print(f"[analysis] No data for scatter {x_key} vs {y_key}; skipping.")
        return

    x = np.asarray(xs, float)
    y = np.asarray(ys, float)

    plt.figure(figsize=(5.6, 5.2))
    plt.scatter(x, y, s=12, alpha=0.6)
    plt.axvline(x_thr, linestyle="--")
    plt.axhline(y_thr, linestyle="--")
    plt.xlabel(x_key)
    plt.ylabel(y_key)
    plt.title(f"{x_key} vs {y_key}")

    # annotate quadrant count (low rouge/high contextual)
    try:
        q = int(np.sum((x < x_thr) & (y > y_thr)))
        plt.text(0.02, 0.98, f"low-x/high-y: {q}", transform=plt.gca().transAxes,
                 va="top", ha="left")
    except Exception:
        pass

    _savefig(out_dir, fname)
    plt.close()


def plot_per_annotator_bars(rows: List[Dict], out_dir: Path, metric: str, min_n: int = 15) -> None:
    by_ann: Dict[str, List[float]] = {}
    for r in rows:
        a = str(r.get("annotator", ""))
        v = r.get(metric, None)
        if not a or v is None:
            continue
        try:
            by_ann.setdefault(a, []).append(float(v))
        except Exception:
            continue

    ann = []
    mean = []
    n = []
    for a, vals in by_ann.items():
        if len(vals) < min_n:
            continue
        ann.append(a)
        mean.append(float(np.mean(vals)))
        n.append(len(vals))

    if not ann:
        print(f"[analysis] Not enough per-annotator data for {metric}; skipping.")
        return

    order = np.argsort(np.asarray(mean))
    ann = [ann[i] for i in order]
    mean = [mean[i] for i in order]
    n = [n[i] for i in order]

    plt.figure(figsize=(10.5, 4.2))
    plt.bar(range(len(ann)), mean)
    plt.xticks(range(len(ann)), ann, rotation=60, ha="right")
    plt.ylabel(f"Mean {metric}")
    plt.title(f"Per-annotator {metric} (min_n={min_n})")

    # show n above bars lightly
    for i, (m, nn) in enumerate(zip(mean, n)):
        plt.text(i, m, str(nn), ha="center", va="bottom", fontsize=8)

    _savefig(out_dir, f"fig_per_annotator_{metric}.png")
    plt.close()


def plot_rouge_vs_sem_with_density_hint(rows: List[Dict], out_dir: Path) -> None:
    # A slightly nicer scatter: same as quadrant but uses ROUGE vs semantic
    plot_quadrant_scatter(
        rows,
        out_dir,
        x_key="rougeL_f1",
        y_key="semantic_sim",
        x_thr=0.30,
        y_thr=0.75,
        fname="fig_rouge_vs_sem.png",
    )


def plot_rouge_vs_nli(rows: List[Dict], out_dir: Path) -> None:
    plot_quadrant_scatter(
        rows,
        out_dir,
        x_key="rougeL_f1",
        y_key="nli_entail_prob",
        x_thr=0.30,
        y_thr=0.70,
        fname="fig_rouge_vs_nli.png",
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_json", required=True, help="Detailed eval JSON from eval_truthfulness.py --save_detailed_json")
    ap.add_argument("--out_dir", default="figs", help="Directory to save figures")
    ap.add_argument("--min_n_per_annotator", type=int, default=15)
    args = ap.parse_args()

    p = Path(args.eval_json)
    data = json.loads(p.read_text(encoding="utf-8"))

    rows = data.get("rows")
    if not rows or not isinstance(rows, list):
        raise SystemExit(
            "No 'rows' found. Re-run eval_truthfulness.py with --save_detailed_json <path>."
        )

    out_dir = Path(args.out_dir)

    # Core paper figures
    plot_metric_distributions(rows, out_dir)
    plot_rouge_vs_sem_with_density_hint(rows, out_dir)
    plot_rouge_vs_nli(rows, out_dir)

    # Annotator variability (paper-friendly)
    plot_per_annotator_bars(rows, out_dir, "nli_entail_prob", min_n=args.min_n_per_annotator)
    plot_per_annotator_bars(rows, out_dir, "semantic_sim", min_n=args.min_n_per_annotator)
    plot_per_annotator_bars(rows, out_dir, "rougeL_f1", min_n=args.min_n_per_annotator)

    print(f"[analysis] Saved figures to: {out_dir}")


if __name__ == "__main__":
    main()