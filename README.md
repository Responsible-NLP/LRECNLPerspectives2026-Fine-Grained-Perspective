# LRECNLPerspectives2026-Fine-Grained-Perspective
Code for LREC-NLPerspectives 5th Workshop publication "Fine-Grained Perspectives: Modeling Explanations with Annotator-Specific Rationales"

This repository contains the code for our paper submitted to the NLPerspectivist workshop at LREC 2026 (5th Workshop on Perspectivist Approaches to NLP) and implements the full pipeline described in our publication. If you use this repository, please cite our paper.

```
@inproceedings{sarumi-2026-Fine-Grained-perspective,
  title = {Fine-Grained Perspectives: Modeling Explanations with Annotator-Specific Rationales},
  author = {Sarumi, Olufunke and Welch, Charles and Braun, Daniel},
  booktitle = {Proceedings of the 5th Workshop on Perspectivist Approaches to NLP (NLPerspectives) @ LREC 2026},
  year = {2026},
}
```

## Running the Models

1. **CLF**: annotator-aware (User passport) multi-label classifier over {C,E,N}
2. ** the retrieval**: top-K similar training examples + explanation snippets
3. **LLM explainer**: generates **annotator** explanations (using the annotator's rationales)

## Data files
from [Learning With Disagreement 2025 (LeWiDi 2025)](https://www.codabench.org/competitions/7192/#/phases-tab)

- VariErrNLI_train.json
- VariErrNLI_dev.json
- VariErrNLI_annotators_meta.json (note: has trailing commas; code auto-fixes)

## Quickstart

You can run all scripts directly using the source directory.

```bash
PYTHONPATH=src python ...
```

### 1) Train the classifier (CLF)

The CLF uses **Focal BCE + pos_weight** to reduce class-imbalance (especially helpful when Neutral dominates).

```bash
python scripts/train_clf.py   --train_json ./dataset/VariErrNLI_train.json   --dev_json ./dataset/VariErrNLI_dev.json   --annotator_meta_json ./dataset/VariErrNLI_annotators_meta.json   --model_name microsoft/deberta-v3-base   --out_dir runs/clf_deberta  --epochs 50 --batch_size 32
```

### 2) Tune thresholds τ (maps probs → label-set)

```bash
python scripts/tune_thresholds.py   --ckpt_dir runs/clf_deberta   --dev_json ./dataset/VariErrNLI_dev.json   --annotator_meta_json ./dataset/VariErrNLI_annotators_meta.json
```
### 3) Build retrieval index (training set only)

```bash
python scripts/build_retrieval_index.py   --train_json ./dataset/VariErrNLI_train.json   --out_dir runs/retrieval
```

### 4a) Train the post-hoc explainer model (seq2seq; annotator-consistent)

```bash
python scripts/train_explainer.py   --train_json ./dataset/VariErrNLI_train.json   --dev_json   ./dataset/VariErrNLI_dev.json   --annotator_meta_json ./dataset/VariErrNLI_annotators_meta.json   --model_name google/flan-t5-base   --out_dir runs-experiment1/explainer_flan_t5 --batch_size 32 --epochs 50
```

### 4b) (Recommended) Train the explainer with _strong coupling_ to the classifier via a prefix bridge

This implements a **twin-encoder** architecture:
the CLF produces an internal representation per annotator, and a small MLP projects it into
`P` learned prefix embeddings that are prepended to the T5 encoder inputs.

```bash
python scripts/train_explainer_prefix.py   --train_json ./dataset/VariErrNLI_train.json --dev_json  ./dataset/VariErrNLI_dev.json   --annotator_meta_json ./dataset/VariErrNLI_annotators_meta.json   --clf_ckpt_dir runs/clf_deberta   --t5_model_name google/flan-t5-base   --out_dir runs/explainer_prefix_flan_t5   --prefix_len 16 --batch_size 32 --epochs 50
```

### 5) End-to-end inference (CLF → (Retrieval) → Explainer)

```bash
python scripts/infer.py   --ckpt_dir runs/clf_deberta   --thresholds_json runs/clf_deberta/thresholds.json   --explainer_dir runs/explainer_prefix_flan_t5   --retrieval_dir runs/retrieval   --use_retrieval 1   --top_k 5   --context "A person is in a kitchen, chopping vegetables on a cutting board."   --statement "The person is cooking dinner."
```

If you pass an explainer dir that contains bridge_config.json (trained via _4b_), the inference script automatically uses the prefix bridge.

### 6) Truthfulness evaluation for explanations (recommended)

### a) We train a small _judge_ model that predicts the label-set from (context, statement + explanation).
Then we score generated explanations by whether the judge recovers the intended label-set.

```bash
python scripts/train_truthfulness_judge.py   --train_json ./dataset/VariErrNLI_train.json   --dev_json   ./dataset/VariErrNLI_dev.json --annotator_meta_json ./dataset/VariErrNLI_annotators_meta.json --out_dir runs/judge_deberta
```

### b) Truthfulness Evaluation for Prefixed-bridged Explainer

```bash
python scripts/eval_truthfulness.py --dev_json ./dataset/VariErrNLI_test.json --annotator_meta_json ./dataset/VariErrNLI_annotators_meta.json --clf_ckpt_dir runs/clf_deberta --thresholds_json runs/clf_deberta/thresholds.json --explainer_dir runs/explainer_prefix_flan_t5 --judge_ckpt_dir runs/judge_deberta --save_detailed_json runs/eval_detailed.json --save_text 0
```

```bash
python scripts/analysis.py --eval_json runs/eval_detailed.json --out_dir runs/figs --min_n_per_annotator 15
```

### c) Truthfulness Evaluation for Post-hoc Explainer 

```bash
python scripts/eval_truthfulness.py
  --dev_json ./dataset/VariErrNLI_test.json
  --annotator_meta_json ./dataset/VariErrNLI_annotators_meta.json
  --clf_ckpt_dir runs/clf_deberta
  --thresholds_json runs/clf_deberta/thresholds.json
  --explainer_dir runs-experiment1/explainer_flan_t5
  --judge_ckpt_dir runs/judge_deberta
  --save_detailed_json runs-experiment1/eval_detailed.json
  --save_text 0
```

```bash
python scripts/analysis.py
  --eval_json runs-experiment1/eval_detailed.json
  --out_dir runs-experiment1/figs
  --min_n_per_annotator 15
```
