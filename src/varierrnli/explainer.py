from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from .utils import LABELS, format_labelset, clip_text

LABEL_LONG = {
    "C": "Contradiction (the statement conflicts with the context)",
    "E": "Entailment (the context supports the statement)",
    "N": "Neutral/Unknown (not enough info to entail or contradict)",
}

def _style_guidance_from_profile(profile: Optional[Dict[str, str]]) -> str:
    """
    Use ONLY non-sensitive, non-stereotyping guidance.
    We keep this conservative: only 'Education' affects level of technical detail.
    """
    if not profile:
        return "Write clearly and concisely."

    edu = (profile.get("Education") or "").lower()
    if "postdoc" in edu or "phd" in edu:
        return (
            "Write a precise, slightly more technical explanation. "
            "Be explicit about what is supported vs not supported."
        )
    if "master" in edu:
        return (
            "Write a clear explanation with moderate detail. "
            "Point out key phrases that justify the label-set."
        )
    return "Write a simple, clear explanation."

def build_explainer_prompt(
    context: str,
    statement: str,
    label_set: List[str],
    p_triplet: Optional[np.ndarray] = None,
    retrieved_snippets: Optional[List[str]] = None,
    mode: str = "annotator",              # or "group"
    annotator_id: Optional[str] = None,   # e.g., "Ann1"
    annotator_meta: Optional[Dict[str, Dict[str, str]]] = None,
) -> str:
    # label description
    label_desc = "; ".join([LABEL_LONG[l] for l in label_set])

    # probabilities
    p_line = ""
    if p_triplet is not None:
        p_line = (
            f"Model probabilities (C,E,N) = "
            f"({p_triplet[0]:.2f}, {p_triplet[1]:.2f}, {p_triplet[2]:.2f}).\n"
        )

    # retrieval
    ret_block = ""
    if retrieved_snippets:
        bullets = "\n".join(
            [f"- {clip_text(s, 240)}" for s in retrieved_snippets[:5] if s.strip()]
        )
        if bullets.strip():
            ret_block = (
                "Retrieved human rationale snippets (training set only):\n"
                f"{bullets}\n\n"
            )

    # annotator profile → style only (no stereotypes)
    profile = None
    if annotator_meta and annotator_id and annotator_id in annotator_meta:
        profile = annotator_meta[annotator_id]
    style_hint = _style_guidance_from_profile(profile)

    if mode == "group":
        task = (
            "Write ONE group-level explanation summarizing why the given label-set is acceptable. "
            "Be faithful to the context and do not introduce new facts."
        )
        persona = "Group summarizer."
    else:
        # key: annotator-specific, but without using demographics as “perspective”
        task = (
            "Write an explanation consistent with ONLY the given label-set. "
            "If multiple labels are in the set, explain why each is plausible and where ambiguity comes from. "
            "Do not argue for labels not in the set."
        )
        persona = f"Annotator: {annotator_id or 'Unknown'}."

    prompt = f"""Give your explanation as if you are {persona} Here are the details:


Labels:
- C: {LABEL_LONG['C']}
- E: {LABEL_LONG['E']}
- N: {LABEL_LONG['N']}

Rules:
- Be faithful to the provided context and statement; do not add new facts.
- Do NOT mention demographics (nationality/gender/age/etc.) and do NOT make cultural assumptions.
- Adjust *writing style only* based on expertise level when given.

Style guidance: {style_hint}

Task: {task}
Prefer 2–5 sentences. When helpful, quote a few short phrases from the context as evidence.

Context: {context}
Statement: {statement}
Required label-set: {format_labelset(label_set)}  ({label_desc})
{p_line}
{ret_block}
Explanation:"""
    return prompt

@dataclass
class Explainer:
    tokenizer: AutoTokenizer
    model: AutoModelForSeq2SeqLM
    device: torch.device

    @classmethod
    def load(cls, model_dir_or_name: str, device: Optional[str] = None):
        tok = AutoTokenizer.from_pretrained(model_dir_or_name)
        mdl = AutoModelForSeq2SeqLM.from_pretrained(model_dir_or_name)
        d = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        mdl.to(d)
        mdl.eval()
        return cls(tokenizer=tok, model=mdl, device=d)

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        num_beams: int = 4,
    ) -> str:
        enc = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        out = self.model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=False,
        )
        txt = self.tokenizer.decode(out[0], skip_special_tokens=True)
        return txt.strip()
