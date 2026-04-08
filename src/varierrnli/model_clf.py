from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig

from .utils import LABELS

@dataclass
class ClfOutput:
    p_by_annotator: torch.Tensor   # [B,A,3] probs
    p_soft: torch.Tensor           # [B,3] probs
    logits_by_annotator: torch.Tensor  # [B,A,3] logits

    # pooled encoder representation (before annotator/meta fusion)
    h: torch.Tensor                # [B,H]
    # per-annotator fused representation used by the classifier head
    # (used to condition the explainer via a prefix/soft-prompt bridge)
    rep_by_annotator: torch.Tensor # [B,A,D]

class AnnotatorAwareNLI(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_annotators: int,
        annotator_emb_dim: int = 64,
        meta_feat_dim: int = 0,
        meta_proj_dim: int = 32,
        mlp_hidden: int = 256,
        dropout: float = 0.1,
        freeze_encoder: bool = False,
    ):
        super().__init__()
        self.model_name = model_name
        self.num_annotators = num_annotators

        self.cfg = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        hidden = getattr(self.cfg, "hidden_size", None)
        if hidden is None:
            # e.g. some models use d_model
            hidden = getattr(self.cfg, "d_model")

        self.annotator_emb = nn.Embedding(num_annotators, annotator_emb_dim)

        self.use_meta = meta_feat_dim > 0
        self.meta_feat_dim = meta_feat_dim
        self.meta_proj = None
        if self.use_meta:
            self.meta_proj = nn.Sequential(
                nn.Linear(meta_feat_dim, meta_proj_dim),
                nn.Tanh()
            )
        in_dim = hidden + annotator_emb_dim + (meta_proj_dim if self.use_meta else 0)

        self.head = nn.Sequential(
            nn.Linear(in_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 3),
        )

    def _pool(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # Use CLS token embedding by default
        return last_hidden_state[:, 0]

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        meta_features: Optional[torch.Tensor] = None,   # [A,D] float
    ) -> ClfOutput:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last = out.last_hidden_state
        h = self._pool(last, attention_mask)  # [B,H]

        B = h.size(0)
        A = self.num_annotators
        device = h.device

        ann_ids = torch.arange(A, device=device)  # [A]
        e = self.annotator_emb(ann_ids)           # [A,E]
        e = e.unsqueeze(0).expand(B, A, -1)       # [B,A,E]
        h_rep = h.unsqueeze(1).expand(B, A, -1)   # [B,A,H]

        feats = [h_rep, e]
        if self.use_meta:
            if meta_features is None:
                raise ValueError("meta_features required when meta_feat_dim > 0")
            m = self.meta_proj(meta_features.to(device))  # [A,M]
            m = m.unsqueeze(0).expand(B, A, -1)           # [B,A,M]
            feats.append(m)

        z = torch.cat(feats, dim=-1)  # [B,A,*]
        logits = self.head(z)         # [B,A,3]
        probs = torch.sigmoid(logits) # [B,A,3]
        p_soft = probs.mean(dim=1)    # [B,3]
        return ClfOutput(
            p_by_annotator=probs,
            p_soft=p_soft,
            logits_by_annotator=logits,
            h=h,
            rep_by_annotator=z,
        )

def masked_bce_with_logits(
    logits: torch.Tensor,     # [B,A,3]
    targets: torch.Tensor,    # [B,A,3]
    mask: torch.Tensor,       # [B,A]
) -> torch.Tensor:
    # BCE per element
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")  # [B,A,3]
    # average across labels
    loss = loss.mean(dim=-1)  # [B,A]
    loss = loss * mask
    denom = mask.sum().clamp_min(1.0)
    return loss.sum() / denom


def masked_focal_bce_with_logits(
    logits: torch.Tensor,          # [B,A,3]
    targets: torch.Tensor,         # [B,A,3]
    mask: torch.Tensor,            # [B,A]
    pos_weight: Optional[torch.Tensor] = None,  # [3]
    gamma: float = 2.0,
) -> torch.Tensor:
    """Masked focal BCE for multi-label logits.

    Combines:
    - BCEWithLogits + pos_weight (class imbalance)
    - Focal factor (1 - p_t)^gamma (hard-example mining)
    - Masking for missing annotator labels
    """
    if pos_weight is not None:
        pos_weight = pos_weight.to(logits.device)

    bce = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none",
        pos_weight=pos_weight,
    )  # [B,A,3]

    p = torch.sigmoid(logits)
    p_t = p * targets + (1.0 - p) * (1.0 - targets)
    focal = (1.0 - p_t).clamp_min(1e-6).pow(float(gamma))

    loss = focal * bce
    loss = loss.mean(dim=-1)  # [B,A]
    loss = loss * mask
    denom = mask.sum().clamp_min(1.0)
    return loss.sum() / denom
