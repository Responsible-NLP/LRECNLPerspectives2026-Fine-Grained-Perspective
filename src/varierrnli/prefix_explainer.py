from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class PrefixBridge(nn.Module):
    """Projects a classifier representation -> a sequence of prefix embeddings.

    We keep it intentionally small: a 2-layer MLP.
    """

    def __init__(
        self,
        in_dim: int,
        d_model: int,
        prefix_len: int = 16,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_dim = int(in_dim)
        self.d_model = int(d_model)
        self.prefix_len = int(prefix_len)

        self.net = nn.Sequential(
            nn.Linear(self.in_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.prefix_len * self.d_model),
        )

    def forward(self, rep: torch.Tensor) -> torch.Tensor:
        """rep: [B, in_dim] -> prefix: [B, prefix_len, d_model]"""
        x = self.net(rep)
        return x.view(rep.size(0), self.prefix_len, self.d_model)


@dataclass
class PrefixExplainer:
    """T5/Flan-T5 explainer conditioned by a learned prefix from the CLF representation."""

    tokenizer: AutoTokenizer
    model: AutoModelForSeq2SeqLM
    bridge: PrefixBridge
    device: torch.device

    @classmethod
    def load(cls, explainer_dir: str, device: Optional[str] = None):
        d = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        exp_dir = Path(explainer_dir)
        tok = AutoTokenizer.from_pretrained(exp_dir)
        mdl = AutoModelForSeq2SeqLM.from_pretrained(exp_dir)

        cfg_path = exp_dir / "bridge_config.json"
        if not cfg_path.exists():
            raise FileNotFoundError(
                f"{cfg_path} not found. This directory does not look like a prefix-conditioned explainer." \
                " Train with train_explainer_prefix.py."
            )

        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        bridge = PrefixBridge(
            in_dim=cfg["in_dim"],
            d_model=cfg["d_model"],
            prefix_len=cfg["prefix_len"],
            hidden_dim=cfg.get("hidden_dim", 512),
            dropout=cfg.get("dropout", 0.1),
        )
        bridge.load_state_dict(torch.load(exp_dir / "bridge.pt", map_location="cpu"))

        mdl.to(d)
        bridge.to(d)
        mdl.eval()
        bridge.eval()

        return cls(tokenizer=tok, model=mdl, bridge=bridge, device=d)

    @torch.inference_mode()
    def generate_from_rep(
        self,
        prompt: str,
        rep: torch.Tensor,  # [in_dim] or [1,in_dim]
        max_new_tokens: int = 128,
        num_beams: int = 4,
    ) -> str:
        """Generate explanation conditioned on a classifier representation."""

        if rep.dim() == 1:
            rep = rep.unsqueeze(0)
        rep = rep.to(self.device)

        enc = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        input_ids = enc["input_ids"].to(self.device)
        attn = enc["attention_mask"].to(self.device)

        # build prefix
        prefix = self.bridge(rep)  # [1,P,d_model]

        # concat with token embeddings
        token_emb = self.model.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([prefix, token_emb], dim=1)
        attn2 = torch.cat(
            [torch.ones((attn.size(0), prefix.size(1)), device=self.device, dtype=attn.dtype), attn],
            dim=1,
        )

        # generate via encoder_outputs (more robust than passing inputs_embeds directly into generate)
        encoder = self.model.get_encoder()
        encoder_outputs = encoder(inputs_embeds=inputs_embeds, attention_mask=attn2)

        out = self.model.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=attn2,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=False,
        )
        txt = self.tokenizer.decode(out[0], skip_special_tokens=True)
        return txt.strip()
