"""Split-VFL model: one bottom encoder per party, one top model at the active party.

    party k:        x^(k)  --f_k-->  h_k            (runs inside party k)
    active party:   [h_1, ..., h_K]  --g-->  logit  (runs at the label holder)

A missing block is represented by a per-party token h_bot[k] that replaces h_k.
Because the replacement is done with torch.where, a masked record sends *no*
gradient into f_k - exactly as if party k did not hold that record's data.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def mlp(in_dim: int, hidden: list, out_dim: int) -> nn.Sequential:
    layers, d = [], in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU()]
        d = h
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


class VFLModel(nn.Module):
    def __init__(self, in_dims: list, emb_dim: int = 16, bottom_hidden=(64,), top_hidden=(64,),
                 null_token: str = "mean"):
        super().__init__()
        if null_token not in ("mean", "zero", "learned"):
            raise ValueError("null_token must be 'mean', 'zero' or 'learned'")
        self.K, self.emb_dim, self.null_mode = len(in_dims), emb_dim, null_token
        self.bottoms = nn.ModuleList([mlp(d, list(bottom_hidden), emb_dim) for d in in_dims])
        self.top = mlp(self.K * emb_dim, list(top_hidden), 1)
        if null_token == "learned":
            self.null_emb = nn.Parameter(torch.zeros(self.K, emb_dim))
        else:
            self.register_buffer("null_emb", torch.zeros(self.K, emb_dim))

    # --- the two halves of the split model ---------------------------------
    def embed(self, xs: list) -> torch.Tensor:
        """Bottom models. Returns H with shape [B, K, emb_dim]."""
        return torch.stack([f(x) for f, x in zip(self.bottoms, xs)], dim=1)

    def fuse(self, H: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Top model, after swapping missing blocks for the null token. Returns logits [B]."""
        if mask is not None:
            H = torch.where(mask.unsqueeze(-1), self.null_emb.unsqueeze(0).expand_as(H), H)
        return self.top(H.flatten(1)).squeeze(-1)

    def forward(self, xs: list, mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.fuse(self.embed(xs), mask)

    # --- null token maintenance (only used in 'mean' mode) -------------------
    @torch.no_grad()
    def update_null_ema(self, H: torch.Tensor, mask: torch.Tensor, momentum: float = 0.99):
        if self.null_mode != "mean":
            return
        for k in range(self.K):
            keep = ~mask[:, k]
            if keep.any():
                self.null_emb[k].mul_(momentum).add_((1 - momentum) * H[keep, k].mean(0))


def build_model(in_dims: list, mcfg: dict) -> VFLModel:
    return VFLModel(in_dims, emb_dim=mcfg.get("emb_dim", 16),
                    bottom_hidden=mcfg.get("bottom_hidden", [64]),
                    top_hidden=mcfg.get("top_hidden", [64]),
                    null_token=mcfg.get("null_token", "mean"))
