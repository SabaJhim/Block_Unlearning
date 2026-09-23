"""Training and prediction for the split-VFL model."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from .data import VFLData
from .models import VFLModel, build_model
from .utils import CommCounter, set_seed


def train_vfl(model: VFLModel, data: VFLData, idx: np.ndarray, mask: torch.Tensor, tcfg: dict,
              seed: int, comm: CommCounter | None = None, block_dropout: float = 0.0,
              epochs: int | None = None, lr: float | None = None, verbose: bool = False) -> VFLModel:
    """Standard joint VFL training on records `idx` under missing-block `mask`.

    block_dropout > 0 turns on "erasure-ready" training: each passive party's block
    is randomly replaced by the null token, so the top model learns to predict
    from partial records before any real erasure request ever arrives.
    """
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr or tcfg["lr"],
                           weight_decay=tcfg.get("weight_decay", 0.0))
    bs = tcfg["batch_size"]
    idx_t = torch.as_tensor(idx)
    n_epochs = epochs if epochs is not None else tcfg["epochs"]
    model.train()
    for ep in range(n_epochs):
        perm = idx_t[torch.randperm(len(idx_t), generator=g)]
        total = 0.0
        for s in range(0, len(perm), bs):
            b = perm[s:s + bs]
            xs, y = data.batch(b)
            m = mask[b]
            if block_dropout > 0:
                drop = torch.rand(m.shape, generator=g) < block_dropout
                drop[:, data.active_party] = False
                m = m | drop
            H = model.embed(xs)
            model.update_null_ema(H.detach(), m)
            loss = F.binary_cross_entropy_with_logits(model.fuse(H, m), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
            if comm is not None:
                comm.joint_batch(len(b), model.emb_dim, data.n_parties - 1)
        if verbose and (ep == 0 or (ep + 1) % max(1, n_epochs // 5) == 0):
            print(f"    epoch {ep + 1:3d}/{n_epochs}  train loss {total / len(perm):.4f}")
    model.eval()
    return model


@torch.no_grad()
def finalize_null(model: VFLModel, data: VFLData, idx: np.ndarray, mask: torch.Tensor) -> None:
    """In 'mean' mode, replace the running estimate by the exact mean embedding."""
    if model.null_mode != "mean":
        return
    H = embed_all(model, data, idx)
    m = mask[torch.as_tensor(idx)]
    for k in range(model.K):
        keep = ~m[:, k]
        if keep.any():
            model.null_emb[k] = H[keep, k].mean(0)


def fit_fresh(data: VFLData, idx: np.ndarray, mask: torch.Tensor, cfg: dict, seed: int,
              comm: CommCounter | None = None, verbose: bool = False) -> VFLModel:
    """Build a new model from `seed` and train it from scratch. Same seed -> same
    initialisation and same batch order, which keeps retraining comparisons honest."""
    set_seed(seed)
    model = build_model(data.in_dims, cfg["model"])
    train_vfl(model, data, idx, mask, cfg["train"], seed, comm=comm,
              block_dropout=cfg["train"].get("block_dropout", 0.0), verbose=verbose)
    finalize_null(model, data, idx, mask)
    return model


# --------------------------------------------------------------------------- #
# Prediction helpers
# --------------------------------------------------------------------------- #
@torch.no_grad()
def embed_all(model: VFLModel, data: VFLData, idx, bs: int = 8192) -> torch.Tensor:
    model.eval()
    idx = torch.as_tensor(idx)
    return torch.cat([model.embed(data.batch(idx[s:s + bs])[0]) for s in range(0, len(idx), bs)])


@torch.no_grad()
def predict_logits(model: VFLModel, data: VFLData, idx, mask: torch.Tensor | None,
                   bs: int = 8192) -> torch.Tensor:
    model.eval()
    idx = torch.as_tensor(idx)
    out = []
    for s in range(0, len(idx), bs):
        b = idx[s:s + bs]
        out.append(model(data.batch(b)[0], None if mask is None else mask[b]))
    return torch.cat(out)


def per_record_loss(model, data, idx, mask) -> np.ndarray:
    logits = predict_logits(model, data, idx, mask)
    y = data.y[torch.as_tensor(idx)]
    return F.binary_cross_entropy_with_logits(logits, y, reduction="none").numpy()


def evaluate_utility(model, data, idx, mask=None) -> dict:
    logits = predict_logits(model, data, idx, mask)
    y = data.y[torch.as_tensor(idx)].numpy()
    p = torch.sigmoid(logits).numpy()
    return {"acc": float(((p > 0.5) == y).mean()),
            "auc": float(roc_auc_score(y, p)) if 0 < y.mean() < 1 else float("nan"),
            "bce": float(F.binary_cross_entropy_with_logits(logits, torch.as_tensor(y)).item())}
