"""Block-unlearning methods.

Every method receives the model *before* deletion and returns a model *after*.
Methods that start from scratch get the base seed, so their initialisation and
batch order match the original run as closely as the data allows.

    mask_only          party j deletes the rows and sends the null token. No
                       parameter change. Perfect locality, erases nothing that
                       was already absorbed into the weights. Lower bound.
    masked_retrain     retrain from scratch with the deleted blocks masked.
                       GOLD STANDARD: defines what "correctly forgotten" means.
    full_row_retrain   retrain without the deleted records at all. Over-deletes;
                       shows the cost of ignoring R3 (survival).
    row_ga             gradient ascent on the deleted records' full rows - the
                       current VFL sample-unlearning recipe (VFU-GA style).
    masked_finetune    a few epochs of joint training with the deletions masked.
                       The obvious cheap federated baseline.
    blockedit          OURS. Party j alone edits its encoder: map the deleted
                       blocks to a target, anchor its output on neighbours and a
                       random retain sample. Zero communication, no labels.
    blockedit_plus     blockedit, then a short calibration of the top model so
                       it predicts well for the now-partial records.
    *_nbr              the same, but the erasure target is the neighbours' mean
                       embedding instead of the null token (see README, "open
                       design decisions").
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors

from .data import DeletionSplit, VFLData, base_mask, deleted_mask
from .models import VFLModel
from .train import embed_all, fit_fresh, predict_logits, train_vfl
from .utils import CommCounter, Timer, clone_model, set_seed


@dataclass
class UnlearnResult:
    model: VFLModel
    seconds: float
    comm: CommCounter = field(default_factory=CommCounter)
    notes: dict = field(default_factory=dict)


@dataclass
class Context:
    before: VFLModel
    data: VFLData
    split: DeletionSplit
    cfg: dict
    seed: int

    @property
    def ucfg(self) -> dict:
        return self.cfg["unlearning"]

    def method_cfg(self, name: str) -> dict:
        return self.cfg.get("methods", {}).get(name, {})


# --------------------------------------------------------------------------- #
def mask_only(ctx: Context) -> UnlearnResult:
    with Timer() as t:
        model = clone_model(ctx.before)
    return UnlearnResult(model, t.seconds)


def masked_retrain(ctx: Context) -> UnlearnResult:
    comm = CommCounter()
    with Timer() as t:
        model = fit_fresh(ctx.data, ctx.data.train_idx, deleted_mask(ctx.data, ctx.split),
                          ctx.cfg, ctx.seed, comm=comm)
    return UnlearnResult(model, t.seconds, comm)


def full_row_retrain(ctx: Context) -> UnlearnResult:
    comm = CommCounter()
    keep = np.setdiff1d(ctx.data.train_idx, ctx.split.delete_idx)
    with Timer() as t:
        model = fit_fresh(ctx.data, keep, base_mask(ctx.data, ctx.split), ctx.cfg, ctx.seed, comm=comm)
    return UnlearnResult(model, t.seconds, comm)


def row_ga(ctx: Context) -> UnlearnResult:
    """Gradient ascent on the deleted records, optionally with a retain descent term
    (alpha > 0 gives the 'gradient difference' variant)."""
    c = {"epochs": 3, "lr": 5e-4, "clip": 1.0, "alpha": 0.0, "batch_size": 128} | ctx.method_cfg("row_ga")
    data, split = ctx.data, ctx.split
    model, comm = clone_model(ctx.before), CommCounter()
    mask = base_mask(data, split)
    opt = torch.optim.Adam(model.parameters(), lr=c["lr"])
    g = torch.Generator().manual_seed(ctx.seed + 1)
    del_t, pool_t = torch.as_tensor(split.delete_idx), torch.as_tensor(split.anchor_pool)
    bs = c["batch_size"]
    with Timer() as t:
        model.train()
        for _ in range(c["epochs"]):
            perm = del_t[torch.randperm(len(del_t), generator=g)]
            for s in range(0, len(perm), bs):
                b = perm[s:s + bs]
                xs, y = data.batch(b)
                loss = -F.binary_cross_entropy_with_logits(model(xs, mask[b]), y)
                if c["alpha"] > 0:
                    r = pool_t[torch.randint(len(pool_t), (len(b),), generator=g)]
                    xr, yr = data.batch(r)
                    loss = loss + c["alpha"] * F.binary_cross_entropy_with_logits(model(xr, mask[r]), yr)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), c["clip"])
                opt.step()
                comm.joint_batch(len(b) * (2 if c["alpha"] > 0 else 1), model.emb_dim, data.n_parties - 1)
        model.eval()
    return UnlearnResult(model, t.seconds, comm)


def masked_finetune(ctx: Context) -> UnlearnResult:
    c = {"epochs": 2, "lr": 5e-4} | ctx.method_cfg("masked_finetune")
    model, comm = clone_model(ctx.before), CommCounter()
    with Timer() as t:
        train_vfl(model, ctx.data, ctx.data.train_idx, deleted_mask(ctx.data, ctx.split), ctx.cfg["train"],
                  ctx.seed + 2, comm=comm, epochs=c["epochs"], lr=c["lr"])
    return UnlearnResult(model, t.seconds, comm)


# --------------------------------------------------------------------------- #
# BlockEdit
# --------------------------------------------------------------------------- #
def _blockedit_targets(ctx: Context, before_H: torch.Tensor, mode: str, k: int) -> torch.Tensor:
    """What the deleted blocks' embeddings should become.

    'null'       the missing-block token: the encoder refuses to represent them.
    'neighbors'  the mean before-embedding of the k nearest *other* retain blocks:
                 an estimate of how a model that never saw the record would embed it.
    """
    j, split, data = ctx.split.delete_party, ctx.split, ctx.data
    if mode == "null":
        return ctx.before.null_emb[j].detach().expand(len(split.delete_idx), -1).clone()
    if mode == "neighbors":
        Xj = data.X[j].numpy()
        nn = NearestNeighbors(n_neighbors=k).fit(Xj[split.retain_idx])
        _, ind = nn.kneighbors(Xj[split.delete_idx])      # positions within retain_idx
        return before_H[torch.as_tensor(ind)].mean(1)     # before_H is aligned with retain_idx
    raise ValueError(f"unknown blockedit target '{mode}'")


def blockedit(ctx: Context, calibrate_top: bool = False, target: str | None = None) -> UnlearnResult:
    c = {"steps": 300, "lr": 1e-3, "lam": 1.0, "target": "null", "n_random_anchor": 2000,
         "batch_anchor": 512, "target_k": 10} | ctx.method_cfg("blockedit")
    if target is not None:
        c["target"] = target
    data, split, j = ctx.data, ctx.split, ctx.split.delete_party
    model, comm = clone_model(ctx.before), CommCounter()
    f_j = model.bottoms[j]
    rng = np.random.default_rng(ctx.seed + 3)

    # Anchor set: neighbours (where damage concentrates) + random retain (coverage).
    # Drawn only from records that are NOT used to measure locality.
    pool = split.anchor_pool
    rand = rng.choice(pool, min(c["n_random_anchor"], len(pool)), replace=False)
    anchor_idx = np.union1d(split.anchor_neighbor_idx, rand)

    with Timer() as t:
        with torch.no_grad():
            X_j = data.X[j]
            anchor_x = X_j[torch.as_tensor(anchor_idx)]
            anchor_target = ctx.before.bottoms[j](anchor_x)
            retain_H = ctx.before.bottoms[j](X_j[torch.as_tensor(split.retain_idx)]) \
                if c["target"] == "neighbors" else None
            del_x = X_j[torch.as_tensor(split.delete_idx)]
            del_target = _blockedit_targets(ctx, retain_H, c["target"], c["target_k"])

        opt = torch.optim.Adam(f_j.parameters(), lr=c["lr"])
        g = torch.Generator().manual_seed(ctx.seed + 4)
        f_j.train()
        for _ in range(c["steps"]):
            a = torch.randint(len(anchor_idx), (c["batch_anchor"],), generator=g)
            erase = (f_j(del_x) - del_target).pow(2).sum(1).mean()
            keep = (f_j(anchor_x[a]) - anchor_target[a]).pow(2).sum(1).mean()
            loss = erase + c["lam"] * keep
            opt.zero_grad()
            loss.backward()
            opt.step()
        f_j.eval()

        if calibrate_top:
            _calibrate_top(ctx, model, comm)

    notes = {"final_erase_loss": float(erase.item()), "final_anchor_loss": float(keep.item())}
    return UnlearnResult(model, t.seconds, comm, notes)


def _calibrate_top(ctx: Context, model: VFLModel, comm: CommCounter) -> None:
    """Short top-model pass. Deleted records (now partial) are fit to their labels,
    exactly as masked retraining would; a retain sample is distilled towards the
    before-model's logits so the top model does not drift for everyone else.
    Passive parties send embeddings once (bottoms are frozen); no gradients return."""
    c = {"steps": 150, "lr": 5e-4, "n_retain": 2000, "batch_size": 256, "beta": 1.0} \
        | ctx.method_cfg("blockedit_plus")
    data, split = ctx.data, ctx.split
    rng = np.random.default_rng(ctx.seed + 5)
    pool = split.anchor_pool
    ret = rng.choice(pool, min(c["n_retain"], len(pool)), replace=False)
    mask = deleted_mask(data, split)
    with torch.no_grad():
        H_del = embed_all(model, data, split.delete_idx)
        H_ret = embed_all(model, data, ret)
        y_del = data.y[torch.as_tensor(split.delete_idx)]
        teacher = predict_logits(ctx.before, data, ret, mask)
        m_del, m_ret = mask[torch.as_tensor(split.delete_idx)], mask[torch.as_tensor(ret)]
    n_sent = len(split.delete_idx) + len(ret)
    comm.joint_batch(n_sent, model.emb_dim, data.n_parties - 1, backward=False)

    params = list(model.top.parameters()) + ([model.null_emb] if model.null_mode == "learned" else [])
    opt = torch.optim.Adam(params, lr=c["lr"])
    g = torch.Generator().manual_seed(ctx.seed + 6)
    bs = c["batch_size"]
    model.top.train()
    for _ in range(c["steps"]):
        a = torch.randint(len(H_del), (min(bs, len(H_del)),), generator=g)
        r = torch.randint(len(H_ret), (bs,), generator=g)
        fit = F.binary_cross_entropy_with_logits(model.fuse(H_del[a], m_del[a]), y_del[a])
        distil = (model.fuse(H_ret[r], m_ret[r]) - teacher[r]).pow(2).mean()
        loss = fit + c["beta"] * distil
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.top.eval()


def blockedit_plus(ctx: Context) -> UnlearnResult:
    return blockedit(ctx, calibrate_top=True)


def blockedit_nbr(ctx: Context) -> UnlearnResult:
    return blockedit(ctx, target="neighbors")


def blockedit_plus_nbr(ctx: Context) -> UnlearnResult:
    return blockedit(ctx, calibrate_top=True, target="neighbors")


METHODS = {
    "mask_only": mask_only,
    "masked_retrain": masked_retrain,
    "full_row_retrain": full_row_retrain,
    "row_ga": row_ga,
    "masked_finetune": masked_finetune,
    "blockedit": blockedit,
    "blockedit_plus": blockedit_plus,
    "blockedit_nbr": blockedit_nbr,
    "blockedit_plus_nbr": blockedit_plus_nbr,
}


def run_method(name: str, ctx: Context) -> UnlearnResult:
    if name not in METHODS:
        raise KeyError(f"unknown method '{name}'. Available: {list(METHODS)}")
    set_seed(ctx.seed + 100)
    return METHODS[name](ctx)
