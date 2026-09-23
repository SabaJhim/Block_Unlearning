"""Fast invariant checks (about 20 s). Run with:  pytest -q

These test the *plumbing*, not the research claims: masks do what they say,
methods that claim zero communication really send nothing, the gold standard
is at zero distance from itself, and every metric is finite.
"""
import math
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bvu.data import base_mask, deleted_mask, load_dataset, make_deletion_split  # noqa: E402
from bvu.metrics import evaluate  # noqa: E402
from bvu.models import build_model  # noqa: E402
from bvu.train import fit_fresh  # noqa: E402
from bvu.unlearn import METHODS, Context, run_method  # noqa: E402

CFG = {
    "name": "tiny",
    "data": {"kind": "synthetic", "n": 1500, "n_parties": 3, "features_per_party": 6,
             "latent_dim": 4, "cross_party_corr": 0.3, "label_noise": 0.1, "active_party": 0},
    "model": {"emb_dim": 8, "bottom_hidden": [32], "top_hidden": [32], "null_token": "mean"},
    "train": {"epochs": 8, "batch_size": 64, "lr": 0.003},
    "unlearning": {"delete_party": "p1", "n_delete": 60, "n_control": 60, "n_neighbors": 3,
                   "n_eval_retain": 300},
    "methods": {"blockedit": {"steps": 30, "n_random_anchor": 200},
                "blockedit_plus": {"steps": 20, "n_retain": 200}},
}


@pytest.fixture(scope="module")
def setup():
    data = load_dataset(CFG, 0)
    split = make_deletion_split(data, CFG, 0)
    before = fit_fresh(data, data.train_idx, base_mask(data, split), CFG, 0)
    return data, split, before


def test_split_is_a_partition(setup):
    data, split, _ = setup
    parts = [split.delete_idx, split.control_idx, split.retain_idx]
    assert sum(len(p) for p in parts) == len(data.train_idx)
    assert len(np.intersect1d(split.delete_idx, split.control_idx)) == 0
    # evaluation-only records must never be available for anchoring
    assert len(np.intersect1d(split.anchor_pool, split.eval_retain_idx)) == 0
    assert len(np.intersect1d(split.anchor_pool, split.neighbor_idx)) == 0
    assert len(np.intersect1d(split.anchor_neighbor_idx, split.neighbor_idx)) == 0


def test_masks(setup):
    data, split, _ = setup
    mb, md = base_mask(data, split), deleted_mask(data, split)
    j = split.delete_party
    assert mb[:, j].sum() == len(split.control_idx)
    assert md[:, j].sum() == len(split.control_idx) + len(split.delete_idx)
    assert mb.sum() == mb[:, j].sum()          # nothing masked at other parties


def test_masked_block_sends_no_gradient(setup):
    """A masked record must not update the masked party's encoder."""
    data, split, _ = setup
    torch.manual_seed(0)
    model = build_model(data.in_dims, CFG["model"])
    idx = torch.as_tensor(split.delete_idx[:8])
    m = torch.zeros(len(idx), data.n_parties, dtype=torch.bool)
    m[:, split.delete_party] = True
    xs, y = data.batch(idx)
    torch.nn.functional.binary_cross_entropy_with_logits(model(xs, m), y).backward()
    g = [p.grad for p in model.bottoms[split.delete_party].parameters()]
    assert all(gi is None or torch.count_nonzero(gi) == 0 for gi in g)


@pytest.mark.parametrize("name", list(METHODS))
def test_every_method_runs_and_metrics_are_finite(setup, name):
    data, split, before = setup
    ctx = Context(before, data, split, CFG, 0)
    res = run_method(name, ctx)
    out = evaluate(res.model, before, data, split, seed=0)
    for k, v in out.items():
        assert math.isfinite(v), f"{name}: {k} = {v}"


def test_invariants(setup):
    data, split, before = setup
    ctx = Context(before, data, split, CFG, 0)
    mo = run_method("mask_only", ctx)
    assert evaluate(mo.model, before, data, split)["drift_retain"] == 0.0
    for name in ["mask_only", "blockedit", "blockedit_nbr"]:
        assert run_method(name, ctx).comm.rounds == 0, f"{name} must not communicate"
    ref = run_method("masked_retrain", ctx).model
    assert evaluate(ref, before, data, split, reference=ref)["gap_retrain_del"] == 0.0


def test_blockedit_only_touches_party_j(setup):
    data, split, before = setup
    after = run_method("blockedit", Context(before, data, split, CFG, 0)).model
    for k in range(data.n_parties):
        same = all(torch.equal(a, b) for a, b in zip(after.bottoms[k].parameters(),
                                                      before.bottoms[k].parameters()))
        assert same == (k != split.delete_party)
    assert all(torch.equal(a, b) for a, b in zip(after.top.parameters(), before.top.parameters()))
