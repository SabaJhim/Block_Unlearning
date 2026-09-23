"""Null controls. Run these BEFORE trusting any number from run_experiment.py.

    python scripts/sanity_checks.py --config configs/adult.yaml

Checks
  1. pipeline null     control-vs-control erasure test on the original model ~ 0.5
                       (if not, the test is biased and every erasure number is meaningless)
  2. memorisation      original model: deleted-vs-control erasure test clearly > 0.5
                       (if not, there is nothing to erase and no method can look different)
  3. gold standard     masked retrain: deleted-vs-control ~ 0.5
                       (the definition of "forgotten" must actually pass its own test)
  4. mask_only         zero prediction drift on retain records (plumbing check)
  5. context           noise floor and duplicate-block rate, printed for interpretation

"~ 0.5" means inside a 3-sigma band for an AUC under the null, computed from the group sizes.
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bvu.data import base_mask, load_dataset, make_deletion_split  # noqa: E402
from bvu.metrics import erasure_metrics, pipeline_null_auc, prediction_drift  # noqa: E402
from bvu.train import evaluate_utility, fit_fresh  # noqa: E402
from bvu.unlearn import Context, run_method  # noqa: E402
from bvu.utils import load_config  # noqa: E402


def null_band(n1: int, n2: int, k: float = 3.0) -> float:
    return k * math.sqrt((n1 + n2 + 1) / (12 * n1 * n2))


def report(ok: bool | None, name: str, detail: str, advice: str = ""):
    tag = {True: "PASS", False: "FAIL", None: "WARN"}[ok]
    print(f"[{tag}] {name}: {detail}")
    if ok is not True and advice:
        print(f"       -> {advice}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    torch.set_num_threads(cfg.get("threads", torch.get_num_threads()))
    data = load_dataset(cfg, args.seed)
    split = make_deletion_split(data, cfg, args.seed)
    print(data.describe())
    print(split.describe())
    nd, nc = len(split.delete_idx), len(split.control_idx)

    print("\ntraining original model ...")
    before = fit_fresh(data, data.train_idx, base_mask(data, split), cfg, args.seed)
    u = evaluate_utility(before, data, split.test_idx)
    print(f"original model: test acc {u['acc']:.4f}, AUC {u['auc']:.4f}\n")

    # 1. pipeline null
    auc = pipeline_null_auc(before, data, split, args.seed)
    band = null_band(nc // 2, nc - nc // 2)
    report(abs(auc - 0.5) <= band, "pipeline null",
           f"control-vs-control AUC = {auc:.3f} (allowed 0.5 ± {band:.3f})",
           "the erasure test is biased; check that control blocks are masked in training")

    # 2. memorisation exists
    e_before = erasure_metrics(before, data, split)["erasure_auc"]
    band_dc = null_band(nd, nc)
    strong = e_before - 0.5 > 2 * band_dc
    report(True if strong else (None if e_before - 0.5 > band_dc else False), "memorisation",
           f"original model erasure AUC = {e_before:.3f} (needs to clear 0.5 + {band_dc:.3f})",
           "the model barely remembers individual blocks, so erasure cannot be measured. "
           "Increase capacity (emb_dim, hidden sizes), train longer, or set data.subsample_train "
           "to a smaller training set.")

    # 3. gold standard
    ctx = Context(before, data, split, cfg, args.seed)
    print("\ntraining masked retrain ...")
    retrain = run_method("masked_retrain", ctx).model
    e_retrain = erasure_metrics(retrain, data, split)["erasure_auc"]
    report(abs(e_retrain - 0.5) <= band_dc, "gold standard",
           f"masked retrain erasure AUC = {e_retrain:.3f} (allowed 0.5 ± {band_dc:.3f})",
           "retraining without the blocks should be indistinguishable from never seeing them; "
           "if it is not, the deleted and control groups differ systematically")

    # 4. mask_only plumbing
    mo = run_method("mask_only", ctx).model
    drift, _ = prediction_drift(mo, before, data, split.eval_retain_idx)
    report(drift == 0.0, "mask_only plumbing", f"retain drift = {drift:.2e} (must be exactly 0)")

    # 5. context
    print("\ntraining noise-floor model ...")
    twin = fit_fresh(data, data.train_idx, base_mask(data, split), cfg, args.seed + 1000)
    noise, flips = prediction_drift(twin, before, data, split.eval_retain_idx)
    print(f"[INFO] noise floor: mean |dp| between two independent originals = {noise:.4f} "
          f"({flips:.1%} of predictions flip)")
    print(f"[INFO] {split.dup_frac:.1%} of deleted blocks have an exact duplicate in retain")
    if split.dup_frac > 0.3:
        print("       -> many deleted blocks are identical to someone else's. An encoder cannot map "
              "one copy to a new value without moving the other, so encoder-only methods will show "
              "locality damage on those twins by necessity. See README, 'open design decisions'.")


if __name__ == "__main__":
    main()
