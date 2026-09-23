"""Run the full block-unlearning comparison for one config.

    python scripts/run_experiment.py --config configs/adult.yaml
    python scripts/run_experiment.py --config configs/adult.yaml --seeds 0 1 2 3 4
    python scripts/run_experiment.py --config configs/synthetic.yaml --methods mask_only blockedit

For each seed:
  1. build the deletion split (which blocks get erased, which form the control group)
  2. train the original model (cached under results/<name>/seed<k>/base.pt)
  3. train a second original model with another seed -> noise floor for locality
  4. run every requested method, starting from the original model
  5. evaluate each result against the original and against the masked retrain
Writes results/<name>/metrics_raw.csv and metrics_summary.csv.
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bvu.data import base_mask, load_dataset, make_deletion_split  # noqa: E402
from bvu.metrics import evaluate, prediction_drift  # noqa: E402
from bvu.models import build_model  # noqa: E402
from bvu.train import evaluate_utility, fit_fresh  # noqa: E402
from bvu.unlearn import METHODS, Context, run_method  # noqa: E402
from bvu.utils import CommCounter, load_config, save_json  # noqa: E402

DEFAULT_METHODS = ["mask_only", "masked_retrain", "full_row_retrain", "row_ga",
                   "masked_finetune", "blockedit", "blockedit_plus", "blockedit_nbr",
                   "blockedit_plus_nbr"]

SUMMARY_COLS = ["test_acc", "surv_acc", "surv_test_acc", "drift_retain", "drift_nbr", "nn_ratio",
                "erasure_auc", "gap_retrain_del", "seconds", "comm_rounds"]


def get_base_model(cfg, data, split, seed, run_dir, force=False):
    path = os.path.join(run_dir, "base.pt")
    mask = base_mask(data, split)
    if os.path.exists(path) and not force:
        model = build_model(data.in_dims, cfg["model"])
        model.load_state_dict(torch.load(path))
        model.eval()
        print(f"  loaded cached original model from {path}")
        return model, None
    print("  training original model")
    comm = CommCounter()
    model = fit_fresh(data, data.train_idx, mask, cfg, seed, comm=comm, verbose=True)
    os.makedirs(run_dir, exist_ok=True)
    torch.save(model.state_dict(), path)
    return model, comm


def run_seed(cfg, seed, methods, out_dir, force_base=False):
    print(f"\n=== {cfg['name']} | seed {seed} ===")
    data = load_dataset(cfg, seed)
    split = make_deletion_split(data, cfg, seed)
    print(data.describe())
    print(split.describe())

    run_dir = os.path.join(out_dir, f"seed{seed}")
    before, _ = get_base_model(cfg, data, split, seed, run_dir, force_base)
    u = evaluate_utility(before, data, split.test_idx)
    print(f"  original model: test acc {u['acc']:.4f}, AUC {u['auc']:.4f}")

    # Noise floor: how much do predictions move between two independent trainings?
    print("  training noise-floor model (different seed)")
    twin = fit_fresh(data, data.train_idx, base_mask(data, split), cfg, seed + 1000)
    noise_retain, _ = prediction_drift(twin, before, data, split.eval_retain_idx)

    ctx = Context(before, data, split, cfg, seed)
    results, reference = {}, None
    ordered = (["masked_retrain"] if "masked_retrain" in methods else []) + \
              [m for m in methods if m != "masked_retrain"]
    if "masked_retrain" not in methods:
        print("  (masked_retrain not requested: gap_retrain_* columns will be empty)")
    for name in ordered:
        print(f"  running {name} ...", end=" ", flush=True)
        res = run_method(name, ctx)
        if name == "masked_retrain":
            reference = res.model
        m = evaluate(res.model, before, data, split, reference=reference, seed=seed)
        m |= {"method": name, "seed": seed, "seconds": res.seconds, **res.comm.as_dict(),
              "noise_floor_retain": noise_retain, "dup_frac": split.dup_frac, **res.notes}
        results[name] = m
        print(f"done in {res.seconds:.1f}s  erasure_auc={m['erasure_auc']:.3f}  "
              f"drift_retain={m['drift_retain']:.4f}  surv_acc={m['surv_acc']:.3f}")
    save_json(results, os.path.join(run_dir, "metrics.json"))
    return list(results.values())


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    order = [m for m in DEFAULT_METHODS if m in df["method"].unique()] + \
            [m for m in df["method"].unique() if m not in DEFAULT_METHODS]
    cols = [c for c in SUMMARY_COLS if c in df.columns]
    g = df.groupby("method")[cols]
    mean, std = g.mean().loc[order], g.std().loc[order]
    if df["seed"].nunique() == 1:
        return mean
    return mean.round(4).astype(str) + " ± " + std.round(4).astype(str)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--methods", nargs="+", default=DEFAULT_METHODS, choices=list(METHODS))
    ap.add_argument("--out", default="results")
    ap.add_argument("--force-base", action="store_true", help="retrain the original model even if cached")
    args = ap.parse_args()

    cfg = load_config(args.config)
    torch.set_num_threads(cfg.get("threads", torch.get_num_threads()))
    out_dir = os.path.join(args.out, cfg["name"])
    rows = []
    for seed in args.seeds:
        rows += run_seed(cfg, seed, args.methods, out_dir, args.force_base)

    df = pd.DataFrame(rows)
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "metrics_raw.csv"), index=False)
    summary = summarise(df)
    summary.to_csv(os.path.join(out_dir, "metrics_summary.csv"))

    pd.set_option("display.width", 200, "display.max_columns", 20)
    print(f"\n=== summary over seeds {args.seeds} ===")
    print(summary.to_string(float_format=lambda v: f"{v:.4f}"))
    print(f"\nnoise floor (drift between two independent original models): "
          f"{df['noise_floor_retain'].mean():.4f}")
    print(f"results written to {out_dir}/")


if __name__ == "__main__":
    main()
