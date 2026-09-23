"""Datasets, vertical party splits, and the block-deletion split.

Vocabulary used throughout the codebase
---------------------------------------
record i        one row, i.e. one person, aligned across all parties
party k         one organisation; holds a *block* of columns for every record
block (i, j)    record i's columns at party j  ->  x_i^(j)
mask M[i, k]    True  = party k's block for record i is missing, so the
                        model sees the missing-block token h_bot instead
                False = the real block is used

The deletion split partitions the training records into:
    delete   blocks (i, j) that receive an erasure request (the target)
    control  blocks (i, j) that were masked from the very start of training,
             i.e. party j never saw them. This is the "never seen" reference
             group that the erasure test compares against.
    retain   everyone else
    neighbor retain records closest to a deleted record in party j's feature
             space - where collateral damage is expected to concentrate.
"""
from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors


# --------------------------------------------------------------------------- #
# Containers
# --------------------------------------------------------------------------- #
@dataclass
class VFLData:
    X: list                    # list[FloatTensor [N, d_k]], one per party
    y: torch.Tensor            # FloatTensor [N] with values in {0, 1}
    party_names: list
    feature_names: list        # list[list[str]]
    active_party: int          # index of the party that holds the labels
    train_idx: np.ndarray
    test_idx: np.ndarray

    @property
    def n(self) -> int:
        return len(self.y)

    @property
    def n_parties(self) -> int:
        return len(self.X)

    @property
    def in_dims(self) -> list:
        return [x.shape[1] for x in self.X]

    def batch(self, idx):
        return [x[idx] for x in self.X], self.y[idx]

    def describe(self) -> str:
        lines = [f"{self.n} records, {self.n_parties} parties, "
                 f"{len(self.train_idx)} train / {len(self.test_idx)} test, "
                 f"positive rate {self.y.mean():.3f}"]
        for k, (name, d) in enumerate(zip(self.party_names, self.in_dims)):
            tag = "  (active, holds labels)" if k == self.active_party else ""
            lines.append(f"  party {k} '{name}': {d} input dims{tag}")
        return "\n".join(lines)


@dataclass
class DeletionSplit:
    delete_party: int
    delete_idx: np.ndarray
    control_idx: np.ndarray
    retain_idx: np.ndarray
    neighbor_idx: np.ndarray           # held-out neighbours: evaluation only
    test_idx: np.ndarray
    dup_frac: float            # share of deleted blocks with an exact duplicate in retain
    anchor_neighbor_idx: np.ndarray = None   # neighbours methods may anchor on
    eval_retain_idx: np.ndarray = None       # random retain records: evaluation only

    @property
    def anchor_pool(self) -> np.ndarray:
        """Retain records a method may train/anchor on without contaminating evaluation."""
        return np.setdiff1d(self.retain_idx, np.union1d(self.eval_retain_idx, self.neighbor_idx))

    def describe(self) -> str:
        return (f"delete party {self.delete_party}: {len(self.delete_idx)} deleted blocks, "
                f"{len(self.control_idx)} control blocks, {len(self.retain_idx)} retain "
                f"({len(self.eval_retain_idx)} held out for locality), "
                f"{len(self.neighbor_idx)}+{len(self.anchor_neighbor_idx)} eval/anchor neighbours; "
                f"{self.dup_frac:.1%} of deleted blocks have an exact duplicate in retain")


# --------------------------------------------------------------------------- #
# Masks
# --------------------------------------------------------------------------- #
def base_mask(data: VFLData, split: DeletionSplit) -> torch.Tensor:
    """Mask used to train the original model: only control blocks are missing."""
    m = torch.zeros(data.n, data.n_parties, dtype=torch.bool)
    m[torch.as_tensor(split.control_idx), split.delete_party] = True
    return m


def deleted_mask(data: VFLData, split: DeletionSplit) -> torch.Tensor:
    """Mask after the erasure: control blocks AND deleted blocks are missing.

    This is both the training mask of the gold-standard retrain and the
    inference mask after deletion (party j no longer holds the data, so it can
    only send the missing-block token for those records).
    """
    m = base_mask(data, split)
    m[torch.as_tensor(split.delete_idx), split.delete_party] = True
    return m


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #
def load_dataset(cfg: dict, seed: int) -> VFLData:
    kind = cfg["data"]["kind"]
    if kind == "synthetic":
        return load_synthetic(cfg["data"], seed)
    if kind == "tabular":
        return load_tabular(cfg["data"], seed)
    raise ValueError(f"unknown data.kind '{kind}'")


def _train_test_split(n: int, test_frac: float, seed: int):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_test = int(round(test_frac * n))
    return np.sort(perm[n_test:]), np.sort(perm[:n_test])


def load_tabular(d: dict, seed: int) -> VFLData:
    """Generic CSV loader driven entirely by the config's column lists."""
    if not os.path.exists(d["path"]):
        raise FileNotFoundError(
            f"{d['path']} not found. Run `python scripts/prepare_data.py <dataset>` first.")
    df = pd.read_csv(d["path"])
    y = df[d["label_col"]].astype(str).str.strip().isin(
        set(map(str, d["positive_labels"]))).astype(np.float32).values

    split_seed = d.get("split_seed", 0)       # data split fixed across experiment seeds
    train_idx, test_idx = _train_test_split(len(df), d.get("test_frac", 0.2), split_seed)
    if d.get("subsample_train"):
        rng = np.random.default_rng(split_seed + 1)
        train_idx = np.sort(rng.choice(train_idx, d["subsample_train"], replace=False))

    categorical = set(d.get("categorical", []))
    log1p = set(d.get("log1p", []))
    X_parts, fnames = [], []
    for pname, cols in d["parties"].items():
        blocks, names = [], []
        for c in cols:
            if c in categorical:
                dummies = pd.get_dummies(df[c].astype(str).str.strip(), prefix=c, dtype=np.float32)
                blocks.append(dummies.values)
                names += list(dummies.columns)
            else:
                v = pd.to_numeric(df[c], errors="coerce").astype(np.float64).values
                if c in log1p:
                    v = np.log1p(np.clip(v, 0, None))
                mu, sd = np.nanmean(v[train_idx]), np.nanstd(v[train_idx]) + 1e-8   # train stats only
                v = (np.where(np.isnan(v), mu, v) - mu) / sd
                blocks.append(v[:, None].astype(np.float32))
                names.append(c)
        X_parts.append(torch.tensor(np.concatenate(blocks, axis=1)))
        fnames.append(names)

    names = list(d["parties"].keys())
    return VFLData(X_parts, torch.tensor(y), names, fnames, names.index(d["active_party"]),
                   train_idx, test_idx)


def load_synthetic(d: dict, seed: int) -> VFLData:
    """Synthetic VFL data with a dial for how redundant the parties are.

    Each party k observes a latent z'_k = sqrt(rho) * z_shared + sqrt(1-rho) * z_k,
    projected into its own feature space. rho = cross_party_corr:
        rho -> 0  parties carry independent information (deleting a block hurts)
        rho -> 1  parties see the same underlying signal (others can compensate)
    The label depends on every party's latent, with label noise so that the
    model has something individual to memorise (and therefore to erase).
    """
    rng = np.random.default_rng(d.get("data_seed", 0))
    n, K = d["n"], d["n_parties"]
    L, F = d.get("latent_dim", 6), d["features_per_party"]
    rho = float(d["cross_party_corr"])

    z_shared = rng.normal(size=(n, L))
    logit = np.zeros(n)
    X_parts = []
    for _ in range(K):
        z = np.sqrt(rho) * z_shared + np.sqrt(1 - rho) * rng.normal(size=(n, L))
        A = rng.normal(size=(L, F)) / np.sqrt(L)
        x = np.tanh(z @ A) + d.get("feature_noise", 0.1) * rng.normal(size=(n, F))
        X_parts.append(torch.tensor(x, dtype=torch.float32))
        logit += z @ rng.normal(size=L)
    logit = d.get("signal_scale", 1.5) * logit / np.sqrt(K * L)
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(np.float32)
    flip = rng.random(n) < d.get("label_noise", 0.05)
    y[flip] = 1 - y[flip]

    train_idx, test_idx = _train_test_split(n, d.get("test_frac", 0.2), d.get("split_seed", 0))
    names = [f"p{k}" for k in range(K)]
    return VFLData(X_parts, torch.tensor(y), names, [[f"{nm}_f{i}" for i in range(F)] for nm in names],
                   d.get("active_party", 0), train_idx, test_idx)


# --------------------------------------------------------------------------- #
# Deletion split
# --------------------------------------------------------------------------- #
def make_deletion_split(data: VFLData, cfg: dict, seed: int) -> DeletionSplit:
    u = cfg["unlearning"]
    dp = u["delete_party"]
    j = data.party_names.index(dp) if isinstance(dp, str) else int(dp)

    rng = np.random.default_rng(seed + 17)
    tr = rng.permutation(data.train_idx)
    nd, nc = u["n_delete"], u["n_control"]
    if nd + nc >= len(tr):
        raise ValueError("n_delete + n_control must be smaller than the training set")
    delete_idx = np.sort(tr[:nd])
    control_idx = np.sort(tr[nd:nd + nc])
    retain_idx = np.sort(tr[nd + nc:])

    Xj = data.X[j].numpy()
    nn = NearestNeighbors(n_neighbors=u.get("n_neighbors", 5)).fit(Xj[retain_idx])
    dist, ind = nn.kneighbors(Xj[delete_idx])
    all_nbr = rng.permutation(np.unique(retain_idx[ind.ravel()]))
    half = len(all_nbr) // 2
    neighbor_idx, anchor_neighbor_idx = np.sort(all_nbr[:half]), np.sort(all_nbr[half:])
    dup_frac = float((dist[:, 0] < 1e-9).mean())

    others = np.setdiff1d(retain_idx, all_nbr)
    n_eval = min(u.get("n_eval_retain", 5000), len(others) // 2)
    eval_retain_idx = np.sort(rng.choice(others, n_eval, replace=False))

    return DeletionSplit(j, delete_idx, control_idx, retain_idx, neighbor_idx,
                         data.test_idx, dup_frac, anchor_neighbor_idx, eval_retain_idx)


# --------------------------------------------------------------------------- #
# Dataset preparation (download + clean into a single CSV)
# --------------------------------------------------------------------------- #
ADULT_COLUMNS = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
                 "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
                 "hours-per-week", "native-country", "income"]
ADULT_UCI = ["https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
             "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"]
ADULT_MIRROR = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/adult-all.csv"


def _fetch(url: str, dest: str) -> None:
    """Download to a temp name first, so a failed download never leaves a partial file."""
    print(f"  downloading {url}")
    tmp = dest + ".part"
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, dest)


def prepare_adult(raw_dir: str = "data/raw", out: str = "data/adult.csv") -> str:
    os.makedirs(raw_dir, exist_ok=True)
    frames = []
    try:
        for url in ADULT_UCI:
            dest = os.path.join(raw_dir, os.path.basename(url))
            if not os.path.exists(dest):
                _fetch(url, dest)
            frames.append(pd.read_csv(dest, header=None, names=ADULT_COLUMNS, skipinitialspace=True,
                                      comment="|", na_values="?"))
    except Exception as e:  # UCI occasionally unreachable -> GitHub mirror of the same 48,842 rows
        print(f"  UCI download failed ({e}); using mirror")
        dest = os.path.join(raw_dir, "adult-all.csv")
        if not os.path.exists(dest):
            _fetch(ADULT_MIRROR, dest)
        frames = [pd.read_csv(dest, header=None, names=ADULT_COLUMNS, skipinitialspace=True,
                              na_values="?")]
    df = pd.concat(frames, ignore_index=True)
    df["income"] = df["income"].astype(str).str.strip().str.rstrip(".")
    df = df.dropna(subset=["income"]).reset_index(drop=True)
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].fillna("missing")
    df.to_csv(out, index=False)
    print(f"  wrote {out}: {len(df)} rows, positive rate {(df['income'] == '>50K').mean():.3f}")
    return out


def prepare_credit_default(src: str, out: str = "data/credit_default.csv") -> str:
    """UCI 'default of credit card clients' (.xls, two header rows). Manual download."""
    df = pd.read_excel(src, header=1)
    df = df.rename(columns={"default payment next month": "default"}).drop(columns=["ID"])
    df.to_csv(out, index=False)
    print(f"  wrote {out}: {len(df)} rows")
    return out


def prepare_bank_marketing(src: str, out: str = "data/bank_marketing.csv") -> str:
    """UCI bank-additional-full.csv (semicolon separated). Manual download.
    `duration` is dropped: it is only known after the call and leaks the label."""
    df = pd.read_csv(src, sep=";").drop(columns=["duration"])
    df.to_csv(out, index=False)
    print(f"  wrote {out}: {len(df)} rows")
    return out
