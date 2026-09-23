"""Small shared utilities: seeding, timing, communication accounting, config IO."""
from __future__ import annotations

import copy
import json
import os
import random
import time
from dataclasses import dataclass

import numpy as np
import torch
import yaml


def set_seed(seed: int) -> None:
    """Seed every RNG we touch so that two runs with the same seed are identical."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("name", os.path.splitext(os.path.basename(path))[0])
    return cfg


def save_json(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=float)


def clone_model(model: torch.nn.Module) -> torch.nn.Module:
    return copy.deepcopy(model)


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self.t0


@dataclass
class CommCounter:
    """Counts cross-party traffic in split VFL.

    One *joint batch* = every passive party sends its embeddings up to the active
    party, and the active party sends embedding-gradients back down. Work done
    entirely inside one party (e.g. BlockEdit) never touches this counter, which
    is how we measure the "zero communication" claim.
    """

    rounds: int = 0          # number of joint forward/backward exchanges
    messages: int = 0        # individual party-to-party messages
    bytes: int = 0           # payload size, float32

    def joint_batch(self, batch_size: int, emb_dim: int, n_passive: int, backward: bool = True):
        per_msg = batch_size * emb_dim * 4
        n_msgs = n_passive * (2 if backward else 1)
        self.rounds += 1
        self.messages += n_msgs
        self.bytes += n_msgs * per_msg

    def as_dict(self) -> dict:
        return {"comm_rounds": self.rounds, "comm_messages": self.messages, "comm_MB": self.bytes / 1e6}
