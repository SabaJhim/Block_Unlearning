"""Evaluation for block unlearning: the three requirements plus utility and cost.

R1  ERASURE  (counterfactual reinsertion test)
    Hand the model the deleted block back and watch how it reacts. The model's
    *reaction* to a block is the reinsertion gain:
        gain_i = loss(record i with block j masked) - loss(record i with block j present)
    If the block was truly forgotten, the model should react to it exactly as it
    reacts to a block it never saw. The control group supplies those never-seen
    blocks. erasure_auc = AUC separating deleted gains from control gains.
        0.5   indistinguishable from never-seen            (the target)
        > 0.5 the model still recognises the deleted blocks (under-forgetting)
        < 0.5 the deleted blocks are now *less* familiar than never-seen ones -
              over-forgetting, itself a detectable fingerprint (Streisand effect)
    erasure_auc_loss is the same test on raw reinsertion loss instead of gain.

R2  LOCALITY
    Prediction drift |p_after - p_before| on records that were NOT deleted and that
    no method was allowed to anchor on: a held-out random retain sample, the
    held-out half of the deleted records' nearest neighbours in party j's feature
    space, and the test set. nn_ratio > 1 means damage concentrates near
    the deleted records. Compare against noise_floor (drift between two base
    models trained with different seeds) to judge what "small" means.

R3  SURVIVAL
    surv_acc: accuracy on the deleted records, evaluated as partial records
    (block j replaced by the null token) - they must still be scorable. Retraining
    scores high here partly because it *memorised* those partial records.
    surv_test_acc: accuracy on the test set with block j masked for everyone - the
    memorisation-free version: can the model handle partial records at all?
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from .data import DeletionSplit, VFLData, deleted_mask
from .train import evaluate_utility, per_record_loss, predict_logits


def _probs(model, data, idx, mask):
    return torch.sigmoid(predict_logits(model, data, idx, mask)).numpy()


def _auc(pos: np.ndarray, neg: np.ndarray) -> float:
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    return float(roc_auc_score(y, np.r_[pos, neg]))


def reinsertion_scores(model, data: VFLData, idx: np.ndarray, j: int) -> tuple:
    """Per-record reinsertion loss and gain for block j of the records in idx."""
    present = per_record_loss(model, data, idx, None)
    m = torch.zeros(data.n, data.n_parties, dtype=torch.bool)
    m[:, j] = True
    absent = per_record_loss(model, data, idx, m)
    return present, absent - present


def erasure_metrics(model, data: VFLData, split: DeletionSplit) -> dict:
    j = split.delete_party
    loss_d, gain_d = reinsertion_scores(model, data, split.delete_idx, j)
    loss_c, gain_c = reinsertion_scores(model, data, split.control_idx, j)
    return {"erasure_auc": _auc(gain_d, gain_c),
            "erasure_auc_loss": _auc(-loss_d, -loss_c),
            "reinsert_gain_del": float(gain_d.mean()),
            "reinsert_gain_ctrl": float(gain_c.mean())}


def prediction_drift(a, b, data, idx, mask=None) -> tuple:
    pa, pb = _probs(a, data, idx, mask), _probs(b, data, idx, mask)
    return float(np.abs(pa - pb).mean()), float(((pa > 0.5) != (pb > 0.5)).mean())


def evaluate(after, before, data: VFLData, split: DeletionSplit, reference=None,
             seed: int = 0) -> dict:
    ret = split.eval_retain_idx          # never used by any method for anchoring
    post = deleted_mask(data, split)                  # inference mask after the erasure

    out = {}
    u = evaluate_utility(after, data, split.test_idx)
    out |= {"test_acc": u["acc"], "test_auc": u["auc"]}

    s = evaluate_utility(after, data, split.delete_idx, post)
    out |= {"surv_acc": s["acc"], "surv_bce": s["bce"]}
    # Survival without memorisation: can the model score *unseen* partial records at all?
    m_test = torch.zeros(data.n, data.n_parties, dtype=torch.bool)
    m_test[:, split.delete_party] = True
    out["surv_test_acc"] = evaluate_utility(after, data, split.test_idx, m_test)["acc"]

    out["drift_retain"], out["flip_retain"] = prediction_drift(after, before, data, ret)
    out["drift_nbr"], _ = prediction_drift(after, before, data, split.neighbor_idx)
    out["drift_test"], _ = prediction_drift(after, before, data, split.test_idx)
    out["nn_ratio"] = out["drift_nbr"] / max(out["drift_retain"], 1e-12)

    out |= erasure_metrics(after, data, split)

    if reference is not None:
        out["gap_retrain_del"] = float(np.abs(_probs(after, data, split.delete_idx, post)
                                              - _probs(reference, data, split.delete_idx, post)).mean())
        out["gap_retrain_test"] = float(np.abs(_probs(after, data, split.test_idx, None)
                                               - _probs(reference, data, split.test_idx, None)).mean())
    return out


def pipeline_null_auc(model, data: VFLData, split: DeletionSplit, seed: int = 0) -> float:
    """Sanity check: split the CONTROL group into two random halves and run the
    erasure test between them. Must be ~0.5, otherwise the test itself is biased."""
    rng = np.random.default_rng(seed + 11)
    ctrl = rng.permutation(split.control_idx)
    half = len(ctrl) // 2
    _, g1 = reinsertion_scores(model, data, ctrl[:half], split.delete_party)
    _, g2 = reinsertion_scores(model, data, ctrl[half:], split.delete_party)
    return _auc(g1, g2)
