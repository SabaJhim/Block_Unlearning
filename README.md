# Block unlearning in vertical federated learning

Core Idea: Erasing **one party's features for one record** in a vertical
federated learning (VFL) model, while the record itself stays in the system.

Example: In a VFL system, Nadia, a user, asks the hospital to erase her. She is still a customer of the bank and the telecom. The joint model must forget her hospital block, keep working for everyone else, and still be able to score her from what remains.

The gap: Existing VFL unlearning removes a whole party, a whole feature, or a whole record. This repository implements the missing case, the baselines to compare against,
and an evaluation built around three requirements:

 1.Erasure: If we hand the model her deleted block again, does it react as if it had never seen it? 
 2.Locality : Did erasing her damage predictions for anyone else, especially people similar to her? 
 3.Survival: Can the model still score her from the remaining parties' blocks? 


## Contents

1. [Install](#install)
2. [Quickstart](#quickstart)
3. [Step-by-step: what to run and why](#step-by-step-what-to-run-and-why)
4. [How the pieces fit](#how-the-pieces-fit)
5. [Methods](#methods)
6. [Reading the output](#reading-the-output)
7. [First results (one seed)](#first-results-one-seed)
8. [Findings that affect the proposal](#findings-that-affect-the-proposal)
9. [Open design decisions](#open-design-decisions)
10. [Datasets](#datasets)
11. [Extending the code](#extending-the-code)
12. [Project layout](#project-layout)
13. [Next steps](#next-steps)

---

## Install

Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Everything runs on a laptop CPU. The largest configuration used so far takes
about 2.5 minutes per seed on a single core. No GPU is needed at this stage.

---

## Quickstart

```bash
pytest -q                                                    # 1. plumbing works (~10 s)
python scripts/sanity_checks.py --config configs/synthetic.yaml   # 2. null controls (~40 s)
python scripts/run_experiment.py --config configs/synthetic.yaml  # 3. all methods (~1 min)

python scripts/prepare_data.py adult                         # 4. real data
python scripts/sanity_checks.py --config configs/adult_memorise.yaml
python scripts/run_experiment.py --config configs/adult_memorise.yaml --seeds 0 1 2 3 4
```

Results land in `results/<config name>/`: `metrics_raw.csv` (one row per method
per seed), `metrics_summary.csv` (mean ± std), and per-seed `metrics.json`.
Trained original models are cached as `results/<name>/seed<k>/base.pt`; pass
`--force-base` to retrain them.

---

## Step-by-step: what to run and why

Do these in order. Each step exists because a later number is meaningless if it fails.

### Step 1 — Smoke tests

```bash
pytest -q
```

**Why.** Checks the plumbing rather than the science: masks hide exactly the
blocks they should, a masked record sends zero gradient into the masked party's
encoder, methods that claim zero communication really send nothing, BlockEdit
modifies only party *j*'s encoder, and every metric is finite. If any of these
fail, stop — every downstream result would be built on a broken mask.

### Step 2 — Sanity checks (null controls)

```bash
python scripts/sanity_checks.py --config configs/synthetic.yaml
```

**Why.** An erasure test is only meaningful if it gives the right answer in
cases where we already know the answer. This script builds those cases:

| Check | Known answer | If it fails |
|---|---|---|
| Pipeline null | Two halves of the control group are identical in kind → AUC ≈ 0.5 | The test itself is biased. Nothing else can be trusted. |
| Memorisation | The original model has seen the deleted blocks but not the control blocks → AUC clearly > 0.5 | There is nothing to erase, so every method will look the same. Make the model overfit more (see Step 4). |
| Gold standard | Retraining with the deleted blocks masked → deleted and control are equivalent → AUC ≈ 0.5 | Your definition of "forgotten" fails its own test; the groups differ systematically. |
| mask_only plumbing | No parameters change → drift exactly 0 | A bug in evaluation. |

"≈ 0.5" is judged with a 3-sigma band for an AUC under the null, computed from
the group sizes, so it automatically tightens as `n_delete` and `n_control` grow.

The script also prints two numbers for context: the **noise floor** (how much
predictions move between two original models trained with different seeds — any
drift below this is invisible in practice) and the **duplicate rate** (see
[Findings](#findings-that-affect-the-proposal)).

### Step 3 — Full comparison on synthetic data

```bash
python scripts/run_experiment.py --config configs/synthetic.yaml
```

**Why synthetic first.** It is fast and it has a dial real data does not:
`cross_party_corr` controls how much the parties' information overlaps. At 0 the
parties are independent, so deleting one block removes information nobody else
has; at 1 they are redundant, so the others can compensate. This is the variable
the proposal expects locality and survival difficulty to depend on, and the one
you will want to sweep.

### Step 4 — Adult, in both regimes

```bash
python scripts/prepare_data.py adult
python scripts/sanity_checks.py --config configs/adult.yaml
python scripts/sanity_checks.py --config configs/adult_memorise.yaml
python scripts/run_experiment.py --config configs/adult_memorise.yaml
```

**Why two configs.** `adult.yaml` is the standard setup: modest model, full
training set, good generalisation (test accuracy ≈ 0.857). It **fails the
memorisation check** — the model has not memorised anything individual, so
there is nothing measurable to erase and every method, including retraining,
lands at the same erasure AUC. `adult_memorise.yaml` uses a larger model on
8,000 records so that it overfits (train ≈ 0.97, test ≈ 0.81); memorisation
then becomes measurable and the methods separate.

This is the same phenomenon the membership-inference literature documents:
privacy leakage requires memorisation. Report both regimes in the paper — the
well-generalised one is an honest finding, not a failure.

### Step 5 — Multiple seeds before believing anything

```bash
python scripts/run_experiment.py --config configs/adult_memorise.yaml --seeds 0 1 2 3 4
```

**Why.** With a few hundred deleted blocks, an AUC has a standard error of
roughly ±0.02. Differences smaller than about 0.05 between methods on a single
seed are noise. The summary table reports mean ± std across seeds.

---

## How the pieces fit

```
                       ┌──────────── split (per seed) ─────────────┐
 training records ───► │ delete   blocks that receive the request  │
                       │ control  blocks masked from the start     │  ← "never seen" reference
                       │ retain   everyone else                    │
                       │   ├ eval_retain   drift measured here     │  ← no method may touch
                       │   ├ neighbours    half eval, half anchor  │
                       │   └ anchor_pool   methods may use these   │
                       └───────────────────────────────────────────┘

original model  = trained with control blocks masked
      │
      ├── mask_only / row_ga / masked_finetune / blockedit ...   (start from original)
      └── masked_retrain / full_row_retrain                      (start from scratch, same seed)
                │
                ▼
      evaluate(after, before, reference = masked_retrain)
```

**The missing-block token.** When a block is missing, its party's embedding is
replaced by a token `h_⊥` (`model.null_emb[k]`). Because this is done with
`torch.where`, a masked record sends no gradient into that party's encoder —
exactly as if the party never had the data. Three choices via `model.null_token`:
`mean` (average embedding; default), `zero`, or `learned` (a trained parameter).

**Inference after deletion.** Once party *j* erases Nadia's row, it no longer has
her data, so it can only send `h_⊥` for her. All post-deletion predictions for
deleted records are therefore made with block *j* masked (`deleted_mask`).

**Why the control group exists.** To ask "does the model react to her block as
if it never saw it?", we need blocks the model genuinely never saw, belonging to
records that are otherwise treated identically. The control blocks are masked
from the first step of training, so they are exactly that. This costs a small
amount of realism — the original model has seen a few hundred masked records —
and is documented under [Open design decisions](#open-design-decisions).

**Same-seed retraining.** Retrain baselines use the original seed, so their
initialisation and batch order match the original run. The only difference is
the deletion itself, which keeps their drift numbers interpretable.

---

## Methods

| Method | Starts from | What it does | Communication | Role |
|---|---|---|---|---|
| `mask_only` | original | Party *j* deletes the rows and sends `h_⊥`. No parameter change. | none | Lower bound: perfect locality, erases nothing already absorbed into the weights. |
| `masked_retrain` | scratch | Retrain with deleted blocks masked. | full training | **Gold standard.** Defines "correctly forgotten". |
| `full_row_retrain` | scratch | Retrain without the deleted records at all. | full training | Over-deletion reference; shows the cost of ignoring survival. |
| `row_ga` | original | Gradient ascent on the deleted records' full rows (VFU-GA style). `alpha > 0` adds a retain term. | a few rounds | Current VFL sample-unlearning recipe, adapted. |
| `masked_finetune` | original | A few epochs of joint training with deletions masked. | tens of rounds | The obvious cheap federated baseline. |
| `blockedit` | original | **Ours.** Party *j* alone edits its encoder: deleted blocks → target, anchored on neighbours + random retain. | **zero** | Local, label-free erasure. |
| `blockedit_plus` | original | `blockedit`, then a short top-model calibration (fit deleted-as-partial records, distil retain towards the original). | one forward pass | Addresses the top-model residual (see Findings). |
| `blockedit_nbr`, `blockedit_plus_nbr` | original | Same, but the target is the mean embedding of the block's nearest retain neighbours instead of `h_⊥`. | as above | Tests the target choice (see Open design decisions). |

The BlockEdit objective, all local to party *j*:

```
min over θ_j   ‖ f_j(x_del; θ_j) − target ‖²                          erase
             + λ · mean_{a ∈ anchors} ‖ f_j(x_a; θ_j) − f_j(x_a; θ_j^before) ‖²    keep
```

`λ` (`methods.blockedit.lam`) is the erasure–locality knob. On synthetic data,
λ = 1 / 10 / 100 gives retain drift of roughly 0.10 / 0.02 / 0.003 with the null
target: it behaves exactly as designed.

Hyperparameter defaults live at the top of each method in `bvu/unlearn.py` and
can be overridden per config under `methods:`.

---

## Reading the output

| Column | Meaning | What good looks like |
|---|---|---|
| `test_acc`, `test_auc` | Utility on full test records | Close to the original model |
| `surv_acc` | Accuracy on the deleted records, scored as partial records | Close to `masked_retrain`. Note retraining scores high partly because it *memorised* those partial records |
| `surv_test_acc` | Accuracy on the test set with block *j* masked for everyone | The memorisation-free survival measure: can the model handle partial records at all? |
| `drift_retain` | Mean \|Δp\| vs. the original, on held-out retain records | Well below `noise_floor_retain` |
| `flip_retain` | Share of those records whose predicted class changes | Near 0 |
| `drift_nbr` | Same, on held-out neighbours of the deleted blocks | Near `drift_retain` |
| `nn_ratio` | `drift_nbr / drift_retain` | ≈ 1. Above 1 means damage concentrates near the deleted records |
| `erasure_auc` | **R1.** Reinsertion test, deleted vs. control | **0.5.** Above = still remembers; below = over-forgets (itself detectable) |
| `erasure_auc_loss` | Same test on raw reinsertion loss | 0.5 |
| `reinsert_gain_del`, `reinsert_gain_ctrl` | Mean reaction to reinsertion, per group | Equal to each other (this is what the AUC compares) |
| `gap_retrain_del`, `gap_retrain_test` | Mean \|Δp\| vs. `masked_retrain` | Small |
| `seconds`, `comm_rounds`, `comm_MB` | Cost | Far below `masked_retrain` |
| `noise_floor_retain` | Drift between two independent original models | Context for every drift number |
| `dup_frac` | Share of deleted blocks with an exact duplicate in retain | Context; see Findings |

**The erasure test in one paragraph.** For every record, the *reinsertion gain*
is how much the model's loss improves when block *j* is revealed versus masked
— the model's reaction to that block. A block it genuinely never saw (control)
produces one distribution of reactions. If a deleted block has truly been
forgotten, its reactions should come from the same distribution. `erasure_auc`
measures how well the two can be told apart. This is the strict,
retrain-indistinguishability version of "forgotten".

---

## First results (one seed)

`configs/adult_memorise.yaml`, seed 0, demographic party as the deleting party.
**One seed only — treat as a direction, not a result.**

| Method | test acc | surv test acc | drift retain | nn ratio | erasure AUC | comm rounds |
|---|---|---|---|---|---|---|
| mask_only | 0.815 | 0.797 | 0.000 | – | 0.633 | 0 |
| masked_retrain | 0.821 | 0.801 | 0.056 | 0.90 | **0.491** | 3200 |
| full_row_retrain | 0.814 | 0.794 | 0.049 | 0.66 | 0.569 | 3000 |
| row_ga | **0.717** | 0.674 | 0.208 | 0.85 | 0.566 | 12 |
| masked_finetune | 0.818 | 0.787 | 0.032 | 0.61 | 0.638 | 64 |
| blockedit | 0.818 | 0.797 | 0.104 | 0.92 | **0.555** | **0** |
| blockedit_plus | 0.824 | 0.802 | 0.088 | 0.97 | 0.567 | 1 |
| blockedit_nbr | 0.814 | 0.797 | 0.003 | 2.15 | 0.624 | 0 |
| blockedit_plus_nbr | 0.815 | 0.796 | 0.011 | 1.05 | 0.628 | 1 |

Noise floor: 0.070.

What this says, provisionally:

- **BlockEdit (null target) moves most of the way to retraining** on erasure
  (0.633 → 0.555, target 0.491) with zero communication and no utility loss.
  That is the core claim, and it survives a first look.
- **But its locality cost is real**: drift 0.104 is above the noise floor. λ
  controls this; the trade-off curve (erasure AUC vs. drift as λ varies) is
  probably the paper's central figure.
- **Row-GA erases by damaging the model** (test accuracy 0.815 → 0.717). This
  is the failure mode the paper argues against, reproduced.
- **Masked fine-tuning barely erases** (0.638) despite 64 rounds of
  communication — cheap federated fixes are not enough.
- **The neighbour target preserves locality but erases little.** See below.

---

## Findings that affect the proposal

Four things surfaced in the confirmation runs that the proposal should absorb.

### 1. Erasure is only measurable in a memorisation regime

Standard Adult fails the memorisation check: every method, including
retraining, scores about 0.526. The standard setup generalises too well to have
memorised anything individual. Only the overfit configuration produces a
measurable gap. **Implication:** the paper needs to report both regimes and to
state plainly that block unlearning is close to vacuous for well-generalised
models. That is a finding reviewers will respect, and it pre-empts the question.

### 2. Most "sensitive" blocks are not unique

On Adult, **89% of deleted demographic blocks have an exact twin** among the
retained records. An encoder is a function: it cannot move one copy of an input
without moving every other copy. Encoder-only erasure of a duplicated block
therefore damages the twins by necessity — a lower bound on locality cost, not a
tuning problem. **Implication:** worth a short proposition in the paper, and a
reason to report results split by unique vs. duplicated blocks. Memorisation
still shows up for duplicated blocks because the *top model* memorises the
combination of a person's blocks plus their label.

### 3. Much of the residual lives in the top model

On synthetic data, encoder-only editing barely moves the erasure AUC, and the
per-group reinsertion gains show why. Under retraining, *both* groups have
negative gain: the retrained top model memorised the deleted records **as
partial records**, so revealing the block actually hurts. Encoder editing can at
best drive the deleted group's gain to zero; only changing the top model can
make it negative. On Adult, encoder editing does much better, so the split
between encoder and top-model residual depends on the data. **Implication:** the
"zero communication" claim may need to become "one forward pass" (BlockEdit+),
and the calibration step deserves more attention than the proposal gave it.
Measuring the encoder/top split per dataset is itself a contribution.

### 4. The target choice matters more than expected

The `h_⊥` target (proposal default) erases more but damages locality; the
neighbour target barely damages locality but barely erases. See the next section
for why, and treat this as a design question to settle early.

---

## Open design decisions

These are genuine research choices, not bugs. Each is a config switch.

**What should the deleted block map to?** The proposal says `h_⊥`. But a model
retrained without her block would not map her features to `h_⊥` — it would
embed them like any unseen input. Mapping to `h_⊥` is *stronger* than retraining
(the encoder refuses to represent her), and a stronger-than-retrain edit is
itself a detectable signature, which is exactly Paper 2's territory. The
`neighbors` target approximates the retrained behaviour instead. Current
evidence: `h_⊥` erases more, `neighbors` is gentler. Worth deciding on
principle, then defending.

**Which definition of "forgotten"?** `erasure_auc` implements the strict,
retrain-indistinguishability definition. A weaker, privacy-oriented definition
is "no information about her block remains" — tested, for example, by whether
the model still prefers her true block over a random one when both are
reinserted. The two can disagree (BlockEdit with `h_⊥` plausibly passes the
weak test and fails the strict one). Adding the weak test is a natural next
metric and would sharpen the paper's claims.

**The control group changes the original model slightly.** The original model
is trained with a few hundred blocks masked. This is realistic — real VFL data
has missing blocks — and necessary for the erasure test. Keep `n_control` small
relative to the training set and mention it in the experimental setup.

**Null token.** `mean`, `zero` and `learned` all run; `block_dropout > 0`
("erasure-ready" training) also runs. Neither has been compared systematically
yet. The proposal promises this ablation.

---

## Datasets

| Config | Status | How to get the data |
|---|---|---|
| `synthetic.yaml` | ✅ verified | Generated on the fly |
| `adult.yaml` | ✅ verified | `python scripts/prepare_data.py adult` (UCI, with a GitHub mirror fallback) |
| `adult_memorise.yaml` | ✅ verified | Same file as above |
| `credit_default.yaml` | ⚠️ written, not yet run | Download *default of credit card clients* (.xls) from the UCI repository, then `python scripts/prepare_data.py credit_default --src <file>`. Needs `pip install xlrd`. |
| `bank_marketing.yaml` | ⚠️ written, not yet run | Download *Bank Marketing* from UCI, extract `bank-additional-full.csv`, then `python scripts/prepare_data.py bank_marketing --src <file>`. `duration` is dropped because it leaks the label. |

For the two unverified configs, check the column names against the file on
first use and run the sanity checks before anything else. Both will probably
need a memorisation-regime variant, as Adult did.

Adult party split: `employment` (active, holds labels) ·
`demographic` (the deleting party, playing the hospital) · `education`.

---

## Extending the code

**Add a tabular dataset.** Write a config with `data.kind: tabular`, the CSV
path, label column and positive labels, the categorical and `log1p` columns,
and a `parties:` mapping from party name to column list. No code changes needed.
Then run the sanity checks.

**Add a method.** Write a function `my_method(ctx: Context) -> UnlearnResult` in
`bvu/unlearn.py`, register it in `METHODS`, and it is picked up by the runner
and the smoke tests automatically. Use `ctx.split.anchor_pool` for any retain
records the method trains on — never `retain_idx` directly — or the locality
numbers will be contaminated. Count any cross-party traffic with a
`CommCounter`.

**Use from a notebook.**

```python
import sys; sys.path.insert(0, "path/to/block-unlearning-vfl")
from bvu.utils import load_config
from bvu.data import load_dataset, make_deletion_split, base_mask
from bvu.train import fit_fresh
from bvu.unlearn import Context, run_method
from bvu.metrics import evaluate

cfg   = load_config("configs/adult_memorise.yaml")
data  = load_dataset(cfg, seed=0)
split = make_deletion_split(data, cfg, seed=0)
before = fit_fresh(data, data.train_idx, base_mask(data, split), cfg, seed=0)
ctx   = Context(before, data, split, cfg, seed=0)
ref   = run_method("masked_retrain", ctx).model
res   = run_method("blockedit", ctx)
print(evaluate(res.model, before, data, split, reference=ref))
```

---

## Project layout

```
block-unlearning-vfl/
├── bvu/
│   ├── data.py        datasets, party splits, deletion split, masks, dataset preparation
│   ├── models.py      split-VFL model: bottom encoders, top model, missing-block token
│   ├── train.py       joint training, erasure-ready training, prediction, utility
│   ├── unlearn.py     all methods + the METHODS registry
│   ├── metrics.py     erasure (reinsertion test), locality, survival, gap to retrain
│   └── utils.py       seeding, timing, communication counter, config IO
├── configs/           one YAML per dataset / regime
├── scripts/
│   ├── prepare_data.py     download and clean datasets
│   ├── sanity_checks.py    null controls — run first
│   └── run_experiment.py   main entry point
├── tests/test_smoke.py     14 invariant tests
├── data/                   (created) raw and prepared CSVs
└── results/                (created) metrics and cached models
```

---

## Next steps

Roughly in the order the proposal's timeline expects:

1. **Five seeds on `adult_memorise`** to see whether the BlockEdit result holds.
2. **The λ sweep** — erasure AUC against locality drift, one curve per method.
   This is likely the paper's main figure. A `scripts/sweep.py` is the obvious
   next file to write.
3. **Split results by unique vs. duplicated blocks** (Finding 2).
4. **Decide the target question** (`h_⊥` vs. neighbours) and add the weak,
   information-removal erasure test.
5. **Strengthen BlockEdit+** — the top-model calibration is currently short and
   simple; Finding 3 says it matters.
6. **Sweep `cross_party_corr`** on synthetic data.
7. **Run credit default and bank marketing**, then NUS-WIDE (needs a
   multi-class extension of the top model and loss).
8. **Null-token and erasure-ready training ablation.**
