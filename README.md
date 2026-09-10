# Counterfactual Explanations for ICU In-Hospital Mortality

This repository trains an XGBoost model to predict in-hospital mortality from
the first 48 hours of a MIMIC-IV ICU stay, and then generates and evaluates
counterfactual explanations for that model ("what would need to be
different about this patient's observed trajectory for the model to no
longer predict death?") using three different counterfactual-generation
approaches:

- **DiCE** (Diverse Counterfactual Explanations, genetic search), run three
  ways depending on how missing feature values are filled in before search
  (mean, median, or KNN imputation): `dice_mean.py`, `dice_median.py`,
  `dice_knn.py`.
- **FACET-style** search: a from-scratch beam search over the trained
  XGBoost ensemble's own decision regions ("facets"): `facet_style.py`.
- **OCEAN-style MILP**: an exact mixed-integer-programming formulation of
  the trained tree ensemble, solved with PuLP/CBC: `ocean_style.py`.

All three approaches respect a shared feature-mutability policy (some
features, e.g. age, treatments already given, are excluded from the
search) and are scored on the same validity / proximity / sparsity /
plausibility metrics, so their outputs are directly comparable.

## Data: MIMIC-IV must be provided

MIMIC-IV is a credentialed dataset distributed by PhysioNet
(<https://physionet.org/content/mimiciv/>); its license does not allow
redistributing the data, so this repository does not, and cannot,
include it. Running `preprocessing.py` requires completed PhysioNet
credentialing and a local copy of the `hosp/` and `icu/` CSV folders from a
MIMIC-IV release.

By default the scripts look for that data in `data/mimic_raw/` next to this
README, laid out exactly as PhysioNet distributes it:

```
data/mimic_raw/
  hosp/patients.csv, admissions.csv, transfers.csv, labevents.csv, prescriptions.csv, d_labitems.csv
  icu/icustays.csv, chartevents.csv, inputevents.csv, outputevents.csv, procedureevents.csv, d_items.csv
```

If the data is elsewhere, either move it there or pass
`--data-dir /path/to/mimic-iv` (see below).

## Repository contents

| File | What it does |
|---|---|
| `preprocessing.py` | Builds the 48-hour-window ICU cohort from raw MIMIC-IV, splits it into train/val/test by patient, selects a frequent-enough set of chart/lab/input/output/procedure/drug concepts using the train split only, extracts per-stay features, and writes `X_/y_/meta_{train,val,test}` tables plus a `counterfactual_feature_metadata.csv` that records each feature's plausible range and a mutability label (used by every counterfactual script later). |
| `train_xgboost.py` | Loads those tables, trains a single `XGBClassifier` (with a class-imbalance-aware `scale_pos_weight`), picks a decision threshold on the validation split by F1, and saves the model, per-split predictions, metrics, and feature importances. |
| `cf_experiment_utils.py` | Shared library, not run directly: all five counterfactual scripts import it. It does three things so each script doesn't have to reimplement them: (1) loads the trained model, feature tables, and feature-mutability metadata into a consistent format; (2) picks which patients each script will generate counterfactuals for (by default the highest-risk predicted-positive patients), or an exact fixed set of patients if `--query-ids-path` is passed (see "Building a fixed factual cohort" below); and (3) after counterfactuals are generated, scores them on validity (did the prediction flip?), proximity (distance from the original patient), sparsity (how many features changed), and plausibility (does the counterfactual look like a realistic patient, via nearest-neighbor distance and outlier detection against the training data). A row_id assigned during step 2 is reused in step 3 to match each generated counterfactual back to the right original patient. |
| `make_counterfactual_query_batches.py` | Builds the row_id CSV(s) that `--query-ids-path` consumes, from a `predictions_<split>` file written by `train_xgboost.py`: filters by predicted class and (optionally) true label, then samples one or more fixed batches of a given size. This is how a reusable, reproducible factual cohort is produced to use (e.g. "the same 300 true-positive patients") across every CF method. See "Building a fixed factual cohort" below. |
| `dice_mean.py` / `dice_median.py` / `dice_knn.py` | Run DiCE's genetic-search counterfactual generator against the trained model. Because DiCE needs a complete (non-missing) feature vector to search over, each variant fills originally-missing values with a different placeholder (train-split mean / median / KNN-imputed value) *only* for the search itself; the model is still queried with the original missingness pattern restored (`NativeMissingPredictionWrapper`), and originally-missing features are excluded from that patient's mutable search space and restored to `NaN` again before evaluation. `dice_mean.py` and `dice_median.py` are otherwise identical; `dice_knn.py` additionally supports `--imputed-data-dir` to reuse a pre-computed KNN-imputation cache instead of recomputing it on every run (optional as a normal `dice_knn.py` run doesn't need this; it imputes on the fly). |
| `build_dice_knn_imputed_dataset.py` | Builds that KNN-imputation cache once (writes `x_train_imputed.pkl`, `selected_factuals_native.pkl`, `selected_factuals_imputed.pkl`, `metadata.json` to `--out-dir`) so repeated `dice_knn.py` runs with `--imputed-data-dir` skip refitting the KNN imputer every time. It is useful for re-running `dice_knn.py` (e.g. sweeping other hyperparameters) to amortize that cost; skip it otherwise. |
| `facet_style.py` | A dependency-free beam search: it enumerates the leaves ("facets") of every tree in the trained ensemble as a set of feature-range constraints, ranks the most promising ones per tree, and beam-searches combinations of them (respecting mutability and each feature's observed range) until the prediction flips or the step budget runs out. |
| `ocean_style.py` | Encodes the *entire* trained XGBoost ensemble as a mixed-integer program (one binary "leaf selected" variable per tree, split thresholds as linear constraints, XGBoost's own `default_left` used to route originally-missing features exactly as the real model would) and solves it with PuLP for a proximity/sparsity-minimal counterfactual. It needs a MILP solver (CBC ships with PuLP; Gurobi/HiGHS are also supported via `--solver-name`). |
| `pipeline.py` | Orchestrates all of the above end to end, or any subset of it, with one shared, consistent directory layout. See below. |
| `requirements.txt` | Everything needed to `pip install`. |

## Setup

Python 3.10+ is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate          # .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

`pulp` (used by `ocean_style.py`) bundles the open-source CBC solver, so no
separate solver installation is required for a default run. `dice-ml` and
`raiutils` are only needed for the three `dice_*.py` scripts; `facet_style.py`
needs nothing beyond the core stack.

## Running everything: `pipeline.py`

```bash
# Run the whole pipeline:
python pipeline.py

# Rebuild the model from data already preprocessed and generate FACET counterfactuals:
python pipeline.py --stages train,facet

# Fast end-to-end sanity check (few queries per CF method) before a full run:
python pipeline.py --cf-args "--max-queries 20"

# See every command it would run, without running:
python pipeline.py --dry-run

# List the available stage names:
python pipeline.py --list
```

By default `pipeline.py` runs, in order: `preprocess -> train -> query_ids ->
facet -> dice_mean -> dice_median -> dice_knn -> ocean`, writing to:

```
data/mimic_raw/                         (input provided)
data/processed/                         (preprocessing.py output)
data/model/                             (train_xgboost.py output)
data/query_ids/                         (make_counterfactual_query_batches.py output)
results/counterfactual_experiments/<method-name>/   (one folder per CF method)
```

`preprocess` and `train` are load-bearing: if either fails, the pipeline
stops immediately (their output is required by everything after them). The
`query_ids` stage and the five counterfactual stages are independent of each
other; by default a failure there is reported and the pipeline moves on
to the next stage, so e.g. running without `dice-ml` installed still reports
`facet` and `ocean` results. Pass `--strict` to stop on the first failure of
any stage instead.

`query_ids` runs by default (with `--query-ids-args` defaulting to a
300-patient true-positive cohort) but its output isn't used by the CF
stages unless it's passed in via `--cf-args`, since `--cf-args` is one shared
string applied to every CF stage identically:

```bash
--cf-args "--query-ids-path data/query_ids/query_ids_batch001.csv --max-queries 300"
```

`query_ids_batch001.csv` is `query_ids`'s default output filename (see
"Script parameters" below); see "Building a fixed factual cohort" below for
why `--max-queries` needs to be passed alongside `--query-ids-path` there.

`--preprocess-args`, `--train-args`, `--query-ids-args`, and `--cf-args` pass
extra flags straight through to the underlying script, see "Script
parameters" below for what each script accepts. For example, to reduce the
XGBoost training run and cap every CF stage at 20 queries in the same
command:

```bash
python pipeline.py \
  --train-args "--n-estimators 500 --learning-rate 0.05" \
  --cf-args "--max-queries 20 --allow-treatments"
```

`python pipeline.py --dry-run` prints the exact command each selected stage
will run without running anything, which is a quick way to check that
per-stage arguments landed where expected. Every script is also runnable
completely on its own; run any of them with `--help` for its full flag list.

## Running stages by hand

```bash
# 1. Build the cohort and feature tables (needs a local MIMIC-IV extract).
python preprocessing.py --data-dir data/mimic_raw --out-dir data/processed

# 2. Train the mortality model.
python train_xgboost.py --data-dir data/processed --out-dir data/model

# 3. (Optional) Build a fixed, reusable set of factual patients.
python make_counterfactual_query_batches.py \
  --predictions data/model/predictions_test.parquet --metrics data/model/metrics.json \
  --source-label 1 --true-label-one-only --num-samples 300 --num-batches 1 \
  --out-dir data/query_ids

# 4. Generate counterfactuals with any/all of the five methods:
python facet_style.py    --data-dir data/processed --model-dir data/model --out-root results/counterfactual_experiments
python dice_mean.py      --data-dir data/processed --model-dir data/model --out-root results/counterfactual_experiments
python dice_median.py    --data-dir data/processed --model-dir data/model --out-root results/counterfactual_experiments
python dice_knn.py       --data-dir data/processed --model-dir data/model --out-root results/counterfactual_experiments
python ocean_style.py    --data-dir data/processed --model-dir data/model --out-root results/counterfactual_experiments
# ...or, to reuse the fixed cohort from step 3 across all five (--max-queries must be repeated here even though step 3 already
# fixed the batch size):
python facet_style.py --data-dir data/processed --model-dir data/model --out-root results/counterfactual_experiments \
  --query-ids-path data/query_ids/query_ids_batch001.csv --max-queries 300
```

Each CF script writes its own subfolder under `--out-root` (named after the
method, e.g. `results/counterfactual_experiments/facet_style_xgboost_point_preserve_missing/`)
containing `selected_factuals.csv`, `counterfactuals.csv`, `failed_queries.csv`,
`summary.json`, and `run_config.json`. Useful shared flags on every CF
script: `--max-queries N` (how many patients to explain), `--query-split
{train,val,test}`, `--allow-treatments` (allow treatment/exposure features
into the search space, not just clinical measurements), and
`--query-ids-path some.csv` (reuse an exact, fixed set of patients across
methods for a like-for-like comparison). Run any script with
`--help` for the complete, method-specific list (e.g. `--total-cfs`,
`--solver-name`, `--knn-neighbors`).

## Building a fixed factual cohort (`--query-ids-path`)

By default, each CF script independently ranks patients the same way (same
model, same `pred_proba`, sorted descending) but applies its own
`--max-queries` cutoff to that ranking: 250 for `facet_style.py` and the
`dice_*.py` scripts, 50 for `ocean_style.py`. That means, by default,
`ocean_style.py` only explains a subset of the patients `facet_style.py`
does. The same ranking can be reproduced across scripts by passing matching `--max-queries`,
`--min-query-proba`, and `--random-query-sample` values to each one
explicitly, but that has to be done consistently on every run. Fixing the
cohort with `--query-ids-path` instead removes the need to keep those
values in sync by hand, and is also how the cohort gets restricted to true
positives specifically (patients the model correctly predicted would die)
rather than just "predicted positive".

`make_counterfactual_query_batches.py` builds that fixed cohort as a
`row_id` CSV from a `predictions_<split>` file (it reads either the
`.parquet` or `.csv` form):

```bash
python make_counterfactual_query_batches.py \
  --predictions data/model/predictions_test.parquet \
  --metrics data/model/metrics.json \
  --source-label 1 --true-label-one-only \
  --num-samples 300 --num-batches 1 \
  --out-dir data/query_ids
```

- `--source-label 1` keeps only rows the model predicted positive (at the
  tuned threshold in `metrics.json`); `--true-label-one-only` additionally
  keeps only rows whose true label was positive, i.e. this reproduces a
  true-positive cohort. Drop `--true-label-one-only` to sample from all
  predicted-positive patients instead (true and false positives together).
- `--num-samples 300 --num-batches 1` writes one CSV of 300 row_ids
  (`query_ids_batch001.csv`); raise `--num-batches` to get several
  non-overlapping batches (or `--allow-overlap` to let them overlap) in one
  call.
- The output CSV's `row_id` column is a positional index into that split's
  feature table as `cf_experiment_utils.py` loads it, not `subject_id`/`hadm_id`/`stay_id`. It's only meaningful for
  the exact `--data-dir`/split it was built from; the manifest JSON written
  alongside it (`query_ids_manifest.json`) records the source predictions
  file and settings used, for traceability.

Then pass that file to any CF script via `--query-ids-path`:

```bash
python facet_style.py --data-dir data/processed --model-dir data/model \
  --out-root results/counterfactual_experiments \
  --query-ids-path data/query_ids/query_ids_batch001.csv --max-queries 300
```

**Important:** always pass `--max-queries` explicitly alongside
`--query-ids-path`, set to at least the number of ids in the file (300 in
the example above). Each CF script applies its own `--max-queries` cap
*after* loading the fixed id list, and their defaults (`facet_style.py`,
`dice_mean.py`, `dice_median.py`, `dice_knn.py`: 250; `ocean_style.py`: 50)
are all below 300, so without an explicit override, a 300-patient cohort
is silently truncated to the default that script happens to use, and
different CF methods end up explaining different numbers of
patients from the same file. `pipeline.py`'s own `--cf-args` doesn't set
this automatically either, for the same reason: it's one shared string
applied to every CF stage, so `--max-queries` needs to be included in it
explicitly whenever `--query-ids-path` is used.

## Script parameters

Every script also accepts `--help` at the command line for this same
information. The five counterfactual scripts share a large common set of
flags (inherited from `cf_experiment_utils.py`'s base config), listed once
below and then followed by each script's own additional flags.

### preprocessing.py

| Flag | Default | What it does |
|---|---|---|
| `--data-dir` | `data/mimic_raw` | Folder containing the raw MIMIC-IV `hosp/` and `icu/` CSVs. |
| `--out-dir` | `data/processed` | Folder to write the `X_/y_/meta_{train,val,test}` tables and `counterfactual_feature_metadata.csv` to. |
| `--hours-window` | `48` | Prediction landmark: only data from the first N hours of the ICU stay is used as features. |
| `--test-size` | `0.15` | Fraction of patients held out for the test split. |
| `--val-size` | `0.15` | Fraction of patients held out for the validation split. |
| `--random-state` | `42` | Random seed for the train/val/test split. |

### train_xgboost.py

| Flag | Default | What it does |
|---|---|---|
| `--data-dir` | `data/processed` | Folder containing `X_/y_{train,val,test}` tables from `preprocessing.py`. |
| `--out-dir` | `data/model` | Folder to write the model, metrics, feature importances, and per-split predictions to. |
| `--n-estimators` | `2000` | Maximum number of boosting rounds (early stopping on the validation AUC/AUCPR/logloss can stop it sooner). |
| `--learning-rate` | `0.02` | XGBoost's `eta`. |
| `--max-depth` | `4` | Maximum tree depth. |
| `--random-state` | `42` | Random seed for model training. |

Only the four flags above are exposed on the command line. `build_model()`
also sets `min_child_weight = 15`, `subsample = 0.7`, `colsample_bytree =
0.5`, `reg_alpha = 1`, `reg_lambda = 10`, `early_stopping_rounds = 100`, and
`eval_metric = ("auc", "aucpr", "logloss")` on the `XGBClassifier`, plus
`threshold_metric = "f1"` for the decision-threshold search, none of these
have a corresponding CLI flag, so changing any of them means editing the
`TrainConfig` defaults directly in `train_xgboost.py`. `scale_pos_weight` is
the one exception: it's computed automatically
from the training split's class balance (`neg/pos`) each run.

### Shared counterfactual-script flags

`facet_style.py`, `dice_mean.py`, `dice_median.py`, `dice_knn.py`, and
`ocean_style.py` all accept:

| Flag | Default | What it does |
|---|---|---|
| `--data-dir` | `data/processed` | Feature tables from `preprocessing.py`. |
| `--model-dir` | `data/model` | Trained model and metrics from `train_xgboost.py`. |
| `--out-root` | `results/counterfactual_experiments` | Root folder; each script writes its own named subfolder under it. |
| `--out-name` | one per method, e.g. `facet_style_xgboost_point_preserve_missing` | Name of that subfolder. |
| `--query-split` | `test` | Which split (`train`, `val`, or `test`) to select factual patients from. |
| `--max-queries` | `250` for facet/dice_*, `50` for ocean_style.py | Maximum number of factual patients to explain. Applied *after* `--query-ids-path` loads a fixed id list. |
| `--query-offset` | `0` | Skip this many rows from the selected/fixed factual cohort before starting (for splitting a run into chunks). |
| `--min-query-proba` | `0.5` | Minimum predicted probability for a patient to be eligible as a factual (ignored when `--query-ids-path` is set). |
| `--query-ids-path` | none | CSV of `row_id` values (from `make_counterfactual_query_batches.py`) to reuse an exact, fixed set of factual patients instead of picking the highest-risk ones. |
| `--random-query-sample` | off | Randomly sample `--max-queries` eligible factuals (seeded by `--random-state`) instead of taking the highest-risk ones first. |
| `--max-features-to-vary` | `0` for facet/dice_* (unlimited), `20` for ocean_style.py | Cap on how many mutable features the search may change. |
| `--feature-subset-strategy` | `all` for facet/dice_*, `top` for ocean_style.py | Which mutable features are offered to the search (`all` = every mutable feature; `top` = the model's top features by gain, capped by `--max-features-to-vary`). |
| `--allow-treatments` | off | Allow treatment/exposure features (e.g. drugs given, procedures performed) into the search space, not just clinical measurements. |
| `--random-state` | `42` | Random seed for the search itself. |
| `--total-cfs` | `8` for dice_*, `1` for ocean_style.py | Counterfactuals requested per patient (`facet_style.py` accepts this flag for output-format compatibility but always produces exactly one). |

### facet_style.py additional flags

| Flag | Default | What it does |
|---|---|---|
| `--max-steps` | `8` | Beam-search step budget per patient. |
| `--beam-width` | `20` | Number of candidate partial solutions kept at each step. |
| `--top-facets` | `400` | How many leaf-derived "facets" (feature-range constraints) are considered overall. |
| `--top-facets-per-tree` | `3` | How many facets are kept per tree when building that candidate pool. |
| `--max-trees` | `500` | How many trees of the ensemble are used to derive facets. |

### dice_mean.py / dice_median.py / dice_knn.py additional flags

| Flag | Default | What it does |
|---|---|---|
| `--desired-class` | `0` | Target class for the counterfactual (0 = survive). |
| `--sparsity-weight` | `0.2` | Weight on the sparsity term in DiCE's genetic-search objective. |
| `--proximity-weight` | `1.0` | Weight on the proximity term. |
| `--diversity-weight` | `1.0` | Weight on the diversity term (spread across the `--total-cfs` counterfactuals returned per patient). |

`dice_knn.py` additionally accepts:

| Flag | Default | What it does |
|---|---|---|
| `--knn-neighbors` | `5` | Number of neighbors used to fit the KNN imputer on the training feature matrix. |
| `--imputed-data-dir` | none | Directory produced by `build_dice_knn_imputed_dataset.py`; when set, loads its cached imputed frames instead of recomputing the KNN imputation on every run. |

### ocean_style.py additional flags

| Flag | Default | What it does |
|---|---|---|
| `--max-trees` | `0` | How many trees of the ensemble are encoded into the MILP; `0` means all of them. |
| `--solver-name` | `cbc` | MILP solver to use (CBC ships with PuLP; Gurobi/HiGHS also supported if installed). |
| `--time-limit-seconds` | `180` | Solver time limit per counterfactual. |
| `--mip-gap` | `0.02` | Solver optimality-gap tolerance. |
| `--proximity-weight` | `1.0` | Weight on the proximity term in the MILP objective. |
| `--sparsity-weight` | `0.05` | Weight on the sparsity term. |
| `--milp-target-proba` | `0.0` (means: use `--min-query-proba`) | Internal probability target used by the partial-ensemble MILP when `--max-trees` is less than the full model. |
| `--min-changed-features` | `0` | Require at least this many mutable features to change in the solution. |

### make_counterfactual_query_batches.py

| Flag | Default | What it does |
|---|---|---|
| `--predictions` | required | Path to a `predictions_<split>` file from `train_xgboost.py` (`.parquet` or `.csv`, or the bare path with neither extension). |
| `--labels` | none | Optional `y_<split>` file with a `label` column; only needed if `--predictions` doesn't already have a `label`/`y_true` column. |
| `--metrics` | none | Optional `metrics.json`; used to read the tuned decision threshold when `pred_label` is missing from `--predictions` and `--threshold` isn't given directly. |
| `--threshold` | none | Explicit probability threshold to derive `pred_label` from `pred_proba`, if `pred_label` isn't already present. |
| `--source-label` | `1` | Predicted class to sample from (0 or 1). |
| `--include-all-predictions` | off | Skip filtering by predicted class before sampling. |
| `--true-label-one-only` | off | Additionally restrict to rows whose true label is 1 (a true-positive cohort when combined with `--source-label 1`). |
| `--num-samples` | required | Number of `row_id`s per batch. |
| `--num-batches` | `1` | How many batch CSVs to write in one call. |
| `--allow-overlap` | off | Allow the same `row_id` to appear in more than one batch. |
| `--random-state` | `42` | Random seed for sampling. |
| `--out-dir` | required | Folder to write the batch CSV(s) and manifest JSON to. |
| `--prefix` | `query_ids` | Filename prefix for the output CSVs and manifest. |

### build_dice_knn_imputed_dataset.py

Shares `--data-dir`, `--model-dir` (both required, no default),
`--query-split` (default `test`), `--max-queries` (default `250`),
`--query-offset` (default `0`), `--min-query-proba` (default `0.5`),
`--query-ids-path`, `--random-query-sample`, `--random-state` (default
`42`), and `--knn-neighbors` (default `5`) with the same meaning as the
flags above -- it selects the same factual set `dice_knn.py` would, then
builds and caches the KNN imputation for it. Additionally, `--out-dir` is the directory to write the cached imputed frames (`x_train_imputed.pkl`, `selected_factuals_native.pkl`, `selected_factuals_imputed.pkl`, `metadata.json`) to.

### pipeline.py

| Flag | Default | What it does |
|---|---|---|
| `--stages` | `all` | Comma-separated stage names to run, in order (`preprocess`, `train`, `query_ids`, `facet`, `dice_mean`, `dice_median`, `dice_knn`, `ocean`), or `all`. |
| `--data-raw-dir` | `data/mimic_raw` | Passed as `--data-dir` to the `preprocess` stage. |
| `--data-processed-dir` | `data/processed` | Passed as `--out-dir` to `preprocess` and `--data-dir` to every later stage. |
| `--model-dir` | `data/model` | Passed as `--out-dir` to `train` and `--model-dir` to every later stage. |
| `--results-dir` | `results/counterfactual_experiments` | Passed as `--out-root` to every CF stage. |
| `--query-ids-dir` | `data/query_ids` | Passed as `--out-dir` to the `query_ids` stage. |
| `--preprocess-args` | empty | Extra raw arguments appended to the `preprocess` stage command. |
| `--train-args` | empty | Extra raw arguments appended to the `train` stage command. |
| `--query-ids-args` | `--source-label 1 --true-label-one-only --num-samples 300 --num-batches 1` | Extra raw arguments appended to the `query_ids` stage command. |
| `--cf-args` | empty | Extra raw arguments appended to every CF stage command (`facet`, `dice_mean`, `dice_median`, `dice_knn`, `ocean`) identically. |
| `--python` | current interpreter | Python interpreter used to run each stage. |
| `--strict` | off | Stop the whole pipeline on the first failing stage, including the non-core ones. |
| `--dry-run` | off | Print each selected stage's command without running it. |
| `--list` | -- | Print the available stage names and exit. |

### Example: one pipeline.py call, custom parameters per stage

```bash
python pipeline.py \
  --preprocess-args "--hours-window 48 --test-size 0.15 --val-size 0.15" \
  --train-args "--n-estimators 2000 --learning-rate 0.02 --max-depth 4" \
  --query-ids-args "--source-label 1 --true-label-one-only --num-samples 300 --num-batches 1" \
  --cf-args "--max-queries 300 --query-ids-path data/query_ids/query_ids_batch001.csv --allow-treatments"
```

This reproduces the default preprocessing/training configuration explicitly,
builds a 300-patient true-positive cohort, and runs every CF stage against
that exact cohort with treatment/exposure features included in the search
space.
