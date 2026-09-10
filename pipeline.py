#!/usr/bin/env python3
"""
pipeline.py -- run the whole thesis pipeline (or any subset of it) in one go.

Stages, in the order they normally run:

  1. preprocess   preprocessing.py                     raw MIMIC-IV CSVs      -> train/val/test feature tables
  2. train        train_xgboost.py                     feature tables          -> XGBoost model + metrics
  3. query_ids    make_counterfactual_query_batches.py  model predictions       -> fixed row_id CSV(s) for a reusable factual cohort
  4. facet        facet_style.py                       model + feature tables  -> FACET-style counterfactuals
  5. dice_mean    dice_mean.py                         model + feature tables  -> DiCE-genetic counterfactuals (mean-imputed)
  6. dice_median  dice_median.py                       model + feature tables  -> DiCE-genetic counterfactuals (median-imputed)
  7. dice_knn     dice_knn.py                          model + feature tables  -> DiCE-genetic counterfactuals (KNN-imputed)
  8. ocean        ocean_style.py                       model + feature tables  -> OCEAN-style MILP counterfactuals

Stages 3-8 are independent of each other. Stage 1
needs a MIMIC-IV extract provided (see README.md); stage 2
needs stage 1's output; stages 3-8 need stage 2's output. Stage 3 is optional:
skip it and each CF stage (4-8) will pick its own factual queries (highest
predicted-risk patients) instead of a fixed, reusable cohort.

Every stage is just a normal standalone script; this file
only chains them together with a shared, consistent directory layout.

Examples
--------
  # Everything, with the optional dependencies (dice-ml, pulp) installed:
  python pipeline.py

  # Only rebuild the model from already-preprocessed data:
  python pipeline.py --stages train,facet

  # See what would run without running it:
  python pipeline.py --dry-run

  # Point at data that lives outside this checkout, and cap each CF method to
  # 20 queries for a fast end-to-end check:
  python pipeline.py --data-raw-dir /path/to/mimic-iv --cf-args "--max-queries 20"

  # Build a fixed 300-patient true-positive factual cohort, then reuse the
  # exact same 300 patients across every CF method (see README.md for why
  # --max-queries must be passed explicitly here):
  python pipeline.py --stages preprocess,train,query_ids
  python pipeline.py --stages facet,dice_mean,dice_median,dice_knn,ocean \\
      --cf-args "--query-ids-path data/query_ids/query_ids_batch001.csv --max-queries 300"
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent

# stage name -> (script filename, is core stage)
# Core stages (preprocess, train) are load-bearing: if they fail, everything
# after them is meaningless, so the pipeline stops. The counterfactual stages
# are independent of each other and depend on optional third-party packages
# (dice-ml, pulp) that may not be installed, so by default a failure there is
# reported and the pipeline moves on to the next stage -- pass --strict to
# stop on the first failure instead.
STAGES: dict[str, tuple[str, bool]] = {
    "preprocess": ("preprocessing.py", True),
    "train": ("train_xgboost.py", True),
    "query_ids": ("make_counterfactual_query_batches.py", False),
    "facet": ("facet_style.py", False),
    "dice_mean": ("dice_mean.py", False),
    "dice_median": ("dice_median.py", False),
    "dice_knn": ("dice_knn.py", False),
    "ocean": ("ocean_style.py", False),
}
ALL_STAGES = list(STAGES.keys())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the counterfactual-explanation thesis pipeline end to end.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--stages",
        default="all",
        help=f"Comma-separated stage names to run, in the order given, or 'all' (default). Choices: {', '.join(ALL_STAGES)}.",
    )
    parser.add_argument("--data-raw-dir", default=str(PROJECT_DIR / "data" / "mimic_raw"), help="Raw MIMIC-IV CSVs (input to 'preprocess'). Default: data/mimic_raw")
    parser.add_argument("--data-processed-dir", default=str(PROJECT_DIR / "data" / "processed"), help="Preprocessed feature tables (output of 'preprocess', input to everything else). Default: data/processed")
    parser.add_argument("--model-dir", default=str(PROJECT_DIR / "data" / "model"), help="Trained model + metrics (output of 'train', input to the CF stages). Default: data/model")
    parser.add_argument("--results-dir", default=str(PROJECT_DIR / "results" / "counterfactual_experiments"), help="Root folder each CF stage writes its own subfolder into. Default: results/counterfactual_experiments")
    parser.add_argument("--query-ids-dir", default=str(PROJECT_DIR / "data" / "query_ids"), help="Output folder for the fixed-cohort row_id CSV(s) written by the 'query_ids' stage. Default: data/query_ids")
    parser.add_argument("--preprocess-args", default="", help="Extra raw arguments appended to the preprocess stage, e.g. \"--hours-window 24\".")
    parser.add_argument("--train-args", default="", help="Extra raw arguments appended to the train stage, e.g. \"--n-estimators 500\".")
    parser.add_argument(
        "--query-ids-args",
        default="--source-label 1 --true-label-one-only --num-samples 300 --num-batches 1",
        help=(
            "Extra raw arguments appended to the query_ids stage. The default "
            "reproduces a 300-patient true-positive factual cohort from the test "
            "split (predicted positive AND actually died at the model's tuned "
            "threshold). Override this for a different cohort definition/size, "
            "e.g. \"--include-all-predictions --num-samples 50\". This stage will "
            "fail (non-fatally -- see --strict) if fewer than --num-samples "
            "eligible rows exist, e.g. on a small smoke-test dataset."
        ),
    )
    parser.add_argument("--cf-args", default="", help="Extra raw arguments appended to every CF stage (facet/dice_*/ocean), e.g. \"--max-queries 50 --allow-treatments\".")
    parser.add_argument("--python", default=sys.executable, help="Python interpreter to run each stage with. Default: the interpreter running this script.")
    parser.add_argument("--strict", action="store_true", help="Stop the whole pipeline on the first failing stage, including optional CF stages (default: only the two core stages, preprocess/train, are stop-on-failure).")
    parser.add_argument("--dry-run", action="store_true", help="Print the command for each selected stage without running anything.")
    parser.add_argument("--list", action="store_true", help="Print the available stage names and exit.")
    return parser.parse_args()


def resolve_stage_list(spec: str) -> list[str]:
    if spec.strip().lower() == "all":
        return list(ALL_STAGES)
    names = [s.strip() for s in spec.split(",") if s.strip()]
    unknown = [n for n in names if n not in STAGES]
    if unknown:
        raise SystemExit(f"Unknown stage(s): {', '.join(unknown)}. Choices: {', '.join(ALL_STAGES)}")
    return names


def build_command(python: str, stage: str, args: argparse.Namespace) -> list[str]:
    script, _ = STAGES[stage]
    cmd = [python, str(PROJECT_DIR / script)]

    if stage == "preprocess":
        cmd += ["--data-dir", args.data_raw_dir, "--out-dir", args.data_processed_dir]
        cmd += shlex.split(args.preprocess_args)
    elif stage == "train":
        cmd += ["--data-dir", args.data_processed_dir, "--out-dir", args.model_dir]
        cmd += shlex.split(args.train_args)
    elif stage == "query_ids":
        # Bare stem (no suffix): make_counterfactual_query_batches.py resolves
        # this to predictions_test.parquet or .csv, whichever train_xgboost.py
        # actually wrote (parquet if pyarrow is installed, csv otherwise).
        cmd += [
            "--predictions", str(Path(args.model_dir) / "predictions_test"),
            "--metrics", str(Path(args.model_dir) / "metrics.json"),
            "--out-dir", args.query_ids_dir,
        ]
        cmd += shlex.split(args.query_ids_args)
    else:  # facet, dice_mean, dice_median, dice_knn, ocean
        cmd += [
            "--data-dir", args.data_processed_dir,
            "--model-dir", args.model_dir,
            "--out-root", args.results_dir,
        ]
        cmd += shlex.split(args.cf_args)
    return cmd


def run_stage(stage: str, cmd: list[str], dry_run: bool) -> int:
    header = f" Stage: {stage} ".center(78, "=")
    print(f"\n{header}")
    print(" ".join(shlex.quote(p) for p in cmd))
    if dry_run:
        return 0

    start = time.monotonic()
    result = subprocess.run(cmd, cwd=str(PROJECT_DIR))
    elapsed = time.monotonic() - start
    status = "OK" if result.returncode == 0 else f"FAILED (exit code {result.returncode})"
    print(f"-- {stage}: {status} in {elapsed:.1f}s")
    return result.returncode


def main() -> None:
    args = parse_args()

    if args.list:
        for name, (script, core) in STAGES.items():
            kind = "core" if core else "counterfactual method"
            print(f"  {name:<12} -> {script:<20} ({kind})")
        return

    stages = resolve_stage_list(args.stages)

    print("Pipeline directories:")
    print("  raw data      :", args.data_raw_dir)
    print("  processed data:", args.data_processed_dir)
    print("  model         :", args.model_dir)
    print("  query ids     :", args.query_ids_dir)
    print("  CF results    :", args.results_dir)
    print("Stages to run   :", ", ".join(stages))

    failures: list[str] = []
    for stage in stages:
        _, is_core = STAGES[stage]
        cmd = build_command(args.python, stage, args)
        code = run_stage(stage, cmd, args.dry_run)
        if code != 0:
            failures.append(stage)
            if is_core or args.strict:
                print(f"\nStopping: '{stage}' failed and is a stop-on-failure stage "
                      f"({'core stage' if is_core else '--strict was set'}).")
                sys.exit(code)
            print(f"'{stage}' failed but is not stop-on-failure; continuing "
                  f"(pass --strict to stop on any failure).")

    print("\n" + "=" * 78)
    if failures:
        print(f"Pipeline finished with {len(failures)} failed stage(s): {', '.join(failures)}")
        sys.exit(1)
    print("Pipeline finished: all selected stages succeeded.")


if __name__ == "__main__":
    main()
