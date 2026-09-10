from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict

import pandas as pd

from cf_experiment_utils import ensure_dir, load_training_artifacts, select_query_set

from dice_knn import (
    DiceGeneticNativeMissingKNNWrapperConfig,
    build_imputation_values,
    impute_with_training_values,
)


CACHE_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a reusable KNN-imputed dataset for "
            "dice_knn.py."
        )
    )
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--out-dir", required=True, help="Directory where the cached imputed frames will be written.")
    parser.add_argument("--query-split", default=None, choices=["train", "val", "test"])
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--query-offset", type=int, default=None)
    parser.add_argument("--min-query-proba", type=float, default=None)
    parser.add_argument("--query-ids-path", default=None)
    parser.add_argument("--random-query-sample", action="store_true")
    parser.add_argument("--random-state", type=int, default=None)
    parser.add_argument("--knn-neighbors", type=int, default=None)
    return parser.parse_args()


def apply_args(cfg: DiceGeneticNativeMissingKNNWrapperConfig, args: argparse.Namespace) -> None:
    cfg.data_dir = args.data_dir
    cfg.model_dir = args.model_dir
    if args.query_split is not None:
        cfg.query_split = args.query_split
    if args.max_queries is not None:
        cfg.max_queries = args.max_queries
    if args.query_offset is not None:
        cfg.query_offset = args.query_offset
    if args.min_query_proba is not None:
        cfg.min_query_proba = args.min_query_proba
    if args.query_ids_path is not None:
        cfg.query_ids_path = args.query_ids_path
    if args.random_query_sample:
        cfg.random_query_sample = True
    if args.random_state is not None:
        cfg.random_state = args.random_state
    if args.knn_neighbors is not None:
        cfg.knn_neighbors = args.knn_neighbors
    if int(cfg.knn_neighbors) <= 0:
        raise ValueError("knn_neighbors must be a positive integer.")


def write_frame(df: pd.DataFrame, out_dir: Path, name: str) -> str:
    filename = f"{name}.pkl"
    df.to_pickle(out_dir / filename)
    return filename


def main() -> None:
    args = parse_args()
    cfg = DiceGeneticNativeMissingKNNWrapperConfig()
    apply_args(cfg, args)

    artifacts = load_training_artifacts(cfg)
    cfg.min_query_proba = float(artifacts["model_threshold"])
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    model = artifacts["model"]
    x_train_native = artifacts["x_train"].copy()
    x_query_native = artifacts[f"x_{cfg.query_split}"].copy()
    y_query = artifacts[f"y_{cfg.query_split}"]
    x_query_raw = artifacts[f"x_{cfg.query_split}_raw"]

    print("DEBUG data_dir:", cfg.data_dir)
    print("DEBUG model_dir:", cfg.model_dir)
    print("DEBUG loaded threshold:", cfg.min_query_proba)
    print("DEBUG building KNN imputer with neighbors:", cfg.knn_neighbors)

    imputation_values = build_imputation_values(x_train_native, cfg.knn_neighbors)
    x_train_imputed = impute_with_training_values(x_train_native, imputation_values)
    factuals_native = select_query_set(x_query_raw, x_query_native, y_query, model, cfg)
    factuals_imputed = impute_with_training_values(factuals_native, imputation_values)

    files: Dict[str, str] = {
        "x_train_imputed": write_frame(x_train_imputed, out_dir, "x_train_imputed"),
        "selected_factuals_native": write_frame(factuals_native, out_dir, "selected_factuals_native"),
        "selected_factuals_imputed": write_frame(factuals_imputed, out_dir, "selected_factuals_imputed"),
    }

    metadata = {
        "cache_version": CACHE_VERSION,
        "files": files,
        "data_dir": str(Path(cfg.data_dir)),
        "model_dir": str(Path(cfg.model_dir)),
        "query_split": cfg.query_split,
        "query_ids_path": cfg.query_ids_path,
        "query_offset": int(getattr(cfg, "query_offset", 0) or 0),
        "max_queries": int(cfg.max_queries),
        "random_query_sample": bool(cfg.random_query_sample),
        "random_state": int(cfg.random_state),
        "model_threshold": float(cfg.min_query_proba),
        "knn_neighbors": int(cfg.knn_neighbors),
        "knn_scaled": True,
        "knn_scaler": "StandardScaler",
        "knn_fit_scope": "whole_training_feature_matrix_only_no_labels",
        "knn_fallback_strategy": "train_median_else_0_for_fully_missing_features",
        "num_training_rows": int(len(x_train_imputed)),
        "num_training_features": int(len(x_train_imputed.columns)),
        "num_selected_factuals": int(len(factuals_imputed)),
        "feature_columns": list(x_train_imputed.columns),
        "num_fully_missing_training_features": int(len(imputation_values.fully_missing_features)),
        "fully_missing_training_features": list(imputation_values.fully_missing_features),
        "builder_config": asdict(cfg),
    }
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("Wrote KNN-imputed dataset cache:", out_dir)
    print("Selected factual rows:", len(factuals_imputed))
    print("Training feature matrix shape:", x_train_imputed.shape)


if __name__ == "__main__":
    main()
