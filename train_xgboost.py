from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    roc_auc_score,
)
from xgboost import XGBClassifier

# Defaults are relative to this script's folder and match preprocessing.py's
# out_dir, so `python preprocessing.py && python train_xgboost.py` works with
# no flags. Override with --data-dir / --out-dir to point elsewhere.
PROJECT_DIR = Path(__file__).resolve().parent


@dataclass
class TrainConfig:
    data_dir: str = str(PROJECT_DIR / "data" / "processed")
    out_dir: str = str(PROJECT_DIR / "data" / "model")
    random_state: int = 42

    # Columns to drop from the design matrix if present.
    id_columns: Tuple[str, ...] = ("stay_id", "subject_id", "hadm_id")

    # XGBoost hyperparameters
    n_estimators: int = 2000
    learning_rate: float = 0.02
    max_depth: int = 4
    min_child_weight: float = 15
    subsample: float = 0.7
    colsample_bytree: float = 0.5
    reg_alpha: float = 1
    reg_lambda: float = 10
    early_stopping_rounds: int = 100
    eval_metric: Tuple[str, ...] = ("auc", "aucpr", "logloss")
    threshold_metric: str = "f1"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def read_split_table(data_dir: Path, stem: str) -> pd.DataFrame:
    parquet_path = data_dir / f"{stem}.parquet"
    csv_path = data_dir / f"{stem}.csv"

    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path)

    raise FileNotFoundError(f"Could not find {parquet_path.name} or {csv_path.name}")

def load_split(data_dir: Path, split: str) -> Tuple[pd.DataFrame, pd.Series]:
    x = read_split_table(data_dir, f"X_{split}")
    y = read_split_table(data_dir, f"y_{split}")
    return x, y["label"].astype(int)

def write_table(df: pd.DataFrame, path_without_suffix: Path) -> str:
    parquet_path = path_without_suffix.with_suffix(".parquet")
    csv_path = path_without_suffix.with_suffix(".csv")
    try:
        df.to_parquet(parquet_path, index=False)
        return parquet_path.name
    except ImportError:
        df.to_csv(csv_path, index=False)
        return csv_path.name


def prepare_matrix(df: pd.DataFrame, cfg: TrainConfig, reference_columns: pd.Index | None = None) -> pd.DataFrame:
    x = df.copy()
    drop_cols = [c for c in cfg.id_columns if c in x.columns]
    if drop_cols:
        x = x.drop(columns=drop_cols)

    bool_cols = x.select_dtypes(include=["bool"]).columns
    if len(bool_cols) > 0:
        x[bool_cols] = x[bool_cols].astype(np.int8)

    if reference_columns is not None:
        x = x.reindex(columns=reference_columns, fill_value=0)

    return x


def compute_scale_pos_weight(y: pd.Series) -> float:
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    if pos == 0:
        return 1.0
    return neg / pos


def find_best_threshold(y_true: pd.Series, y_prob: np.ndarray, metric: str = "f1") -> Tuple[float, float]:
    thresholds = np.linspace(0.05, 0.95, 181)
    best_threshold = 0.5
    best_score = -np.inf

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        if metric == "f1":
            score = f1_score(y_true, y_pred, zero_division=0)
        else:
            raise ValueError(f"Unsupported threshold metric: {metric}")

        if score > best_score:
            best_score = score
            best_threshold = float(threshold)

    return best_threshold, float(best_score)


def evaluate_split(y_true: pd.Series, y_prob: np.ndarray, threshold: float) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "average_precision": float(average_precision_score(y_true, y_prob)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, y_prob, labels=[0, 1])),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "threshold": float(threshold),
        "prevalence": float(np.mean(y_true)),
        "predicted_positive_rate": float(np.mean(y_pred)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def build_model(cfg: TrainConfig, scale_pos_weight: float) -> XGBClassifier:
    return XGBClassifier(
        objective="binary:logistic",
        n_estimators=cfg.n_estimators,
        learning_rate=cfg.learning_rate,
        max_depth=cfg.max_depth,
        min_child_weight=cfg.min_child_weight,
        subsample=cfg.subsample,
        colsample_bytree=cfg.colsample_bytree,
        reg_alpha=cfg.reg_alpha,
        reg_lambda=cfg.reg_lambda,
        scale_pos_weight=scale_pos_weight,
        random_state=cfg.random_state,
        tree_method="hist",
        enable_categorical=False,
        eval_metric=list(cfg.eval_metric),
        early_stopping_rounds=cfg.early_stopping_rounds,
    )


def save_predictions(
    out_dir: Path,
    split: str,
    x_raw: pd.DataFrame,
    y_true: pd.Series,
    y_prob: np.ndarray,
    threshold: float,
) -> None:
    pred_df = pd.DataFrame(
        {
            "label": y_true.values,
            "pred_proba": y_prob,
            "pred_label": (y_prob >= threshold).astype(int),
        }
    )

    for c in ["stay_id", "subject_id", "hadm_id"]:
        if c in x_raw.columns:
            pred_df.insert(len(pred_df.columns), c, x_raw[c].values)

    write_table(pred_df, out_dir / f"predictions_{split}")




def save_feature_importance(out_dir: Path, model: XGBClassifier, feature_names: pd.Index) -> None:
    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "importance_gain": model.feature_importances_,
        }
    ).sort_values("importance_gain", ascending=False)
    importance.to_csv(out_dir / "feature_importance_gain.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the XGBoost in-hospital mortality model on preprocessed splits.")
    parser.add_argument("--data-dir", default=None, help="Folder containing X_/y_{train,val,test} tables from preprocessing.py.")
    parser.add_argument("--out-dir", default=None, help="Folder to write the model, metrics, and predictions to.")
    parser.add_argument("--n-estimators", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--random-state", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    cfg = TrainConfig()
    args = parse_args()
    if args.data_dir is not None:
        cfg.data_dir = args.data_dir
    if args.out_dir is not None:
        cfg.out_dir = args.out_dir
    if args.n_estimators is not None:
        cfg.n_estimators = args.n_estimators
    if args.learning_rate is not None:
        cfg.learning_rate = args.learning_rate
    if args.max_depth is not None:
        cfg.max_depth = args.max_depth
    if args.random_state is not None:
        cfg.random_state = args.random_state

    data_dir = Path(cfg.data_dir)
    out_dir = Path(cfg.out_dir)
    ensure_dir(out_dir)

    print("Loading processed splits...")
    x_train_raw, y_train = load_split(data_dir, "train")
    x_val_raw, y_val = load_split(data_dir, "val")
    x_test_raw, y_test = load_split(data_dir, "test")

    x_train = prepare_matrix(x_train_raw, cfg)
    x_val = prepare_matrix(x_val_raw, cfg, reference_columns=x_train.columns)
    x_test = prepare_matrix(x_test_raw, cfg, reference_columns=x_train.columns)
    
    x_train, rename_map = sanitize_feature_names(x_train)
    x_val = x_val.rename(columns=rename_map)
    x_test = x_test.rename(columns=rename_map)

    scale_pos_weight = compute_scale_pos_weight(y_train)
    print(f"Train rows: {len(x_train)} | Features: {x_train.shape[1]}")
    print(f"Train prevalence: {y_train.mean():.4f}")
    print(f"scale_pos_weight: {scale_pos_weight:.4f}")

    model = build_model(cfg, scale_pos_weight)
    print("Training XGBoost...")
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_train, y_train), (x_val, y_val)],
        verbose=100,
    )

    print("Scoring splits...")
    train_prob = model.predict_proba(x_train)[:, 1]
    val_prob = model.predict_proba(x_val)[:, 1]
    test_prob = model.predict_proba(x_test)[:, 1]

    best_threshold, best_val_threshold_score = find_best_threshold(
        y_true=y_val,
        y_prob=val_prob,
        metric=cfg.threshold_metric,
    )
    print(f"Best validation threshold ({cfg.threshold_metric}): {best_threshold:.3f}")

    metrics = {
        "train": evaluate_split(y_train, train_prob, best_threshold),
        "val": evaluate_split(y_val, val_prob, best_threshold),
        "test": evaluate_split(y_test, test_prob, best_threshold),
        "threshold_selection": {
            "metric": cfg.threshold_metric,
            "best_threshold": best_threshold,
            "best_validation_score": best_val_threshold_score,
        },
        "best_iteration": int(getattr(model, "best_iteration", model.n_estimators)),
        "scale_pos_weight": float(scale_pos_weight),
    }

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    with open(out_dir / "train_config_used.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    print("Saving outputs...")
    model.save_model(out_dir / "xgboost_model.json")
    save_predictions(out_dir, "train", x_train_raw, y_train, train_prob, best_threshold)
    save_predictions(out_dir, "val", x_val_raw, y_val, val_prob, best_threshold)
    save_predictions(out_dir, "test", x_test_raw, y_test, test_prob, best_threshold)
    save_feature_importance(out_dir, model, x_train.columns)

    with open(out_dir / "feature_name_map.json", "w") as f:
        json.dump(rename_map, f, indent=2)

    print("Done.")
    print(f"Saved outputs to: {out_dir}")
    print(json.dumps(metrics["val"], indent=2))
    print(json.dumps(metrics["test"], indent=2))

def sanitize_feature_names(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    rename_map = {}
    seen = set()

    for col in df.columns:
        new_col = str(col)
        new_col = new_col.replace("[", "_")
        new_col = new_col.replace("]", "_")
        new_col = new_col.replace("<", "_lt_")
        new_col = new_col.replace(">", "_gt_")
        new_col = new_col.replace(" ", "_")
        new_col = new_col.replace("/", "_")
        new_col = new_col.replace("\\", "_")
        new_col = new_col.replace("(", "_")
        new_col = new_col.replace(")", "_")
        new_col = new_col.replace(",", "_")
        new_col = new_col.replace(":", "_")
        new_col = new_col.replace(";", "_")
        new_col = new_col.replace("-", "_")

        while "__" in new_col:
            new_col = new_col.replace("__", "_")
        new_col = new_col.strip("_")

        if new_col == "":
            new_col = "feature"

        base = new_col
        i = 1
        while new_col in seen:
            new_col = f"{base}_{i}"
            i += 1

        seen.add(new_col)
        rename_map[col] = new_col

    return df.rename(columns=rename_map), rename_map


if __name__ == "__main__":
    main()
