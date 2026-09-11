from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

PROJECT_DIR = Path(__file__).resolve().parent


@dataclass
class CFExperimentConfig:
    data_dir: str = str(PROJECT_DIR / "data" / "processed")
    model_dir: str = str(PROJECT_DIR / "data" / "model")
    out_root: str = str(PROJECT_DIR / "results" / "counterfactual_experiments")
    random_state: int = 42
    id_columns: Tuple[str, ...] = ("stay_id", "subject_id", "hadm_id")
    target_positive_label: int = 1
    query_split: str = "test"
    max_queries: int = 100
    query_offset: int = 0
    min_query_proba: float = 0.5
    mutable_labels: Tuple[str, ...] = ("mutable_with_clinical_constraints",)
    allow_treatments: bool = False
    desired_class: str = "opposite"
    query_ids_path: str = ""
    random_query_sample: bool = False


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_table(data_dir: Path, stem: str) -> pd.DataFrame:
    parquet_path = data_dir / f"{stem}.parquet"
    csv_path = data_dir / f"{stem}.csv"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path)
    raise FileNotFoundError(f"Could not find {parquet_path.name} or {csv_path.name}")


def load_split(data_dir: Path, split: str) -> Tuple[pd.DataFrame, pd.Series]:
    x = read_table(data_dir, f"X_{split}")
    y = read_table(data_dir, f"y_{split}")
    return x, y["label"].astype(int)


def prepare_matrix(
    df: pd.DataFrame,
    id_columns: Sequence[str],
    reference_columns: Optional[pd.Index] = None,
) -> pd.DataFrame:
    x = df.copy()
    drop_cols = [c for c in id_columns if c in x.columns]
    if drop_cols:
        x = x.drop(columns=drop_cols)

    bool_cols = x.select_dtypes(include=["bool"]).columns
    if len(bool_cols) > 0:
        x[bool_cols] = x[bool_cols].astype(np.int8)

    if reference_columns is not None:
        x = x.reindex(columns=reference_columns, fill_value=0)

    return x


def sanitize_feature_names(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    rename_map: dict[str, str] = {}
    seen = set()

    for col in df.columns:
        new_col = str(col)
        for old, new in [
            ("[", "_"),
            ("]", "_"),
            ("<", "_lt_"),
            (">", "_gt_"),
            (" ", "_"),
            ("/", "_"),
            ("\\", "_"),
            ("(", "_"),
            (")", "_"),
            (",", "_"),
            (":", "_"),
            (";", "_"),
            ("-", "_"),
        ]:
            new_col = new_col.replace(old, new)

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
        rename_map[str(col)] = new_col

    return df.rename(columns=rename_map), rename_map


def load_feature_name_map(model_dir: Path) -> Dict[str, str]:
    path = model_dir / "feature_name_map.json"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def apply_feature_name_map(df: pd.DataFrame, feature_name_map: Dict[str, str]) -> pd.DataFrame:
    if not feature_name_map:
        sanitized, _ = sanitize_feature_names(df)
        return sanitized
    return df.rename(columns=feature_name_map)


def load_counterfactual_metadata(data_dir: Path, feature_name_map: Dict[str, str]) -> pd.DataFrame:
    metadata = pd.read_csv(data_dir / "counterfactual_feature_metadata.csv")
    metadata["feature_original"] = metadata["feature"].astype(str)
    metadata["feature"] = metadata["feature"].astype(str).map(feature_name_map).fillna(metadata["feature"].astype(str))
    return metadata


def load_preprocessing_spec(data_dir: Path) -> Dict[str, object]:
    with open(data_dir / "preprocessing_spec.json", "r", encoding="utf-8") as f:
        return json.load(f)


def load_model(model_dir: Path) -> XGBClassifier:
    model = XGBClassifier()
    model.load_model(model_dir / "xgboost_model.json")
    return model


def load_training_artifacts(cfg: CFExperimentConfig) -> Dict[str, object]:
    data_dir = Path(cfg.data_dir)
    model_dir = Path(cfg.model_dir)

    x_train_raw, y_train = load_split(data_dir, "train")
    x_val_raw, y_val = load_split(data_dir, "val")
    x_test_raw, y_test = load_split(data_dir, "test")

    x_train = prepare_matrix(x_train_raw, cfg.id_columns)
    x_val = prepare_matrix(x_val_raw, cfg.id_columns, reference_columns=x_train.columns)
    x_test = prepare_matrix(x_test_raw, cfg.id_columns, reference_columns=x_train.columns)

    feature_name_map = load_feature_name_map(model_dir)
    if feature_name_map:
        x_train = apply_feature_name_map(x_train, feature_name_map)
        x_val = apply_feature_name_map(x_val, feature_name_map)
        x_test = apply_feature_name_map(x_test, feature_name_map)
    else:
        x_train, inferred_map = sanitize_feature_names(x_train)
        x_val = x_val.rename(columns=inferred_map)
        x_test = x_test.rename(columns=inferred_map)
        feature_name_map = inferred_map

    cf_metadata = load_counterfactual_metadata(data_dir, feature_name_map)
    preprocessing_spec = load_preprocessing_spec(data_dir)
    model = load_model(model_dir)
    model_threshold = load_model_threshold(model_dir, default_threshold=cfg.min_query_proba)


    model_feature_names = model.get_booster().feature_names
    if model_feature_names is not None:
        model_feature_names = list(model_feature_names)

        x_train = x_train.reindex(columns=model_feature_names, fill_value=0)
        x_val = x_val.reindex(columns=model_feature_names, fill_value=0)
        x_test = x_test.reindex(columns=model_feature_names, fill_value=0)

        cf_metadata = cf_metadata[cf_metadata["feature"].isin(model_feature_names)].copy()

    return {
        "data_dir": data_dir,
        "model_dir": model_dir,
        "x_train_raw": x_train_raw,
        "x_val_raw": x_val_raw,
        "x_test_raw": x_test_raw,
        "x_train": x_train,
        "x_val": x_val,
        "x_test": x_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "feature_name_map": feature_name_map,
        "cf_metadata": cf_metadata,
        "preprocessing_spec": preprocessing_spec,
        "model": model,
        "model_threshold": model_threshold,
    }



def mutable_feature_mask(metadata: pd.DataFrame, cfg: CFExperimentConfig) -> pd.Series:
    allowed_labels = set(cfg.mutable_labels)

    if cfg.allow_treatments:
        allowed_labels.add("non_actionable_by_default")
        allowed_labels.add("non_actionable_or_constrained")

    return (
        metadata["mutability"].isin(allowed_labels)
        & ~metadata["feature"].astype(str).str.endswith("_missing")
    )


def get_mutable_features(metadata: pd.DataFrame, cfg: CFExperimentConfig) -> List[str]:
    return metadata.loc[mutable_feature_mask(metadata, cfg), "feature"].astype(str).tolist()


def get_immutable_features(metadata: pd.DataFrame, cfg: CFExperimentConfig) -> List[str]:
    return metadata.loc[
        ~mutable_feature_mask(metadata, cfg),
        "feature",
    ].astype(str).tolist()


def get_permitted_range(metadata: pd.DataFrame, features: Iterable[str]) -> Dict[str, List[float]]:
    feature_set = set(features)
    subset = metadata[metadata["feature"].isin(feature_set)].copy()
    ranges: Dict[str, List[float]] = {}
    for row in subset.itertuples(index=False):
        if pd.isna(row.train_min) or pd.isna(row.train_max):
            continue
        lo = float(row.train_min)
        hi = float(row.train_max)
        if lo == hi:
            continue
        ranges[str(row.feature)] = [lo, hi]
    return ranges

def load_model_threshold(model_dir: Path, default_threshold: float = 0.5) -> float:
    metrics_path = model_dir / "metrics.json"
    if not metrics_path.exists():
        return float(default_threshold)

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    threshold = (
        metrics.get("threshold_selection", {})
        .get("best_threshold", default_threshold)
    )
    return float(threshold)

def load_query_row_ids(path: str | Path) -> List[int]:
    query_ids = pd.read_csv(path)
    if "row_id" in query_ids.columns:
        series = query_ids["row_id"]
    else:
        series = query_ids.iloc[:, 0]
    return [int(x) for x in series.dropna().tolist()]


def apply_query_window(selected: pd.DataFrame, cfg: CFExperimentConfig) -> pd.DataFrame:
    query_offset = max(0, int(getattr(cfg, "query_offset", 0) or 0))
    max_queries = int(cfg.max_queries) if cfg.max_queries is not None else 0
    if query_offset <= 0 and max_queries <= 0:
        return selected
    if max_queries > 0:
        return selected.iloc[query_offset : query_offset + max_queries].copy()
    return selected.iloc[query_offset:].copy()


def select_query_set(
    x_query_raw: pd.DataFrame,
    x_query: pd.DataFrame,
    y_query: pd.Series,
    model,
    cfg: CFExperimentConfig,
) -> pd.DataFrame:
    split = str(cfg.query_split)

    # Recompute predictions directly on the exact feature matrix being used.
    # This avoids row-order mismatch between X_test.csv and predictions_test.csv.
    pred_proba = model.predict_proba(x_query)[:, 1]
    pred_label = (pred_proba >= float(cfg.min_query_proba)).astype(int)

    df = x_query.copy().reset_index(drop=True)
    df["row_id"] = np.arange(len(df))
    df["y_true"] = y_query.reset_index(drop=True).to_numpy()
    df["pred_proba"] = pred_proba
    df["pred_label"] = pred_label

    for col in ["stay_id", "subject_id", "hadm_id"]:
        if col in x_query_raw.columns:
            df[col] = x_query_raw.reset_index(drop=True)[col].to_numpy()

    query_ids_path = str(getattr(cfg, "query_ids_path", "") or "")
    if query_ids_path:
        row_ids = load_query_row_ids(query_ids_path)
        selected = df.set_index("row_id", drop=False).reindex(row_ids)
        missing = selected[selected["row_id"].isna()].index.tolist()
        if missing:
            raise ValueError(f"Query id file contains row_ids not present in {split}: {missing[:10]}")
        selected = apply_query_window(selected, cfg)
        print("DEBUG total rows in split:", len(df))
        print("DEBUG loaded fixed query ids:", len(row_ids))
        print("DEBUG query_offset:", int(getattr(cfg, "query_offset", 0) or 0))
        print("DEBUG max_queries:", cfg.max_queries)
        print("DEBUG selected fixed query ids after window:", len(selected))
        print("DEBUG query_ids_path:", query_ids_path)
        return selected.reset_index(drop=True)

    keep = (
        (df["pred_label"] == cfg.target_positive_label)
        & (df["pred_proba"] >= cfg.min_query_proba)
    )

    selected = df.loc[keep].sort_values("pred_proba", ascending=False)
    max_queries = int(cfg.max_queries) if cfg.max_queries is not None else 0
    if max_queries > 0:
        if getattr(cfg, "random_query_sample", False) and len(selected) > max_queries:
            selected = selected.sample(n=max_queries, random_state=int(cfg.random_state))
            selected = selected.sort_values("pred_proba", ascending=False)
        else:
            selected = selected.head(max_queries)

    print("DEBUG total rows in split:", len(df))
    print("DEBUG predicted positives at threshold:", int(keep.sum()))
    print("DEBUG selected after keep:", int(len(selected)))
    print("DEBUG max_queries:", cfg.max_queries)
    print("DEBUG random_query_sample:", bool(getattr(cfg, "random_query_sample", False)))

    return selected.reset_index(drop=True)


def load_saved_predictions(model_dir: Path, split: str) -> pd.DataFrame:
    path = model_dir / f"predictions_{split}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing saved predictions file: {path}")
    return pd.read_csv(path)

def flatten_counterfactual_examples(
    factuals: pd.DataFrame,
    cf_frames: Sequence[pd.DataFrame],
    mutable_features: Sequence[str],
    model: XGBClassifier,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    mutable = list(mutable_features)

    for factual, cf_df in zip(factuals.to_dict(orient="records"), cf_frames):
        factual_features = pd.Series(factual)
        for cf_rank, (_, cf_row) in enumerate(cf_df.iterrows(), start=1):
            cf_features = cf_row.reindex(mutable).copy()
            full_row = factual_features.copy()
            for feature in mutable:
                if feature in cf_features.index:
                    full_row[feature] = cf_features[feature]

            changed_features = []
            total_abs_shift = 0.0
            for feature in mutable:
                if feature not in factual_features.index or feature not in cf_features.index:
                    continue
                old_value = factual_features[feature]
                new_value = cf_features[feature]
                if pd.isna(old_value) and pd.isna(new_value):
                    continue
                if float(old_value) != float(new_value):
                    changed_features.append(feature)
                    total_abs_shift += abs(float(new_value) - float(old_value))

            rows.append(
                {
                    "row_id": factual.get("row_id"),
                    "cf_rank": cf_rank,
                    "pred_proba_factual": factual.get("pred_proba"),
                    "num_changed_features": len(changed_features),
                    "changed_features": json.dumps(changed_features),
                    "total_abs_shift": total_abs_shift,
                }
            )

    out = pd.DataFrame(rows)
    if not out.empty:
        out["pred_proba_counterfactual"] = np.nan
    return out


def summarize_run(
    factuals: pd.DataFrame,
    counterfactuals: pd.DataFrame,
    model: XGBClassifier,
    x_reference: pd.DataFrame,
) -> Dict[str, object]:
    summary: Dict[str, object] = {
        "num_queries": int(len(factuals)),
        "num_counterfactual_rows": int(len(counterfactuals)),
    }
    if len(x_reference) > 0:
        summary["reference_auc"] = float(roc_auc_score(factuals["label"], factuals["pred_proba"])) if "label" in factuals else None
    if not counterfactuals.empty:
        summary["mean_num_changed_features"] = float(counterfactuals["num_changed_features"].mean())
        summary["mean_total_abs_shift"] = float(counterfactuals["total_abs_shift"].mean())
    return summary


def save_run_outputs(
    out_dir: Path,
    factuals: pd.DataFrame,
    counterfactuals: pd.DataFrame,
    summary: Dict[str, object],
    config_dict: Dict[str, object],
) -> None:
    ensure_dir(out_dir)
    factuals.to_csv(out_dir / "selected_factuals.csv", index=False)
    counterfactuals.to_csv(out_dir / "counterfactuals.csv", index=False)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2)


def _safe_numeric_frame(df: pd.DataFrame, feature_cols: Sequence[str]) -> pd.DataFrame:
    out = df.loc[:, [c for c in feature_cols if c in df.columns]].copy()
    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.fillna(0.0)


def _compute_scale_stats(reference: pd.DataFrame) -> Dict[str, pd.Series]:
    median = reference.median(axis=0)
    q1 = reference.quantile(0.25, axis=0)
    q3 = reference.quantile(0.75, axis=0)
    iqr = (q3 - q1).replace(0.0, np.nan)
    mad = (reference.sub(median, axis=1).abs().median(axis=0) * 1.4826).replace(0.0, np.nan)
    q01 = reference.quantile(0.01, axis=0)
    q99 = reference.quantile(0.99, axis=0)

    iqr = iqr.fillna(1.0)
    mad = mad.fillna(1.0)

    return {
        "median": median,
        "iqr": iqr,
        "mad": mad,
        "q01": q01,
        "q99": q99,
    }


def _scaled_frame(reference: pd.DataFrame, center: pd.Series, scale: pd.Series) -> pd.DataFrame:
    denom = scale.replace(0.0, 1.0).fillna(1.0)
    return reference.sub(center, axis=1).div(denom, axis=1).fillna(0.0)


def _infer_decimal_precision(series: pd.Series, max_precision: int = 6) -> int:
    """Infer how many decimal places this column's values are normally
    recorded to, mirroring the convention DiCE's own genetic search uses
    (round_to_precision(), applied to its final output) -- so a range bound
    can be compared at the same precision the counterfactual value was
    itself rounded to, rather than against the bound's full float64
    precision. Without this, a value that legitimately satisfied
    [train_min, train_max] before DiCE's own final rounding step can appear
    to fall just outside it afterwards (e.g. a value at exactly 654.36 that
    DiCE rounds to 654.4 for display, since that column is normally
    recorded to one decimal place).

    Returns the most common number of decimal places across the column's
    non-null values (after first rounding to max_precision to strip float64
    representation noise), defaulting to max_precision if the column is
    empty.
    """
    vals = pd.to_numeric(series, errors="coerce").dropna()
    if vals.empty:
        return max_precision
    counts: Dict[int, int] = {}
    for v in vals:
        s = f"{float(v):.{max_precision}f}".rstrip("0")
        n_decimals = len(s.split(".")[1]) if "." in s else 0
        counts[n_decimals] = counts.get(n_decimals, 0) + 1
    return max(counts.items(), key=lambda kv: kv[1])[0]


def _precision_aware_bounds(
    reference: pd.DataFrame,
    permitted_range: Dict[str, List[float]],
) -> Dict[str, List[float]]:
    """Round each feature's [lo, hi] bound to that feature's own typical
    decimal precision in `reference` (see _infer_decimal_precision), so a
    downstream range check is comparing at the same precision DiCE's own
    round_to_precision() output is expressed at."""
    rounded: Dict[str, List[float]] = {}
    for feature, (lo, hi) in permitted_range.items():
        if feature not in reference.columns:
            rounded[feature] = [lo, hi]
            continue
        precision = _infer_decimal_precision(reference[feature])
        rounded[feature] = [round(lo, precision), round(hi, precision)]
    return rounded


def _parse_changed_features(raw_value: object) -> List[str]:
    if raw_value is None or (isinstance(raw_value, float) and pd.isna(raw_value)):
        return []
    if isinstance(raw_value, list):
        return [str(x) for x in raw_value]
    if isinstance(raw_value, str):
        try:
            parsed = json.loads(raw_value)
        except json.JSONDecodeError:
            return []
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
    return []


def _parse_changes_dict(raw_value: object) -> Dict[str, Dict[str, float]]:
    if raw_value is None or (isinstance(raw_value, float) and pd.isna(raw_value)):
        return {}
    if isinstance(raw_value, dict):
        return raw_value
    if isinstance(raw_value, str):
        try:
            parsed = json.loads(raw_value)
        except json.JSONDecodeError:
            return {}
        if isinstance(parsed, dict):
            return parsed
    return {}


def reconstruct_counterfactual_matrix(
    counterfactuals: pd.DataFrame,
    factuals: pd.DataFrame,
    feature_cols: Sequence[str],
) -> pd.DataFrame:
    factual_by_row = factuals.set_index("row_id")
    feature_cols = [c for c in feature_cols if c in factuals.columns]

    if set(feature_cols).issubset(counterfactuals.columns):
        return counterfactuals.loc[:, feature_cols].copy().reset_index(drop=True)

    rows: List[pd.Series] = []
    for cf_row in counterfactuals.to_dict(orient="records"):
        row_id = int(cf_row["row_id"])
        factual_row = factual_by_row.loc[row_id, feature_cols].copy()
        changes = _parse_changes_dict(cf_row.get("changes"))
        for feature, payload in changes.items():
            if feature in factual_row.index and isinstance(payload, dict) and "to" in payload:
                factual_row[feature] = payload["to"]
        rows.append(factual_row)

    if not rows:
        return pd.DataFrame(columns=feature_cols)
    return pd.DataFrame(rows).reset_index(drop=True)


def evaluate_counterfactuals(
    factuals: pd.DataFrame,
    counterfactuals: pd.DataFrame,
    x_reference: pd.DataFrame,
    metadata: pd.DataFrame,
    mutable_features: Sequence[str],
    threshold: float,
    feature_cols: Optional[Sequence[str]] = None,
    plausibility_features: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    if counterfactuals.empty:
        return counterfactuals.copy(), {
            "num_counterfactual_rows": 0,
            "validity_flip_rate": 0.0,
        }

    if feature_cols is None:
        feature_cols = [c for c in x_reference.columns if c in factuals.columns]
    else:
        feature_cols = [c for c in feature_cols if c in x_reference.columns and c in factuals.columns]

    if plausibility_features is None:
        plausibility_features = [c for c in mutable_features if c in x_reference.columns]
        if not plausibility_features:
            plausibility_features = list(feature_cols)
    else:
        plausibility_features = [c for c in plausibility_features if c in x_reference.columns]

    factual_by_row = factuals.set_index("row_id")
    cf_matrix = reconstruct_counterfactual_matrix(counterfactuals, factuals, feature_cols)
    factual_matrix = pd.DataFrame(
        [factual_by_row.loc[int(row_id), feature_cols] for row_id in counterfactuals["row_id"].tolist()],
        columns=feature_cols,
    ).reset_index(drop=True)

    reference_num = _safe_numeric_frame(x_reference, feature_cols)
    factual_num = _safe_numeric_frame(factual_matrix, feature_cols)
    cf_num = _safe_numeric_frame(cf_matrix, feature_cols)

    scale_stats = _compute_scale_stats(reference_num)
    diff = cf_num.sub(factual_num, axis=0)
    abs_diff = diff.abs()

    l1_mad = abs_diff.div(scale_stats["mad"], axis=1).sum(axis=1)
    l2_mad = np.sqrt((diff.div(scale_stats["mad"], axis=1) ** 2).sum(axis=1))
    l1_iqr = abs_diff.div(scale_stats["iqr"], axis=1).sum(axis=1)
    l2_iqr = np.sqrt((diff.div(scale_stats["iqr"], axis=1) ** 2).sum(axis=1))

    mutable_feature_set = set(str(x) for x in mutable_features)
    num_total_features = max(len(feature_cols), 1)

    plaus_ref = _safe_numeric_frame(x_reference, plausibility_features)
    plaus_cf = _safe_numeric_frame(cf_matrix, plausibility_features)
    plaus_stats = _compute_scale_stats(plaus_ref)
    plaus_ref_scaled = _scaled_frame(plaus_ref, plaus_stats["median"], plaus_stats["iqr"])
    plaus_cf_scaled = _scaled_frame(plaus_cf, plaus_stats["median"], plaus_stats["iqr"])

    nn = NearestNeighbors(
        metric="euclidean",
        n_neighbors=min(max(1, 5), max(1, len(plaus_ref_scaled))),
    )
    nn.fit(plaus_ref_scaled)
    knn_distances, _ = nn.kneighbors(plaus_cf_scaled)
    knn_mean_distance = knn_distances.mean(axis=1)

    if len(plaus_ref_scaled) >= 3:
        lof = LocalOutlierFactor(
            n_neighbors=min(35, max(2, len(plaus_ref_scaled) - 1)),
            novelty=True,
        )
        lof.fit(plaus_ref_scaled)
        lof_score = lof.score_samples(plaus_cf_scaled)
    else:
        lof_score = np.full(len(plaus_cf_scaled), np.nan)

    cov = np.cov(plaus_ref_scaled.to_numpy(), rowvar=False)
    if cov.ndim == 0:
        cov = np.asarray([[float(cov)]])
    cov = cov + np.eye(cov.shape[0]) * 1e-6
    cov_inv = np.linalg.pinv(cov)
    centered_cf = plaus_cf_scaled.to_numpy()
    mahalanobis = np.sqrt(np.einsum("ij,jk,ik->i", centered_cf, cov_inv, centered_cf))

    # Training-range bounds for the quantile-violation check
    # A "violation" is defined as falling outside [train_min, train_max]
    # -- the same range the counterfactual search was itself constrained to
    raw_plausibility_bounds = get_permitted_range(metadata, plausibility_features)
    plausibility_bounds = _precision_aware_bounds(x_reference, raw_plausibility_bounds)
    plausibility_precisions = {
        feature: _infer_decimal_precision(x_reference[feature])
        for feature in plausibility_bounds
        if feature in x_reference.columns
    }

    evaluated = counterfactuals.copy().reset_index(drop=True)
    changed_feature_lists: List[List[str]] = []
    actionability_violations: List[int] = []
    quantile_violations: List[int] = []
    plausibility_feature_set = set(plausibility_features)
    for idx, row in evaluated.iterrows():
        changes_dict = _parse_changes_dict(row.get("changes"))
        if changes_dict:
            changed = []
            for feature, payload in changes_dict.items():
                if not isinstance(payload, dict):
                    changed.append(str(feature))
                    continue
                old = payload.get("from")
                new = payload.get("to")
                if pd.isna(old) and pd.isna(new):
                    continue
                changed.append(str(feature))
        elif "changed_features" in evaluated.columns:
            changed = _parse_changed_features(row.get("changed_features"))
        else:
            changed = [
                feature
                for feature in feature_cols
                if not np.isclose(
                    cf_num.iloc[idx][feature],
                    factual_num.iloc[idx][feature],
                    equal_nan=True,
                )
            ]

        changed_feature_lists.append(changed)
        actionability_violations.append(sum(1 for feature in changed if feature not in mutable_feature_set))

        def _is_range_violation(feature: str) -> bool:
            if feature not in plausibility_feature_set or feature not in plausibility_bounds:
                return False
            raw_value = plaus_cf.iloc[idx][feature]
            precision = plausibility_precisions.get(feature)
            value = round(float(raw_value), precision) if precision is not None else raw_value
            lo, hi = plausibility_bounds[feature]
            return value < lo or value > hi

        quantile_violations.append(sum(1 for feature in changed if _is_range_violation(feature)))

    pred_proba_factual = evaluated["pred_proba_factual"].astype(float) if "pred_proba_factual" in evaluated.columns else factuals.set_index("row_id").loc[evaluated["row_id"], "pred_proba"].reset_index(drop=True).astype(float)
    pred_proba_cf = evaluated["pred_proba_counterfactual"].astype(float)

    evaluated["validity"] = (pred_proba_cf < float(threshold)).astype(int)
    evaluated["risk_reduction"] = pred_proba_factual - pred_proba_cf
    evaluated["proximity_l1_mad"] = l1_mad.to_numpy()
    evaluated["proximity_l2_mad"] = l2_mad.to_numpy()
    evaluated["proximity_l1_iqr"] = l1_iqr.to_numpy()
    evaluated["proximity_l2_iqr"] = l2_iqr.to_numpy()
    evaluated["sparsity_num_changed"] = [len(x) for x in changed_feature_lists]
    evaluated["sparsity_frac_changed"] = evaluated["sparsity_num_changed"] / float(num_total_features)
    evaluated["actionability_violation_count"] = actionability_violations
    evaluated["actionability_all_mutable"] = (evaluated["actionability_violation_count"] == 0).astype(int)
    evaluated["plausibility_knn_distance"] = knn_mean_distance
    evaluated["plausibility_lof_score"] = lof_score
    evaluated["plausibility_mahalanobis"] = mahalanobis
    evaluated["plausibility_quantile_violation_count"] = quantile_violations

    summary: Dict[str, object] = {
        "num_counterfactual_rows": int(len(evaluated)),
        "validity_flip_rate": float(evaluated["validity"].mean()),
        "mean_risk_reduction": float(evaluated["risk_reduction"].mean()),
        "median_risk_reduction": float(evaluated["risk_reduction"].median()),
        "mean_proximity_l1_mad": float(evaluated["proximity_l1_mad"].mean()),
        "median_proximity_l1_mad": float(evaluated["proximity_l1_mad"].median()),
        "mean_proximity_l2_mad": float(evaluated["proximity_l2_mad"].mean()),
        "median_proximity_l2_mad": float(evaluated["proximity_l2_mad"].median()),
        "mean_proximity_l1_iqr": float(evaluated["proximity_l1_iqr"].mean()),
        "median_proximity_l1_iqr": float(evaluated["proximity_l1_iqr"].median()),
        "mean_proximity_l2_iqr": float(evaluated["proximity_l2_iqr"].mean()),
        "median_proximity_l2_iqr": float(evaluated["proximity_l2_iqr"].median()),
        "mean_sparsity_num_changed": float(evaluated["sparsity_num_changed"].mean()),
        "median_sparsity_num_changed": float(evaluated["sparsity_num_changed"].median()),
        "mean_sparsity_frac_changed": float(evaluated["sparsity_frac_changed"].mean()),
        "median_sparsity_frac_changed": float(evaluated["sparsity_frac_changed"].median()),
        "actionability_violation_rate": float((evaluated["actionability_violation_count"] > 0).mean()),
        "mean_actionability_violation_count": float(evaluated["actionability_violation_count"].mean()),
        "median_actionability_violation_count": float(evaluated["actionability_violation_count"].median()),
        "mean_plausibility_knn_distance": float(evaluated["plausibility_knn_distance"].mean()),
        "median_plausibility_knn_distance": float(evaluated["plausibility_knn_distance"].median()),
        "mean_plausibility_lof_score": float(np.nanmean(evaluated["plausibility_lof_score"])),
        "median_plausibility_lof_score": float(np.nanmedian(evaluated["plausibility_lof_score"])),
        "mean_plausibility_mahalanobis": float(evaluated["plausibility_mahalanobis"].mean()),
        "median_plausibility_mahalanobis": float(evaluated["plausibility_mahalanobis"].median()),
        "mean_plausibility_quantile_violation_count": float(evaluated["plausibility_quantile_violation_count"].mean()),
        "median_plausibility_quantile_violation_count": float(evaluated["plausibility_quantile_violation_count"].median()),
        "threshold_used_for_validity": float(threshold),
    }
    return evaluated, summary