from __future__ import annotations

import argparse
import inspect
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from cf_experiment_utils import (
    CFExperimentConfig,
    evaluate_counterfactuals,
    ensure_dir,
    get_mutable_features,
    get_permitted_range,
    load_training_artifacts,
    select_query_set,
)


try:
    from raiutils.exceptions import UserConfigValidationException
except ImportError:
    class UserConfigValidationException(Exception):
        pass


try:
    import dice_ml
except ImportError as exc:  # pragma: no cover
    dice_ml = None
    DICE_IMPORT_ERROR = exc
else:
    DICE_IMPORT_ERROR = None


@dataclass
class DiceGeneticNativeMissingWrapperConfig(CFExperimentConfig):
    out_name: str = "dice_genetic_native_missing_wrapper"
    max_queries: int = 250
    total_cfs: int = 8
    sparsity_weight: float = 0.2
    proximity_weight: float = 1.0
    diversity_weight: float = 1.0
    categorical_features: tuple[str, ...] = ()
    max_features_to_vary: int = 0
    feature_subset_strategy: str = "all"
    desired_class: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DiCE genetic counterfactuals with native-missing prediction wrapper.")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--out-root", default=None)
    parser.add_argument("--out-name", default=None)
    parser.add_argument("--query-split", default=None, choices=["train", "val", "test"])
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--query-offset", type=int, default=None, help="Skip this many rows from the selected/fixed factual cohort before running DiCE.")
    parser.add_argument("--min-query-proba", type=float, default=None)
    parser.add_argument("--total-cfs", type=int, default=None)
    parser.add_argument("--desired-class", type=int, default=None)
    parser.add_argument("--sparsity-weight", type=float, default=None)
    parser.add_argument("--proximity-weight", type=float, default=None)
    parser.add_argument("--diversity-weight", type=float, default=None)
    parser.add_argument("--max-features-to-vary", type=int, default=None)
    parser.add_argument("--feature-subset-strategy", choices=["all", "top"], default=None)
    parser.add_argument("--allow-treatments", action="store_true")
    parser.add_argument("--random-state", type=int, default=None)
    parser.add_argument("--query-ids-path", default=None, help="CSV containing row_id values to reuse the exact same factual patients.")
    parser.add_argument("--random-query-sample", action="store_true", help="Randomly sample max_queries eligible factuals using random_state.")
    return parser.parse_args()


def select_ranked_mutable_features(cfg: DiceGeneticNativeMissingWrapperConfig, metadata: pd.DataFrame, model_dir: Path) -> List[str]:
    """Return the full ranked mutable list; apply max_features_to_vary per patient."""
    mutable_features = enforce_mutability_policy(metadata, cfg)

    if cfg.feature_subset_strategy == "all":
        return mutable_features

    importance_path = model_dir / "feature_importance_gain.csv"
    if not importance_path.exists():
        return mutable_features

    importance_df = pd.read_csv(importance_path)
    ranked_mutable = [
        str(f)
        for f in importance_df["feature"].astype(str).tolist()
        if str(f) in set(mutable_features)
    ]
    return ranked_mutable if ranked_mutable else mutable_features


def cap_observed_features(
    ranked_features: List[str],
    missing_features: List[str],
    max_features_to_vary: int,
) -> List[str]:
    missing_set = set(missing_features)
    observed = [feature for feature in ranked_features if feature not in missing_set]
    if max_features_to_vary and max_features_to_vary > 0:
        return observed[:max_features_to_vary]
    return observed


def build_imputation_values(train_df: pd.DataFrame) -> Dict[str, float]:
    values: Dict[str, float] = {}
    for col in train_df.columns:
        series = pd.to_numeric(train_df[col], errors="coerce")
        if series.notna().any():
            values[col] = float(series.median())
        else:
            values[col] = 0.0
    return values


def impute_with_training_values(
    df: pd.DataFrame,
    imputation_values: Dict[str, float],
) -> pd.DataFrame:
    out = df.copy()
    for feature, value in imputation_values.items():
        if feature in out.columns:
            out[feature] = pd.to_numeric(out[feature], errors="coerce").fillna(value)
    return out


class NativeMissingPredictionWrapper:
    def __init__(self, model, feature_cols, missing_mask, threshold):
        self.model = model
        self.feature_cols = list(feature_cols)
        self.missing_mask = missing_mask
        self.threshold = float(threshold)
        self.classes_ = getattr(model, "classes_", np.array([0, 1]))

    def _restore_missingness(self, X):
        if isinstance(X, pd.DataFrame):
            X_eval = X.copy()
            X_eval = X_eval.loc[:, self.feature_cols]
        else:
            X_eval = pd.DataFrame(X, columns=self.feature_cols)

        mask = self.missing_mask
        if isinstance(mask, pd.Series):
            missing_features = mask[mask].index.tolist()
        else:
            missing_features = list(mask)

        for feature in missing_features:
            if feature in X_eval.columns:
                X_eval[feature] = np.nan

        return X_eval

    def predict_proba(self, X):
        X_eval = self._restore_missingness(X)
        return self.model.predict_proba(X_eval)

    def predict(self, X):
        proba = self.predict_proba(X)[:, 1]
        return (proba >= self.threshold).astype(int)


def build_dice_generation_kwargs(cfg: DiceGeneticNativeMissingWrapperConfig) -> Dict[str, float]:
    signature = inspect.signature(dice_ml.Dice.generate_counterfactuals)
    supported = set(signature.parameters)
    weights = {
        "proximity_weight": cfg.proximity_weight,
        "diversity_weight": cfg.diversity_weight,
        "sparsity_weight": cfg.sparsity_weight,
    }
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return {key: float(value) for key, value in weights.items()}
    return {key: float(value) for key, value in weights.items() if key in supported}


def dice_weight_args_summary(cfg: DiceGeneticNativeMissingWrapperConfig) -> tuple[List[str], str]:
    passed = sorted(build_dice_generation_kwargs(cfg))
    requested = ["diversity_weight", "proximity_weight", "sparsity_weight"]
    missing = sorted(set(requested) - set(passed))
    if not missing:
        return passed, "Configured DiCE weight arguments are passed to generate_counterfactuals()."
    return passed, (
        "This installed DiCE generate_counterfactuals() signature does not expose "
        + ", ".join(missing)
        + "; those config values are retained for reproducibility but are not reported as used."
    )

def annotate_counterfactuals(
    factuals: pd.DataFrame,
    counterfactuals: pd.DataFrame,
    feature_cols: List[str],
    model,
    threshold: float,
) -> pd.DataFrame:
    if counterfactuals.empty:
        return counterfactuals

    factual_by_row_id = factuals.set_index("row_id")
    enriched_rows = []
    for row in counterfactuals.to_dict(orient="records"):
        row_id = int(row["row_id"])
        factual = factual_by_row_id.loc[row_id]
        changed_features = []
        changes = {}
        total_abs_shift = 0.0

        for feature in feature_cols:
            if feature not in row or feature not in factual.index:
                continue
            old_value = factual[feature]
            new_value = row[feature]
            if pd.isna(old_value) and pd.isna(new_value):
                continue
            try:
                old_float = float(old_value)
                new_float = float(new_value)
                if not np.isclose(old_float, new_float, equal_nan=True):
                    changed_features.append(feature)
                    changes[feature] = {"from": old_float, "to": new_float}
                    total_abs_shift += abs(new_float - old_float)
            except Exception:
                if str(old_value) != str(new_value):
                    changed_features.append(feature)
                    changes[feature] = {"from": str(old_value), "to": str(new_value)}

        cf_vector = pd.DataFrame([{feature: row.get(feature, np.nan) for feature in feature_cols}])
        pred_proba = float(model.predict_proba(cf_vector)[:, 1][0])
        row["pred_proba_counterfactual"] = pred_proba
        row["pred_label_counterfactual"] = int(pred_proba >= float(threshold))
        row["num_changed_features"] = int(len(changed_features))
        row["changed_features"] = json.dumps(changed_features)
        row["changes"] = json.dumps(changes)
        row["total_abs_shift"] = float(total_abs_shift)
        enriched_rows.append(row)
    return pd.DataFrame(enriched_rows)


def add_imputed_factual_scores(
    factuals_native: pd.DataFrame,
    factuals_imputed: pd.DataFrame,
    feature_cols: List[str],
    model,
    threshold: float,
) -> pd.DataFrame:
    out = factuals_native.copy()
    if factuals_imputed.empty:
        return out

    native_proba = model.predict_proba(factuals_native.loc[:, feature_cols])[:, 1]
    imputed_proba = model.predict_proba(factuals_imputed.loc[:, feature_cols])[:, 1]

    out["pred_proba_native"] = native_proba
    out["pred_label_native"] = (native_proba >= float(threshold)).astype(int)
    out["pred_proba_imputed_factual"] = imputed_proba
    out["pred_label_imputed_factual"] = (imputed_proba >= float(threshold)).astype(int)

    return out


def originally_missing_features(
    factual_native: pd.Series,
    features: List[str],
) -> List[str]:
    return [feature for feature in features if feature in factual_native.index and pd.isna(factual_native[feature])]


def subset_permitted_range(
    permitted_range: Dict[str, List[float]],
    features_to_vary: List[str],
) -> Dict[str, List[float]]:
    return {feature: permitted_range[feature] for feature in features_to_vary if feature in permitted_range}


def restore_native_missing_values(
    counterfactuals: pd.DataFrame,
    factuals_native: pd.DataFrame,
    feature_cols: List[str],
) -> pd.DataFrame:
    if counterfactuals.empty:
        return counterfactuals

    restored = counterfactuals.copy()
    factual_by_row_id = factuals_native.set_index("row_id")
    restored_payloads = []

    for idx, row in restored.iterrows():
        row_id = int(row["row_id"])
        factual_native = factual_by_row_id.loc[row_id]
        missing = originally_missing_features(factual_native, feature_cols)
        restored_payloads.append(missing)
        for feature in missing:
            if feature in restored.columns:
                restored.at[idx, feature] = np.nan

    restored["originally_missing_features_restored"] = [json.dumps(x) for x in restored_payloads]
    restored["num_originally_missing_features_restored"] = [len(x) for x in restored_payloads]
    return restored


def main() -> None:
    if dice_ml is None:
        raise ImportError(
            "dice_ml is not installed. Install DiCE first, then rerun this script."
        ) from DICE_IMPORT_ERROR

    cfg = DiceGeneticNativeMissingWrapperConfig()
    args = parse_args()
    if args.data_dir is not None:
        cfg.data_dir = args.data_dir
    if args.model_dir is not None:
        cfg.model_dir = args.model_dir
    if args.out_root is not None:
        cfg.out_root = args.out_root
    if args.out_name is not None:
        cfg.out_name = args.out_name
    if args.query_split is not None:
        cfg.query_split = args.query_split
    if args.max_queries is not None:
        cfg.max_queries = args.max_queries
    if args.query_offset is not None:
        cfg.query_offset = args.query_offset
    if args.min_query_proba is not None:
        cfg.min_query_proba = args.min_query_proba
    if args.total_cfs is not None:
        cfg.total_cfs = args.total_cfs
    if args.max_features_to_vary is not None:
        cfg.max_features_to_vary = args.max_features_to_vary
    if args.feature_subset_strategy is not None:
        cfg.feature_subset_strategy = args.feature_subset_strategy
    if args.allow_treatments:
        cfg.allow_treatments = True
    if args.random_state is not None:
        cfg.random_state = args.random_state
    if args.query_ids_path is not None:
        cfg.query_ids_path = args.query_ids_path
    if args.random_query_sample:
        cfg.random_query_sample = True
    if args.desired_class is not None:
        cfg.desired_class = args.desired_class
    if args.sparsity_weight is not None:
        cfg.sparsity_weight = args.sparsity_weight
    if args.proximity_weight is not None:
        cfg.proximity_weight = args.proximity_weight
    if args.diversity_weight is not None:
        cfg.diversity_weight = args.diversity_weight

    artifacts = load_training_artifacts(cfg)
    cfg.min_query_proba = float(artifacts["model_threshold"])
    print("DEBUG data_dir:", cfg.data_dir)
    print("DEBUG model_dir:", cfg.model_dir)
    print("DEBUG loaded threshold:", cfg.min_query_proba)

    model = artifacts["model"]
    x_train_native = artifacts["x_train"].copy()
    metadata = artifacts["cf_metadata"]
    imputation_values = build_imputation_values(x_train_native)
    x_train_imputed = impute_with_training_values(x_train_native, imputation_values)

    ranked_mutable_features = select_ranked_mutable_features(cfg, metadata, Path(cfg.model_dir))
    permitted_range = get_permitted_range(metadata, ranked_mutable_features)
    x_query_native = artifacts[f"x_{cfg.query_split}"].copy()
    y_query = artifacts[f"y_{cfg.query_split}"]
    x_query_raw = artifacts[f"x_{cfg.query_split}_raw"]
    factuals_native = select_query_set(x_query_raw, x_query_native, y_query, model, cfg)
    factuals_imputed = impute_with_training_values(factuals_native, imputation_values)

    out_dir = Path(cfg.out_root) / cfg.out_name
    ensure_dir(out_dir)

    feature_frame = x_train_imputed.copy()
    feature_frame["label"] = artifacts["y_train"].to_numpy()
    full_feature_ranges = build_full_numeric_range(x_train_imputed)
    feature_cols = [c for c in feature_frame.columns if c != "label"]

    factuals_output = add_imputed_factual_scores(
        factuals_native=factuals_native,
        factuals_imputed=factuals_imputed,
        feature_cols=feature_cols,
        model=model,
        threshold=cfg.min_query_proba,
    )

    num_selected_for_diagnostics = int(len(factuals_output))
    if num_selected_for_diagnostics > 0:
        native_labels = factuals_output["pred_label_native"].astype(int)
        imputed_labels = factuals_output["pred_label_imputed_factual"].astype(int)
        num_imputation_flipped_factuals = int((native_labels != imputed_labels).sum())
        pct_imputation_flipped_factuals = float(num_imputation_flipped_factuals / num_selected_for_diagnostics)
        num_imputed_factuals_already_desired = int((imputed_labels == int(cfg.desired_class)).sum())
        pct_imputed_factuals_already_desired = float(
            num_imputed_factuals_already_desired / num_selected_for_diagnostics
        )
    else:
        num_imputation_flipped_factuals = 0
        pct_imputation_flipped_factuals = 0.0
        num_imputed_factuals_already_desired = 0
        pct_imputed_factuals_already_desired = 0.0

    dice_weight_args_passed, dice_weight_args_note = dice_weight_args_summary(cfg)

    continuous_features = [
        c for c in feature_cols
        if c not in set(cfg.categorical_features)
        and pd.api.types.is_numeric_dtype(feature_frame[c])
    ]

    data = dice_ml.Data(
        dataframe=feature_frame,
        continuous_features=continuous_features,
        outcome_name="label",
    )
    factual_feature_cols = x_query_native.columns.tolist()
    factuals_input = factuals_imputed[factual_feature_cols].reset_index(drop=True)
    factuals_native_by_pos = factuals_native.reset_index(drop=True)

    cf_rows: List[pd.DataFrame] = []
    failed_rows: List[Dict[str, object]] = []
    missing_search_counts: List[int] = []

    for row_idx in range(len(factuals_input)):
        query = factuals_input.iloc[[row_idx]].copy()
        query = impute_with_training_values(query, imputation_values)
        query = clip_query_to_feature_range(query=query, feature_ranges=full_feature_ranges)
        row_id = int(factuals_imputed.iloc[row_idx]["row_id"])
        factual_native = factuals_native_by_pos.iloc[row_idx]
        missing_features = originally_missing_features(factual_native, ranked_mutable_features)
        row_mutable_features = cap_observed_features(
            ranked_features=ranked_mutable_features,
            missing_features=missing_features,
            max_features_to_vary=int(cfg.max_features_to_vary),
        )
        row_permitted_range = subset_permitted_range(permitted_range, row_mutable_features)
        missing_search_counts.append(len(missing_features))

        if not row_mutable_features:
            failed_rows.append(
                {
                    "row_id": row_id,
                    "error_type": "NoMutableObservedFeatures",
                    "error_message": "All selected DiCE features_to_vary were originally missing for this row.",
                    "num_missing_search_features": len(missing_features),
                    "missing_search_features": json.dumps(missing_features),
                }
            )
            continue

        try:
            missing_mask = factual_native.loc[factual_feature_cols].isna()
            wrapped_model = NativeMissingPredictionWrapper(
                model=model,
                feature_cols=factual_feature_cols,
                missing_mask=missing_mask,
                threshold=cfg.min_query_proba,
            )
            backend_model = dice_ml.Model(model=wrapped_model, backend="sklearn")
            exp = dice_ml.Dice(data, backend_model, method="genetic")
            generation_kwargs = build_dice_generation_kwargs(cfg)
            result = exp.generate_counterfactuals(
                query,
                total_CFs=cfg.total_cfs,
                desired_class=int(cfg.desired_class),
                features_to_vary=row_mutable_features,
                permitted_range=row_permitted_range,
                **generation_kwargs,
            )
        except UserConfigValidationException as e:
            print(f"WARNING: no counterfactuals found for row_id={row_id}: {e}")
            failed_rows.append(
                {
                    "row_id": row_id,
                    "error_type": "UserConfigValidationException",
                    "error_message": str(e),
                    "num_missing_search_features": len(missing_features),
                    "missing_search_features": json.dumps(missing_features),
                }
            )
            continue
        except Exception as e:
            print(f"WARNING: unexpected failure for row_id={row_id}: {e}")
            failed_rows.append(
                {
                    "row_id": row_id,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "num_missing_search_features": len(missing_features),
                    "missing_search_features": json.dumps(missing_features),
                }
            )
            continue

        if not result.cf_examples_list:
            print(f"WARNING: no cf_examples_list for row_id={row_id}")
            failed_rows.append(
                {
                    "row_id": row_id,
                    "error_type": "EmptyCFExamplesList",
                    "error_message": "DiCE returned an empty cf_examples_list.",
                    "num_missing_search_features": len(missing_features),
                    "missing_search_features": json.dumps(missing_features),
                }
            )
            continue

        cf_df = result.cf_examples_list[0].final_cfs_df
        if cf_df is None or cf_df.empty:
            print(f"WARNING: empty counterfactual dataframe for row_id={row_id}")
            failed_rows.append(
                {
                    "row_id": row_id,
                    "error_type": "EmptyCounterfactuals",
                    "error_message": "DiCE returned no counterfactual rows.",
                    "num_missing_search_features": len(missing_features),
                    "missing_search_features": json.dumps(missing_features),
                }
            )
            continue

        cf_df = cf_df.copy()
        cf_df.insert(0, "row_id", row_id)
        cf_df["num_missing_search_features_frozen"] = len(missing_features)
        cf_df["missing_search_features_frozen"] = json.dumps(missing_features)
        cf_df["row_search_features"] = json.dumps(row_mutable_features)
        cf_df["num_row_search_features"] = len(row_mutable_features)
        cf_rows.append(cf_df)

    counterfactuals = pd.concat(cf_rows, ignore_index=True) if cf_rows else pd.DataFrame()

    counterfactuals = enforce_frozen_features(
        counterfactuals=counterfactuals,
        factuals=factuals_imputed,
        feature_cols=factual_feature_cols,
        mutable_features=ranked_mutable_features,
    )
    counterfactuals = restore_native_missing_values(
        counterfactuals=counterfactuals,
        factuals_native=factuals_native,
        feature_cols=factual_feature_cols,
    )
    counterfactuals = annotate_counterfactuals(
        factuals=factuals_output,
        counterfactuals=counterfactuals,
        feature_cols=factual_feature_cols,
        model=model,
        threshold=cfg.min_query_proba,
    )

    counterfactuals, eval_summary = evaluate_counterfactuals(
        factuals=factuals_output,
        counterfactuals=counterfactuals,
        x_reference=x_train_native,
        metadata=metadata,
        mutable_features=ranked_mutable_features,
        threshold=cfg.min_query_proba,
        feature_cols=factual_feature_cols,
        plausibility_features=ranked_mutable_features,
    )

    num_selected_queries = int(len(factuals_native))
    failed_query_ids = set(int(r["row_id"]) for r in failed_rows)
    successful_query_ids = set(counterfactuals["row_id"].astype(int).tolist()) if not counterfactuals.empty else set()
    summary: Dict[str, object] = {
        "method": "dice_genetic_native_missing_wrapper",
        "num_queries": num_selected_queries,
        "num_counterfactual_rows": int(len(counterfactuals)),
        "mutable_features": len(ranked_mutable_features),
        "ranked_search_features": ranked_mutable_features,
        "max_features_to_vary_per_row": int(cfg.max_features_to_vary),
        "feature_subset_strategy": cfg.feature_subset_strategy,
        "num_failed_queries": int(len(failed_query_ids - successful_query_ids)),
        "num_successful_queries": int(len(successful_query_ids)),
        "query_success_rate": (
            float(len(successful_query_ids) / num_selected_queries)
            if num_selected_queries > 0 else 0.0
        ),
        "mean_missing_search_features_frozen": float(np.mean(missing_search_counts)) if missing_search_counts else 0.0,
        "median_missing_search_features_frozen": float(np.median(missing_search_counts)) if missing_search_counts else 0.0,
        "num_imputation_flipped_factuals": num_imputation_flipped_factuals,
        "pct_imputation_flipped_factuals": pct_imputation_flipped_factuals,
        "num_imputed_factuals_already_desired": num_imputed_factuals_already_desired,
        "pct_imputed_factuals_already_desired": pct_imputed_factuals_already_desired,
        "dice_weight_args_passed": dice_weight_args_passed,
        "dice_weight_args_note": dice_weight_args_note,
        "note": (
            "DiCE genetic with native-missing prediction wrapper: median values are used only as "
            "placeholders for DiCE's complete-vector search. For every model prediction made "
            "during DiCE generation, the original factual missing mask is restored before calling "
            "the trained model. Originally missing features are excluded from the row-specific "
            "mutable feature set and restored to NaN before final evaluation."
        ),
    }
    summary.update(eval_summary)

    factuals_output.to_csv(out_dir / "selected_factuals.csv", index=False)
    counterfactuals.to_csv(out_dir / "counterfactuals.csv", index=False)
    pd.DataFrame(failed_rows).to_csv(out_dir / "failed_queries.csv", index=False)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

def clip_query_to_training_range(
    query: pd.DataFrame,
    permitted_range: Dict[str, List[float]],
    features_to_vary: List[str],
) -> pd.DataFrame:
    clipped = query.copy()
    for feature in features_to_vary:
        if feature not in clipped.columns:
            continue
        if feature not in permitted_range:
            continue
        lo, hi = permitted_range[feature]
        clipped[feature] = clipped[feature].clip(lower=lo, upper=hi)
    return clipped

def enforce_mutability_policy(
    metadata: pd.DataFrame,
    cfg: DiceGeneticNativeMissingWrapperConfig,
) -> List[str]:
    allowed_labels = set(cfg.mutable_labels)
    if cfg.allow_treatments:
        allowed_labels.add("non_actionable_by_default")
        allowed_labels.add("non_actionable_or_constrained")

    filtered = metadata.loc[
        metadata["mutability"].isin(allowed_labels)
        & ~metadata["feature"].astype(str).str.endswith("_missing"),
        "feature",
    ].astype(str).tolist()

    return filtered


def build_full_numeric_range(train_df: pd.DataFrame) -> Dict[str, List[float]]:
    ranges: Dict[str, List[float]] = {}
    for col in train_df.columns:
        series = pd.to_numeric(train_df[col], errors="coerce")
        if series.notna().sum() == 0:
            continue
        lo = float(series.min())
        hi = float(series.max())
        if np.isfinite(lo) and np.isfinite(hi):
            ranges[col] = [lo, hi]
    return ranges


def clip_query_to_feature_range(
    query: pd.DataFrame,
    feature_ranges: Dict[str, List[float]],
) -> pd.DataFrame:
    clipped = query.copy()
    for feature, bounds in feature_ranges.items():
        if feature not in clipped.columns:
            continue
        lo, hi = bounds
        series = pd.to_numeric(clipped[feature], errors="coerce")
        if series.notna().any():
            clipped[feature] = series.clip(lower=lo, upper=hi)
    return clipped

def _row_allowed_features(row: pd.Series, default_features: set[str]) -> set[str]:
    raw_value = row.get("row_search_features")
    if raw_value is None or (isinstance(raw_value, float) and pd.isna(raw_value)):
        return default_features
    try:
        parsed = json.loads(raw_value) if isinstance(raw_value, str) else raw_value
    except json.JSONDecodeError:
        return default_features
    if isinstance(parsed, list):
        return {str(feature) for feature in parsed}
    return default_features


def enforce_frozen_features(
    counterfactuals: pd.DataFrame,
    factuals: pd.DataFrame,
    feature_cols: List[str],
    mutable_features: List[str],
) -> pd.DataFrame:
    if counterfactuals.empty:
        return counterfactuals

    frozen = counterfactuals.copy()
    factual_by_row_id = factuals.set_index("row_id")
    default_mutable_set = set(mutable_features)

    for idx, row in frozen.iterrows():
        row_id = int(row["row_id"])
        factual_row = factual_by_row_id.loc[row_id]
        allowed_features = _row_allowed_features(row, default_mutable_set)

        for feature in feature_cols:
            if feature not in frozen.columns:
                continue
            if feature not in factual_row.index:
                continue
            if feature not in allowed_features:
                frozen.at[idx, feature] = factual_row[feature]

    return frozen

if __name__ == "__main__":
    main()







