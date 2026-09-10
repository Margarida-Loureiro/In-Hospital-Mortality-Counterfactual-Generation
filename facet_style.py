from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
import pandas as pd

from cf_experiment_utils import (
    CFExperimentConfig,
    evaluate_counterfactuals,
    ensure_dir,
    load_training_artifacts,
    select_query_set,
)


@dataclass
class FacetConfig(CFExperimentConfig):
    out_name: str = "facet_style_xgboost_point_preserve_missing"
    max_queries: int = 250
    max_steps: int = 8
    beam_width: int = 20
    top_facets: int = 400
    top_facets_per_tree: int = 3
    max_trees: int = 500
    feature_subset_strategy: str = "all"
    max_features_to_vary: int = 0


@dataclass(frozen=True)
class Constraint:
    lower: float = -math.inf
    upper: float = math.inf
    allow_missing: bool = False


@dataclass(frozen=True)
class Facet:
    tree_idx: int
    leaf_id: int
    leaf_value: float
    constraints: Mapping[str, Constraint]


@dataclass
class BeamState:
    constraints: Dict[str, Constraint]
    facets: List[Tuple[int, int]]
    x_cf: pd.Series
    proba: float
    pred_label: int
    distance: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FACET-style counterfactuals for the trained XGBoost mortality model.")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--out-root", default=None)
    parser.add_argument("--out-name", default=None)
    parser.add_argument("--query-split", default=None, choices=["train", "val", "test"])
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--query-offset", type=int, default=None, help="Skip this many rows from the selected/fixed factual cohort before running FACET.")
    parser.add_argument("--min-query-proba", type=float, default=None)
    parser.add_argument("--total-cfs", type=int, default=None, help="Accepted for sweep compatibility; FACET produces one counterfactual per query.")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--beam-width", type=int, default=None)
    parser.add_argument("--top-facets", type=int, default=None)
    parser.add_argument("--top-facets-per-tree", type=int, default=None)
    parser.add_argument("--max-trees", type=int, default=None)
    parser.add_argument("--max-features-to-vary", type=int, default=None)
    parser.add_argument("--feature-subset-strategy", choices=["all", "top"], default=None)
    parser.add_argument("--allow-treatments", action="store_true")
    parser.add_argument("--random-state", type=int, default=None)
    parser.add_argument("--query-ids-path", default=None, help="CSV containing row_id values to reuse the exact same factual patients.")
    parser.add_argument("--random-query-sample", action="store_true", help="Randomly sample max_queries eligible factuals using random_state.")
    return parser.parse_args()


def enforce_mutability_policy(metadata: pd.DataFrame, cfg: FacetConfig) -> List[str]:
    allowed_labels = set(cfg.mutable_labels)
    if cfg.allow_treatments:
        allowed_labels.add("non_actionable_by_default")
        allowed_labels.add("non_actionable_or_constrained")

    return metadata.loc[
        metadata["mutability"].isin(allowed_labels)
        & ~metadata["feature"].astype(str).str.endswith("_missing"),
        "feature",
    ].astype(str).tolist()


def select_search_features(cfg: FacetConfig, metadata: pd.DataFrame, model_dir: Path) -> List[str]:
    mutable_features = enforce_mutability_policy(metadata, cfg)
    if cfg.feature_subset_strategy == "all" or cfg.max_features_to_vary <= 0:
        return mutable_features

    importance_path = model_dir / "feature_importance_gain.csv"
    if not importance_path.exists():
        return mutable_features[: cfg.max_features_to_vary]

    importance_df = pd.read_csv(importance_path)
    mutable_features = [feature for feature in mutable_features if not str(feature).endswith("_missing")]
    mutable_set = set(mutable_features)
    ranked_mutable = [
        str(feature)
        for feature in importance_df["feature"].astype(str).tolist()
        if str(feature) in mutable_set
    ]
    if not ranked_mutable:
        return mutable_features[: cfg.max_features_to_vary]
    return ranked_mutable[: cfg.max_features_to_vary]


def iter_leaf_facets(model, max_trees: int | None = None) -> List[Facet]:
    booster = model.get_booster()
    dump = booster.get_dump(dump_format="json", with_stats=True)
    if max_trees is not None and max_trees > 0:
        dump = dump[:max_trees]
    facets: List[Facet] = []

    def walk(
        node: Mapping[str, object],
        tree_idx: int,
        constraints: MutableMapping[str, Constraint],
    ) -> None:
        if "leaf" in node:
            facets.append(
                Facet(
                    tree_idx=tree_idx,
                    leaf_id=int(node["nodeid"]),
                    leaf_value=float(node["leaf"]),
                    constraints=dict(constraints),
                )
            )
            return

        split = str(node["split"])
        threshold = float(node["split_condition"])
        yes_id = int(node["yes"])
        no_id = int(node["no"])
        missing_id = int(node["missing"])
        children = {int(child["nodeid"]): child for child in node["children"]}  # type: ignore[index]

        for child_id, child in children.items():
            old = constraints.get(split)
            if child_id == yes_id:
                branch = Constraint(
                    upper=np.nextafter(threshold, -math.inf),
                    allow_missing=child_id == missing_id,
                )
            elif child_id == no_id:
                branch = Constraint(
                    lower=threshold,
                    allow_missing=child_id == missing_id,
                )
            else:
                continue

            if old is None:
                new = branch
            else:
                new = Constraint(
                    lower=max(old.lower, branch.lower),
                    upper=min(old.upper, branch.upper),
                    allow_missing=old.allow_missing and branch.allow_missing,
                )

            if new.lower <= new.upper or new.allow_missing:
                constraints[split] = new
                walk(child, tree_idx, constraints)

            if old is None:
                constraints.pop(split, None)
            else:
                constraints[split] = old

    for tree_idx, tree_json in enumerate(dump):
        walk(json.loads(tree_json), tree_idx, {})
    return facets


def rank_candidate_facets(
    facets: Sequence[Facet],
    target_label: int,
    top_per_tree: int,
    top_global: int,
) -> List[Facet]:
    by_tree: Dict[int, List[Facet]] = {}
    for facet in facets:
        by_tree.setdefault(facet.tree_idx, []).append(facet)

    reverse = target_label == 1
    candidates: List[Facet] = []
    for tree_facets in by_tree.values():
        candidates.extend(sorted(tree_facets, key=lambda f: f.leaf_value, reverse=reverse)[:top_per_tree])
    return sorted(candidates, key=lambda f: f.leaf_value, reverse=reverse)[:top_global]


def binary_features(x: pd.DataFrame) -> set[str]:
    binary: set[str] = set()
    for col in x.columns:
        vals = pd.Series(x[col].dropna().unique())
        if len(vals) <= 2 and vals.isin([0, 1, 0.0, 1.0]).all():
            binary.add(col)
    return binary


def feature_ranges(x: pd.DataFrame) -> pd.Series:
    ranges = x.max(numeric_only=True) - x.min(numeric_only=True)
    ranges = ranges.reindex(x.columns).fillna(1.0).astype(float)
    ranges[ranges <= 0] = 1.0
    return ranges


def feature_bounds(x: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    train_min = x.min(numeric_only=True, skipna=True).reindex(x.columns)
    train_max = x.max(numeric_only=True, skipna=True).reindex(x.columns)
    return train_min.astype(float), train_max.astype(float)


def predict_one(model, x: pd.Series, threshold: float) -> Tuple[float, int]:
    proba = float(model.predict_proba(x.to_frame().T)[:, 1][0])
    return proba, int(proba >= threshold)


def combine_constraints(
    left: Mapping[str, Constraint],
    right: Mapping[str, Constraint],
) -> Dict[str, Constraint] | None:
    merged = dict(left)
    for feature, constraint in right.items():
        old = merged.get(feature)
        if old is None:
            merged[feature] = constraint
            continue

        new = Constraint(
            lower=max(old.lower, constraint.lower),
            upper=min(old.upper, constraint.upper),
            allow_missing=old.allow_missing and constraint.allow_missing,
        )
        if new.lower > new.upper and not new.allow_missing:
            return None
        merged[feature] = new
    return merged


def finite_bound(value: float | int | np.floating | None) -> float | None:
    if value is None or pd.isna(value):
        return None
    value = float(value)
    if not np.isfinite(value):
        return None
    return value


def project_binary_value(lower: float, upper: float, original_value: float) -> float | None:
    candidates = sorted([0.0, 1.0], key=lambda value: abs(value - original_value))
    for candidate in candidates:
        if lower <= candidate <= upper:
            return candidate
    return None


def project_to_constraints(
    x_original: pd.Series,
    constraints: Mapping[str, Constraint],
    mutable_features: set[str],
    binary: set[str],
    train_min: pd.Series,
    train_max: pd.Series,
) -> pd.Series | None:
    x_cf = x_original.copy()
    for feature, constraint in constraints.items():
        if feature not in x_cf.index:
            return None

        original_value = x_original[feature]
        if pd.isna(original_value):
            if constraint.allow_missing:
                x_cf[feature] = np.nan
                continue
            return None

        value = float(original_value)
        if feature not in mutable_features:
            if constraint.lower <= value <= constraint.upper:
                continue
            return None

        final_lower = constraint.lower
        final_upper = constraint.upper
        lower_bound = finite_bound(train_min.get(feature))
        upper_bound = finite_bound(train_max.get(feature))
        if lower_bound is not None:
            final_lower = max(final_lower, lower_bound)
        if upper_bound is not None:
            final_upper = min(final_upper, upper_bound)
        if final_lower > final_upper:
            return None

        if feature in binary:
            projected = project_binary_value(final_lower, final_upper, value)
            if projected is None:
                return None
        else:
            projected = min(max(value, final_lower), final_upper)
            if not (final_lower <= projected <= final_upper):
                return None

        if pd.isna(projected):
            return None
        x_cf[feature] = float(projected)

    missing_before = x_original.isna()
    missing_after = x_cf.isna()
    if not missing_before.equals(missing_after):
        return None
    return x_cf


def normalized_l1(x0: pd.Series, x1: pd.Series, ranges: pd.Series) -> float:
    delta = (x1.astype(float) - x0.astype(float)).abs()
    return float((delta / ranges).sum())


def state_sort_key(state: BeamState, target_label: int) -> Tuple[float, float, float]:
    target_score = state.proba if target_label == 1 else 1.0 - state.proba
    return (-target_score, state.distance, float(len(state.facets)))


def generate_counterfactual(
    model,
    x_original: pd.Series,
    candidates: Sequence[Facet],
    threshold: float,
    target_label: int,
    ranges: pd.Series,
    mutable_features: set[str],
    binary: set[str],
    train_min: pd.Series,
    train_max: pd.Series,
    max_steps: int,
    beam_width: int,
) -> BeamState:
    proba, pred_label = predict_one(model, x_original, threshold)
    initial = BeamState({}, [], x_original.copy(), proba, pred_label, 0.0)
    beam = [initial]
    best = initial
    seen: set[Tuple[Tuple[int, int], ...]] = {tuple()}

    for _ in range(max_steps):
        expanded: List[BeamState] = []
        for state in beam:
            used_trees = {tree_idx for tree_idx, _ in state.facets}
            for facet in candidates:
                if facet.tree_idx in used_trees:
                    continue
                merged = combine_constraints(state.constraints, facet.constraints)
                if merged is None:
                    continue
                x_cf = project_to_constraints(x_original, merged, mutable_features, binary, train_min, train_max)
                if x_cf is None:
                    continue
                facets_used = state.facets + [(facet.tree_idx, facet.leaf_id)]
                key = tuple(sorted(facets_used))
                if key in seen:
                    continue
                seen.add(key)

                cf_proba, cf_label = predict_one(model, x_cf, threshold)
                distance = normalized_l1(x_original, x_cf, ranges)
                expanded.append(BeamState(merged, facets_used, x_cf, cf_proba, cf_label, distance))

        if not expanded:
            break

        expanded.sort(key=lambda s: state_sort_key(s, target_label))
        beam = expanded[:beam_width]
        if state_sort_key(beam[0], target_label) < state_sort_key(best, target_label):
            best = beam[0]

        successful = [s for s in expanded if s.pred_label == target_label]
        if successful:
            successful.sort(key=lambda s: (s.distance, len(s.facets)))
            return successful[0]

    return best


def changed_feature_payload(x0: pd.Series, x1: pd.Series) -> Tuple[List[str], Dict[str, Dict[str, float]], float]:
    changed = []
    changes: Dict[str, Dict[str, float]] = {}
    total_abs_shift = 0.0
    for feature in x0.index:
        old_value = float(x0[feature])
        new_value = float(x1[feature])

        if pd.isna(old_value) and pd.isna(new_value):
            continue

        if not np.isclose(old_value, new_value, equal_nan=True):
            changed.append(feature)
            changes[feature] = {"from": old_value, "to": new_value}
            total_abs_shift += abs(new_value - old_value)

    return changed, changes, float(total_abs_shift)


def validate_counterfactual(
    x_original: pd.Series,
    x_cf: pd.Series,
    feature_cols: Sequence[str],
    mutable_features: set[str],
    target_label: int,
    pred_label: int,
) -> Tuple[str, str] | None:
    for feature in feature_cols:
        if pd.isna(x_original[feature]) != pd.isna(x_cf[feature]):
            return "MissingnessChanged", f"Missingness changed for feature {feature}."

    for feature in feature_cols:
        if not str(feature).endswith("_missing"):
            continue
        old_value = float(x_original[feature])
        new_value = float(x_cf[feature])
        if not np.isclose(old_value, new_value, equal_nan=True):
            return "MissingIndicatorChanged", f"Missingness indicator changed for feature {feature}."

    if int(pred_label) != int(target_label):
        return "PredictionDidNotFlip", "Counterfactual prediction is not the target label."

    for feature in feature_cols:
        old_value = float(x_original[feature])
        new_value = float(x_cf[feature])
        if np.isclose(old_value, new_value, equal_nan=True):
            continue
        if feature not in mutable_features:
            return "ImmutableFeatureChanged", f"Immutable feature changed: {feature}."

    return None


def rescore_factuals_with_loaded_model(
    factuals: pd.DataFrame,
    feature_cols: Sequence[str],
    model,
    cfg: FacetConfig,
) -> pd.DataFrame:
    rescored = factuals.copy()
    if rescored.empty:
        return rescored

    saved_proba = rescored["pred_proba"].astype(float).to_numpy()
    current_proba = model.predict_proba(rescored.loc[:, feature_cols])[:, 1]
    current_label = (current_proba >= float(cfg.min_query_proba)).astype(int)

    rescored["pred_proba_saved"] = saved_proba
    rescored["pred_label_saved"] = rescored["pred_label"].astype(int).to_numpy()
    rescored["pred_proba"] = current_proba
    rescored["pred_label"] = current_label

    keep = (
        (rescored["pred_label"] == int(cfg.target_positive_label))
        & (rescored["pred_proba"] >= float(cfg.min_query_proba))
    )
    dropped = int((~keep).sum())
    if dropped:
        print(
            "WARNING dropped factuals whose saved predictions disagree with the loaded model:",
            dropped,
        )
    return rescored.loc[keep].reset_index(drop=True)


def main() -> None:
    cfg = FacetConfig()
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
    if args.max_steps is not None:
        cfg.max_steps = args.max_steps
    if args.beam_width is not None:
        cfg.beam_width = args.beam_width
    if args.top_facets is not None:
        cfg.top_facets = args.top_facets
    if args.top_facets_per_tree is not None:
        cfg.top_facets_per_tree = args.top_facets_per_tree
    if args.max_trees is not None:
        cfg.max_trees = args.max_trees
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

    artifacts = load_training_artifacts(cfg)
    if args.min_query_proba is None:
        cfg.min_query_proba = float(artifacts["model_threshold"])
    print("DEBUG data_dir:", cfg.data_dir)
    print("DEBUG model_dir:", cfg.model_dir)
    print("DEBUG decision threshold:", cfg.min_query_proba)

    model = artifacts["model"]
    x_train = artifacts["x_train"]
    metadata = artifacts["cf_metadata"]
    mutable_features = select_search_features(cfg, metadata, Path(cfg.model_dir))
    mutable_features = [feature for feature in mutable_features if not str(feature).endswith("_missing")]
    mutable_set = set(mutable_features)

    x_query = artifacts[f"x_{cfg.query_split}"]
    y_query = artifacts[f"y_{cfg.query_split}"]
    x_query_raw = artifacts[f"x_{cfg.query_split}_raw"]
    factuals = select_query_set(x_query_raw, x_query, y_query, model, cfg)
    feature_cols = x_query.columns.tolist()
    factuals = rescore_factuals_with_loaded_model(factuals, feature_cols, model, cfg)

    out_dir = Path(cfg.out_root) / cfg.out_name
    ensure_dir(out_dir)

    facets = iter_leaf_facets(model, max_trees=cfg.max_trees)
    target_label = 0 if int(cfg.target_positive_label) == 1 else 1
    candidates = rank_candidate_facets(
        facets,
        target_label=target_label,
        top_per_tree=cfg.top_facets_per_tree,
        top_global=cfg.top_facets,
    )
    ranges = feature_ranges(x_train)
    train_min, train_max = feature_bounds(x_train)
    binary = binary_features(x_train)

    counterfactual_rows: List[Dict[str, object]] = []
    failed_rows: List[Dict[str, object]] = []

    factual_by_row = factuals.set_index("row_id")
    for position, factual in enumerate(factuals.to_dict(orient="records"), start=1):
        row_id = int(factual["row_id"])
        print(f"[{position}/{len(factuals)}] explaining row_id={row_id}")
        x_original = factual_by_row.loc[row_id, feature_cols].astype(float)

        result = generate_counterfactual(
            model=model,
            x_original=x_original,
            candidates=candidates,
            threshold=cfg.min_query_proba,
            target_label=target_label,
            ranges=ranges,
            mutable_features=mutable_set,
            binary=binary,
            train_min=train_min,
            train_max=train_max,
            max_steps=cfg.max_steps,
            beam_width=cfg.beam_width,
        )

        if result.pred_label != target_label:
            failed_rows.append(
                {
                    "row_id": row_id,
                    "error_type": "NoValidCounterfactual",
                    "error_message": "FACET search did not find a threshold-flipping candidate.",
                    "best_pred_proba_counterfactual": float(result.proba),
                }
            )
            continue

        sanity_error = validate_counterfactual(
            x_original=x_original,
            x_cf=result.x_cf,
            feature_cols=feature_cols,
            mutable_features=mutable_set,
            target_label=target_label,
            pred_label=result.pred_label,
        )
        if sanity_error is not None:
            error_type, error_message = sanity_error
            failed_rows.append(
                {
                    "row_id": row_id,
                    "error_type": error_type,
                    "error_message": error_message,
                    "best_pred_proba_counterfactual": float(result.proba),
                }
            )
            continue

        changed_features, changes, total_abs_shift = changed_feature_payload(x_original, result.x_cf)
        row: Dict[str, object] = {
            "row_id": row_id,
            "cf_rank": 1,
            "pred_proba_factual": float(factual["pred_proba"]),
            "pred_proba_counterfactual": float(result.proba),
            "num_changed_features": int(len(changed_features)),
            "changed_features": json.dumps(changed_features),
            "changes": json.dumps(changes),
            "total_abs_shift": total_abs_shift,
            "facet_steps": int(len(result.facets)),
            "facets": json.dumps(result.facets),
        }
        for feature in feature_cols:
            row[feature] = result.x_cf[feature]
        counterfactual_rows.append(row)

    counterfactuals = pd.DataFrame(counterfactual_rows)
    counterfactuals, eval_summary = evaluate_counterfactuals(
        factuals=factuals,
        counterfactuals=counterfactuals,
        x_reference=x_train,
        metadata=metadata,
        mutable_features=mutable_features,
        threshold=cfg.min_query_proba,
        feature_cols=feature_cols,
        plausibility_features=mutable_features,
    )

    num_selected_queries = int(len(factuals))
    num_failed_queries = int(len(failed_rows))
    num_successful_queries = int(num_selected_queries - num_failed_queries)
    summary: Dict[str, object] = {
        "method": "facet_style_xgboost_point_preserve_missing",
        "num_queries": num_selected_queries,
        "num_counterfactual_rows": int(len(counterfactuals)),
        "mutable_features": len(mutable_features),
        "search_features": mutable_features,
        "feature_subset_strategy": cfg.feature_subset_strategy,
        "num_failed_queries": num_failed_queries,
        "num_successful_queries": num_successful_queries,
        "query_success_rate": (
            float(num_successful_queries / num_selected_queries)
            if num_selected_queries > 0 else 0.0
        ),
        "num_model_facets": int(len(facets)),
        "num_candidate_facets": int(len(candidates)),
    }
    summary.update(eval_summary)

    factuals.to_csv(out_dir / "selected_factuals.csv", index=False)
    counterfactuals.to_csv(out_dir / "counterfactuals.csv", index=False)
    pd.DataFrame(failed_rows).to_csv(out_dir / "failed_queries.csv", index=False)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)


if __name__ == "__main__":
    main()



