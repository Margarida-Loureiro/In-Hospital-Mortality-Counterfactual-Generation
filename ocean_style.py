from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from cf_experiment_utils import (
    CFExperimentConfig,
    ensure_dir,
    evaluate_counterfactuals,
    get_immutable_features,
    get_mutable_features,
    get_permitted_range,
    load_training_artifacts,
    select_query_set,
)

try:
    import pulp
    from pulp.apis.core import PulpSolverError
except ImportError as exc:  # pragma: no cover
    pulp = None
    PULP_IMPORT_ERROR = exc
else:
    PULP_IMPORT_ERROR = None

# NOTE: this used to be parents[1] (the *parent* of this script's folder),
# which silently pointed outside the project on a normal checkout. It must
# match cf_experiment_utils.PROJECT_DIR (== this script's own folder).
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "data" / "model"
DEFAULT_OUT_ROOT = PROJECT_ROOT / "results" / "counterfactual_experiments"


@dataclass
class OceanMILPXGBMissingRoutingStrictConfig(CFExperimentConfig):
    """OCEAN MILP with explicit XGBoost native missing routing and strict split semantics."""

    data_dir: str = str(DEFAULT_DATA_DIR)
    model_dir: str = str(DEFAULT_MODEL_DIR)
    out_root: str = str(DEFAULT_OUT_ROOT)
    out_name: str = "ocean_milp_xgb_missing_routing_strict"
    max_queries: int = 50
    query_offset: int = 0
    total_cfs: int = 1
    max_trees: int = 0
    max_features_to_vary: int = 20
    feature_subset_strategy: str = "top"
    solver_name: str = "cbc"
    time_limit_seconds: int = 180
    mip_gap: float = 0.02
    proximity_weight: float = 1.0
    sparsity_weight: float = 0.05
    validity_epsilon: float = 1e-5
    split_epsilon: float = 1e-6
    change_tolerance: float = 1e-6
    diversity_min_l1: float = 1e-3
    milp_target_proba: float = 0.0
    min_changed_features: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run OCEAN-style MILP counterfactuals for the trained XGBoost "
            "in-hospital mortality model."
        )
    )
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--out-root", default=None)
    parser.add_argument("--out-name", default=None)
    parser.add_argument("--query-split", choices=["train", "val", "test"], default=None)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--query-offset", type=int, default=None, help="Skip this many selected high-risk factuals before running OCEAN.")
    parser.add_argument("--min-query-proba", type=float, default=None)
    parser.add_argument("--total-cfs", type=int, default=None)
    parser.add_argument("--max-trees", type=int, default=None, help="0 means use all trees in the XGBoost JSON model.")
    parser.add_argument("--max-features-to-vary", type=int, default=None)
    parser.add_argument("--feature-subset-strategy", choices=["all", "top"], default=None)
    parser.add_argument("--solver-name", choices=["cbc", "gurobi", "highs"], default=None)
    parser.add_argument("--time-limit-seconds", type=int, default=None)
    parser.add_argument("--mip-gap", type=float, default=None)
    parser.add_argument(
        "--milp-target-proba",
        type=float,
        default=None,
        help="Internal probability target for the partial MILP ensemble. 0 means use min-query-proba.",
    )
    parser.add_argument(
        "--min-changed-features",
        type=int,
        default=None,
        help="Require at least this many changed mutable features.",
    )
    parser.add_argument("--proximity-weight", type=float, default=None)
    parser.add_argument("--sparsity-weight", type=float, default=None)
    parser.add_argument("--allow-treatments", action="store_true")
    parser.add_argument("--random-state", type=int, default=None)
    parser.add_argument("--query-ids-path", default=None, help="CSV containing row_id values to reuse the exact same factual patients.")
    parser.add_argument("--random-query-sample", action="store_true", help="Randomly sample max_queries eligible factuals using random_state.")
    return parser.parse_args()


def apply_args(
    cfg: OceanMILPXGBMissingRoutingStrictConfig,
    args: argparse.Namespace,
) -> OceanMILPXGBMissingRoutingStrictConfig:
    for key in [
        "data_dir",
        "model_dir",
        "out_root",
        "out_name",
        "query_split",
        "max_queries",
        "query_offset",
        "min_query_proba",
        "total_cfs",
        "max_trees",
        "max_features_to_vary",
        "feature_subset_strategy",
        "solver_name",
        "time_limit_seconds",
        "mip_gap",
        "milp_target_proba",
        "min_changed_features",
        "proximity_weight",
        "sparsity_weight",
        "random_state",
        "query_ids_path",
    ]:
        value = getattr(args, key, None)
        if value is not None:
            setattr(cfg, key, value)
    if args.allow_treatments:
        cfg.allow_treatments = True
    if args.random_query_sample:
        cfg.random_query_sample = True
    return cfg


def require_pulp() -> None:
    if pulp is None:
        raise ImportError(
            "experiment_ocean_milp.py requires PuLP for the MILP model. "
            "Install it in the Python environment you use for this project, for example: "
            "python3 -m pip install pulp."
        ) from PULP_IMPORT_ERROR


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def logit(p: float) -> float:
    p = min(max(float(p), 1e-12), 1.0 - 1e-12)
    return math.log(p / (1.0 - p))


def parse_base_margin(model_json: Dict[str, object]) -> float:
    raw = model_json["learner"]["learner_model_param"].get("base_score", "[5E-1]")
    if isinstance(raw, str):
        raw = raw.strip("[]")
    return logit(float(raw))


def load_xgboost_json(model_dir: Path) -> Dict[str, object]:
    path = model_dir / "xgboost_model.json"
    if not path.exists():
        path = model_dir / "best_xgboost_model.json"
    if not path.exists():
        raise FileNotFoundError(f"Could not find xgboost_model.json or best_xgboost_model.json in {model_dir}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def select_ranked_mutable_features(
    cfg: OceanMILPXGBMissingRoutingStrictConfig,
    metadata: pd.DataFrame,
    model_dir: Path,
) -> List[str]:
    """Return the full ranked mutable list; apply max_features_to_vary per patient."""
    mutable_features = get_mutable_features(metadata, cfg)
    if cfg.feature_subset_strategy == "all":
        return mutable_features

    importance_path = model_dir / "feature_importance_gain.csv"
    if not importance_path.exists():
        return mutable_features

    importance_df = pd.read_csv(importance_path)
    mutable_set = set(mutable_features)
    ranked = [str(f) for f in importance_df["feature"].astype(str).tolist() if str(f) in mutable_set]
    return ranked if ranked else mutable_features


def cap_observed_features(
    ranked_features: Sequence[str],
    missing_features: Sequence[str],
    max_features_to_vary: int,
) -> List[str]:
    missing_set = set(missing_features)
    observed = [feature for feature in ranked_features if feature not in missing_set]
    if max_features_to_vary and max_features_to_vary > 0:
        return observed[:max_features_to_vary]
    return observed


def compute_feature_bounds(
    x_train: pd.DataFrame,
    metadata: pd.DataFrame,
    mutable_features: Sequence[str],
) -> Dict[str, Tuple[float, float]]:
    permitted = get_permitted_range(metadata, mutable_features)
    bounds: Dict[str, Tuple[float, float]] = {}
    for feature in mutable_features:
        if feature in permitted:
            lo, hi = permitted[feature]
        elif feature in x_train.columns:
            series = pd.to_numeric(x_train[feature], errors="coerce")
            lo, hi = float(series.min()), float(series.max())
        else:
            continue
        if np.isfinite(lo) and np.isfinite(hi) and lo < hi:
            bounds[feature] = (float(lo), float(hi))
    return bounds


def compute_feature_scales(x_train: pd.DataFrame, features: Sequence[str]) -> Dict[str, float]:
    scales: Dict[str, float] = {}
    for feature in features:
        series = pd.to_numeric(x_train[feature], errors="coerce") if feature in x_train.columns else pd.Series(dtype=float)
        q1 = float(series.quantile(0.25)) if not series.empty else 0.0
        q3 = float(series.quantile(0.75)) if not series.empty else 0.0
        scale = q3 - q1
        if not np.isfinite(scale) or scale <= 0:
            scale = float(series.std()) if not series.empty else 1.0
        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0
        scales[feature] = float(scale)
    return scales


def validate_xgboost_missing_routing_metadata(
    trees: Sequence[Dict[str, object]],
    feature_names: Sequence[str],
) -> None:
    if not feature_names:
        raise ValueError("XGBoost JSON does not expose feature_names; split_indices cannot be aligned to columns.")

    for tree_idx, tree in enumerate(trees):
        for key in ["left_children", "right_children", "split_indices", "split_conditions", "base_weights"]:
            if key not in tree:
                raise ValueError(f"Tree {tree_idx} is missing required XGBoost JSON field '{key}'.")
        if "default_left" not in tree:
            raise ValueError(
                f"Tree {tree_idx} does not expose default_left; native XGBoost missing routing cannot be encoded."
            )

        node_count = len(tree["left_children"])
        if len(tree["right_children"]) != node_count or len(tree["split_indices"]) != node_count:
            raise ValueError(f"Tree {tree_idx} has incompatible child/split array lengths.")
        if len(tree["split_conditions"]) != node_count or len(tree["base_weights"]) != node_count:
            raise ValueError(f"Tree {tree_idx} has incompatible split/value array lengths.")
        if len(tree["default_left"]) < node_count:
            raise ValueError(
                f"Tree {tree_idx} default_left length {len(tree['default_left'])} is shorter than node count {node_count}."
            )

        left = tree["left_children"]
        right = tree["right_children"]
        split_indices = tree["split_indices"]
        for node in range(node_count):
            if int(left[node]) == -1 and int(right[node]) == -1:
                continue
            split_idx = int(split_indices[node])
            if split_idx < 0 or split_idx >= len(feature_names):
                raise ValueError(
                    f"Tree {tree_idx} node {node} split index {split_idx} does not align with "
                    f"{len(feature_names)} feature names."
                )


def enumerate_leaf_paths(tree: Dict[str, object], feature_names: Sequence[str]) -> List[Dict[str, object]]:
    left = tree["left_children"]
    right = tree["right_children"]
    split_indices = tree["split_indices"]
    split_conditions = tree["split_conditions"]
    default_left = tree.get("default_left")
    if default_left is None:
        raise ValueError("XGBoost JSON tree is missing default_left; native missing routing cannot be encoded.")
    leaf_values = tree["base_weights"]
    paths: List[Dict[str, object]] = []

    def walk(node: int, constraints: List[Dict[str, object]]) -> None:
        if int(left[node]) == -1 and int(right[node]) == -1:
            paths.append({"node": node, "value": float(leaf_values[node]), "constraints": list(constraints)})
            return
        feature = str(feature_names[int(split_indices[node])])
        threshold = float(split_conditions[node])
        default_branch = "left" if xgb_default_left_bool(default_left[node]) else "right"
        walk(
            int(left[node]),
            constraints
            + [
                {
                    "node_id": int(node),
                    "feature": feature,
                    "threshold": threshold,
                    "branch_taken": "left",
                    "xgb_condition": "lt",
                    "default_branch": default_branch,
                }
            ],
        )
        walk(
            int(right[node]),
            constraints
            + [
                {
                    "node_id": int(node),
                    "feature": feature,
                    "threshold": threshold,
                    "branch_taken": "right",
                    "xgb_condition": "ge",
                    "default_branch": default_branch,
                }
            ],
        )

    walk(0, [])
    return paths


def make_solver(cfg: OceanMILPXGBMissingRoutingStrictConfig):
    name = cfg.solver_name.lower()
    if name == "cbc":
        return pulp.PULP_CBC_CMD(msg=True, timeLimit=cfg.time_limit_seconds, gapRel=cfg.mip_gap)
    if name == "gurobi":
        return pulp.GUROBI_CMD(msg=True, timeLimit=cfg.time_limit_seconds, gapRel=cfg.mip_gap)
    if name == "highs" and hasattr(pulp, "HiGHS_CMD"):
        return pulp.HiGHS_CMD(msg=True, timeLimit=cfg.time_limit_seconds, gapRel=cfg.mip_gap)
    raise ValueError(f"Unsupported or unavailable PuLP solver: {cfg.solver_name}")


def constant_satisfies(value: float, xgb_condition: str, threshold: float, eps: float) -> bool:
    if xgb_condition == "lt":
        return value < threshold
    if xgb_condition == "ge":
        return value >= threshold
    raise ValueError(f"Unsupported XGBoost split condition '{xgb_condition}'.")


def xgb_default_left_bool(value: object) -> bool:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true"}:
            return True
        if normalized in {"0", "false"}:
            return False
        raise ValueError(f"Unsupported default_left value '{value}'.")
    return bool(value)


def missing_route_satisfies(branch_taken: str, default_branch: str) -> bool:
    return branch_taken == default_branch


def add_split_constraint(problem, z, expr, xgb_condition: str, threshold: float, lo: float, hi: float, eps: float) -> None:
    if xgb_condition == "lt":
        boundary = threshold - eps
        m = max(hi - boundary, 0.0) + 1.0
        problem += expr <= boundary + m * (1 - z)
    elif xgb_condition == "ge":
        boundary = threshold
        m = max(boundary - lo, 0.0) + 1.0
        problem += expr >= boundary - m * (1 - z)
    else:
        raise ValueError(f"Unsupported XGBoost split condition '{xgb_condition}'.")


def solve_one_counterfactual(
    factual_imputed: pd.Series,
    factual_native: pd.Series,
    row_id: int,
    cf_rank: int,
    tree_leaf_paths: Sequence[List[Dict[str, object]]],
    base_margin: float,
    search_features: Sequence[str],
    bounds: Dict[str, Tuple[float, float]],
    scales: Dict[str, float],
    threshold: float,
    cfg: OceanMILPXGBMissingRoutingStrictConfig,
    previous_solutions: Sequence[Dict[str, float]],
) -> Tuple[Optional[Dict[str, object]], Optional[str]]:
    problem = pulp.LpProblem(f"ocean_row_{row_id}_cf_{cf_rank}", pulp.LpMinimize)
    x_vars = {}
    abs_vars = {}
    changed_vars = {}
    missing_routing_checked = 0
    missing_routing_satisfied = 0
    selected_leaf_candidates = []

    for feature in search_features:
        if feature not in factual_native.index:
            return None, {
                "error_type": "MissingFeatureInNativeFactual",
                "error_message": f"Feature '{feature}' is absent from factual_native and cannot be optimized.",
            }
        if pd.isna(factual_native[feature]):
            return None, {
                "error_type": "MissingFeatureMarkedMutable",
                "error_message": f"Feature '{feature}' is originally missing but was included as a MILP decision variable.",
            }
        lo, hi = bounds[feature]
        factual_value = float(factual_imputed[feature])
        x = pulp.LpVariable(f"x__{feature}", lowBound=lo, upBound=hi, cat="Continuous")
        d = pulp.LpVariable(f"abs__{feature}", lowBound=0.0, cat="Continuous")
        c = pulp.LpVariable(f"chg__{feature}", lowBound=0, upBound=1, cat="Binary")
        big_m = max(abs(hi - factual_value), abs(factual_value - lo), hi - lo, 1.0)
        problem += d >= x - factual_value
        problem += d >= factual_value - x
        problem += d <= big_m * c
        x_vars[feature] = x
        abs_vars[feature] = d
        changed_vars[feature] = c

    margin_terms = []
    for tree_idx, leaves in enumerate(tree_leaf_paths):
        leaf_vars = []
        tree_leaf_candidates = []
        for leaf_pos, leaf in enumerate(leaves):
            feasible_for_fixed = True
            z = pulp.LpVariable(f"z__t{tree_idx}__l{leaf_pos}", lowBound=0, upBound=1, cat="Binary")
            for constraint in leaf["constraints"]:
                feature = str(constraint["feature"])
                xgb_condition = str(constraint["xgb_condition"])
                split_value = float(constraint["threshold"])
                if feature in x_vars:
                    if feature in factual_native.index and pd.isna(factual_native[feature]):
                        return None, {
                            "error_type": "MissingFeatureMarkedMutable",
                            "error_message": (
                                f"Feature '{feature}' is originally missing but was included as a MILP decision variable."
                            ),
                        }
                    lo, hi = bounds[feature]
                    add_split_constraint(
                        problem,
                        z,
                        x_vars[feature],
                        xgb_condition,
                        split_value,
                        lo,
                        hi,
                        cfg.split_epsilon,
                    )
                else:
                    if feature not in factual_native.index:
                        return None, {
                            "error_type": "MissingFeatureInNativeFactual",
                            "error_message": (
                                f"Feature '{feature}' from tree path is absent from factual_native; refusing to use a "
                                "silent numeric fallback."
                            ),
                        }
                    if pd.isna(factual_native[feature]):
                        missing_routing_checked += 1
                        if missing_route_satisfies(
                            branch_taken=str(constraint["branch_taken"]),
                            default_branch=str(constraint["default_branch"]),
                        ):
                            missing_routing_satisfied += 1
                            continue
                        feasible_for_fixed = False
                        break
                    fixed_value = float(factual_native[feature])
                    if not np.isfinite(fixed_value):
                        return None, {
                            "error_type": "NonFiniteObservedFixedFeature",
                            "error_message": (
                                f"Feature '{feature}' is observed but non-finite in factual_native; cannot evaluate "
                                "numeric split feasibility."
                            ),
                        }
                    if not constant_satisfies(fixed_value, xgb_condition, split_value, cfg.split_epsilon):
                        feasible_for_fixed = False
                        break
            if feasible_for_fixed:
                leaf_vars.append((z, float(leaf["value"])))
                tree_leaf_candidates.append((z, leaf_pos, leaf))
            else:
                problem += z == 0
        if not leaf_vars:
            return None, "No feasible leaf remains after immutable/fixed-feature filtering."
        problem += pulp.lpSum(z for z, _ in leaf_vars) == 1
        margin_terms.append(pulp.lpSum(value * z for z, value in leaf_vars))
        selected_leaf_candidates.append(tree_leaf_candidates)

    effective_target = float(cfg.milp_target_proba) if float(cfg.milp_target_proba) > 0 else float(threshold)
    target_margin = logit(effective_target) - cfg.validity_epsilon
    problem += base_margin + pulp.lpSum(margin_terms) <= target_margin

    if int(cfg.min_changed_features) > 0:
        problem += pulp.lpSum(changed_vars.values()) >= int(cfg.min_changed_features)

    for prev_idx, previous in enumerate(previous_solutions):
        diversity_terms = []
        for feature in search_features:
            prev_value = float(previous[feature])
            div = pulp.LpVariable(f"div__{prev_idx}__{feature}", lowBound=0.0, cat="Continuous")
            problem += div >= x_vars[feature] - prev_value
            problem += div >= prev_value - x_vars[feature]
            diversity_terms.append(div / scales[feature])
        problem += pulp.lpSum(diversity_terms) >= cfg.diversity_min_l1

    objective = []
    for feature in search_features:
        objective.append(cfg.proximity_weight * abs_vars[feature] / scales[feature])
        objective.append(cfg.sparsity_weight * changed_vars[feature])
    problem += pulp.lpSum(objective)

    try:
        status = problem.solve(make_solver(cfg))
    except PulpSolverError as exc:
        return None, {
            "error_type": "PulpSolverError",
            "error_message": str(exc),
            "solver_name": cfg.solver_name,
            "time_limit_seconds": cfg.time_limit_seconds,
            "mip_gap": cfg.mip_gap,
            "max_trees": cfg.max_trees,
            "max_features_to_vary": cfg.max_features_to_vary,
        }
    status_name = pulp.LpStatus.get(status, str(status))
    if status_name != "Optimal":
        return None, {
            "error_type": "SolverNonOptimalStatus",
            "error_message": f"Solver returned status {status_name}.",
            "solver_status": status_name,
            "solver_name": cfg.solver_name,
            "time_limit_seconds": cfg.time_limit_seconds,
            "mip_gap": cfg.mip_gap,
            "max_trees": cfg.max_trees,
            "max_features_to_vary": cfg.max_features_to_vary,
        }

    cf_values = factual_imputed.copy()
    for feature, var in x_vars.items():
        value = var.value()
        if value is None:
            return None, "Solver returned no value for a decision variable."
        cf_values[feature] = float(value)

    selected_leaf_variables = []
    selected_leaf_nodes = []
    selected_leaf_values = []
    missing_routing_on_selected_paths = 0
    missing_routing_satisfied_on_selected_paths = 0
    for tree_idx, tree_candidates in enumerate(selected_leaf_candidates):
        selected = [(z, leaf_pos, leaf) for z, leaf_pos, leaf in tree_candidates if (z.value() or 0.0) > 0.5]
        if len(selected) != 1:
            return None, {
                "error_type": "SelectedLeafAuditError",
                "error_message": f"Expected one selected leaf for tree {tree_idx}, found {len(selected)}.",
            }
        z, leaf_pos, leaf = selected[0]
        selected_leaf_variables.append(z.name)
        selected_leaf_nodes.append(int(leaf["node"]))
        selected_leaf_values.append(float(leaf["value"]))
        for constraint in leaf["constraints"]:
            feature = str(constraint["feature"])
            if feature in x_vars:
                continue
            if feature not in factual_native.index:
                return None, {
                    "error_type": "MissingFeatureInNativeFactual",
                    "error_message": f"Feature '{feature}' from selected path is absent from factual_native.",
                }
            if pd.isna(factual_native[feature]):
                missing_routing_on_selected_paths += 1
                if missing_route_satisfies(
                    branch_taken=str(constraint["branch_taken"]),
                    default_branch=str(constraint["default_branch"]),
                ):
                    missing_routing_satisfied_on_selected_paths += 1

    selected_leaf_sum = float(sum(selected_leaf_values))
    subset_margin = float(base_margin + selected_leaf_sum)
    margin_from_expression = float(base_margin + sum(pulp.value(term) for term in margin_terms))
    if not np.isclose(subset_margin, margin_from_expression, rtol=0.0, atol=1e-8):
        return None, {
            "error_type": "SelectedLeafAuditError",
            "error_message": "Selected leaf values do not match the solved MILP margin expression.",
            "selected_leaf_sum": selected_leaf_sum,
            "base_margin": float(base_margin),
            "ocean_margin_subset": subset_margin,
            "margin_from_expression": margin_from_expression,
        }
    if not np.isclose(selected_leaf_sum, subset_margin - float(base_margin), rtol=0.0, atol=1e-8):
        return None, {
            "error_type": "SelectedLeafAuditError",
            "error_message": "selected_leaf_sum does not match ocean_margin_subset - base_margin.",
            "selected_leaf_sum": selected_leaf_sum,
            "base_margin": float(base_margin),
            "ocean_margin_subset": subset_margin,
        }

    changed_features = []
    changes = {}
    total_abs_shift = 0.0
    for feature in search_features:
        before = float(factual_imputed[feature])
        after = float(cf_values[feature])
        if abs(after - before) > cfg.change_tolerance:
            changed_features.append(feature)
            changes[feature] = {"from": before, "to": after}
            total_abs_shift += abs(after - before)

    row = cf_values.to_dict()
    def first_present(*columns: str) -> object:
        for col in columns:
            if col in factual_imputed.index and pd.notna(factual_imputed[col]):
                return factual_imputed[col]
        raise ValueError(f"None of the factual audit columns are available: {columns}")

    pred_proba_factual_native = float(first_present("pred_proba_native", "pred_proba"))
    pred_label_factual_native = int(first_present("pred_label_native", "pred_label"))
    pred_proba_factual_imputed = float(first_present("pred_proba_imputed_factual", "pred_proba"))
    pred_label_factual_imputed = int(first_present("pred_label_imputed_factual", "pred_label"))
    row.update(
        {
            "row_id": row_id,
            "cf_rank": cf_rank,
            "pred_proba_factual": pred_proba_factual_native,
            "pred_proba_factual_native": pred_proba_factual_native,
            "pred_proba_factual_imputed": pred_proba_factual_imputed,
            "pred_label_factual_native": pred_label_factual_native,
            "pred_label_factual_imputed": pred_label_factual_imputed,
            "factual_prediction_audit_note": (
                "Native and imputed factual prediction audits use explicit factual score columns when available; "
                "otherwise they fall back to the existing pred_proba/pred_label fields."
            ),
            "num_changed_features": len(changed_features),
            "changed_features": json.dumps(changed_features),
            "changes": json.dumps(changes),
            "total_abs_shift": float(total_abs_shift),
            "milp_status": status_name,
            "milp_objective": float(pulp.value(problem.objective)),
            "base_margin": float(base_margin),
            "selected_leaf_values": json.dumps(selected_leaf_values),
            "selected_leaf_sum": selected_leaf_sum,
            "ocean_margin_subset": subset_margin,
            "ocean_proba_subset": float(sigmoid(subset_margin)),
            "num_missing_routing_constraints_checked_during_leaf_filtering": int(missing_routing_checked),
            "num_missing_routing_constraints_satisfied_during_leaf_filtering": int(missing_routing_satisfied),
            "num_missing_routing_constraints_on_selected_paths": int(missing_routing_on_selected_paths),
            "num_missing_routing_constraints_satisfied_on_selected_paths": int(
                missing_routing_satisfied_on_selected_paths
            ),
            "selected_leaf_variables": json.dumps(selected_leaf_variables),
            "selected_leaf_nodes": json.dumps(selected_leaf_nodes),
            "missing_routing_note": (
                "Originally missing fixed features were routed using XGBoost default_left branches, not median "
                "imputation. Leaf-filtering counts are accumulated while testing feasible leaves and are not "
                "restricted to the final selected leaf; selected-path counts are reported separately."
            ),
        }
    )
    return row, None


def compute_selected_leaf_explicit_routing(
    tree: Dict[str, object],
    feature_names: Sequence[str],
    factual_native: pd.Series,
    factual_imputed: pd.Series,
) -> int:
    """Return the leaf selected by explicit XGBoost numeric/default-missing routing."""
    left = tree["left_children"]
    right = tree["right_children"]
    split_indices = tree["split_indices"]
    split_conditions = tree["split_conditions"]
    default_left = tree.get("default_left")
    if default_left is None:
        raise ValueError("XGBoost JSON tree is missing default_left; native missing routing cannot be encoded.")

    node = 0
    while int(left[node]) != -1 or int(right[node]) != -1:
        feature = str(feature_names[int(split_indices[node])])
        if feature not in factual_native.index:
            raise ValueError(f"Feature '{feature}' is absent from factual_native while routing tree.")
        if pd.isna(factual_native[feature]):
            node = int(left[node]) if xgb_default_left_bool(default_left[node]) else int(right[node])
            continue
        value = float(factual_native[feature])
        if not np.isfinite(value):
            raise ValueError(f"Feature '{feature}' is observed but non-finite while routing tree.")
        node = int(left[node]) if value < float(split_conditions[node]) else int(right[node])
    return int(node)


def compute_selected_leaves_explicit_routing(
    trees: Sequence[Dict[str, object]],
    feature_names: Sequence[str],
    factual_native: pd.Series,
    factual_imputed: pd.Series,
) -> List[int]:
    return [
        compute_selected_leaf_explicit_routing(tree, feature_names, factual_native, factual_imputed)
        for tree in trees
    ]


def predict_counterfactuals(counterfactuals: pd.DataFrame, model, feature_cols: Sequence[str]) -> pd.DataFrame:
    if counterfactuals.empty:
        return counterfactuals
    out = counterfactuals.copy()
    matrix = out.loc[:, list(feature_cols)].copy()
    out["pred_proba_counterfactual"] = model.predict_proba(matrix)[:, 1]
    out["pred_label_counterfactual"] = model.predict(matrix).astype(int)
    return out


def add_final_evaluation_audits(counterfactuals: pd.DataFrame, threshold: float) -> pd.DataFrame:
    if counterfactuals.empty:
        return counterfactuals
    out = counterfactuals.copy()
    out["milp_valid_subset"] = out["ocean_proba_subset"].astype(float) < float(threshold)
    out["native_xgb_valid"] = out["pred_proba_counterfactual"].astype(float) < float(threshold)
    out["milp_native_probability_gap"] = (
        out["pred_proba_counterfactual"].astype(float) - out["ocean_proba_subset"].astype(float)
    )
    out["validity_audit_note"] = (
        "Validity audits use probability < threshold for the negative-class counterfactual target."
    )
    return out


def select_query_batch(
    x_query_raw: pd.DataFrame,
    x_query: pd.DataFrame,
    y_query: pd.Series,
    model,
    cfg: OceanMILPXGBMissingRoutingStrictConfig,
) -> pd.DataFrame:
    requested_max_queries = int(cfg.max_queries) if cfg.max_queries is not None else 0
    query_offset = max(0, int(cfg.query_offset))

    if str(getattr(cfg, "query_ids_path", "") or ""):
        factuals = select_query_set(x_query_raw, x_query, y_query, model, cfg)
        cfg.max_queries = requested_max_queries
        return factuals.reset_index(drop=True)

    if query_offset > 0 and requested_max_queries > 0:
        cfg.max_queries = query_offset + requested_max_queries

    factuals = select_query_set(x_query_raw, x_query, y_query, model, cfg)

    if query_offset > 0:
        if requested_max_queries > 0:
            factuals = factuals.iloc[query_offset : query_offset + requested_max_queries].copy()
        else:
            factuals = factuals.iloc[query_offset:].copy()

    cfg.max_queries = requested_max_queries
    return factuals.reset_index(drop=True)


def impute_model_matrices_with_train_median(artifacts: Dict[str, object]) -> Dict[str, object]:
    """Make OCEAN MILP matrices finite. MILP solvers cannot handle NaN/inf."""
    x_train = artifacts["x_train"].replace([np.inf, -np.inf], np.nan)
    medians = x_train.median(axis=0, numeric_only=True).fillna(0.0)

    for key in ["x_train", "x_val", "x_test"]:
        x = artifacts[key].replace([np.inf, -np.inf], np.nan)
        artifacts[key] = x.fillna(medians).fillna(0.0)

    return artifacts


def overlay_imputed_features_on_selected_factuals(
    factuals: pd.DataFrame,
    x_query_imputed: pd.DataFrame,
    feature_cols: Sequence[str],
    model,
    threshold: float,
) -> pd.DataFrame:
    """Keep the native-XGBoost selected cohort, but make feature values finite for MILP."""
    if factuals.empty:
        return factuals

    out = factuals.copy()
    model_feature_cols = [c for c in feature_cols if c in out.columns and c in x_query_imputed.columns]
    row_ids = out["row_id"].astype(int).to_numpy()

    out["pred_proba_native"] = out["pred_proba"].astype(float)
    out["pred_label_native"] = out["pred_label"].astype(int)

    imputed_rows = x_query_imputed.iloc[row_ids].reset_index(drop=True)
    out.loc[:, model_feature_cols] = imputed_rows.loc[:, model_feature_cols].to_numpy()

    imputed_proba = model.predict_proba(out.loc[:, list(feature_cols)])[:, 1]
    out["pred_proba_imputed_factual"] = imputed_proba
    out["pred_label_imputed_factual"] = (imputed_proba >= float(threshold)).astype(int)
    return out


def add_imputed_factual_scores(
    factuals_native: pd.DataFrame,
    factuals_imputed: pd.DataFrame,
) -> pd.DataFrame:
    """Save native factual values while retaining the imputed factual scores for auditing."""
    out = factuals_native.copy()
    for col in ["pred_proba_native", "pred_label_native", "pred_proba_imputed_factual", "pred_label_imputed_factual"]:
        if col in factuals_imputed.columns:
            out[col] = factuals_imputed[col].to_numpy()
    return out


def originally_missing_features(
    factual_native: pd.Series,
    features: Sequence[str],
) -> List[str]:
    return [feature for feature in features if feature in factual_native.index and pd.isna(factual_native[feature])]


def restore_native_missing_values(
    counterfactuals: pd.DataFrame,
    factuals_native: pd.DataFrame,
    feature_cols: Sequence[str],
) -> pd.DataFrame:
    """Restore features that were originally NaN before native-XGBoost evaluation."""
    if counterfactuals.empty:
        return counterfactuals

    restored = counterfactuals.copy()
    factual_by_row = factuals_native.set_index("row_id")
    missing_payloads = []

    for idx, row in restored.iterrows():
        row_id = int(row["row_id"])
        factual_native = factual_by_row.loc[row_id]
        missing = originally_missing_features(factual_native, feature_cols)
        missing_payloads.append(missing)
        for feature in missing:
            if feature in restored.columns:
                restored.at[idx, feature] = np.nan

    restored["originally_missing_features_restored"] = [json.dumps(x) for x in missing_payloads]
    restored["num_originally_missing_features_restored"] = [len(x) for x in missing_payloads]
    return restored


def main() -> None:
    require_pulp()
    cfg = apply_args(OceanMILPXGBMissingRoutingStrictConfig(), parse_args())
    artifacts = load_training_artifacts(cfg)
    cfg.min_query_proba = float(artifacts["model_threshold"])

    model = artifacts["model"]
    model_dir = Path(cfg.model_dir)
    metadata = artifacts["cf_metadata"]

    x_train_native = artifacts["x_train"].copy()
    x_query_native = artifacts[f"x_{cfg.query_split}"].copy()
    y_query = artifacts[f"y_{cfg.query_split}"]
    x_query_raw = artifacts[f"x_{cfg.query_split}_raw"]
    factuals_native = select_query_batch(x_query_raw, x_query_native, y_query, model, cfg)

    artifacts_imputed = impute_model_matrices_with_train_median(dict(artifacts))
    x_train_imputed = artifacts_imputed["x_train"]
    x_query_imputed = artifacts_imputed[f"x_{cfg.query_split}"]

    mutable_features = get_mutable_features(metadata, cfg)
    immutable_features = get_immutable_features(metadata, cfg)
    ranked_search_features = select_ranked_mutable_features(cfg, metadata, model_dir)
    feature_cols = list(x_train_imputed.columns)

    factuals_imputed = overlay_imputed_features_on_selected_factuals(
        factuals=factuals_native,
        x_query_imputed=x_query_imputed,
        feature_cols=feature_cols,
        model=model,
        threshold=cfg.min_query_proba,
    )
    factuals_output = add_imputed_factual_scores(factuals_native, factuals_imputed)

    ranked_search_features = [f for f in ranked_search_features if f in feature_cols]
    bounds = compute_feature_bounds(x_train_imputed, metadata, ranked_search_features)
    ranked_search_features = [f for f in ranked_search_features if f in bounds]
    scales = compute_feature_scales(x_train_imputed, ranked_search_features)

    model_json = load_xgboost_json(model_dir)
    booster_model = model_json["learner"]["gradient_booster"]["model"]
    feature_names = [str(f) for f in model_json["learner"].get("feature_names", feature_cols)]
    all_trees = booster_model["trees"]
    num_total_trees_in_model = int(len(all_trees))
    trees = all_trees
    if cfg.max_trees and cfg.max_trees > 0:
        trees = trees[: cfg.max_trees]
    validate_xgboost_missing_routing_metadata(trees, feature_names)
    tree_leaf_paths = [enumerate_leaf_paths(tree, feature_names) for tree in trees]
    num_trees_encoded = int(len(trees))
    uses_partial_ensemble = bool(
        int(cfg.max_trees) > 0 and num_trees_encoded < num_total_trees_in_model
    )
    base_margin = parse_base_margin(model_json)

    out_dir = Path(cfg.out_root) / cfg.out_name
    ensure_dir(out_dir)
    with open(out_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)
    checkpoint_path = out_dir / "checkpoint_rows.jsonl"
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    factual_native_by_row = factuals_native.set_index("row_id")
    cf_rows: List[Dict[str, object]] = []
    failed_rows: List[Dict[str, object]] = []
    missing_search_counts: List[int] = []

    for factual in factuals_imputed.to_dict(orient="records"):
        row_id = int(factual["row_id"])
        factual_imputed_series = pd.Series(factual).reindex(
            feature_cols
            + [
                "row_id",
                "y_true",
                "pred_proba",
                "pred_label",
                "pred_proba_native",
                "pred_label_native",
                "pred_proba_imputed_factual",
                "pred_label_imputed_factual",
            ]
        )
        factual_native = factual_native_by_row.loc[row_id].reindex(feature_cols + ["row_id", "y_true", "pred_proba", "pred_label"])
        missing_search_features = originally_missing_features(factual_native, ranked_search_features)
        row_search_features = cap_observed_features(
            ranked_features=ranked_search_features,
            missing_features=missing_search_features,
            max_features_to_vary=int(cfg.max_features_to_vary),
        )
        row_bounds = {feature: bounds[feature] for feature in row_search_features}
        row_scales = {feature: scales[feature] for feature in row_search_features}
        missing_search_counts.append(len(missing_search_features))

        if not row_search_features:
            failed_rows.append(
                {
                    "row_id": row_id,
                    "cf_rank": 1,
                    "error_type": "NoMutableObservedFeatures",
                    "error_message": "All selected OCEAN search features were originally missing for this row.",
                    "num_missing_search_features": len(missing_search_features),
                    "missing_search_features": json.dumps(missing_search_features),
                }
            )
            continue

        previous_solutions: List[Dict[str, float]] = []
        for cf_rank in range(1, int(cfg.total_cfs) + 1):
            row, error = solve_one_counterfactual(
                factual_imputed=factual_imputed_series,
                factual_native=factual_native,
                row_id=row_id,
                cf_rank=cf_rank,
                tree_leaf_paths=tree_leaf_paths,
                base_margin=base_margin,
                search_features=row_search_features,
                bounds=row_bounds,
                scales=row_scales,
                threshold=cfg.min_query_proba,
                cfg=cfg,
                previous_solutions=previous_solutions,
            )
            if row is None:
                failure = {
                    "row_id": row_id,
                    "cf_rank": cf_rank,
                    "error_type": "MILPFailure",
                    "error_message": error,
                    "num_missing_search_features": len(missing_search_features),
                    "missing_search_features": json.dumps(missing_search_features),
                }
                if isinstance(error, dict):
                    failure.update(error)
                failed_rows.append(failure)
                # NOTE: this branch previously logged {"status": "success", ...}
                # here even though `row is None` on failure -- a copy/paste bug.
                # It now records the actual failure, and a matching entry is
                # appended on the success path below so this file is a genuine
                # per-attempt log (e.g. for monitoring a long OCEAN run).
                with open(checkpoint_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"status": "failed", "row_id": row_id, "cf_rank": cf_rank, "error": error}, default=str) + "\n")
                break
            row["num_missing_search_features_frozen"] = len(missing_search_features)
            row["missing_search_features_frozen"] = json.dumps(missing_search_features)
            row["row_search_features"] = json.dumps(row_search_features)
            row["num_row_search_features"] = len(row_search_features)
            previous_solutions.append({feature: float(row[feature]) for feature in row_search_features})
            cf_rows.append(row)
            with open(checkpoint_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"status": "success", "row_id": row_id, "cf_rank": cf_rank}, default=str) + "\n")

    counterfactuals = pd.DataFrame(cf_rows)
    counterfactuals = restore_native_missing_values(counterfactuals, factuals_native, feature_cols)
    counterfactuals = predict_counterfactuals(counterfactuals, model, feature_cols)
    counterfactuals = add_final_evaluation_audits(counterfactuals, cfg.min_query_proba)
    counterfactuals, eval_summary = evaluate_counterfactuals(
        factuals=factuals_output,
        counterfactuals=counterfactuals,
        x_reference=x_train_native,
        metadata=metadata,
        mutable_features=mutable_features,
        threshold=cfg.min_query_proba,
        feature_cols=feature_cols,
        plausibility_features=ranked_search_features,
    )

    num_selected = int(len(factuals_native))
    failed_query_ids = set(int(r["row_id"]) for r in failed_rows)
    successful_query_ids = set(counterfactuals["row_id"].astype(int).tolist()) if not counterfactuals.empty else set()
    summary: Dict[str, object] = {
        "method": "ocean_milp_xgb_missing_routing_strict",
        "num_queries": num_selected,
        "num_counterfactual_rows": int(len(counterfactuals)),
        "num_failed_rows": int(len(failed_rows)),
        "num_failed_queries": int(len(failed_query_ids - successful_query_ids)),
        "num_successful_queries": int(len(successful_query_ids)),
        "query_success_rate": float(len(successful_query_ids) / num_selected) if num_selected else 0.0,
        "query_offset": int(cfg.query_offset),
        "mutable_features": int(len(mutable_features)),
        "immutable_features": int(len(immutable_features)),
        "ranked_search_features": list(ranked_search_features),
        "max_features_to_vary_per_row": int(cfg.max_features_to_vary),
        "feature_subset_strategy": cfg.feature_subset_strategy,
        "mean_missing_search_features_frozen": float(np.mean(missing_search_counts)) if missing_search_counts else 0.0,
        "median_missing_search_features_frozen": float(np.median(missing_search_counts)) if missing_search_counts else 0.0,
        "num_total_trees_in_model": num_total_trees_in_model,
        "num_trees_encoded": num_trees_encoded,
        "uses_partial_ensemble": uses_partial_ensemble,
        "solver_name": cfg.solver_name,
        "threshold_used": float(cfg.min_query_proba),
        "milp_target_proba": float(cfg.milp_target_proba),
        "min_changed_features": int(cfg.min_changed_features),
        "note": (
            "OCEAN-style MILP for XGBoost with native missing-value routing: originally missing "
            "values are preserved and are routed through XGBoost's learned default_left branches "
            "inside the MILP leaf feasibility logic. Numeric splits use XGBoost-style left < "
            "threshold and right >= threshold semantics, with the MILP left branch encoded as "
            "x <= threshold - split_epsilon. Train-median-imputed values are used only for finite "
            "bounds, scales, and observed mutable search variables, not for missing-value path "
            "feasibility. If uses_partial_ensemble is true, the MILP target is optimized on the "
            "truncated encoded ensemble; final validity is always evaluated using the full native "
            "XGBoost model with NaNs restored."
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


if __name__ == "__main__":
    main()




