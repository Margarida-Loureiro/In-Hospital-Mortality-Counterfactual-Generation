from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_DIR = Path(__file__).resolve().parent


# ==========================================================
# Configuration
# ==========================================================


@dataclass
class Config:
    # Folder containing the raw MIMIC-IV CSVs (hosp/ and icu/ subfolders), as
    # downloaded from PhysioNet. You must provide this yourself; see README.md.
    data_dir: str = str(PROJECT_DIR / "data" / "mimic_raw")
    out_dir: str = str(PROJECT_DIR / "data" / "processed")

    # MIMIC-IV CSV paths
    patients_file: str = "hosp/patients.csv"
    admissions_file: str = "hosp/admissions.csv"
    transfers_file: str = "hosp/transfers.csv"
    icustays_file: str = "icu/icustays.csv"
    chartevents_file: str = "icu/chartevents.csv"
    labevents_file: str = "hosp/labevents.csv"
    inputevents_file: str = "icu/inputevents.csv"
    outputevents_file: str = "icu/outputevents.csv"
    procedureevents_file: str = "icu/procedureevents.csv"
    prescriptions_file: str = "hosp/prescriptions.csv"
    d_items_file: str = "icu/d_items.csv"
    d_labitems_file: str = "hosp/d_labitems.csv"

    # Cohort / target
    adult_age: int = 18
    hours_window: int = 48
    target: str = "hospital_mortality"  # in-hospital death after the prediction landmark
    keep_first_icu_per_hadm: bool = True
    # Clean prospective landmark setup: use only data available through ICU intime + hours_window.
    # Patients who die or are discharged before the landmark are excluded because they are no
    # longer at risk for a prediction made at that time.
    require_at_risk_at_window_end: bool = True

    # Split
    test_size: float = 0.15
    val_size: float = 0.15
    random_state: int = 42

    # Feature selection: frequent concepts from train split only
    top_chart_items: Optional[int] = None
    top_lab_items: Optional[int] = None
    top_input_items: Optional[int] = None
    top_output_items: Optional[int] = None
    top_proc_items: Optional[int] = None
    top_drugs: Optional[int] = None
    min_chart_item_prevalence: float = 0.005
    min_lab_item_prevalence: float = 0.005
    min_input_item_prevalence: float = 0.002
    min_output_item_prevalence: float = 0.002
    min_proc_item_prevalence: float = 0.001
    min_drug_prevalence: float = 0.001

    # Raw-value clipping for chart/lab before aggregation
    clip_lower_q: float = 0.01
    clip_upper_q: float = 0.99
    max_quantile_sample_per_item: int = 200_000

    # Optional hard plausibility ranges for very common bedside variables.
    # These are only applied if the itemid is mapped below.
    use_hard_ranges: bool = False

    # Missingness handling
    add_missing_indicators: bool = False

    # CSV reading
    chunksize_large: int = 1_000_000
    chunksize_medium: int = 500_000

    # Tables to include
    # Treatment/care-process features usually improve prediction. Keep their
    # counterfactual mutability constrained later because they often encode
    # clinician response to severity.
    include_inputs: bool = True
    include_outputs: bool = True
    include_procedures: bool = True
    include_prescriptions: bool = True
    use_procedure_categories: bool = False
    add_prescription_other_group: bool = True
    categorical_min_prevalence: float = 0.01
    categorical_max_levels: int = 12
    exposure_min_prevalence: float = 0.002


# ==========================================================
# Utilities
# ==========================================================


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_to_datetime(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")


def safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def compute_age(anchor_age: pd.Series, anchor_year: pd.Series, dt: pd.Series) -> pd.Series:
    return anchor_age + (dt.dt.year - anchor_year)


def load_item_metadata(path: Path, itemids: Sequence[int]) -> Dict[str, Dict[str, object]]:
    if not path.exists() or len(itemids) == 0:
        return {}

    wanted = set(int(x) for x in itemids)
    meta = pd.read_csv(path, low_memory=False)
    if "itemid" not in meta.columns:
        return {}

    keep_cols = [c for c in ["itemid", "label", "category", "unitname", "fluid"] if c in meta.columns]
    meta = meta[keep_cols].copy()
    meta["itemid"] = safe_numeric(meta["itemid"]).astype("Int64")
    meta = meta.dropna(subset=["itemid"])
    meta["itemid"] = meta["itemid"].astype(int)
    meta = meta[meta["itemid"].isin(wanted)]

    out: Dict[str, Dict[str, object]] = {}
    for row in meta.to_dict(orient="records"):
        itemid = int(row.pop("itemid"))
        out[str(itemid)] = {k: (None if pd.isna(v) else v) for k, v in row.items()}
    return out


def select_frequent_keys(counter: Counter, min_count: int, top_n: Optional[int] = None) -> List:
    selected = [k for k, v in counter.most_common() if v >= min_count]
    if top_n is not None and top_n > 0:
        selected = selected[:top_n]
    return selected


def min_count_from_prevalence(n_train_stays: int, prevalence: float) -> int:
    prevalence = float(prevalence)
    if prevalence <= 0:
        return 1
    return max(1, int(math.ceil(n_train_stays * prevalence)))


def update_stay_level_counter(counter: Counter, stay_ids: pd.Series, keys: pd.Series) -> None:
    pairs = pd.DataFrame({"stay_id": stay_ids, "key": keys}).dropna().drop_duplicates()
    if pairs.empty:
        return
    counter.update(pairs["key"].tolist())


def load_d_items_category_map(path: Path) -> Dict[int, str]:
    if not path.exists():
        return {}

    df = pd.read_csv(path, usecols=[c for c in ["itemid", "category"]], low_memory=False)
    if "itemid" not in df.columns or "category" not in df.columns:
        return {}

    df["itemid"] = safe_numeric(df["itemid"]).astype("Int64")
    df = df.dropna(subset=["itemid"])
    df["itemid"] = df["itemid"].astype(int)
    df["category"] = df["category"].fillna("UNKNOWN").astype(str).str.strip().str.upper()
    df["category"] = df["category"].replace({"": "UNKNOWN", "NAN": "UNKNOWN"})
    return dict(zip(df["itemid"], df["category"]))


def fit_frequency_encoder(
    values: pd.Series,
    train_mask: pd.Series,
    min_count: int,
    max_levels: int,
) -> Tuple[pd.Series, Dict[str, object]]:
    values = values.fillna("UNKNOWN").astype(str)
    train_values = values.loc[train_mask]

    level_counts = train_values.value_counts()
    kept_levels = level_counts[level_counts >= int(min_count)].index.tolist()
    if max_levels > 0:
        kept_levels = kept_levels[:max_levels]

    special_levels = ["UNKNOWN", "RARE"]
    for level in special_levels:
        if level not in kept_levels:
            kept_levels.append(level)

    mapped = values.where(values.isin(kept_levels), "RARE")
    train_mapped = mapped.loc[train_mask]
    frequencies = train_mapped.value_counts(normalize=True).to_dict()
    encoded = mapped.map(frequencies).fillna(0.0).astype(float)

    spec = {
        "encoding": "train_frequency_with_rare_bucket",
        "min_count": int(min_count),
        "max_levels": int(max_levels),
        "kept_levels": [str(x) for x in kept_levels],
        "frequencies": {str(k): float(v) for k, v in frequencies.items()},
    }
    return encoded, spec

def write_table(df: pd.DataFrame, path_without_suffix: Path) -> str:
    parquet_path = path_without_suffix.with_suffix(".parquet")
    csv_path = path_without_suffix.with_suffix(".csv")
    try:
        df.to_parquet(parquet_path, index=False)
        return parquet_path.name
    except ImportError:
        df.to_csv(csv_path, index=False)
        return csv_path.name


def train_constant_columns(df: pd.DataFrame, exclude: Sequence[str]) -> List[str]:
    cols = [c for c in df.columns if c not in exclude]
    dead = []
    for c in cols:
        if df[c].nunique(dropna=False) <= 1:
            dead.append(c)
    return dead


def sparse_exposure_columns(
    df: pd.DataFrame,
    exposure_cols: Sequence[str],
    min_prevalence: float,
) -> List[str]:
    if len(df) == 0:
        return []

    dead = []
    for c in exposure_cols:
        prevalence = float((df[c].fillna(0) != 0).mean())
        if prevalence < float(min_prevalence):
            dead.append(c)
    return dead



class ReservoirSampler:
    """
    Per-key reservoir sampler for approximate train-only quantiles.
    Keeps at most max_size values per key.
    """

    def __init__(self, max_size: int, random_state: int = 42):
        self.max_size = int(max_size)
        self.rng = np.random.default_rng(random_state)
        self.samples: Dict[int, List[float]] = defaultdict(list)
        self.counts: Dict[int, int] = defaultdict(int)

    def update_many(self, key: int, values: Sequence[float]) -> None:
        if len(values) == 0:
            return
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return

        sample_list = self.samples[key]
        count = self.counts[key]

        for v in arr:
            count += 1
            if len(sample_list) < self.max_size:
                sample_list.append(float(v))
            else:
                j = self.rng.integers(0, count)
                if j < self.max_size:
                    sample_list[int(j)] = float(v)

        self.counts[key] = count

    def quantile_bounds(self, lower_q: float, upper_q: float) -> Dict[int, Tuple[Optional[float], Optional[float]]]:
        out: Dict[int, Tuple[Optional[float], Optional[float]]] = {}
        for k, vals in self.samples.items():
            if len(vals) == 0:
                out[k] = (None, None)
                continue
            arr = np.asarray(vals, dtype=float)
            lo = float(np.quantile(arr, lower_q))
            hi = float(np.quantile(arr, upper_q))
            out[k] = (lo, hi)
        return out


class NumericSummaryAgg:
    """
    Maintains first, last, min, max, mean for each stay_id x itemid.
    first/last are time ordered on the raw cleaned values.
    """

    def __init__(self) -> None:
        self.data: Dict[Tuple[int, int], Dict[str, float]] = {}

    def update(self, stay_id: int, itemid: int, event_time: pd.Timestamp, value: float) -> None:
        if pd.isna(event_time) or not np.isfinite(value):
            return
        key = (int(stay_id), int(itemid))
        if key not in self.data:
            self.data[key] = {
                "first_time": event_time.value,
                "first": float(value),
                "last_time": event_time.value,
                "last": float(value),
                "min": float(value),
                "max": float(value),
                "sum": float(value),
                "count": 1.0,
            }
            return

        d = self.data[key]
        t = event_time.value
        if t < d["first_time"]:
            d["first_time"] = t
            d["first"] = float(value)
        if t > d["last_time"]:
            d["last_time"] = t
            d["last"] = float(value)
        if value < d["min"]:
            d["min"] = float(value)
        if value > d["max"]:
            d["max"] = float(value)
        d["sum"] += float(value)
        d["count"] += 1.0

    def to_wide(self, prefix: str) -> pd.DataFrame:
        rows = []
        for (stay_id, itemid), d in self.data.items():
            rows.append(
                {
                    "stay_id": stay_id,
                    "itemid": itemid,
                    f"{prefix}_first": d["first"],
                    f"{prefix}_last": d["last"],
                    f"{prefix}_min": d["min"],
                    f"{prefix}_max": d["max"],
                    f"{prefix}_mean": d["sum"] / max(d["count"], 1.0),
                }
            )
        if not rows:
            return pd.DataFrame(columns=["stay_id"])

        long_df = pd.DataFrame(rows)
        parts = []
        stat_cols = [f"{prefix}_first", f"{prefix}_last", f"{prefix}_min", f"{prefix}_max", f"{prefix}_mean"]
        for stat in stat_cols:
            tmp = long_df.pivot(index="stay_id", columns="itemid", values=stat)
            base = stat.replace(prefix + "_", "")
            tmp.columns = [f"{prefix}_{base}_item_{int(c)}" for c in tmp.columns]
            parts.append(tmp)
        return pd.concat(parts, axis=1).reset_index()


class CountSumAgg:
    """
    Generic aggregator for exposure-style features.
    Stores count plus optional sum/max/mean.
    """

    def __init__(self) -> None:
        self.data: Dict[Tuple[int, int], Dict[str, float]] = {}

    def update(self, stay_id: int, itemid: int, value: Optional[float] = None, value2: Optional[float] = None) -> None:
        key = (int(stay_id), int(itemid))
        if key not in self.data:
            self.data[key] = {
                "count": 0.0,
                "sum": 0.0,
                "sum2": 0.0,
                "n_sum": 0.0,
                "max": -np.inf,
            }
        d = self.data[key]
        d["count"] += 1.0
        if value is not None and np.isfinite(value):
            d["sum"] += float(value)
            d["n_sum"] += 1.0
            if value > d["max"]:
                d["max"] = float(value)
        if value2 is not None and np.isfinite(value2):
            d["sum2"] += float(value2)

    def to_wide(self, prefix: str, include_sum: bool = True, include_mean: bool = True, include_max: bool = True, second_sum_name: Optional[str] = None) -> pd.DataFrame:
        rows = []
        for (stay_id, itemid), d in self.data.items():
            row = {
                "stay_id": stay_id,
                "itemid": itemid,
                f"{prefix}_cnt": d["count"],
            }
            if include_sum:
                row[f"{prefix}_sum"] = d["sum"]
            if include_mean:
                row[f"{prefix}_mean"] = d["sum"] / d["n_sum"] if d["n_sum"] > 0 else np.nan
            if include_max:
                row[f"{prefix}_max"] = d["max"] if np.isfinite(d["max"]) else np.nan
            if second_sum_name is not None:
                row[f"{prefix}_{second_sum_name}"] = d["sum2"]
            rows.append(row)

        if not rows:
            return pd.DataFrame(columns=["stay_id"])

        long_df = pd.DataFrame(rows)
        stat_cols = [c for c in long_df.columns if c not in ["stay_id", "itemid"]]
        parts = []
        for stat in stat_cols:
            tmp = long_df.pivot(index="stay_id", columns="itemid", values=stat)
            base = stat.replace(prefix + "_", "")
            tmp.columns = [f"{prefix}_{base}_item_{int(c)}" for c in tmp.columns]
            parts.append(tmp)
        return pd.concat(parts, axis=1).reset_index()


# Optional hard ranges for a few common concepts if you later map itemids.
HARD_RANGE_BY_ITEMID: Dict[int, Tuple[float, float]] = {}


# ==========================================================
# Cohort
# ==========================================================


def build_cohort(cfg: Config) -> pd.DataFrame:
    data_dir = Path(cfg.data_dir)

    patients = pd.read_csv(data_dir / cfg.patients_file, low_memory=False)
    admissions = pd.read_csv(data_dir / cfg.admissions_file, low_memory=False)
    icustays = pd.read_csv(data_dir / cfg.icustays_file, low_memory=False)
    transfers = pd.read_csv(data_dir / cfg.transfers_file, low_memory=False)
    print("Initial icustays:", len(icustays))
    patients["dod"] = safe_to_datetime(patients.get("dod")) if "dod" in patients.columns else pd.NaT
    admissions["admittime"] = safe_to_datetime(admissions["admittime"])
    admissions["dischtime"] = safe_to_datetime(admissions["dischtime"])
    admissions["deathtime"] = safe_to_datetime(admissions["deathtime"])
    icustays["intime"] = safe_to_datetime(icustays["intime"])
    icustays["outtime"] = safe_to_datetime(icustays["outtime"])
    transfers["intime"] = safe_to_datetime(transfers["intime"])
    transfers["outtime"] = safe_to_datetime(transfers["outtime"])

    cohort = (
        icustays.merge(admissions, on=["subject_id", "hadm_id"], how="left", suffixes=("", "_adm"))
        .merge(patients, on="subject_id", how="left", suffixes=("", "_pat"))
    )
    print("After merge:", len(cohort))
    cohort["anchor_age"] = safe_numeric(cohort["anchor_age"])
    cohort["anchor_year"] = safe_numeric(cohort["anchor_year"])
    cohort["age"] = compute_age(cohort["anchor_age"], cohort["anchor_year"], cohort["intime"])
    cohort = cohort[cohort["age"] >= cfg.adult_age].copy()
    print("After adult filter:", len(cohort))
    print(cohort["age"].describe())
    print((cohort["age"] < 18).sum())
    print(cohort["age"].min(), cohort["age"].max())
    cohort = cohort.dropna(subset=["subject_id", "hadm_id", "stay_id", "intime"])
    print("After non-missing IDs/intime:", len(cohort))
    if cfg.keep_first_icu_per_hadm:
        cohort = (
            cohort.sort_values(["subject_id", "hadm_id", "intime"])
            .drop_duplicates(["hadm_id"], keep="first")
            .copy()
        )
    print("After first ICU per hadm:", len(cohort))
    # ICU mortality
    cohort["icu_mortality"] = (
        cohort["deathtime"].notna()
        & (cohort["deathtime"] >= cohort["intime"])
        & (cohort["outtime"].isna() | (cohort["deathtime"] <= cohort["outtime"]))
    ).astype(int)
    print("Before 48h at-risk filter:", len(cohort))
    # In-hospital mortality after ICU admission
    if "hospital_expire_flag" in cohort.columns:
        cohort["hospital_mortality"] = np.where(
            cohort["deathtime"].notna(),
            (
                (cohort["deathtime"] >= cohort["intime"])
                & cohort["dischtime"].notna()
                & (cohort["deathtime"] <= cohort["dischtime"])
            ).astype(int),
            cohort["hospital_expire_flag"].fillna(0).astype(int),
        )
    else:
        cohort["hospital_mortality"] = (
            cohort["deathtime"].notna()
            & (cohort["deathtime"] >= cohort["intime"])
            & cohort["dischtime"].notna()
            & (cohort["deathtime"] <= cohort["dischtime"])
        ).astype(int)

    cohort["feature_end"] = cohort["intime"] + pd.to_timedelta(cfg.hours_window, unit="h")
    cohort["observed_hours"] = float(cfg.hours_window)

    if cfg.require_at_risk_at_window_end:
        alive_at_landmark = cohort["deathtime"].isna() | (cohort["deathtime"] > cohort["feature_end"])
        still_admitted_at_landmark = cohort["dischtime"].isna() | (cohort["dischtime"] > cohort["feature_end"])
        cohort = cohort[alive_at_landmark & still_admitted_at_landmark].copy()

        if "hospital_expire_flag" in cohort.columns:
            cohort["hospital_mortality"] = np.where(
                cohort["deathtime"].notna(),
                (
                    (cohort["deathtime"] > cohort["feature_end"])
                    & cohort["dischtime"].notna()
                    & (cohort["deathtime"] <= cohort["dischtime"])
                ).astype(int),
                cohort["hospital_expire_flag"].fillna(0).astype(int),
            )
        else:
            cohort["hospital_mortality"] = (
                cohort["deathtime"].notna()
                & (cohort["deathtime"] > cohort["feature_end"])
                & cohort["dischtime"].notna()
                & (cohort["deathtime"] <= cohort["dischtime"])
            ).astype(int)
    print("After 48h at-risk filter:", len(cohort))
    # Transfer features before ICU admission
    tr = transfers[transfers["hadm_id"].isin(cohort["hadm_id"].unique())].copy()
    temp = cohort[["stay_id", "hadm_id", "intime"]].rename(columns={"intime": "icu_intime"})
    tr = tr.merge(temp, on="hadm_id", how="inner")
    tr = tr[tr["intime"] < tr["icu_intime"]].copy()

    n_transfers = tr.groupby("stay_id").size().rename("n_transfers_before_icu")
    last_pre_icu = (
        tr.sort_values(["stay_id", "intime"])
        .groupby("stay_id")
        .tail(1)[["stay_id", "careunit"]]
        .rename(columns={"careunit": "pre_icu_careunit"})
    )

    cohort = cohort.merge(n_transfers, on="stay_id", how="left")
    cohort = cohort.merge(last_pre_icu, on="stay_id", how="left")
    cohort["n_transfers_before_icu"] = cohort["n_transfers_before_icu"].fillna(0).astype(int)
    cohort["pre_icu_careunit"] = cohort["pre_icu_careunit"].fillna("UNKNOWN")

    keep_cols = [
        "subject_id",
        "hadm_id",
        "stay_id",
        "intime",
        "outtime",
        "admittime",
        "dischtime",
        "deathtime",
        "age",
        "gender",
        "race",
        "language",
        "marital_status",
        "insurance",
        "admission_type",
        "admission_location",
        "discharge_location",
        "first_careunit",
        "last_careunit",
        "pre_icu_careunit",
        "n_transfers_before_icu",
        "feature_end",
        "observed_hours",
        "hospital_mortality",
        "icu_mortality",
    ]
    keep_cols = [c for c in keep_cols if c in cohort.columns]
    return cohort[keep_cols].reset_index(drop=True)


# ==========================================================
# Split
# ==========================================================


def subject_level_split(cohort: pd.DataFrame, target_col: str, test_size: float, val_size: float, random_state: int) -> pd.DataFrame:
    subj = cohort.groupby("subject_id")[target_col].max().reset_index()

    train_val_subj, test_subj = train_test_split(
        subj,
        test_size=test_size,
        random_state=random_state,
        stratify=subj[target_col],
    )

    val_frac = val_size / (1.0 - test_size)
    train_subj, val_subj = train_test_split(
        train_val_subj,
        test_size=val_frac,
        random_state=random_state,
        stratify=train_val_subj[target_col],
    )

    split_map = pd.concat(
        [
            train_subj.assign(split="train"),
            val_subj.assign(split="val"),
            test_subj.assign(split="test"),
        ],
        ignore_index=True,
    )

    cohort = cohort.merge(split_map[["subject_id", "split"]], on="subject_id", how="left")
    return cohort


# ==========================================================
# Lookup maps for fast chunk filtering
# ==========================================================


def make_stay_map(cohort: pd.DataFrame) -> Dict[int, Dict[str, object]]:
    m = {}
    for r in cohort[["stay_id", "intime", "feature_end", "split"]].itertuples(index=False):
        m[int(r.stay_id)] = {"intime": r.intime, "feature_end": r.feature_end, "split": r.split}
    return m


def make_hadm_map(cohort: pd.DataFrame) -> Dict[int, Dict[str, object]]:
    # hadm -> stay_id, intime, feature_end, split
    m = {}
    for r in cohort[["hadm_id", "stay_id", "intime", "feature_end", "split"]].itertuples(index=False):
        m[int(r.hadm_id)] = {
            "stay_id": int(r.stay_id),
            "intime": r.intime,
            "feature_end": r.feature_end,
            "split": r.split,
        }
    return m


# ==========================================================
# Pass 1: concept selection from train split only
# ==========================================================


def count_top_chart_items(cfg: Config, cohort: pd.DataFrame) -> List[int]:
    data_dir = Path(cfg.data_dir)
    train_cohort = cohort[cohort["split"] == "train"]
    stay_map = make_stay_map(train_cohort)
    valid_stays = set(stay_map.keys())
    counter: Counter = Counter()
    min_count = min_count_from_prevalence(len(train_cohort), cfg.min_chart_item_prevalence)

    usecols = ["stay_id", "charttime", "itemid", "valuenum"]
    for chunk in pd.read_csv(data_dir / cfg.chartevents_file, usecols=usecols, chunksize=cfg.chunksize_large, low_memory=False):
        chunk = chunk[chunk["stay_id"].isin(valid_stays)].copy()
        if chunk.empty:
            continue
        chunk["charttime"] = safe_to_datetime(chunk["charttime"])
        chunk["valuenum"] = safe_numeric(chunk["valuenum"])
        chunk = chunk.dropna(subset=["stay_id", "itemid", "charttime", "valuenum"])
        chunk["stay_id"] = chunk["stay_id"].astype(int)
        chunk["itemid"] = chunk["itemid"].astype(int)

        chunk["intime"] = chunk["stay_id"].map(lambda x: stay_map[int(x)]["intime"])
        chunk["feature_end"] = chunk["stay_id"].map(lambda x: stay_map[int(x)]["feature_end"])
        chunk = chunk[(chunk["charttime"] >= chunk["intime"]) & (chunk["charttime"] <= chunk["feature_end"])]
        if chunk.empty:
            continue
        update_stay_level_counter(counter, chunk["stay_id"], chunk["itemid"])

    print(f"Chart item stay-prevalence threshold: >= {min_count} train stays ({cfg.min_chart_item_prevalence:.3%})")
    return select_frequent_keys(counter, min_count, cfg.top_chart_items)


def count_top_lab_items(cfg: Config, cohort: pd.DataFrame) -> List[int]:
    data_dir = Path(cfg.data_dir)
    train_cohort = cohort[cohort["split"] == "train"]
    hadm_map = make_hadm_map(train_cohort)
    valid_hadm = set(hadm_map.keys())
    counter: Counter = Counter()
    min_count = min_count_from_prevalence(len(train_cohort), cfg.min_lab_item_prevalence)

    usecols = ["hadm_id", "charttime", "itemid", "valuenum"]
    for chunk in pd.read_csv(data_dir / cfg.labevents_file, usecols=usecols, chunksize=cfg.chunksize_medium, low_memory=False):
        chunk = chunk[chunk["hadm_id"].isin(valid_hadm)].copy()
        if chunk.empty:
            continue
        chunk["charttime"] = safe_to_datetime(chunk["charttime"])
        chunk["valuenum"] = safe_numeric(chunk["valuenum"])
        chunk = chunk.dropna(subset=["hadm_id", "itemid", "charttime", "valuenum"])
        chunk["hadm_id"] = chunk["hadm_id"].astype(int)
        chunk["itemid"] = chunk["itemid"].astype(int)

        chunk["stay_id"] = chunk["hadm_id"].map(lambda x: hadm_map[int(x)]["stay_id"])
        chunk["intime"] = chunk["hadm_id"].map(lambda x: hadm_map[int(x)]["intime"])
        chunk["feature_end"] = chunk["hadm_id"].map(lambda x: hadm_map[int(x)]["feature_end"])
        chunk = chunk[(chunk["charttime"] >= chunk["intime"]) & (chunk["charttime"] <= chunk["feature_end"])]
        if chunk.empty:
            continue
        update_stay_level_counter(counter, chunk["stay_id"], chunk["itemid"])

    print(f"Lab item stay-prevalence threshold: >= {min_count} train stays ({cfg.min_lab_item_prevalence:.3%})")
    return select_frequent_keys(counter, min_count, cfg.top_lab_items)


def count_top_input_items(cfg: Config, cohort: pd.DataFrame) -> List[int]:
    data_dir = Path(cfg.data_dir)
    train_cohort = cohort[cohort["split"] == "train"]
    stay_map = make_stay_map(train_cohort)
    valid_stays = set(stay_map.keys())
    counter: Counter = Counter()
    min_count = min_count_from_prevalence(len(train_cohort), cfg.min_input_item_prevalence)

    usecols = ["stay_id", "itemid", "starttime", "endtime"]
    for chunk in pd.read_csv(data_dir / cfg.inputevents_file, usecols=usecols, chunksize=cfg.chunksize_medium, low_memory=False):
        chunk = chunk[chunk["stay_id"].isin(valid_stays)].copy()
        if chunk.empty:
            continue
        chunk["starttime"] = safe_to_datetime(chunk["starttime"])
        chunk["endtime"] = safe_to_datetime(chunk["endtime"])
        chunk["stay_id"] = chunk["stay_id"].astype(int)
        chunk["itemid"] = safe_numeric(chunk["itemid"]).astype("Int64")
        chunk = chunk.dropna(subset=["stay_id", "itemid"])
        chunk["event_start"] = chunk["starttime"].fillna(chunk["endtime"])
        chunk["event_end"] = chunk["endtime"].fillna(chunk["starttime"])
        chunk["intime"] = chunk["stay_id"].map(lambda x: stay_map[int(x)]["intime"])
        chunk["feature_end"] = chunk["stay_id"].map(lambda x: stay_map[int(x)]["feature_end"])
        chunk = chunk[(chunk["event_start"] <= chunk["feature_end"]) & (chunk["event_end"] >= chunk["intime"])]
        if chunk.empty:
            continue
        update_stay_level_counter(counter, chunk["stay_id"], chunk["itemid"].astype(int))

    print(f"Input item stay-prevalence threshold: >= {min_count} train stays ({cfg.min_input_item_prevalence:.3%})")
    return select_frequent_keys(counter, min_count, cfg.top_input_items)


def count_top_output_items(cfg: Config, cohort: pd.DataFrame) -> List[int]:
    data_dir = Path(cfg.data_dir)
    train_cohort = cohort[cohort["split"] == "train"]
    stay_map = make_stay_map(train_cohort)
    valid_stays = set(stay_map.keys())
    counter: Counter = Counter()
    min_count = min_count_from_prevalence(len(train_cohort), cfg.min_output_item_prevalence)

    usecols = ["stay_id", "charttime", "itemid", "value"]
    for chunk in pd.read_csv(data_dir / cfg.outputevents_file, usecols=usecols, chunksize=cfg.chunksize_medium, low_memory=False):
        chunk = chunk[chunk["stay_id"].isin(valid_stays)].copy()
        if chunk.empty:
            continue
        chunk["charttime"] = safe_to_datetime(chunk["charttime"])
        chunk["value"] = safe_numeric(chunk["value"])
        chunk = chunk.dropna(subset=["stay_id", "itemid", "charttime", "value"])
        chunk["stay_id"] = chunk["stay_id"].astype(int)
        chunk["itemid"] = chunk["itemid"].astype(int)
        chunk["intime"] = chunk["stay_id"].map(lambda x: stay_map[int(x)]["intime"])
        chunk["feature_end"] = chunk["stay_id"].map(lambda x: stay_map[int(x)]["feature_end"])
        chunk = chunk[(chunk["charttime"] >= chunk["intime"]) & (chunk["charttime"] <= chunk["feature_end"])]
        if chunk.empty:
            continue
        update_stay_level_counter(counter, chunk["stay_id"], chunk["itemid"])

    print(f"Output item stay-prevalence threshold: >= {min_count} train stays ({cfg.min_output_item_prevalence:.3%})")
    return select_frequent_keys(counter, min_count, cfg.top_output_items)


def count_top_proc_items(cfg: Config, cohort: pd.DataFrame) -> List[str]:
    data_dir = Path(cfg.data_dir)
    train_cohort = cohort[cohort["split"] == "train"]
    stay_map = make_stay_map(train_cohort)
    valid_stays = set(stay_map.keys())
    counter: Counter = Counter()
    proc_category_map = load_d_items_category_map(data_dir / cfg.d_items_file) if cfg.use_procedure_categories else {}
    min_count = min_count_from_prevalence(len(train_cohort), cfg.min_proc_item_prevalence)

    usecols = ["stay_id", "itemid", "starttime", "endtime"]
    for chunk in pd.read_csv(data_dir / cfg.procedureevents_file, usecols=usecols, chunksize=cfg.chunksize_medium, low_memory=False):
        chunk = chunk[chunk["stay_id"].isin(valid_stays)].copy()
        if chunk.empty:
            continue
        chunk["starttime"] = safe_to_datetime(chunk["starttime"])
        chunk["endtime"] = safe_to_datetime(chunk["endtime"])
        chunk["stay_id"] = chunk["stay_id"].astype(int)
        chunk["itemid"] = safe_numeric(chunk["itemid"]).astype("Int64")
        chunk = chunk.dropna(subset=["stay_id", "itemid"])
        chunk["event_start"] = chunk["starttime"].fillna(chunk["endtime"])
        chunk["event_end"] = chunk["endtime"].fillna(chunk["starttime"])
        chunk["intime"] = chunk["stay_id"].map(lambda x: stay_map[int(x)]["intime"])
        chunk["feature_end"] = chunk["stay_id"].map(lambda x: stay_map[int(x)]["feature_end"])
        chunk = chunk[(chunk["event_start"] <= chunk["feature_end"]) & (chunk["event_end"] >= chunk["intime"])]
        if chunk.empty:
            continue

        if cfg.use_procedure_categories:
            proc_keys = chunk["itemid"].astype(int).map(lambda x: proc_category_map.get(int(x), "UNKNOWN"))
            proc_keys = proc_keys.fillna("UNKNOWN").astype(str)
            update_stay_level_counter(counter, chunk["stay_id"], proc_keys)
        else:
            update_stay_level_counter(counter, chunk["stay_id"], chunk["itemid"].astype(int))
    print(f"Procedure stay-prevalence threshold: >= {min_count} train stays ({cfg.min_proc_item_prevalence:.3%})")
    return select_frequent_keys(counter, min_count, cfg.top_proc_items)


def count_top_drugs(cfg: Config, cohort: pd.DataFrame) -> List[str]:
    data_dir = Path(cfg.data_dir)
    train_cohort = cohort[cohort["split"] == "train"]
    hadm_map = make_hadm_map(train_cohort)
    valid_hadm = set(hadm_map.keys())
    counter: Counter = Counter()
    min_count = min_count_from_prevalence(len(train_cohort), cfg.min_drug_prevalence)

    usecols = [c for c in ["hadm_id", "drug", "formulary_drug_cd", "starttime", "stoptime"]]
    for chunk in pd.read_csv(data_dir / cfg.prescriptions_file, usecols=usecols, chunksize=cfg.chunksize_medium, low_memory=False):
        chunk = chunk[chunk["hadm_id"].isin(valid_hadm)].copy()
        if chunk.empty:
            continue
        chunk["starttime"] = safe_to_datetime(chunk["starttime"])
        chunk["stoptime"] = safe_to_datetime(chunk["stoptime"])
        chunk["hadm_id"] = chunk["hadm_id"].astype(int)
        chunk["event_start"] = chunk["starttime"].fillna(chunk["stoptime"])
        chunk["event_end"] = chunk["stoptime"].fillna(chunk["starttime"])
        chunk["intime"] = chunk["hadm_id"].map(lambda x: hadm_map[int(x)]["intime"])
        chunk["feature_end"] = chunk["hadm_id"].map(lambda x: hadm_map[int(x)]["feature_end"])
        chunk = chunk[(chunk["event_start"] <= chunk["feature_end"]) & (chunk["event_end"] >= chunk["intime"])]
        if chunk.empty:
            continue

        chunk["stay_id"] = chunk["hadm_id"].map(lambda x: hadm_map[int(x)]["stay_id"])
        drug_key = chunk["drug"].fillna(chunk["formulary_drug_cd"]).astype(str).str.strip().str.upper()
        drug_key = drug_key.replace({"": np.nan, "NAN": np.nan})
        update_stay_level_counter(counter, chunk["stay_id"], drug_key)

    print(f"Drug stay-prevalence threshold: >= {min_count} train stays ({cfg.min_drug_prevalence:.3%})")
    return select_frequent_keys(counter, min_count, cfg.top_drugs)


# ==========================================================
# Pass 2: raw train-only clipping bounds for chart/lab
# ==========================================================


def estimate_chart_clip_bounds(cfg: Config, cohort: pd.DataFrame, chart_itemids: Sequence[int]) -> Dict[int, Tuple[Optional[float], Optional[float]]]:
    data_dir = Path(cfg.data_dir)
    stay_map = make_stay_map(cohort[cohort["split"] == "train"])
    valid_stays = set(stay_map.keys())
    itemids = set(int(x) for x in chart_itemids)
    sampler = ReservoirSampler(cfg.max_quantile_sample_per_item, random_state=cfg.random_state)

    usecols = ["stay_id", "charttime", "itemid", "valuenum"]
    for chunk in pd.read_csv(data_dir / cfg.chartevents_file, usecols=usecols, chunksize=cfg.chunksize_large, low_memory=False):
        chunk = chunk[chunk["stay_id"].isin(valid_stays)].copy()
        if chunk.empty:
            continue
        chunk["charttime"] = safe_to_datetime(chunk["charttime"])
        chunk["valuenum"] = safe_numeric(chunk["valuenum"])
        chunk = chunk.dropna(subset=["stay_id", "itemid", "charttime", "valuenum"])
        chunk["stay_id"] = chunk["stay_id"].astype(int)
        chunk["itemid"] = chunk["itemid"].astype(int)
        chunk = chunk[chunk["itemid"].isin(itemids)]
        if chunk.empty:
            continue
        chunk["intime"] = chunk["stay_id"].map(lambda x: stay_map[int(x)]["intime"])
        chunk["feature_end"] = chunk["stay_id"].map(lambda x: stay_map[int(x)]["feature_end"])
        chunk = chunk[(chunk["charttime"] >= chunk["intime"]) & (chunk["charttime"] <= chunk["feature_end"])]
        if chunk.empty:
            continue

        for itemid, sub in chunk.groupby("itemid"):
            sampler.update_many(int(itemid), sub["valuenum"].values)

    bounds = sampler.quantile_bounds(cfg.clip_lower_q, cfg.clip_upper_q)
    if cfg.use_hard_ranges:
        for itemid, (lo, hi) in HARD_RANGE_BY_ITEMID.items():
            if itemid in bounds:
                qlo, qhi = bounds[itemid]
                lo_final = lo if qlo is None else max(lo, qlo)
                hi_final = hi if qhi is None else min(hi, qhi)
                bounds[itemid] = (lo_final, hi_final)
    return bounds



def estimate_lab_clip_bounds(cfg: Config, cohort: pd.DataFrame, lab_itemids: Sequence[int]) -> Dict[int, Tuple[Optional[float], Optional[float]]]:
    data_dir = Path(cfg.data_dir)
    hadm_map = make_hadm_map(cohort[cohort["split"] == "train"])
    valid_hadm = set(hadm_map.keys())
    itemids = set(int(x) for x in lab_itemids)
    sampler = ReservoirSampler(cfg.max_quantile_sample_per_item, random_state=cfg.random_state)

    usecols = ["hadm_id", "charttime", "itemid", "valuenum"]
    for chunk in pd.read_csv(data_dir / cfg.labevents_file, usecols=usecols, chunksize=cfg.chunksize_medium, low_memory=False):
        chunk = chunk[chunk["hadm_id"].isin(valid_hadm)].copy()
        if chunk.empty:
            continue
        chunk["charttime"] = safe_to_datetime(chunk["charttime"])
        chunk["valuenum"] = safe_numeric(chunk["valuenum"])
        chunk = chunk.dropna(subset=["hadm_id", "itemid", "charttime", "valuenum"])
        chunk["hadm_id"] = chunk["hadm_id"].astype(int)
        chunk["itemid"] = chunk["itemid"].astype(int)
        chunk = chunk[chunk["itemid"].isin(itemids)]
        if chunk.empty:
            continue
        chunk["intime"] = chunk["hadm_id"].map(lambda x: hadm_map[int(x)]["intime"])
        chunk["feature_end"] = chunk["hadm_id"].map(lambda x: hadm_map[int(x)]["feature_end"])
        chunk = chunk[(chunk["charttime"] >= chunk["intime"]) & (chunk["charttime"] <= chunk["feature_end"])]
        if chunk.empty:
            continue

        for itemid, sub in chunk.groupby("itemid"):
            sampler.update_many(int(itemid), sub["valuenum"].values)

    bounds = sampler.quantile_bounds(cfg.clip_lower_q, cfg.clip_upper_q)
    if cfg.use_hard_ranges:
        for itemid, (lo, hi) in HARD_RANGE_BY_ITEMID.items():
            if itemid in bounds:
                qlo, qhi = bounds[itemid]
                lo_final = lo if qlo is None else max(lo, qlo)
                hi_final = hi if qhi is None else min(hi, qhi)
                bounds[itemid] = (lo_final, hi_final)
    return bounds


# ==========================================================
# Pass 3: feature extraction with raw clipping before aggregation
# ==========================================================


def extract_chart_features(cfg: Config, cohort: pd.DataFrame, chart_itemids: Sequence[int], clip_bounds: Dict[int, Tuple[Optional[float], Optional[float]]]) -> pd.DataFrame:
    data_dir = Path(cfg.data_dir)
    stay_map = make_stay_map(cohort)
    valid_stays = set(stay_map.keys())
    itemids = set(int(x) for x in chart_itemids)
    agg = NumericSummaryAgg()

    usecols = ["stay_id", "charttime", "itemid", "valuenum"]
    for chunk in pd.read_csv(data_dir / cfg.chartevents_file, usecols=usecols, chunksize=cfg.chunksize_large, low_memory=False):
        chunk = chunk[chunk["stay_id"].isin(valid_stays)].copy()
        if chunk.empty:
            continue
        chunk["charttime"] = safe_to_datetime(chunk["charttime"])
        chunk["valuenum"] = safe_numeric(chunk["valuenum"])
        chunk = chunk.dropna(subset=["stay_id", "itemid", "charttime", "valuenum"])
        chunk["stay_id"] = chunk["stay_id"].astype(int)
        chunk["itemid"] = chunk["itemid"].astype(int)
        chunk = chunk[chunk["itemid"].isin(itemids)]
        if chunk.empty:
            continue

        chunk["intime"] = chunk["stay_id"].map(lambda x: stay_map[int(x)]["intime"])
        chunk["feature_end"] = chunk["stay_id"].map(lambda x: stay_map[int(x)]["feature_end"])
        chunk = chunk[(chunk["charttime"] >= chunk["intime"]) & (chunk["charttime"] <= chunk["feature_end"])]
        if chunk.empty:
            continue

        for row in chunk[["stay_id", "itemid", "charttime", "valuenum"]].itertuples(index=False):
            itemid = int(row.itemid)
            val = float(row.valuenum)
            lo, hi = clip_bounds.get(itemid, (None, None))
            if lo is not None:
                val = max(val, lo)
            if hi is not None:
                val = min(val, hi)
            agg.update(int(row.stay_id), itemid, row.charttime, val)

    return agg.to_wide(prefix="chart")



def extract_lab_features(cfg: Config, cohort: pd.DataFrame, lab_itemids: Sequence[int], clip_bounds: Dict[int, Tuple[Optional[float], Optional[float]]]) -> pd.DataFrame:
    data_dir = Path(cfg.data_dir)
    hadm_map = make_hadm_map(cohort)
    itemids = set(int(x) for x in lab_itemids)
    valid_hadm = set(hadm_map.keys())
    agg = NumericSummaryAgg()

    usecols = ["hadm_id", "charttime", "itemid", "valuenum"]
    for chunk in pd.read_csv(data_dir / cfg.labevents_file, usecols=usecols, chunksize=cfg.chunksize_medium, low_memory=False):
        chunk = chunk[chunk["hadm_id"].isin(valid_hadm)].copy()
        if chunk.empty:
            continue
        chunk["charttime"] = safe_to_datetime(chunk["charttime"])
        chunk["valuenum"] = safe_numeric(chunk["valuenum"])
        chunk = chunk.dropna(subset=["hadm_id", "itemid", "charttime", "valuenum"])
        chunk["hadm_id"] = chunk["hadm_id"].astype(int)
        chunk["itemid"] = chunk["itemid"].astype(int)
        chunk = chunk[chunk["itemid"].isin(itemids)]
        if chunk.empty:
            continue

        chunk["stay_id"] = chunk["hadm_id"].map(lambda x: hadm_map[int(x)]["stay_id"])
        chunk["intime"] = chunk["hadm_id"].map(lambda x: hadm_map[int(x)]["intime"])
        chunk["feature_end"] = chunk["hadm_id"].map(lambda x: hadm_map[int(x)]["feature_end"])
        chunk = chunk[(chunk["charttime"] >= chunk["intime"]) & (chunk["charttime"] <= chunk["feature_end"])]
        if chunk.empty:
            continue

        for row in chunk[["stay_id", "itemid", "charttime", "valuenum"]].itertuples(index=False):
            itemid = int(row.itemid)
            val = float(row.valuenum)
            lo, hi = clip_bounds.get(itemid, (None, None))
            if lo is not None:
                val = max(val, lo)
            if hi is not None:
                val = min(val, hi)
            agg.update(int(row.stay_id), itemid, row.charttime, val)

    return agg.to_wide(prefix="lab")



def extract_input_features(cfg: Config, cohort: pd.DataFrame, input_itemids: Sequence[int]) -> pd.DataFrame:
    if not cfg.include_inputs or len(input_itemids) == 0:
        return pd.DataFrame(columns=["stay_id"])

    data_dir = Path(cfg.data_dir)
    stay_map = make_stay_map(cohort)
    valid_stays = set(stay_map.keys())
    itemids = set(int(x) for x in input_itemids)
    agg = CountSumAgg()

    usecols = [c for c in ["stay_id", "itemid", "starttime", "endtime", "amount", "rate"]]
    for chunk in pd.read_csv(data_dir / cfg.inputevents_file, usecols=usecols, chunksize=cfg.chunksize_medium, low_memory=False):
        chunk = chunk[chunk["stay_id"].isin(valid_stays)].copy()
        if chunk.empty:
            continue
        chunk["stay_id"] = chunk["stay_id"].astype(int)
        chunk["itemid"] = safe_numeric(chunk["itemid"]).astype("Int64")
        chunk["starttime"] = safe_to_datetime(chunk["starttime"])
        chunk["endtime"] = safe_to_datetime(chunk["endtime"])
        chunk["amount"] = safe_numeric(chunk["amount"])
        chunk["rate"] = safe_numeric(chunk["rate"])
        chunk = chunk.dropna(subset=["stay_id", "itemid"])
        chunk["itemid"] = chunk["itemid"].astype(int)
        chunk = chunk[chunk["itemid"].isin(itemids)]
        if chunk.empty:
            continue
        chunk["event_start"] = chunk["starttime"].fillna(chunk["endtime"])
        chunk["event_end"] = chunk["endtime"].fillna(chunk["starttime"])
        chunk["intime"] = chunk["stay_id"].map(lambda x: stay_map[int(x)]["intime"])
        chunk["feature_end"] = chunk["stay_id"].map(lambda x: stay_map[int(x)]["feature_end"])
        chunk = chunk[(chunk["event_start"] <= chunk["feature_end"]) & (chunk["event_end"] >= chunk["intime"])]
        if chunk.empty:
            continue
        for row in chunk[["stay_id", "itemid", "amount", "rate"]].itertuples(index=False):
            agg.update(int(row.stay_id), int(row.itemid), value=(None if pd.isna(row.amount) else float(row.amount)), value2=(None if pd.isna(row.rate) else float(row.rate)))

    return agg.to_wide(prefix="input", include_sum=True, include_mean=True, include_max=False, second_sum_name="sum_rate")



def extract_output_features(cfg: Config, cohort: pd.DataFrame, output_itemids: Sequence[int]) -> pd.DataFrame:
    if not cfg.include_outputs or len(output_itemids) == 0:
        return pd.DataFrame(columns=["stay_id"])

    data_dir = Path(cfg.data_dir)
    stay_map = make_stay_map(cohort)
    valid_stays = set(stay_map.keys())
    itemids = set(int(x) for x in output_itemids)
    agg = CountSumAgg()

    usecols = ["stay_id", "charttime", "itemid", "value"]
    for chunk in pd.read_csv(data_dir / cfg.outputevents_file, usecols=usecols, chunksize=cfg.chunksize_medium, low_memory=False):
        chunk = chunk[chunk["stay_id"].isin(valid_stays)].copy()
        if chunk.empty:
            continue
        chunk["stay_id"] = chunk["stay_id"].astype(int)
        chunk["itemid"] = safe_numeric(chunk["itemid"]).astype("Int64")
        chunk["charttime"] = safe_to_datetime(chunk["charttime"])
        chunk["value"] = safe_numeric(chunk["value"])
        chunk = chunk.dropna(subset=["stay_id", "itemid", "charttime"])
        chunk["itemid"] = chunk["itemid"].astype(int)
        chunk = chunk[chunk["itemid"].isin(itemids)]
        if chunk.empty:
            continue
        chunk["intime"] = chunk["stay_id"].map(lambda x: stay_map[int(x)]["intime"])
        chunk["feature_end"] = chunk["stay_id"].map(lambda x: stay_map[int(x)]["feature_end"])
        chunk = chunk[(chunk["charttime"] >= chunk["intime"]) & (chunk["charttime"] <= chunk["feature_end"])]
        if chunk.empty:
            continue
        for row in chunk[["stay_id", "itemid", "value"]].itertuples(index=False):
            agg.update(int(row.stay_id), int(row.itemid), value=(None if pd.isna(row.value) else float(row.value)))

    return agg.to_wide(prefix="output", include_sum=True, include_mean=True, include_max=True)



def extract_procedure_features(cfg: Config, cohort: pd.DataFrame, proc_keys: Sequence[str]) -> pd.DataFrame:
    if not cfg.include_procedures or len(proc_keys) == 0:
        return pd.DataFrame(columns=["stay_id"])

    data_dir = Path(cfg.data_dir)
    stay_map = make_stay_map(cohort)
    valid_stays = set(stay_map.keys())
    selected_proc_keys = set(str(x) for x in proc_keys)
    proc_category_map = load_d_items_category_map(data_dir / cfg.d_items_file) if cfg.use_procedure_categories else {}
    agg = CountSumAgg()

    usecols = ["stay_id", "itemid", "starttime", "endtime"]
    for chunk in pd.read_csv(data_dir / cfg.procedureevents_file, usecols=usecols, chunksize=cfg.chunksize_medium, low_memory=False):
        chunk = chunk[chunk["stay_id"].isin(valid_stays)].copy()
        if chunk.empty:
            continue
        chunk["stay_id"] = chunk["stay_id"].astype(int)
        chunk["itemid"] = safe_numeric(chunk["itemid"]).astype("Int64")
        chunk["starttime"] = safe_to_datetime(chunk["starttime"])
        chunk["endtime"] = safe_to_datetime(chunk["endtime"])
        chunk = chunk.dropna(subset=["stay_id", "itemid"])
        chunk["itemid"] = chunk["itemid"].astype(int)
        chunk["event_start"] = chunk["starttime"].fillna(chunk["endtime"])
        chunk["event_end"] = chunk["endtime"].fillna(chunk["starttime"])
        chunk["intime"] = chunk["stay_id"].map(lambda x: stay_map[int(x)]["intime"])
        chunk["feature_end"] = chunk["stay_id"].map(lambda x: stay_map[int(x)]["feature_end"])
        chunk = chunk[(chunk["event_start"] <= chunk["feature_end"]) & (chunk["event_end"] >= chunk["intime"])]
        if chunk.empty:
            continue

        if cfg.use_procedure_categories:
            chunk["proc_key"] = chunk["itemid"].map(lambda x: proc_category_map.get(int(x), "UNKNOWN"))
        else:
            chunk["proc_key"] = chunk["itemid"].astype(str)
        chunk["proc_key"] = chunk["proc_key"].fillna("UNKNOWN").astype(str)
        chunk = chunk[chunk["proc_key"].isin(selected_proc_keys)]
        if chunk.empty:
            continue

        duration_min = (chunk["event_end"] - chunk["event_start"]).dt.total_seconds() / 60.0
        duration_min = duration_min.clip(lower=0)
        chunk = chunk.assign(duration_min=duration_min)
        proc_key_to_idx = {k: i for i, k in enumerate(sorted(selected_proc_keys))}
        for row in chunk[["stay_id", "proc_key", "duration_min"]].itertuples(index=False):
            agg.update(
                int(row.stay_id),
                proc_key_to_idx[str(row.proc_key)],
                value=(None if pd.isna(row.duration_min) else float(row.duration_min)),
            )

    out = agg.to_wide(prefix="proc", include_sum=True, include_mean=True, include_max=True)
    rename_map = {}
    proc_key_to_idx = {k: i for i, k in enumerate(sorted(selected_proc_keys))}
    idx_to_proc_key = {v: k for k, v in proc_key_to_idx.items()}
    for c in out.columns:
        if c.startswith("proc_") and "_item_" in c:
            stat, idx = c.rsplit("_item_", 1)
            rename_map[c] = f"{stat}_{idx_to_proc_key[int(idx)].replace(' ', '_').replace('/', '_')}"
    return out.rename(columns=rename_map)



def extract_prescription_features(cfg: Config, cohort: pd.DataFrame, top_drugs: Sequence[str]) -> pd.DataFrame:
    if not cfg.include_prescriptions or len(top_drugs) == 0:
        return pd.DataFrame(columns=["stay_id"])

    data_dir = Path(cfg.data_dir)
    hadm_map = make_hadm_map(cohort)
    valid_hadm = set(hadm_map.keys())
    drug_set = set(top_drugs)
    counts: Dict[Tuple[int, str], int] = defaultdict(int)

    usecols = [c for c in ["hadm_id", "drug", "formulary_drug_cd", "starttime", "stoptime"]]
    for chunk in pd.read_csv(data_dir / cfg.prescriptions_file, usecols=usecols, chunksize=cfg.chunksize_medium, low_memory=False):
        chunk = chunk[chunk["hadm_id"].isin(valid_hadm)].copy()
        if chunk.empty:
            continue
        chunk["hadm_id"] = chunk["hadm_id"].astype(int)
        chunk["starttime"] = safe_to_datetime(chunk["starttime"])
        chunk["stoptime"] = safe_to_datetime(chunk["stoptime"])
        chunk["event_start"] = chunk["starttime"].fillna(chunk["stoptime"])
        chunk["event_end"] = chunk["stoptime"].fillna(chunk["starttime"])
        chunk["intime"] = chunk["hadm_id"].map(lambda x: hadm_map[int(x)]["intime"])
        chunk["feature_end"] = chunk["hadm_id"].map(lambda x: hadm_map[int(x)]["feature_end"])
        chunk = chunk[(chunk["event_start"] <= chunk["feature_end"]) & (chunk["event_end"] >= chunk["intime"])]
        if chunk.empty:
            continue
        chunk["stay_id"] = chunk["hadm_id"].map(lambda x: hadm_map[int(x)]["stay_id"])
        chunk["drug_key"] = chunk["drug"].fillna(chunk["formulary_drug_cd"]).astype(str).str.strip().str.upper()
        chunk["drug_key"] = chunk["drug_key"].replace({"": np.nan, "NAN": np.nan})
        chunk = chunk.dropna(subset=["drug_key"])
        if chunk.empty:
            continue
        if cfg.add_prescription_other_group:
            chunk["drug_group"] = chunk["drug_key"].where(chunk["drug_key"].isin(drug_set), "OTHER_RARE_DRUG")
        else:
            chunk = chunk[chunk["drug_key"].isin(drug_set)]
            if chunk.empty:
                continue
            chunk["drug_group"] = chunk["drug_key"]
        for row in chunk[["stay_id", "drug_group"]].itertuples(index=False):
            counts[(int(row.stay_id), str(row.drug_group))] += 1

    if not counts:
        return pd.DataFrame(columns=["stay_id"])

    rows = [{"stay_id": k[0], "drug_key": k[1], "cnt": v} for k, v in counts.items()]
    long_df = pd.DataFrame(rows)
    wide = long_df.pivot(index="stay_id", columns="drug_key", values="cnt")
    wide.columns = [f"drug_cnt_{str(c).replace(' ', '_').replace('/', '_')}" for c in wide.columns]
    return wide.reset_index()


# ==========================================================
# Static + preprocessing
# ==========================================================


def build_static_features(cohort: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "stay_id",
        "subject_id",
        "hadm_id",
        "split",
        "intime",
        "feature_end",
        "age",
        "gender",
        "race",
        "language",
        "marital_status",
        "insurance",
        "admission_type",
        "admission_location",
        "first_careunit",
        "pre_icu_careunit",
        "n_transfers_before_icu",
        "hospital_mortality",
        "icu_mortality",
    ]
    cols = [c for c in cols if c in cohort.columns]
    df = cohort[cols].copy()

    cat_cols = [
        c for c in [
            "gender", "race", "language", "marital_status", "insurance",
            "admission_type", "admission_location",
            "first_careunit", "pre_icu_careunit"
        ] if c in df.columns
    ]

    for c in cat_cols:
        df[c] = df[c].fillna("UNKNOWN").astype(str)

    return df



def preprocess_and_save(cfg: Config, cohort: pd.DataFrame, features: pd.DataFrame, specs: Dict[str, object]) -> None:
    out_dir = Path(cfg.out_dir)
    ensure_dir(out_dir)

    target_col = cfg.target

    non_feature_cols = {
        "stay_id", "subject_id", "hadm_id", "split", "intime", "feature_end",
        "hospital_mortality", "icu_mortality", "observed_hours",
        "discharge_location", "last_careunit"
    }
    feature_cols = [c for c in features.columns if c not in non_feature_cols]

    X = features[["stay_id", "subject_id", "hadm_id", "split"] + feature_cols].copy()
    y = features[["stay_id", target_col]].rename(columns={target_col: "label"}).copy()
    meta_cols = [c for c in ["stay_id", "subject_id", "hadm_id", "split", "intime", "feature_end", "hospital_mortality", "icu_mortality"] if c in features.columns]
    meta = features[meta_cols].copy()

    train_mask = X["split"] == "train"
    categorical_encoding: Dict[str, Dict[str, object]] = {}
    cat_cols = [c for c in X.select_dtypes(include=["object", "category"]).columns.tolist() if c != "split"]
    min_category_count = min_count_from_prevalence(len(cohort.loc[cohort["split"] == "train"]), cfg.categorical_min_prevalence)
    compact_cat_cols = set(cat_cols)
    for c in cat_cols:
        encoded, encoder_spec = fit_frequency_encoder(
            values=X[c],
            train_mask=train_mask,
            min_count=min_category_count,
            max_levels=cfg.categorical_max_levels,
        )
        X[c] = encoded
        categorical_encoding[c] = encoder_spec

    numeric_cols = [c for c in X.select_dtypes(include=[np.number, "bool"]).columns.tolist() if c not in ["stay_id", "subject_id", "hadm_id"]]

    # Semantic groups
    chart_lab_cols = [c for c in numeric_cols if c.startswith("chart_") or c.startswith("lab_")]
    exposure_cols = [
        c for c in numeric_cols if (
            c.startswith("input_") or c.startswith("output_") or c.startswith("proc_") or c.startswith("drug_cnt_")
        )
    ]
    zero_fill_cols = exposure_cols + [c for c in ["n_transfers_before_icu"] if c in numeric_cols]
    static_numeric_cols = [c for c in numeric_cols if c not in chart_lab_cols and c not in exposure_cols]

    # Keep chart/lab NaN for XGBoost native missing handling.
    for c in chart_lab_cols + static_numeric_cols:
        X[c] = X[c].replace([np.inf, -np.inf], np.nan)

    # Exposure-style features: absent event => zero
    for c in zero_fill_cols:
        X[c] = X[c].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Drop dead columns using train only.
    train_X = X.loc[train_mask].copy()
    drop_constant = train_constant_columns(
        train_X,
        exclude=["stay_id", "subject_id", "hadm_id", "split"],
    )

    # Prune sparse exposure features using train only.
    train_X_after_constant = train_X.drop(columns=drop_constant, errors="ignore")
    exposure_cols_after_constant = [c for c in exposure_cols if c not in drop_constant]
    drop_sparse_exposure = sparse_exposure_columns(
        train_X_after_constant,
        exposure_cols_after_constant,
        min_prevalence=cfg.exposure_min_prevalence,
    )

    drop_cols = sorted(set(drop_constant) | set(drop_sparse_exposure))
    X = X.drop(columns=drop_cols, errors="ignore")

    model_feature_cols = [c for c in X.columns if c not in ["stay_id", "subject_id", "hadm_id", "split"]]
    cf_rows = []
    for c in model_feature_cols:
        if c.startswith("chart_") or c.startswith("lab_"):
            group = "clinical_measurement"
            mutability = "mutable_with_clinical_constraints"
        elif c.startswith("output_"):
            group = "physiology_response"
            mutability = "non_actionable_or_constrained"
        elif c.startswith("input_") or c.startswith("proc_") or c.startswith("drug_cnt_"):
            group = "treatment_response"
            mutability = "non_actionable_by_default"
        elif c == "age" or c in compact_cat_cols:
            group = "baseline_static"
            mutability = "immutable"
        elif c == "n_transfers_before_icu":
            group = "pre_icu_context"
            mutability = "immutable"
        else:
            group = "other"
            mutability = "review_before_counterfactuals"

        train_values = X.loc[train_mask, c]
        cf_rows.append(
            {
                "feature": c,
                "group": group,
                "mutability": mutability,
                "train_min": float(train_values.min()) if pd.api.types.is_numeric_dtype(train_values) else np.nan,
                "train_max": float(train_values.max()) if pd.api.types.is_numeric_dtype(train_values) else np.nan,
                "train_median": float(train_values.median()) if pd.api.types.is_numeric_dtype(train_values) else np.nan,
                "train_missing_rate": float(train_values.isna().mean()),
            }
        )

    cf_metadata = pd.DataFrame(cf_rows)

    saved_tables = {}

    for split in ["train", "val", "test"]:
        mask = X["split"] == split
        X_split = X.loc[mask].drop(columns=["split"]).reset_index(drop=True)
        y_split = y.loc[mask].reset_index(drop=True)
        meta_split = meta.loc[meta["stay_id"].isin(X_split["stay_id"])].reset_index(drop=True)

        saved_tables[f"X_{split}"] = write_table(X_split, out_dir / f"X_{split}")
        saved_tables[f"y_{split}"] = write_table(y_split, out_dir / f"y_{split}")
        saved_tables[f"meta_{split}"] = write_table(meta_split, out_dir / f"meta_{split}")

    saved_tables["cohort_with_splits"] = write_table(cohort, out_dir / "cohort_with_splits")


    preprocessing_spec = {
        "target": target_col,
        "chart_lab_cols": chart_lab_cols,
        "exposure_cols": exposure_cols,
        "zero_fill_cols": zero_fill_cols,
        "static_numeric_cols": static_numeric_cols,
        "add_missing_indicators": cfg.add_missing_indicators,
        "categorical_encoding": categorical_encoding,
        "drop_constant_columns": drop_constant,
        "drop_sparse_exposure_columns": drop_sparse_exposure,
        "n_constant_dropped": len(drop_constant),
        "n_sparse_exposure_dropped": len(drop_sparse_exposure),
        "exposure_min_prevalence": cfg.exposure_min_prevalence,
        "clip_lower_q": cfg.clip_lower_q,
        "clip_upper_q": cfg.clip_upper_q,
        "max_quantile_sample_per_item": cfg.max_quantile_sample_per_item,
        "saved_tables": saved_tables,
    }

    with open(out_dir / "feature_spec.json", "w") as f:
        json.dump(specs, f, indent=2)
    with open(out_dir / "preprocessing_spec.json", "w") as f:
        json.dump(preprocessing_spec, f, indent=2)
    with open(out_dir / "config_used.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)
    cf_metadata.to_csv(out_dir / "counterfactual_feature_metadata.csv", index=False)
    print(f"Dropped {len(drop_constant)} constant columns")
    print(f"Dropped {len(drop_sparse_exposure)} sparse exposure columns")


# ==========================================================
# Main assembly
# ==========================================================


def build_features(cfg: Config, cohort: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, object]]:
    static_df = build_static_features(cohort)

    print("Selecting frequent chart items on train split...")
    chart_itemids = count_top_chart_items(cfg, cohort)
    print(f"Selected {len(chart_itemids)} chart itemids")

    print("Selecting frequent lab items on train split...")
    lab_itemids = count_top_lab_items(cfg, cohort)
    print(f"Selected {len(lab_itemids)} lab itemids")

    input_itemids = []
    output_itemids = []
    proc_keys = []
    top_drugs = []

    if cfg.include_inputs:
        print("Selecting frequent input itemids on train split...")
        input_itemids = count_top_input_items(cfg, cohort)
        print(f"Selected {len(input_itemids)} input itemids")

    if cfg.include_outputs:
        print("Selecting frequent output itemids on train split...")
        output_itemids = count_top_output_items(cfg, cohort)
        print(f"Selected {len(output_itemids)} output itemids")

    if cfg.include_procedures:
        selector_label = "procedure categories" if cfg.use_procedure_categories else "procedure itemids"
        print(f"Selecting frequent {selector_label} on train split...")
        proc_keys = count_top_proc_items(cfg, cohort)
        print(f"Selected {len(proc_keys)} {selector_label}")

    if cfg.include_prescriptions:
        print("Selecting frequent drugs on train split...")
        top_drugs = count_top_drugs(cfg, cohort)
        print(f"Selected {len(top_drugs)} drug keys")

    print("Estimating train-only raw clipping bounds for chart values...")
    chart_clip_bounds = estimate_chart_clip_bounds(cfg, cohort, chart_itemids)
    print("Estimating train-only raw clipping bounds for lab values...")
    lab_clip_bounds = estimate_lab_clip_bounds(cfg, cohort, lab_itemids)

    print("Extracting chart features...")
    chart_df = extract_chart_features(cfg, cohort, chart_itemids, chart_clip_bounds)
    print("Extracting lab features...")
    lab_df = extract_lab_features(cfg, cohort, lab_itemids, lab_clip_bounds)

    feature_tables = [chart_df, lab_df]

    if cfg.include_inputs:
        print("Extracting input features...")
        feature_tables.append(extract_input_features(cfg, cohort, input_itemids))
    if cfg.include_outputs:
        print("Extracting output features...")
        feature_tables.append(extract_output_features(cfg, cohort, output_itemids))
    if cfg.include_procedures:
        print("Extracting procedure features...")
        feature_tables.append(extract_procedure_features(cfg, cohort, proc_keys))
    if cfg.include_prescriptions:
        print("Extracting prescription features...")
        feature_tables.append(extract_prescription_features(cfg, cohort, top_drugs))

    features = static_df.copy()
    for extra in feature_tables:
        features = features.merge(extra, on="stay_id", how="left")

    data_dir = Path(cfg.data_dir)
    d_items_meta = load_item_metadata(
        data_dir / cfg.d_items_file,
        list(chart_itemids) + list(input_itemids) + list(output_itemids),
    )
    d_labitems_meta = load_item_metadata(data_dir / cfg.d_labitems_file, lab_itemids)

    specs = {
        "chart_itemids": chart_itemids,
        "lab_itemids": lab_itemids,
        "input_itemids": input_itemids,
        "output_itemids": output_itemids,
        "procedure_keys": proc_keys,
        "drug_keys": top_drugs,
        "chart_item_metadata": {str(k): d_items_meta.get(str(k), {}) for k in chart_itemids},
        "lab_item_metadata": {str(k): d_labitems_meta.get(str(k), {}) for k in lab_itemids},
        "input_item_metadata": {str(k): d_items_meta.get(str(k), {}) for k in input_itemids},
        "output_item_metadata": {str(k): d_items_meta.get(str(k), {}) for k in output_itemids},
        "procedure_grouping": ("category_from_d_items" if cfg.use_procedure_categories else "raw_itemid"),
        "chart_clip_bounds": {str(k): [v[0], v[1]] for k, v in chart_clip_bounds.items()},
        "lab_clip_bounds": {str(k): [v[0], v[1]] for k, v in lab_clip_bounds.items()},
    }
    return features, specs



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the 48h-window ICU mortality feature tables from raw MIMIC-IV CSVs.")
    parser.add_argument("--data-dir", default=None, help="Folder containing the raw MIMIC-IV hosp/ and icu/ CSVs.")
    parser.add_argument("--out-dir", default=None, help="Folder to write X/y/meta tables and specs to.")
    parser.add_argument("--hours-window", type=int, default=None, help="Feature/prediction landmark, in hours after ICU admission.")
    parser.add_argument("--test-size", type=float, default=None)
    parser.add_argument("--val-size", type=float, default=None)
    parser.add_argument("--random-state", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    cfg = Config()
    args = parse_args()
    if args.data_dir is not None:
        cfg.data_dir = args.data_dir
    if args.out_dir is not None:
        cfg.out_dir = args.out_dir
    if args.hours_window is not None:
        cfg.hours_window = args.hours_window
    if args.test_size is not None:
        cfg.test_size = args.test_size
    if args.val_size is not None:
        cfg.val_size = args.val_size
    if args.random_state is not None:
        cfg.random_state = args.random_state

    ensure_dir(Path(cfg.out_dir))

    print("Building ICU cohort...")
    cohort = build_cohort(cfg)
    cohort = subject_level_split(
        cohort=cohort,
        target_col=cfg.target,
        test_size=cfg.test_size,
        val_size=cfg.val_size,
        random_state=cfg.random_state,
    )

    print("Cohort size:", len(cohort))
    print(cohort["split"].value_counts())
    print("Target prevalence:")
    print(cohort[cfg.target].value_counts(normalize=True))

    features, specs = build_features(cfg, cohort)
    preprocess_and_save(cfg, cohort, features, specs)

    print("Done.")
    print(f"Saved files to: {cfg.out_dir}")


if __name__ == "__main__":
    main()
