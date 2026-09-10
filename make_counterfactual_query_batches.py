from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create random row_id batches for counterfactual scripts that use "
            "--query-ids-path / query_ids_path."
        )
    )
    parser.add_argument(
        "--predictions",
        required=True,
        help=(
            "Path to predictions_<split> written by train_xgboost.py, e.g. "
            "predictions_test.parquet or predictions_test.csv. train_xgboost.py "
            "writes .parquet when pyarrow is installed (the default with "
            "requirements.txt) and only falls back to .csv otherwise -- either "
            "extension (or the bare path without one) is accepted here."
        ),
    )
    parser.add_argument(
        "--labels",
        help=(
            "Optional y_<split> file (.parquet or .csv, as written by "
            "preprocessing.py) with a label column. Use this when the "
            "predictions file does not already contain label or y_true."
        ),
    )
    parser.add_argument(
        "--metrics",
        help=(
            "Optional metrics.json. Used to read threshold_selection.best_threshold "
            "when pred_label is missing and --threshold is not given."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        help="Prediction threshold to derive pred_label from pred_proba if pred_label is missing.",
    )
    parser.add_argument(
        "--source-label",
        type=int,
        choices=[0, 1],
        default=1,
        help="Predicted class to sample. Default: 1.",
    )
    parser.add_argument(
        "--include-all-predictions",
        action="store_true",
        help="Do not filter by predicted class before sampling.",
    )
    parser.add_argument(
        "--true-label-one-only",
        action="store_true",
        help="Only sample rows whose true label is 1.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        required=True,
        help="Number of row_ids per batch.",
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=1,
        help="How many random batch CSVs to create. Default: 1.",
    )
    parser.add_argument(
        "--allow-overlap",
        action="store_true",
        help="Allow the same row_id to appear in more than one batch.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed. Default: 42.",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Folder where query-id CSVs and manifest will be written.",
    )
    parser.add_argument(
        "--prefix",
        default="query_ids",
        help="Output file prefix. Default: query_ids.",
    )
    return parser.parse_args()


def load_threshold(metrics_path: str | None, explicit_threshold: float | None) -> float:
    if explicit_threshold is not None:
        return float(explicit_threshold)
    if metrics_path:
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        return float(metrics.get("threshold_selection", {}).get("best_threshold", 0.5))
    return 0.5


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_predictions_rows(path: str | Path) -> list[dict]:
    """Load a predictions_<split> table written by train_xgboost.py's
    write_table(), which saves .parquet when pyarrow is installed and only
    falls back to .csv when it is not. Accepts the .parquet path, the .csv
    path, or the bare path with no suffix, and picks whichever file
    actually exists on disk (preferring an exact match to what was passed).
    """
    path = Path(path)
    if path.suffix == ".parquet":
        candidates = [path, path.with_suffix(".csv")]
    elif path.suffix == ".csv":
        candidates = [path, path.with_suffix(".parquet")]
    else:
        candidates = [path.with_suffix(".parquet"), path.with_suffix(".csv")]

    for candidate in candidates:
        if candidate.exists():
            if candidate.suffix == ".parquet":
                return pd.read_parquet(candidate).to_dict("records")
            return read_csv_rows(candidate)

    looked_at = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(f"Could not find a predictions file. Looked for: {looked_at}")


def find_row_id_column(fieldnames: list[str]) -> str | None:
    for column in ["row_id", "row_index", "index"]:
        if column in fieldnames:
            return column
    return None


def attach_true_label(rows: list[dict[str, str]], labels_path: str | None) -> None:
    if not rows:
        return
    if "y_true" in rows[0]:
        for row in rows:
            row["true_label"] = str(int(float(row["y_true"])))
        return
    if "label" in rows[0]:
        for row in rows:
            row["true_label"] = str(int(float(row["label"])))
        return
    if not labels_path:
        return

    labels = load_predictions_rows(labels_path)
    if labels and "label" not in labels[0]:
        raise ValueError(f"{labels_path} must contain a 'label' column.")
    if len(labels) != len(rows):
        raise ValueError(
            f"Labels length ({len(labels)}) does not match predictions length ({len(rows)})."
        )
    for row, label_row in zip(rows, labels):
        row["true_label"] = str(int(float(label_row["label"])))


def ensure_prediction_label(rows: list[dict[str, str]], threshold: float) -> None:
    if not rows:
        return
    if "pred_label" in rows[0]:
        for row in rows:
            row["sample_pred_label"] = str(int(float(row["pred_label"])))
        return

    proba_col = None
    for column in ["pred_proba", "pred_probability", "probability", "proba"]:
        if column in rows[0]:
            proba_col = column
            break
    if proba_col is None:
        raise ValueError(
            "Predictions must contain pred_label, or a probability column such as pred_proba."
        )

    for row in rows:
        row["sample_pred_label"] = str(int(float(row[proba_col]) >= threshold))


def build_pool(args: argparse.Namespace) -> list[dict[str, str]]:
    threshold = load_threshold(args.metrics, args.threshold)
    rows = load_predictions_rows(args.predictions)
    if not rows:
        raise ValueError("Predictions file is empty.")

    fieldnames = list(rows[0].keys())
    row_id_column = find_row_id_column(fieldnames)
    if row_id_column is None:
        row_id_column = "row_id"
        for idx, row in enumerate(rows):
            row[row_id_column] = str(idx)

    attach_true_label(rows, args.labels)
    has_true_label = bool(rows and "true_label" in rows[0])
    ensure_prediction_label(rows, threshold)
    for row in rows:
        row["row_id_for_query"] = str(int(float(row[row_id_column])))

    if not args.include_all_predictions:
        rows = [row for row in rows if int(row["sample_pred_label"]) == int(args.source_label)]
    if args.true_label_one_only:
        if not has_true_label:
            raise ValueError(
                "Cannot use --true-label-one-only because no label/y_true column was found. "
                "Pass --labels y_<split>.csv or use predictions with label/y_true."
            )
        rows = [row for row in rows if int(row["true_label"]) == 1]

    return rows


def value_counts(rows: list[dict[str, str]], column: str) -> dict[int, int]:
    counts = Counter(int(float(row[column])) for row in rows if column in row and row[column] != "")
    return dict(sorted(counts.items()))


def write_row_ids_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["row_id"])
        writer.writeheader()
        for row in rows:
            writer.writerow({"row_id": int(float(row["row_id_for_query"]))})


def write_batches(pool: list[dict[str, str]], args: argparse.Namespace) -> None:
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive.")
    if args.num_batches <= 0:
        raise ValueError("--num-batches must be positive.")

    required = args.num_samples if args.allow_overlap else args.num_samples * args.num_batches
    if len(pool) < required:
        raise ValueError(
            f"Pool has {len(pool)} eligible rows, but {required} are needed. "
            "Use fewer samples/batches or pass --allow-overlap."
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.random_state)
    manifest_rows = []

    if args.allow_overlap:
        batch_indices = [
            rng.sample(range(len(pool)), k=args.num_samples)
            for _ in range(args.num_batches)
        ]
    else:
        shuffled = list(range(len(pool)))
        rng.shuffle(shuffled)
        batch_indices = [
            shuffled[i * args.num_samples : (i + 1) * args.num_samples]
            for i in range(args.num_batches)
        ]

    for batch_number, indices in enumerate(batch_indices, start=1):
        batch = [pool[idx] for idx in indices]
        random.Random(args.random_state + batch_number).shuffle(batch)
        output_path = out_dir / f"{args.prefix}_batch{batch_number:03d}.csv"
        write_row_ids_csv(output_path, batch)
        manifest_rows.append(
            {
                "batch": batch_number,
                "path": str(output_path),
                "num_samples": len(batch),
                "pred_label_counts": value_counts(batch, "sample_pred_label"),
                "true_label_counts": (
                    value_counts(batch, "true_label")
                    if batch and "true_label" in batch[0]
                    else {}
                ),
            }
        )

    manifest = {
        "predictions": str(Path(args.predictions)),
        "labels": str(Path(args.labels)) if args.labels else None,
        "source_label": None if args.include_all_predictions else int(args.source_label),
        "true_label_one_only": bool(args.true_label_one_only),
        "num_eligible_rows": int(len(pool)),
        "num_batches": int(args.num_batches),
        "num_samples_per_batch": int(args.num_samples),
        "allow_overlap": bool(args.allow_overlap),
        "random_state": int(args.random_state),
        "batches": manifest_rows,
    }
    with open(out_dir / f"{args.prefix}_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Eligible rows: {len(pool)}")
    for row in manifest_rows:
        print(f"Wrote {row['num_samples']} row_ids -> {row['path']}")
    print(f"Manifest -> {out_dir / f'{args.prefix}_manifest.json'}")


def main() -> None:
    args = parse_args()
    pool = build_pool(args)
    write_batches(pool, args)


if __name__ == "__main__":
    main()
