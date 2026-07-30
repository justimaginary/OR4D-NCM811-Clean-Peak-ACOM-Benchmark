#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from pymatgen.core import Structure

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from or4d_common import (  # noqa: E402
    cif_path,
    friedel_aware_misorientation_deg,
    load_config,
    proper_point_group_rotations,
    read_jsonl,
    symmetry_aware_misorientation_deg,
)


def summarize(values: np.ndarray, track: str) -> dict:
    if values.size == 0:
        raise ValueError(f"Cannot summarize an empty {track} result set")
    return {
        "track": track,
        "num_samples": int(values.size),
        "mean_misorientation_deg": float(values.mean()),
        "median_misorientation_deg": float(np.median(values)),
        "p90_misorientation_deg": float(np.percentile(values, 90)),
        "p95_misorientation_deg": float(np.percentile(values, 95)),
        "max_misorientation_deg": float(values.max()),
        "accuracy_within_1deg": float(np.mean(values <= 1.0)),
        "accuracy_within_2deg": float(np.mean(values <= 2.0)),
        "accuracy_within_5deg": float(np.mean(values <= 5.0)),
    }


def unique_records_by_id(records: list[dict], *, source: str) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for record in records:
        record_id = record.get("sample_id", record.get("orientation_id"))
        if record_id is None:
            raise ValueError(
                f"Record in {source} has neither sample_id nor orientation_id"
            )
        sample_id = str(record_id)
        if sample_id in result:
            raise ValueError(f"Duplicate sample_id in {source}: {sample_id}")
        normalized = dict(record)
        normalized["sample_id"] = sample_id
        normalized.setdefault("track", "clean")
        result[sample_id] = normalized
    return result


def validate_prediction_matrix(sample_id: str, value: object) -> np.ndarray:
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (3, 3):
        raise ValueError(
            f"{sample_id} orientation matrix has shape {matrix.shape}, expected (3, 3)"
        )
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{sample_id} orientation matrix contains non-finite values")
    if not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-5):
        raise ValueError(f"{sample_id} orientation matrix is not orthonormal")
    if not np.isclose(np.linalg.det(matrix), 1.0, atol=1e-5):
        raise ValueError(f"{sample_id} orientation matrix determinant is not +1")
    return matrix


def summarize_by_role(
    rows: list[dict],
    *,
    error_field: str,
    track: str,
) -> dict[str, dict]:
    roles = sorted({str(row.get("sample_role", "unspecified")) for row in rows})
    return {
        role: summarize(
            np.asarray(
                [
                    row[error_field]
                    for row in rows
                    if str(row.get("sample_role", "unspecified")) == role
                ],
                dtype=float,
            ),
            track,
        )
        for role in roles
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("submission", type=Path)
    parser.add_argument(
        "--track",
        choices=("clean", "dynamical", "all"),
        default="all",
        help="Evaluate only one track or all generated tracks.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports" / "evaluation.json",
    )
    parser.add_argument(
        "--ground-truth-file",
        type=Path,
        help=(
            "Explicit ground-truth JSONL for an external clean dataset. "
            "Records may use sample_id or orientation_id."
        ),
    )
    parser.add_argument(
        "--ground-truth-id-prefix",
        default="",
        help="Explicit prefix added to ground-truth IDs before submission matching.",
    )
    args = parser.parse_args()

    config = load_config()
    predictions = unique_records_by_id(
        read_jsonl(args.submission),
        source=str(args.submission),
    )
    tracks = ("clean", "dynamical") if args.track == "all" else (args.track,)
    ground_truth_records: list[dict] = []
    if args.ground_truth_file:
        if args.track != "clean":
            raise ValueError("--ground-truth-file currently supports --track clean")
        ground_truth_records.extend(read_jsonl(args.ground_truth_file.resolve()))
    else:
        for track in tracks:
            path = ROOT / "private" / f"{track}_ground_truth.jsonl"
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing ground truth for requested track: {path}"
                )
            ground_truth_records.extend(read_jsonl(path))
    ground_truth_unprefixed = unique_records_by_id(
        ground_truth_records,
        source="requested ground-truth tracks",
    )
    ground_truth: dict[str, dict] = {}
    for ground_truth_id, record in ground_truth_unprefixed.items():
        sample_id = f"{args.ground_truth_id_prefix}{ground_truth_id}"
        normalized = dict(record)
        normalized["sample_id"] = sample_id
        normalized["ground_truth_source_id"] = ground_truth_id
        ground_truth[sample_id] = normalized
    if set(predictions) != set(ground_truth):
        missing = sorted(set(ground_truth) - set(predictions))
        extra = sorted(set(predictions) - set(ground_truth))
        raise ValueError(
            f"Submission IDs do not exactly match the requested ground truth: "
            f"missing={missing}, extra={extra}"
        )

    structure = Structure.from_file(cif_path(config))
    symmetries = proper_point_group_rotations(structure)
    strict_errors = []
    primary_errors = []
    per_sample = []
    for sample_id, gt in ground_truth.items():
        R_pred = validate_prediction_matrix(
            sample_id,
            predictions[sample_id]["orientation_matrix_sample_to_crystal"],
        )
        R_gt = np.asarray(gt["orientation_matrix_sample_to_crystal"], dtype=float)
        strict_error = symmetry_aware_misorientation_deg(
            R_pred, R_gt, symmetries
        )
        if gt.get("track") == "clean":
            primary_error = friedel_aware_misorientation_deg(
                R_pred, R_gt, symmetries
            )
            primary_name = "friedel_equivalent_misorientation_deg"
        else:
            primary_error = strict_error
            primary_name = "strict_misorientation_deg"
        strict_errors.append(strict_error)
        primary_errors.append(primary_error)
        per_sample.append(
            {
                "sample_id": sample_id,
                "track": gt.get("track"),
                "sampling_type": gt.get("sampling_type"),
                "sample_role": gt.get("sample_role"),
                "strict_misorientation_deg": strict_error,
                primary_name: primary_error,
                "misorientation_deg": primary_error,
            }
        )

    strict_values = np.asarray(strict_errors, dtype=float)
    primary_values = np.asarray(primary_errors, dtype=float)
    headline_role = str(config["evaluation"]["headline_sample_role"])
    if args.track == "clean":
        headline_rows = [
            row for row in per_sample if row.get("sample_role") == headline_role
        ]
    else:
        headline_rows = per_sample
    headline_primary = np.asarray(
        [row["misorientation_deg"] for row in headline_rows],
        dtype=float,
    )
    headline_strict = np.asarray(
        [row["strict_misorientation_deg"] for row in headline_rows],
        dtype=float,
    )
    disagreement_tolerance = float(
        config["evaluation"]["strict_friedel_disagreement_tolerance_deg"]
    )
    output = {
        "primary_metric": (
            "friedel_equivalent_misorientation_deg"
            if args.track == "clean"
            else "track_dependent_misorientation_deg"
        ),
        "headline_sample_role": headline_role if args.track == "clean" else None,
        "metrics": summarize(headline_primary, args.track),
        "metrics_strict": summarize(headline_strict, args.track),
        "metrics_all_samples": summarize(primary_values, args.track),
        "metrics_strict_all_samples": summarize(strict_values, args.track),
        "metrics_by_sample_role": summarize_by_role(
            per_sample,
            error_field="misorientation_deg",
            track=args.track,
        ),
        "metrics_strict_by_sample_role": summarize_by_role(
            per_sample,
            error_field="strict_misorientation_deg",
            track=args.track,
        ),
        "samples": per_sample,
    }
    if args.track == "clean":
        output["metrics_friedel_equivalent"] = summarize(
            headline_primary, args.track
        )
        output["strict_friedel_disagreement_rate"] = float(
            np.mean(
                np.abs(headline_strict - headline_primary)
                > disagreement_tolerance
            )
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(json.dumps(output["metrics"], indent=2))
    print(f"Evaluation report: {args.output.resolve()}")


if __name__ == "__main__":
    main()
