#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from or4d_common import (  # noqa: E402
    cif_path,
    friedel_aware_misorientation_deg,
    load_config,
    nearest_rotation,
    read_jsonl,
    symmetry_aware_misorientation_deg,
)


def proper_point_group_rotations() -> list[np.ndarray]:
    config = load_config()
    structure = Structure.from_file(cif_path(config))
    operations = SpacegroupAnalyzer(structure).get_point_group_operations(
        cartesian=True
    )
    rotations: list[np.ndarray] = []
    for operation in operations:
        raw = np.asarray(operation.rotation_matrix, dtype=float)
        if np.linalg.det(raw) < 0.0:
            continue
        matrix = nearest_rotation(raw)
        if not any(np.allclose(matrix, existing, atol=1e-8) for existing in rotations):
            rotations.append(matrix)
    if not rotations:
        raise RuntimeError("No proper crystal point-group rotations were found.")
    return rotations


def summarize(values: np.ndarray, track: str) -> dict:
    return {
        "track": track,
        "num_samples": int(values.size),
        "mean_misorientation_deg": float(values.mean()),
        "median_misorientation_deg": float(np.median(values)),
        "p90_misorientation_deg": float(np.percentile(values, 90)),
        "accuracy_within_1deg": float(np.mean(values <= 1.0)),
        "accuracy_within_2deg": float(np.mean(values <= 2.0)),
        "accuracy_within_5deg": float(np.mean(values <= 5.0)),
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
    args = parser.parse_args()

    predictions = {x["sample_id"]: x for x in read_jsonl(args.submission)}
    tracks = ("clean", "dynamical") if args.track == "all" else (args.track,)
    ground_truth = {}
    for track in tracks:
        path = ROOT / "private" / f"{track}_ground_truth.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Missing ground truth for requested track: {path}")
        ground_truth.update({x["sample_id"]: x for x in read_jsonl(path)})

    symmetries = proper_point_group_rotations()
    strict_errors = []
    primary_errors = []
    per_sample = []
    for sample_id, gt in ground_truth.items():
        if sample_id not in predictions:
            raise ValueError(f"Submission is missing {sample_id}")
        R_pred = nearest_rotation(
            np.asarray(
                predictions[sample_id]["orientation_matrix_sample_to_crystal"],
                dtype=float,
            )
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
                "strict_misorientation_deg": strict_error,
                primary_name: primary_error,
                "misorientation_deg": primary_error,
            }
        )

    strict_values = np.asarray(strict_errors, dtype=float)
    primary_values = np.asarray(primary_errors, dtype=float)
    output = {
        "primary_metric": (
            "friedel_equivalent_misorientation_deg"
            if args.track == "clean"
            else "track_dependent_misorientation_deg"
        ),
        "metrics": summarize(primary_values, args.track),
        "metrics_strict": summarize(strict_values, args.track),
        "samples": per_sample,
    }
    if args.track == "clean":
        output["metrics_friedel_equivalent"] = summarize(
            primary_values, args.track
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(json.dumps(output["metrics"], indent=2))
    print(f"Evaluation report: {args.output.resolve()}")


if __name__ == "__main__":
    main()
