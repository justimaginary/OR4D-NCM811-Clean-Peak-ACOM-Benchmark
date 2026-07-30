#!/usr/bin/env python3
"""Compare repeated detector peak files while ignoring runtime-only metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


NUMERICAL_DATASETS = (
    "peaks/offsets",
    "peaks/qx",
    "peaks/qy",
    "peaks/intensity",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument(
        "--pattern",
        default="*.h5",
        help="Glob used in the reference directory; default: *.h5",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def decode_strings(values: np.ndarray) -> list[str]:
    return [
        value.decode() if isinstance(value, bytes) else str(value)
        for value in values
    ]


def numerical_dataset_names(h5: h5py.File) -> list[str]:
    names = list(NUMERICAL_DATASETS)
    if "peaks/diagnostics" in h5:
        names.extend(
            f"peaks/diagnostics/{name}"
            for name in sorted(h5["peaks/diagnostics"].keys())
        )
    return names


def compare_dataset(
    reference: h5py.File, candidate: h5py.File, name: str
) -> dict:
    if name not in candidate:
        return {
            "dataset": name,
            "present_in_both": False,
            "shape_equal": False,
            "exact": False,
            "max_abs_delta": None,
        }
    left = np.asarray(reference[name][:])
    right = np.asarray(candidate[name][:])
    shape_equal = left.shape == right.shape
    if not shape_equal:
        return {
            "dataset": name,
            "present_in_both": True,
            "reference_shape": list(left.shape),
            "candidate_shape": list(right.shape),
            "shape_equal": False,
            "exact": False,
            "max_abs_delta": None,
        }
    exact = bool(np.array_equal(left, right, equal_nan=True))
    if left.size:
        delta = np.abs(
            np.asarray(left, dtype=np.float64)
            - np.asarray(right, dtype=np.float64)
        )
        finite = delta[np.isfinite(delta)]
        max_abs_delta = float(np.max(finite)) if finite.size else 0.0
    else:
        max_abs_delta = 0.0
    return {
        "dataset": name,
        "present_in_both": True,
        "shape": list(left.shape),
        "shape_equal": True,
        "exact": exact,
        "max_abs_delta": max_abs_delta,
    }


def compare_file(reference_path: Path, candidate_path: Path) -> dict:
    with h5py.File(reference_path, "r") as reference, h5py.File(
        candidate_path, "r"
    ) as candidate:
        reference_ids = decode_strings(reference["sample_id"][:])
        candidate_ids = decode_strings(candidate["sample_id"][:])
        dataset_names = numerical_dataset_names(reference)
        diagnostics = [
            compare_dataset(reference, candidate, name)
            for name in dataset_names
        ]
        candidate_extra = sorted(
            set(numerical_dataset_names(candidate)) - set(dataset_names)
        )
        sample_ids_exact = reference_ids == candidate_ids
        all_exact = (
            sample_ids_exact
            and not candidate_extra
            and all(row["exact"] for row in diagnostics)
        )
        return {
            "file": reference_path.name,
            "reference": str(reference_path.resolve()),
            "candidate": str(candidate_path.resolve()),
            "sample_count": len(reference_ids),
            "sample_ids_exact": sample_ids_exact,
            "candidate_extra_numerical_datasets": candidate_extra,
            "datasets": diagnostics,
            "all_exact": all_exact,
        }


def main() -> None:
    args = parse_args()
    reference_dir = args.reference_dir.resolve()
    candidate_dir = args.candidate_dir.resolve()
    reference_files = sorted(reference_dir.glob(args.pattern))
    if not reference_files:
        raise FileNotFoundError(
            f"No reference files match {args.pattern!r} in {reference_dir}"
        )
    comparisons = []
    missing_candidates = []
    for reference_path in reference_files:
        candidate_path = candidate_dir / reference_path.name
        if not candidate_path.is_file():
            missing_candidates.append(reference_path.name)
            continue
        comparisons.append(compare_file(reference_path, candidate_path))
    report = {
        "comparison_scope": (
            "sample IDs, peak offsets/qx/qy/intensity, and all per-peak "
            "diagnostic arrays; runtime-only sample metadata is excluded"
        ),
        "reference_dir": str(reference_dir),
        "candidate_dir": str(candidate_dir),
        "pattern": args.pattern,
        "reference_file_count": len(reference_files),
        "compared_file_count": len(comparisons),
        "missing_candidate_files": missing_candidates,
        "files": comparisons,
        "all_exact": (
            not missing_candidates
            and len(comparisons) == len(reference_files)
            and all(row["all_exact"] for row in comparisons)
        ),
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
