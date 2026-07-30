#!/usr/bin/env python3
"""Evaluate saved Pyxem template-matching orientations without rerunning them."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
from pymatgen.core import Structure

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from or4d_common import (  # noqa: E402
    FRIEDEL_SAMPLE_ROTATION,
    cif_path,
    load_config,
    proper_point_group_rotations,
    read_jsonl,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-file", type=Path, required=True)
    parser.add_argument("--ground-truth-file", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument(
        "--clean-e-details-output",
        type=Path,
        help="Optional 2048-row JSONL diagnostic for the Clean-E group.",
    )
    return parser.parse_args()


def decode(values: np.ndarray) -> list[str]:
    return [
        value.decode() if isinstance(value, bytes) else str(value)
        for value in values
    ]


def rotation_angles_deg(relative: np.ndarray) -> np.ndarray:
    traces = np.trace(relative, axis1=-2, axis2=-1)
    cosine = np.clip((traces - 1.0) / 2.0, -1.0, 1.0)
    return np.rad2deg(np.arccos(cosine))


def condition_errors(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    symmetries: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return raw, crystal-symmetry, and Friedel-aware errors."""
    predicted = np.asarray(predicted, dtype=np.float64)
    ground_truth = np.asarray(ground_truth, dtype=np.float64)
    raw = rotation_angles_deg(
        predicted @ np.swapaxes(ground_truth, -1, -2)
    )
    strict = np.full(len(predicted), np.inf, dtype=np.float64)
    friedel = np.full(len(predicted), np.inf, dtype=np.float64)
    for symmetry in symmetries:
        equivalent = np.einsum("ab,nbc->nac", symmetry, ground_truth)
        strict = np.minimum(
            strict,
            rotation_angles_deg(
                predicted @ np.swapaxes(equivalent, -1, -2)
            ),
        )
        friedel = np.minimum(friedel, strict)
        equivalent_friedel = np.einsum(
            "nab,bc->nac", equivalent, FRIEDEL_SAMPLE_ROTATION
        )
        friedel = np.minimum(
            friedel,
            rotation_angles_deg(
                predicted @ np.swapaxes(equivalent_friedel, -1, -2)
            ),
        )
    invalid = ~np.isfinite(predicted).all(axis=(1, 2))
    raw[invalid] = np.nan
    strict[invalid] = np.nan
    friedel[invalid] = np.nan
    return raw, strict, friedel


def aggregate(
    errors: np.ndarray,
    *,
    total_samples: int,
) -> dict[str, float | int]:
    finite = np.asarray(errors[np.isfinite(errors)], dtype=np.float64)
    result: dict[str, float | int] = {
        "num_samples": int(total_samples),
        "num_valid_predictions": int(finite.size),
        "prediction_coverage": float(finite.size / total_samples),
    }
    if finite.size == 0:
        result.update(
            {
                "median_misorientation_deg": float("nan"),
                "p95_misorientation_deg": float("nan"),
                "max_misorientation_deg": float("nan"),
                "accuracy_within_1deg": 0.0,
                "accuracy_within_2deg": 0.0,
                "accuracy_within_5deg": 0.0,
            }
        )
        return result
    result.update(
        {
            "median_misorientation_deg": float(np.median(finite)),
            "p95_misorientation_deg": float(np.percentile(finite, 95)),
            "max_misorientation_deg": float(np.max(finite)),
            # Invalid predictions count as failures in headline accuracy.
            "accuracy_within_1deg": float(np.sum(finite <= 1.0) / total_samples),
            "accuracy_within_2deg": float(np.sum(finite <= 2.0) / total_samples),
            "accuracy_within_5deg": float(np.sum(finite <= 5.0) / total_samples),
        }
    )
    return result


def labels_for_group(result: h5py.File, name: str) -> list[dict[str, object]]:
    if name == "clean_e":
        return [{"track": "Clean-E"}]
    doses = np.asarray(result["dose_electrons"][:], dtype=np.int64)
    if name == "clean_c_noiseless":
        return [
            {"track": "Clean-C", "dose_electrons": int(dose), "noise": "noiseless"}
            for dose in doses
        ]
    if name == "clean_c_counted":
        levels = decode(result["counted_noise_level_id"][:])
        repeats = int(result.attrs["counted_repeats"])
        return [
            {
                "track": "Clean-C",
                "dose_electrons": int(dose),
                "noise": level,
                "repeat": repeat,
            }
            for dose in doses
            for level in levels
            for repeat in range(repeats)
        ]
    raise ValueError(f"unsupported result group {name}")


def main() -> None:
    args = parse_args()
    config = load_config()
    symmetries = proper_point_group_rotations(
        Structure.from_file(str(cif_path(config)))
    )
    ground_truth_rows = read_jsonl(args.ground_truth_file)
    ground_truth_by_id = {
        row["sample_id"]: np.asarray(
            row["orientation_matrix_sample_to_crystal"], dtype=np.float64
        )
        for row in ground_truth_rows
    }
    summaries: list[dict[str, object]] = []
    clean_e_details: list[dict[str, object]] = []
    with h5py.File(args.result_file, "r") as result:
        sample_ids = decode(result["sample_id"][:])
        ground_truth = np.stack(
            [ground_truth_by_id[sample_id] for sample_id in sample_ids]
        )
        for group_name in ("clean_e", "clean_c_noiseless", "clean_c_counted"):
            if group_name not in result:
                continue
            group = result[group_name]
            complete = np.asarray(group["condition_complete"][:], dtype=bool)
            matrices = group["orientation_matrix_sample_to_crystal"]
            correlations = group["correlation"]
            mirrored = group["mirrored_template"]
            labels = labels_for_group(result, group_name)
            condition_indices = (
                [()] if group_name == "clean_e" else list(np.ndindex(complete.shape))
            )
            if len(labels) != len(condition_indices):
                raise RuntimeError(f"condition label mismatch for {group_name}")
            for label, condition in zip(labels, condition_indices, strict=True):
                completion_index = (0,) if group_name == "clean_e" else condition
                if not bool(complete[completion_index]):
                    continue
                predicted = np.asarray(
                    matrices[condition] if condition else matrices[:],
                    dtype=np.float64,
                )
                raw, strict_error, friedel_error = condition_errors(
                    predicted, ground_truth, symmetries
                )
                row = dict(label)
                row.update(aggregate(friedel_error, total_samples=len(sample_ids)))
                row["group"] = group_name
                row["raw_median_deg"] = float(np.nanmedian(raw))
                row["strict_median_deg"] = float(np.nanmedian(strict_error))
                row["condition_seconds"] = float(
                    group["condition_seconds"][completion_index]
                )
                summaries.append(row)
                if group_name == "clean_e":
                    corr = np.asarray(correlations[:], dtype=float)
                    mirror = np.asarray(mirrored[:], dtype=bool)
                    clean_e_details = [
                        {
                            "sample_id": sample_id,
                            "raw_misorientation_deg": float(raw[index]),
                            "strict_misorientation_deg": float(strict_error[index]),
                            "friedel_equivalent_misorientation_deg": float(
                                friedel_error[index]
                            ),
                            "correlation": float(corr[index]),
                            "mirrored_template": bool(mirror[index]),
                            "predicted_orientation_matrix_sample_to_crystal": (
                                predicted[index].tolist()
                            ),
                        }
                        for index, sample_id in enumerate(sample_ids)
                    ]
        metadata = {
            "method": str(result.attrs.get("method", "")),
            "pyxem_version": str(result.attrs.get("pyxem_version", "")),
            "target": str(result.attrs.get("target", "")),
            "settings": json.loads(str(result.attrs["settings_json"])),
        }
    output = {
        "metadata": metadata,
        "metric": (
            "Minimum misorientation over proper crystal point-group rotations "
            "and the detector-plane Friedel branch."
        ),
        "conditions": summaries,
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, allow_nan=True),
        encoding="utf-8",
    )
    if args.clean_e_details_output is not None and clean_e_details:
        write_jsonl(args.clean_e_details_output, clean_e_details)
    print(f"Pyxem evaluation: {args.summary_output}")


if __name__ == "__main__":
    main()
