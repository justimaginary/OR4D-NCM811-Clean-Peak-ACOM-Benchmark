#!/usr/bin/env python3
"""Compare two Clean image-generation backends without assuming peak order."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from or4d_common import read_peak_h5  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-image", type=Path, required=True)
    parser.add_argument("--candidate-image", type=Path, required=True)
    parser.add_argument("--reference-oracle", type=Path, required=True)
    parser.add_argument("--candidate-oracle", type=Path, required=True)
    parser.add_argument("--reference-trace", type=Path, required=True)
    parser.add_argument("--candidate-trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def decoded(values: np.ndarray) -> list[str]:
    return [
        value.decode() if isinstance(value, bytes) else str(value)
        for value in values
    ]


def compare_images(reference_path: Path, candidate_path: Path) -> dict:
    squared_error_sum = 0.0
    value_count = 0
    max_abs_error = 0.0
    probability_sum_deltas: list[float] = []
    with h5py.File(reference_path, "r") as reference, h5py.File(
        candidate_path, "r"
    ) as candidate:
        reference_ids = decoded(reference["sample_id"][:])
        candidate_ids = decoded(candidate["sample_id"][:])
        if reference_ids != candidate_ids:
            raise ValueError("image files have different sample ordering")
        reference_images = reference["expectation/intensity"]
        candidate_images = candidate["expectation/intensity"]
        if reference_images.shape != candidate_images.shape:
            raise ValueError("image arrays have different shapes")
        image_shape = list(reference_images.shape)
        for index in range(reference_images.shape[0]):
            reference_image = np.asarray(
                reference_images[index], dtype=np.float64
            )
            candidate_image = np.asarray(
                candidate_images[index], dtype=np.float64
            )
            delta = candidate_image - reference_image
            squared_error_sum += float(np.sum(delta * delta))
            value_count += delta.size
            max_abs_error = max(max_abs_error, float(np.max(np.abs(delta))))
            probability_sum_deltas.append(
                float(candidate_image.sum() - reference_image.sum())
            )
    return {
        "num_samples": len(reference_ids),
        "shape": image_shape,
        "pixel_rmse": float(np.sqrt(squared_error_sum / value_count)),
        "pixel_max_abs_error": max_abs_error,
        "probability_sum_max_abs_delta": float(
            np.max(np.abs(probability_sum_deltas))
        ),
    }


def compare_oracles(reference_path: Path, candidate_path: Path) -> dict:
    reference = {
        str(sample["sample_id"]): sample
        for sample in read_peak_h5(reference_path)
    }
    candidate = {
        str(sample["sample_id"]): sample
        for sample in read_peak_h5(candidate_path)
    }
    if reference.keys() != candidate.keys():
        raise ValueError("oracle files contain different sample IDs")
    squared_q_errors: list[float] = []
    intensity_errors: list[float] = []
    peak_count_mismatches = 0
    for sample_id, reference_sample in reference.items():
        candidate_sample = candidate[sample_id]
        reference_q = np.column_stack(
            (reference_sample["qx"], reference_sample["qy"])
        )
        candidate_q = np.column_stack(
            (candidate_sample["qx"], candidate_sample["qy"])
        )
        if len(reference_q) != len(candidate_q):
            peak_count_mismatches += 1
            continue
        distances = np.linalg.norm(
            reference_q[:, None, :] - candidate_q[None, :, :], axis=2
        )
        reference_indices, candidate_indices = linear_sum_assignment(distances)
        squared_q_errors.extend(
            distances[reference_indices, candidate_indices] ** 2
        )
        intensity_errors.extend(
            np.asarray(candidate_sample["intensity"])[candidate_indices]
            - np.asarray(reference_sample["intensity"])[reference_indices]
        )
    if peak_count_mismatches:
        raise ValueError(
            f"{peak_count_mismatches} samples have different oracle peak counts"
        )
    return {
        "matched_peak_count": len(squared_q_errors),
        "peak_count_mismatch_samples": peak_count_mismatches,
        "q_rmse_Ainv": float(np.sqrt(np.mean(squared_q_errors))),
        "q_max_error_Ainv": float(np.sqrt(np.max(squared_q_errors))),
        "normalized_intensity_rmse": float(
            np.sqrt(np.mean(np.square(intensity_errors)))
        ),
        "normalized_intensity_max_abs_error": float(
            np.max(np.abs(intensity_errors))
        ),
    }


def trace_rows(path: Path) -> dict[str, dict[str, np.ndarray]]:
    rows: dict[str, dict[str, np.ndarray]] = {}
    with h5py.File(path, "r") as h5:
        sample_ids = decoded(h5["sample_id"][:])
        offsets = h5["reflections/offsets"][:]
        for index, sample_id in enumerate(sample_ids):
            start, stop = int(offsets[index]), int(offsets[index + 1])
            rows[sample_id] = {
                "qx": h5["reflections/qx_Ainv"][start:stop],
                "qy": h5["reflections/qy_Ainv"][start:stop],
                "hkl": h5["reflections/hkl"][start:stop],
            }
    return rows


def compare_hkl(reference_path: Path, candidate_path: Path) -> dict:
    reference = trace_rows(reference_path)
    candidate = trace_rows(candidate_path)
    if reference.keys() != candidate.keys():
        raise ValueError("trace files contain different sample IDs")
    matched = 0
    hkl_equal = 0
    for sample_id, reference_sample in reference.items():
        candidate_sample = candidate[sample_id]
        reference_q = np.column_stack(
            (reference_sample["qx"], reference_sample["qy"])
        )
        candidate_q = np.column_stack(
            (candidate_sample["qx"], candidate_sample["qy"])
        )
        if len(reference_q) != len(candidate_q):
            raise ValueError(
                f"{sample_id} has different trace reflection counts"
            )
        distances = np.linalg.norm(
            reference_q[:, None, :] - candidate_q[None, :, :], axis=2
        )
        reference_indices, candidate_indices = linear_sum_assignment(distances)
        matched += len(reference_indices)
        hkl_equal += int(
            np.sum(
                np.all(
                    reference_sample["hkl"][reference_indices]
                    == candidate_sample["hkl"][candidate_indices],
                    axis=1,
                )
            )
        )
    return {
        "matched_reflection_count": matched,
        "matched_hkl_equal_count": hkl_equal,
        "matched_hkl_agreement": hkl_equal / matched if matched else 1.0,
    }


def main() -> None:
    args = parse_args()
    report = {
        "reference": {
            "image": str(args.reference_image.resolve()),
            "oracle": str(args.reference_oracle.resolve()),
            "trace": str(args.reference_trace.resolve()),
        },
        "candidate": {
            "image": str(args.candidate_image.resolve()),
            "oracle": str(args.candidate_oracle.resolve()),
            "trace": str(args.candidate_trace.resolve()),
        },
        "images": compare_images(
            args.reference_image.resolve(), args.candidate_image.resolve()
        ),
        "oracles": compare_oracles(
            args.reference_oracle.resolve(), args.candidate_oracle.resolve()
        ),
        "trace_identity": compare_hkl(
            args.reference_trace.resolve(), args.candidate_trace.resolve()
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Report: {args.output.resolve()}")


if __name__ == "__main__":
    main()
