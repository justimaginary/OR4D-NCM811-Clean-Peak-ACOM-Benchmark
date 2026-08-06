#!/usr/bin/env python3
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
    cif_path,
    load_config,
    proper_point_group_rotations,
    read_jsonl,
)
from v6_polar_metrics import (  # noqa: E402
    beam_polar_coordinates,
    candidate_polar_errors,
    summarize_topk_polar_errors,
)
from v6_runtime import enforce_server_write_scope  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate saved V6 ACOM Top-K candidates in beam polar coordinates."
    )
    parser.add_argument(
        "--config", type=Path, default=ROOT / "config" / "benchmark_v6.yaml"
    )
    parser.add_argument("--candidate-file", type=Path, required=True)
    parser.add_argument("--ground-truth-file", type=Path, required=True)
    parser.add_argument("--output-h5", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def decode(values: np.ndarray) -> list[str]:
    return [
        value.decode() if isinstance(value, bytes) else str(value)
        for value in values
    ]


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    settings = config["v6"]["evaluation"]["polar_metrics"]
    if not bool(settings["enabled"]):
        raise RuntimeError("V6 polar metrics are disabled in the selected config")
    ground_truth_rows = read_jsonl(args.ground_truth_file.resolve())
    ground_truth = {str(row["sample_id"]): row for row in ground_truth_rows}
    if len(ground_truth) != len(ground_truth_rows):
        raise ValueError("V6 polar ground truth contains duplicate sample IDs")
    symmetries = proper_point_group_rotations(
        Structure.from_file(cif_path(config))
    )
    candidate_path = args.candidate_file.resolve()
    with h5py.File(candidate_path, "r") as source:
        sample_ids = decode(source["sample_id"][:])
        matrices = np.asarray(
            source["orientation_matrix_sample_to_crystal"][:], dtype=np.float64
        )
    if matrices.ndim != 4 or matrices.shape[2:] != (3, 3):
        raise ValueError("ACOM candidates must have shape [sample,rank,3,3]")
    if len(sample_ids) != len(matrices) or len(set(sample_ids)) != len(sample_ids):
        raise ValueError("ACOM candidate sample IDs are inconsistent")
    missing = sorted(set(sample_ids) - set(ground_truth))
    if missing:
        raise KeyError(f"Candidate IDs are absent from ground truth: {missing[:3]}")

    sample_count, rank_count = matrices.shape[:2]
    scalar_names = (
        "equivalent_misorientation_deg",
        "beam_direction_error_deg",
        "tilt_error_deg",
        "azimuth_error_deg",
        "ground_truth_tilt_deg",
        "ground_truth_azimuth_deg",
        "predicted_aligned_tilt_deg",
        "predicted_aligned_azimuth_deg",
    )
    scalars = {
        name: np.empty((sample_count, rank_count), dtype=np.float64)
        for name in scalar_names
    }
    friedel_used = np.empty((sample_count, rank_count), dtype=np.bool_)
    symmetry_index = np.empty((sample_count, rank_count), dtype=np.int16)
    aligned = np.empty((sample_count, rank_count, 3, 3), dtype=np.float64)
    for sample_index, sample_id in enumerate(sample_ids):
        gt_matrix = np.asarray(
            ground_truth[sample_id]["orientation_matrix_sample_to_crystal"],
            dtype=np.float64,
        )
        for rank_index in range(rank_count):
            result = candidate_polar_errors(
                matrices[sample_index, rank_index],
                gt_matrix,
                symmetries,
                settings,
            )
            for name in scalar_names:
                scalars[name][sample_index, rank_index] = result[name]
            friedel_used[sample_index, rank_index] = result["friedel_used"]
            symmetry_index[sample_index, rank_index] = result[
                "crystal_symmetry_index"
            ]
            aligned[sample_index, rank_index] = result[
                "aligned_matrix_sample_to_crystal"
            ]

    eligible_total = 0
    for row in ground_truth_rows:
        _, tilt, _ = beam_polar_coordinates(
            np.asarray(row["orientation_matrix_sample_to_crystal"], dtype=np.float64),
            beam_axis_sample=np.asarray(settings["beam_axis_sample"]),
            polar_reference_crystal_axis=np.asarray(
                settings["polar_reference_crystal_axis"]
            ),
            azimuth_reference_crystal_axis=np.asarray(
                settings["azimuth_reference_crystal_axis"]
            ),
        )
        eligible_total += tilt >= float(settings["azimuth_valid_min_tilt_deg"])

    summary = summarize_topk_polar_errors(
        scalars["beam_direction_error_deg"],
        scalars["tilt_error_deg"],
        scalars["azimuth_error_deg"],
        total_input_samples=len(ground_truth_rows),
        total_azimuth_eligible_samples=int(eligible_total),
        beam_thresholds_deg=settings["beam_accuracy_thresholds_deg"],
        azimuth_thresholds_deg=settings["azimuth_accuracy_thresholds_deg"],
    )
    output_h5 = enforce_server_write_scope(args.output_h5, config)
    output_json = enforce_server_write_scope(args.output_json, config)
    output_h5.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_h5, "w") as output:
        output.attrs["schema"] = "or4d-clean-v6-acom-polar-topk-v1"
        output.attrs["candidate_file"] = str(candidate_path)
        output.attrs["ground_truth_file"] = str(args.ground_truth_file.resolve())
        output.create_dataset(
            "sample_id",
            data=np.asarray(sample_ids, dtype=h5py.string_dtype("utf-8")),
        )
        output.create_dataset("rank", data=np.arange(1, rank_count + 1))
        for name, values in scalars.items():
            output.create_dataset(name, data=values, compression="gzip")
        output.create_dataset("friedel_used", data=friedel_used, compression="gzip")
        output.create_dataset(
            "crystal_symmetry_index", data=symmetry_index, compression="gzip"
        )
        output.create_dataset(
            "aligned_matrix_sample_to_crystal", data=aligned, compression="gzip"
        )
    payload = {
        "schema": "or4d-clean-v6-acom-polar-summary-v1",
        "candidate_file": str(candidate_path),
        "ground_truth_file": str(args.ground_truth_file.resolve()),
        "num_input_samples": len(ground_truth_rows),
        "num_indexed_samples": len(sample_ids),
        "num_azimuth_eligible_input_samples": int(eligible_total),
        "azimuth_valid_min_tilt_deg": float(settings["azimuth_valid_min_tilt_deg"]),
        "alignment": "best crystal symmetry plus Friedel branch per candidate",
        "definitions": {
            "beam_direction_error_deg": "angle between aligned predicted and GT beam directions in crystal coordinates",
            "tilt_error_deg": "absolute difference of polar tilt from the configured crystal polar axis",
            "azimuth_error_deg": "circular azimuth difference; N/A below the configured GT tilt",
        },
        "top_k_metrics": summary,
        "output_h5": str(output_h5),
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
