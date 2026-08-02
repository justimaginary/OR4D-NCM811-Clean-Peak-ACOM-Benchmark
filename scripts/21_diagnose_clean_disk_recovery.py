#!/usr/bin/env python3
"""Diagnose Clean disk-recovery misses by intensity, radius, and crowding."""

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

from or4d_common import load_config, read_peak_h5  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-file", type=Path, required=True)
    parser.add_argument("--oracle-file", type=Path, required=True)
    parser.add_argument("--detection-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def records(path: Path) -> dict[str, dict]:
    return {
        str(sample["sample_id"]): sample for sample in read_peak_h5(path)
    }


def quantiles(values: list[float] | np.ndarray) -> dict[str, float | None]:
    array = np.asarray(values, dtype=float)
    probabilities = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0)
    if not len(array):
        return {f"q{probability:g}": None for probability in probabilities}
    return {
        f"q{probability:g}": float(np.quantile(array, probability))
        for probability in probabilities
    }


def binned_recall(
    values: np.ndarray, matched: np.ndarray, edges: np.ndarray
) -> list[dict]:
    rows = []
    for lower, upper in zip(edges[:-1], edges[1:]):
        selected = (values >= lower) & (values < upper)
        count = int(selected.sum())
        rows.append(
            {
                "lower": float(lower),
                "upper": None if np.isinf(upper) else float(upper),
                "oracle_count": count,
                "matched_count": int(np.count_nonzero(matched[selected])),
                "recall": (
                    float(np.mean(matched[selected])) if count else None
                ),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    config = load_config()
    acceptance = config["clean_image"]["acceptance"]
    oracle = records(args.oracle_file.resolve())
    detection_report = json.loads(
        args.detection_report.read_text(encoding="utf-8")
    )
    with h5py.File(args.image_file.resolve(), "r") as h5:
        qx_axis = np.asarray(h5["detector/qx_Ainv"][:], dtype=float)
        q_pixel = float(np.median(np.diff(qx_axis)))
        disk_radius_px = float(h5.attrs["disk_radius_px"])
    tolerance = float(acceptance["match_tolerance_px"]) * q_pixel
    disk_diameter_Ainv = 2.0 * disk_radius_px * q_pixel
    detector_rows = []

    for run in detection_report["runs"]:
        detected = records(Path(run["output"]).resolve())
        matched_intensity: list[float] = []
        missed_intensity: list[float] = []
        matched_radius: list[float] = []
        missed_radius: list[float] = []
        matched_neighbor: list[float] = []
        missed_neighbor: list[float] = []
        all_intensity: list[float] = []
        all_radius: list[float] = []
        all_neighbor: list[float] = []
        all_matched: list[bool] = []
        detected_counts: list[int] = []

        for sample_id, oracle_sample in oracle.items():
            detected_sample = detected[sample_id]
            oracle_xy = np.column_stack(
                (oracle_sample["qx"], oracle_sample["qy"])
            ).astype(float)
            detected_xy = np.column_stack(
                (detected_sample["qx"], detected_sample["qy"])
            ).astype(float)
            matched = np.zeros(len(oracle_xy), dtype=bool)
            if len(oracle_xy) and len(detected_xy):
                distances = np.linalg.norm(
                    oracle_xy[:, None, :] - detected_xy[None, :, :],
                    axis=2,
                )
                oracle_index, detected_index = linear_sum_assignment(
                    distances
                )
                keep = (
                    distances[oracle_index, detected_index] <= tolerance
                )
                matched[oracle_index[keep]] = True
            if len(oracle_xy) > 1:
                pairwise = np.linalg.norm(
                    oracle_xy[:, None, :] - oracle_xy[None, :, :],
                    axis=2,
                )
                np.fill_diagonal(pairwise, np.inf)
                nearest_neighbor = np.min(pairwise, axis=1)
            else:
                nearest_neighbor = np.full(len(oracle_xy), np.inf)
            intensity = np.asarray(
                oracle_sample["intensity"], dtype=float
            )
            radius = np.hypot(oracle_xy[:, 0], oracle_xy[:, 1])
            all_intensity.extend(intensity)
            all_radius.extend(radius)
            all_neighbor.extend(nearest_neighbor)
            all_matched.extend(matched)
            matched_intensity.extend(intensity[matched])
            missed_intensity.extend(intensity[~matched])
            matched_radius.extend(radius[matched])
            missed_radius.extend(radius[~matched])
            matched_neighbor.extend(nearest_neighbor[matched])
            missed_neighbor.extend(nearest_neighbor[~matched])
            detected_counts.append(len(detected_xy))

        intensity_values = np.asarray(all_intensity, dtype=float)
        radius_values = np.asarray(all_radius, dtype=float)
        neighbor_values = np.asarray(all_neighbor, dtype=float)
        matched_values = np.asarray(all_matched, dtype=bool)
        detector_rows.append(
            {
                "detector": run["detector"],
                "num_samples": len(oracle),
                "oracle_peaks": len(all_intensity),
                "matched_peaks": int(np.count_nonzero(matched_values)),
                "recall": float(np.mean(matched_values)),
                "matched_oracle_intensity_quantiles": quantiles(
                    matched_intensity
                ),
                "missed_oracle_intensity_quantiles": quantiles(
                    missed_intensity
                ),
                "matched_radius_Ainv_quantiles": quantiles(
                    matched_radius
                ),
                "missed_radius_Ainv_quantiles": quantiles(missed_radius),
                "matched_nearest_neighbor_Ainv_quantiles": quantiles(
                    matched_neighbor
                ),
                "missed_nearest_neighbor_Ainv_quantiles": quantiles(
                    missed_neighbor
                ),
                "recall_by_oracle_intensity": binned_recall(
                    intensity_values,
                    matched_values,
                    np.asarray(
                        [
                            0.0,
                            1e-4,
                            3e-4,
                            1e-3,
                            3e-3,
                            1e-2,
                            3e-2,
                            1e-1,
                            3e-1,
                            1.01,
                        ]
                    ),
                ),
                "recall_by_radius_Ainv": binned_recall(
                    radius_values,
                    matched_values,
                    np.asarray([0.0, 0.3, 0.6, 0.9, 1.2, 1.51]),
                ),
                "recall_by_nearest_neighbor_disk_diameters": binned_recall(
                    neighbor_values / disk_diameter_Ainv,
                    matched_values,
                    np.asarray([0.0, 0.5, 1.0, 1.5, 2.0, 4.0, np.inf]),
                ),
                "detected_count_quantiles": quantiles(detected_counts),
                "samples_at_or_above_256_detected_peaks": int(
                    np.count_nonzero(np.asarray(detected_counts) >= 256)
                ),
            }
        )

    report = {
        "image_file": str(args.image_file.resolve()),
        "oracle_file": str(args.oracle_file.resolve()),
        "detection_report": str(args.detection_report.resolve()),
        "match_tolerance_px": float(acceptance["match_tolerance_px"]),
        "q_pixel_Ainv": q_pixel,
        "disk_radius_px": disk_radius_px,
        "disk_diameter_Ainv": disk_diameter_Ainv,
        "detectors": detector_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Diagnostic report: {args.output.resolve()}")


if __name__ == "__main__":
    main()
