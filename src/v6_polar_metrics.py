from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from or4d_common import best_friedel_alignment, normalize


def beam_polar_coordinates(
    orientation_sample_to_crystal: np.ndarray,
    *,
    beam_axis_sample: np.ndarray,
    polar_reference_crystal_axis: np.ndarray,
    azimuth_reference_crystal_axis: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """Return beam direction, polar tilt, and azimuth in crystal coordinates."""
    matrix = np.asarray(orientation_sample_to_crystal, dtype=np.float64)
    beam = normalize(matrix @ normalize(beam_axis_sample))
    polar_axis = normalize(polar_reference_crystal_axis)
    azimuth_x = np.asarray(azimuth_reference_crystal_axis, dtype=np.float64)
    azimuth_x = normalize(azimuth_x - np.dot(azimuth_x, polar_axis) * polar_axis)
    azimuth_y = normalize(np.cross(polar_axis, azimuth_x))
    tilt_deg = float(
        np.degrees(np.arccos(np.clip(np.dot(beam, polar_axis), -1.0, 1.0)))
    )
    projected = beam - np.dot(beam, polar_axis) * polar_axis
    if np.linalg.norm(projected) <= 1e-12:
        azimuth_deg = float("nan")
    else:
        projected = normalize(projected)
        azimuth_deg = float(
            np.degrees(
                np.arctan2(
                    np.dot(projected, azimuth_y),
                    np.dot(projected, azimuth_x),
                )
            )
            % 360.0
        )
    return beam, tilt_deg, azimuth_deg


def circular_difference_deg(first: float, second: float) -> float:
    return float(abs((float(first) - float(second) + 180.0) % 360.0 - 180.0))


def candidate_polar_errors(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    symmetry_rotations: Iterable[np.ndarray],
    settings: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate one candidate after the same symmetry/Friedel alignment as ACOM."""
    symmetries = [np.asarray(value, dtype=np.float64) for value in symmetry_rotations]
    alignment = best_friedel_alignment(predicted, ground_truth, symmetries)
    aligned = np.asarray(alignment["aligned_matrix"], dtype=np.float64)
    coordinates = {
        "beam_axis_sample": np.asarray(settings["beam_axis_sample"], dtype=np.float64),
        "polar_reference_crystal_axis": np.asarray(
            settings["polar_reference_crystal_axis"], dtype=np.float64
        ),
        "azimuth_reference_crystal_axis": np.asarray(
            settings["azimuth_reference_crystal_axis"], dtype=np.float64
        ),
    }
    beam_gt, tilt_gt, azimuth_gt = beam_polar_coordinates(
        ground_truth, **coordinates
    )
    beam_predicted, tilt_predicted, azimuth_predicted = beam_polar_coordinates(
        aligned, **coordinates
    )
    beam_error = float(
        np.degrees(
            np.arccos(
                np.clip(np.dot(beam_gt, beam_predicted), -1.0, 1.0)
            )
        )
    )
    azimuth_valid = tilt_gt >= float(settings["azimuth_valid_min_tilt_deg"])
    azimuth_error = (
        circular_difference_deg(azimuth_predicted, azimuth_gt)
        if azimuth_valid
        else float("nan")
    )
    symmetry_index = min(
        range(len(symmetries)),
        key=lambda index: float(
            np.linalg.norm(symmetries[index] - alignment["crystal_symmetry"])
        ),
    )
    return {
        "equivalent_misorientation_deg": float(
            alignment["equivalent_misorientation_deg"]
        ),
        "beam_direction_error_deg": beam_error,
        "tilt_error_deg": abs(tilt_predicted - tilt_gt),
        "azimuth_error_deg": azimuth_error,
        "ground_truth_tilt_deg": tilt_gt,
        "ground_truth_azimuth_deg": azimuth_gt,
        "predicted_aligned_tilt_deg": tilt_predicted,
        "predicted_aligned_azimuth_deg": azimuth_predicted,
        "azimuth_valid": bool(azimuth_valid),
        "friedel_used": bool(alignment["friedel_used"]),
        "crystal_symmetry_index": int(symmetry_index),
        "aligned_matrix_sample_to_crystal": aligned,
    }


def _prefix_summary(
    errors: np.ndarray,
    *,
    k: int,
    denominator: int,
    thresholds: Iterable[float],
) -> dict[str, float | int]:
    prefix = np.where(np.isfinite(errors[:, :k]), errors[:, :k], np.inf)
    best = np.min(prefix, axis=1)
    valid = best[np.isfinite(best)]
    result: dict[str, float | int] = {
        "denominator": int(denominator),
        "num_valid_predictions": int(len(valid)),
        "prediction_coverage": float(len(valid) / denominator) if denominator else 0.0,
        "median_deg_indexed": float(np.median(valid)) if len(valid) else float("nan"),
        "p95_deg_indexed": float(np.percentile(valid, 95)) if len(valid) else float("nan"),
        "max_deg_indexed": float(np.max(valid)) if len(valid) else float("nan"),
    }
    for threshold in thresholds:
        result[f"accuracy_within_{float(threshold):g}deg"] = (
            float(np.sum(valid <= float(threshold)) / denominator)
            if denominator
            else 0.0
        )
    return result


def summarize_topk_polar_errors(
    beam_error_deg: np.ndarray,
    tilt_error_deg: np.ndarray,
    azimuth_error_deg: np.ndarray,
    *,
    total_input_samples: int,
    total_azimuth_eligible_samples: int,
    beam_thresholds_deg: Iterable[float],
    azimuth_thresholds_deg: Iterable[float],
) -> list[dict[str, Any]]:
    beam = np.asarray(beam_error_deg, dtype=np.float64)
    tilt = np.asarray(tilt_error_deg, dtype=np.float64)
    azimuth = np.asarray(azimuth_error_deg, dtype=np.float64)
    if beam.ndim != 2 or beam.shape != tilt.shape or beam.shape != azimuth.shape:
        raise ValueError("polar error arrays must share shape [indexed_sample, rank]")
    if total_input_samples < len(beam) or total_input_samples <= 0:
        raise ValueError("invalid total_input_samples")
    if not 0 <= total_azimuth_eligible_samples <= total_input_samples:
        raise ValueError("invalid total_azimuth_eligible_samples")
    rows = []
    for k in range(1, beam.shape[1] + 1):
        rows.append(
            {
                "k": k,
                "beam_direction": _prefix_summary(
                    beam,
                    k=k,
                    denominator=total_input_samples,
                    thresholds=beam_thresholds_deg,
                ),
                "tilt": _prefix_summary(
                    tilt,
                    k=k,
                    denominator=total_input_samples,
                    thresholds=beam_thresholds_deg,
                ),
                "azimuth": _prefix_summary(
                    azimuth,
                    k=k,
                    denominator=total_azimuth_eligible_samples,
                    thresholds=azimuth_thresholds_deg,
                ),
            }
        )
    return rows
