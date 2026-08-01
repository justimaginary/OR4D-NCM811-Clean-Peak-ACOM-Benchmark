"""Compact, deterministic peak-recovery metrics for the V5 Clean benchmark."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class PeakMatch:
    oracle_count: int
    detected_count: int
    true_positive: int
    high_angle_oracle_count: int
    high_angle_true_positive: int
    distances_Ainv: np.ndarray
    ambiguous: bool


def match_peak_arrays(
    oracle_xy: np.ndarray,
    detected_xy: np.ndarray,
    *,
    tolerance_Ainv: float,
    high_angle_Ainv: float,
) -> PeakMatch:
    """Match peaks one-to-one inside a fixed detector-space tolerance.

    Candidate pairs come from a radius query and are consumed in increasing
    distance order. This is the same deterministic nearest-neighbour rule used
    by the V5 HTML overlays, without constructing a dense O(N*M) matrix.
    ``ambiguous`` records whether any peak had more than one candidate inside
    the tolerance, so the aggregate report can audit the sparse-match regime.
    """

    oracle = np.asarray(oracle_xy, dtype=np.float64).reshape(-1, 2)
    detected = np.asarray(detected_xy, dtype=np.float64).reshape(-1, 2)
    high_mask = (
        np.hypot(oracle[:, 0], oracle[:, 1]) >= float(high_angle_Ainv)
        if len(oracle)
        else np.zeros(0, dtype=bool)
    )
    if not len(oracle) or not len(detected):
        return PeakMatch(
            oracle_count=len(oracle),
            detected_count=len(detected),
            true_positive=0,
            high_angle_oracle_count=int(high_mask.sum()),
            high_angle_true_positive=0,
            distances_Ainv=np.empty(0, dtype=np.float64),
            ambiguous=False,
        )

    neighbours = cKDTree(detected).query_ball_point(
        oracle, r=float(tolerance_Ainv)
    )
    detected_degree = np.zeros(len(detected), dtype=np.int32)
    pairs: list[tuple[float, int, int]] = []
    ambiguous = any(len(indices) > 1 for indices in neighbours)
    for oracle_index, indices in enumerate(neighbours):
        for detected_index in indices:
            detected_degree[detected_index] += 1
            delta = oracle[oracle_index] - detected[detected_index]
            pairs.append(
                (float(np.dot(delta, delta)), oracle_index, detected_index)
            )
    ambiguous = ambiguous or bool(np.any(detected_degree > 1))
    pairs.sort(key=lambda row: (row[0], row[1], row[2]))
    used_oracle: set[int] = set()
    used_detected: set[int] = set()
    matched_oracle: list[int] = []
    distances: list[float] = []
    for distance2, oracle_index, detected_index in pairs:
        if oracle_index in used_oracle or detected_index in used_detected:
            continue
        used_oracle.add(oracle_index)
        used_detected.add(detected_index)
        matched_oracle.append(oracle_index)
        distances.append(distance2**0.5)

    return PeakMatch(
        oracle_count=len(oracle),
        detected_count=len(detected),
        true_positive=len(distances),
        high_angle_oracle_count=int(high_mask.sum()),
        high_angle_true_positive=int(
            high_mask[np.asarray(matched_oracle, dtype=int)].sum()
        ),
        distances_Ainv=np.asarray(distances, dtype=np.float64),
        ambiguous=ambiguous,
    )


def summarize_matches(
    matches: list[PeakMatch], *, q_pixel_Ainv: float
) -> dict:
    oracle_count = sum(row.oracle_count for row in matches)
    detected_count = sum(row.detected_count for row in matches)
    true_positive = sum(row.true_positive for row in matches)
    high_count = sum(row.high_angle_oracle_count for row in matches)
    high_true_positive = sum(
        row.high_angle_true_positive for row in matches
    )
    distance_parts = [row.distances_Ainv for row in matches if row.true_positive]
    distances_px = (
        np.concatenate(distance_parts) / float(q_pixel_Ainv)
        if distance_parts
        else np.empty(0, dtype=np.float64)
    )
    sample_count = len(matches)
    return {
        "num_samples": sample_count,
        "oracle_peaks": oracle_count,
        "detected_peaks": detected_count,
        "true_positive": true_positive,
        "false_positive": detected_count - true_positive,
        "false_negative": oracle_count - true_positive,
        "precision": true_positive / detected_count if detected_count else 0.0,
        "recall": true_positive / oracle_count if oracle_count else 1.0,
        "position_rmse_px": (
            float(np.sqrt(np.mean(np.square(distances_px))))
            if len(distances_px)
            else None
        ),
        "position_p95_px": (
            float(np.percentile(distances_px, 95))
            if len(distances_px)
            else None
        ),
        "high_angle_recall": (
            high_true_positive / high_count if high_count else 1.0
        ),
        "sample_detection_coverage": (
            sum(row.detected_count > 0 for row in matches) / sample_count
            if sample_count
            else 0.0
        ),
        "detected_peaks_per_sample": (
            detected_count / sample_count if sample_count else 0.0
        ),
        "false_positive_per_sample": (
            (detected_count - true_positive) / sample_count
            if sample_count
            else 0.0
        ),
        "false_negative_per_sample": (
            (oracle_count - true_positive) / sample_count
            if sample_count
            else 0.0
        ),
        "ambiguous_sample_count": sum(row.ambiguous for row in matches),
    }
