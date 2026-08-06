from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from v6_polar_metrics import candidate_polar_errors, summarize_topk_polar_errors


SETTINGS = {
    "beam_axis_sample": [0.0, 0.0, 1.0],
    "polar_reference_crystal_axis": [0.0, 0.0, 1.0],
    "azimuth_reference_crystal_axis": [1.0, 0.0, 0.0],
    "azimuth_valid_min_tilt_deg": 1.0,
}


def matrix(tilt_deg: float, azimuth_deg: float) -> np.ndarray:
    return (
        Rotation.from_euler("z", azimuth_deg, degrees=True).as_matrix()
        @ Rotation.from_euler("y", tilt_deg, degrees=True).as_matrix()
    )


def test_polar_errors_use_aligned_orientation() -> None:
    gt = matrix(20.0, 30.0)
    predicted = matrix(22.0, 35.0)
    result = candidate_polar_errors(predicted, gt, [np.eye(3)], SETTINGS)
    assert np.isclose(result["tilt_error_deg"], 2.0, atol=1e-10)
    assert np.isclose(result["azimuth_error_deg"], 5.0, atol=1e-10)
    assert result["beam_direction_error_deg"] > 2.0
    assert result["azimuth_valid"] is True


def test_friedel_equivalent_candidate_has_zero_polar_error() -> None:
    gt = matrix(20.0, 30.0)
    friedel = np.diag([-1.0, -1.0, 1.0])
    result = candidate_polar_errors(gt @ friedel, gt, [np.eye(3)], SETTINGS)
    assert result["friedel_used"] is True
    assert result["equivalent_misorientation_deg"] < 2e-6
    assert result["beam_direction_error_deg"] < 1e-8
    assert result["tilt_error_deg"] < 1e-8
    assert result["azimuth_error_deg"] < 1e-8


def test_topk_polar_summary_keeps_failed_inputs_in_denominator() -> None:
    rows = summarize_topk_polar_errors(
        np.asarray([[3.0, 0.5], [8.0, 7.0]]),
        np.asarray([[2.0, 0.25], [6.0, 5.0]]),
        np.asarray([[10.0, 2.0], [np.nan, np.nan]]),
        total_input_samples=4,
        total_azimuth_eligible_samples=3,
        beam_thresholds_deg=[1.0, 2.0, 5.0],
        azimuth_thresholds_deg=[2.0, 5.0, 10.0],
    )
    assert rows[0]["beam_direction"]["accuracy_within_5deg"] == 0.25
    assert rows[1]["beam_direction"]["accuracy_within_1deg"] == 0.25
    assert rows[1]["azimuth"]["accuracy_within_2deg"] == 1.0 / 3.0
