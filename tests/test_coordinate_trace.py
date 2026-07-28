from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from coordinate_trace import (  # noqa: E402
    crystal_to_sample_reciprocal,
    match_detector_peaks,
    reciprocal_cartesian_from_hkl,
)


class CoordinateTraceTest(unittest.TestCase):
    def test_hkl_to_sample_q_uses_row_basis_and_transpose(self) -> None:
        reciprocal = np.diag([0.5, 0.25, 0.1])
        hkl = np.array([2, -4, 3])
        g_crystal = reciprocal_cartesian_from_hkl(hkl, reciprocal)
        np.testing.assert_allclose(g_crystal, [1.0, -1.0, 0.3])

        matrix_sample_to_crystal = np.array(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        )
        g_sample = crystal_to_sample_reciprocal(
            g_crystal,
            matrix_sample_to_crystal,
        )
        np.testing.assert_allclose(g_sample, [-1.0, -1.0, 0.3])

    def test_peak_matching_is_one_to_one_and_tolerance_limited(self) -> None:
        observed = {
            "qx": np.array([0.0, 1.0]),
            "qy": np.array([0.0, 0.0]),
            "intensity": np.array([1.0, 0.5]),
            "hkl": np.array([[1, 0, 0], [2, 0, 0]]),
        }
        predicted = {
            "qx": np.array([0.01, 1.2]),
            "qy": np.array([0.0, 0.0]),
            "intensity": np.array([0.9, 0.4]),
            "hkl": np.array([[1, 0, 0], [2, 0, 0]]),
        }
        result = match_detector_peaks(observed, predicted, tolerance_Ainv=0.05)
        self.assertEqual(len(result["matches"]), 1)
        self.assertTrue(result["matches"][0]["raw_hkl_equal"])
        self.assertEqual(result["unmatched_observed_indices"], [1])
        self.assertEqual(result["unmatched_predicted_indices"], [1])


if __name__ == "__main__":
    unittest.main()
