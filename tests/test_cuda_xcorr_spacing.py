from __future__ import annotations

import unittest

import numpy as np

from cuda_xcorr_disk_adapter import _deduplicate_by_spacing


class CudaXcorrSpacingTest(unittest.TestCase):
    def test_keeps_brightest_candidate_inside_spacing_radius(self) -> None:
        rows = np.asarray([10.0, 11.0, 30.0])
        cols = np.asarray([10.0, 11.0, 30.0])
        scores = np.asarray([5.0, 4.0, 3.0])
        kept = _deduplicate_by_spacing(
            rows,
            cols,
            scores,
            min_spacing_px=5.0,
            max_num_peaks=10,
        )
        np.testing.assert_array_equal(kept, [0, 2])

    def test_applies_maximum_peak_count_after_score_ordering(self) -> None:
        kept = _deduplicate_by_spacing(
            np.asarray([0.0, 10.0, 20.0]),
            np.zeros(3),
            np.asarray([1.0, 3.0, 2.0]),
            min_spacing_px=1.0,
            max_num_peaks=2,
        )
        np.testing.assert_array_equal(kept, [1, 2])
