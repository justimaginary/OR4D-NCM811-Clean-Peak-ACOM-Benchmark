from __future__ import annotations

import unittest

import numpy as np

from autodisk_adapter import (
    _deduplicate_candidates,
    _sample_bilinear,
    _sample_cubic,
)


class AutoDiskSubpixelTest(unittest.TestCase):
    def test_deduplication_stops_at_same_score_ordered_top_n(self) -> None:
        candidates = [
            (0.0, 0.0, 10.0),
            (0.5, 0.5, 9.0),
            (10.0, 0.0, 8.0),
            (20.0, 0.0, 7.0),
            (30.0, 0.0, 6.0),
        ]
        full = _deduplicate_candidates(candidates, min_spacing_px=2.0)
        limited = _deduplicate_candidates(
            candidates,
            min_spacing_px=2.0,
            max_num_peaks=2,
        )

        self.assertEqual(limited, full[:2])

    def test_cubic_sampling_can_resolve_a_subpixel_maximum(self) -> None:
        rows, cols = np.mgrid[:17, :17]
        expected_row = 8.3
        expected_col = 7.6
        response = np.exp(
            -(
                (rows - expected_row) ** 2 + (cols - expected_col) ** 2
            )
            / (2.0 * 2.0**2)
        )
        offsets = np.arange(-1.0, 1.01, 0.1)
        dy, dx = np.meshgrid(offsets, offsets, indexing="ij")
        sample_rows = 8.0 + dy.ravel()
        sample_cols = 8.0 + dx.ravel()

        bilinear_best = int(
            np.argmax(_sample_bilinear(response, sample_rows, sample_cols))
        )
        cubic_best = int(
            np.argmax(_sample_cubic(response, sample_rows, sample_cols))
        )

        self.assertEqual(sample_rows[bilinear_best], 8.0)
        self.assertEqual(sample_cols[bilinear_best], 8.0)
        self.assertLessEqual(abs(sample_rows[cubic_best] - expected_row), 0.1)
        self.assertLessEqual(abs(sample_cols[cubic_best] - expected_col), 0.1)
