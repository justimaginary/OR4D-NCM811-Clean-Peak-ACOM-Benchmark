from __future__ import annotations

import unittest

import numpy as np

from py4dstem_disk_adapter import _normalize_detection_images


class Py4DSTEMDetectionNormalizationTest(unittest.TestCase):
    def test_max_normalization_is_per_pattern_and_scale_invariant(self) -> None:
        images = np.asarray(
            [
                [[0.0, 2.0], [1.0, 0.5]],
                [[0.0, 20.0], [10.0, 5.0]],
            ],
            dtype=np.float32,
        )
        normalized = _normalize_detection_images(
            images, {"normalize_for_detection": "max"}
        )
        np.testing.assert_allclose(normalized[0], normalized[1])
        np.testing.assert_allclose(normalized.max(axis=(1, 2)), 1.0)

    def test_none_preserves_values(self) -> None:
        image = np.asarray([[1.0, 2.0]], dtype=np.float32)
        np.testing.assert_array_equal(
            _normalize_detection_images(
                image, {"normalize_for_detection": "none"}
            ),
            image,
        )
