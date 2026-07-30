from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clean_counting import (
    add_gaussian_read_noise,
    deterministic_count_seed,
    deterministic_read_noise_seed,
    multinomial_count_image,
    noiseless_expected_count_image,
    poisson_count_image,
)


class CleanCountingTest(unittest.TestCase):
    def test_noiseless_expected_counts_preserve_dose(self) -> None:
        expectation = np.asarray([[1.0, 2.0], [3.0, 4.0]])
        image = noiseless_expected_count_image(expectation, 300)
        self.assertEqual(image.dtype, np.float32)
        self.assertAlmostEqual(float(image.sum()), 300.0, places=5)
        np.testing.assert_allclose(
            image, np.asarray([[30.0, 60.0], [90.0, 120.0]])
        )

    def test_read_noise_is_independent_and_reproducible(self) -> None:
        image = np.ones((8, 8), dtype=np.uint32)
        seed = deterministic_read_noise_seed(
            42, "clean_v5_core_0007", 300, "empad_g2_4frames", 2
        )
        first = add_gaussian_read_noise(
            image, 0.017333333333333333, np.random.default_rng(seed)
        )
        second = add_gaussian_read_noise(
            image, 0.017333333333333333, np.random.default_rng(seed)
        )
        np.testing.assert_array_equal(first, second)
        self.assertFalse(np.array_equal(first, image))
        self.assertEqual(first.dtype, np.float32)

    def test_fixed_total_and_seed_reproducibility(self) -> None:
        expectation = np.asarray([[0.1, 0.2], [0.3, 0.4]])
        first = multinomial_count_image(
            expectation, 10_000, np.random.default_rng(123)
        )
        second = multinomial_count_image(
            expectation, 10_000, np.random.default_rng(123)
        )
        self.assertEqual(first.dtype, np.uint32)
        self.assertEqual(int(first.sum()), 10_000)
        np.testing.assert_array_equal(first, second)

    def test_poisson_expected_total_and_seed_reproducibility(self) -> None:
        expectation = np.asarray([[0.1, 0.2], [0.3, 0.4]])
        seed = deterministic_count_seed(42, "clean_v5_core_0007", 300, 2)
        first = poisson_count_image(
            expectation, 300, np.random.default_rng(seed)
        )
        second = poisson_count_image(
            expectation, 300, np.random.default_rng(seed)
        )
        self.assertEqual(first.dtype, np.uint32)
        np.testing.assert_array_equal(first, second)
        self.assertGreater(int(first.sum()), 0)

    def test_seed_does_not_depend_on_sample_order(self) -> None:
        values = {
            sample_id: deterministic_count_seed(9, sample_id, 1000, 4)
            for sample_id in ("sample_b", "sample_a")
        }
        self.assertEqual(
            values["sample_a"],
            deterministic_count_seed(9, "sample_a", 1000, 4),
        )
        self.assertNotEqual(values["sample_a"], values["sample_b"])


if __name__ == "__main__":
    unittest.main()
