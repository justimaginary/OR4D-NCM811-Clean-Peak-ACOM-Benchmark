from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clean_counting import (
    deterministic_count_seed,
    multinomial_count_image,
    poisson_count_image,
)


class CleanCountingTest(unittest.TestCase):
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
