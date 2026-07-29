from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clean_counting import multinomial_count_image


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


if __name__ == "__main__":
    unittest.main()
