import unittest

import numpy as np

from v5_disk_recovery import match_peak_arrays, summarize_matches


class V5DiskRecoveryTest(unittest.TestCase):
    def test_one_to_one_match_and_high_angle_count(self):
        oracle = np.array([[0.0, 0.0], [1.2, 0.0], [1.4, 0.0]])
        detected = np.array([[0.01, 0.0], [1.21, 0.0], [2.0, 0.0]])
        match = match_peak_arrays(
            oracle,
            detected,
            tolerance_Ainv=0.05,
            high_angle_Ainv=1.1,
        )
        self.assertEqual(match.true_positive, 2)
        self.assertEqual(match.high_angle_oracle_count, 2)
        self.assertEqual(match.high_angle_true_positive, 1)
        summary = summarize_matches([match], q_pixel_Ainv=0.01)
        self.assertAlmostEqual(summary["precision"], 2 / 3)
        self.assertAlmostEqual(summary["recall"], 2 / 3)
        self.assertAlmostEqual(summary["high_angle_recall"], 0.5)
        self.assertAlmostEqual(summary["position_rmse_px"], 1.0)

    def test_conflict_is_reported_and_not_double_counted(self):
        match = match_peak_arrays(
            np.array([[0.0, 0.0], [0.02, 0.0]]),
            np.array([[0.01, 0.0]]),
            tolerance_Ainv=0.02,
            high_angle_Ainv=1.0,
        )
        self.assertTrue(match.ambiguous)
        self.assertEqual(match.true_positive, 1)


if __name__ == "__main__":
    unittest.main()
