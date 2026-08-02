import unittest

import numpy as np

from src.topk_evaluation import summarize_topk_errors


class TopKEvaluationTests(unittest.TestCase):
    def test_topk_uses_prefix_minimum_and_keeps_failure_denominator(self):
        errors = np.array(
            [
                [3.0, 1.5, 0.5],
                [8.0, 7.0, 6.0],
            ]
        )
        rows = summarize_topk_errors(errors, total_input_samples=4)
        self.assertEqual([row["k"] for row in rows], [1, 2, 3])
        self.assertEqual(rows[0]["prediction_coverage"], 0.5)
        self.assertEqual(rows[0]["accuracy_all_inputs_within_2deg"], 0.0)
        self.assertEqual(rows[1]["accuracy_all_inputs_within_2deg"], 0.25)
        self.assertEqual(rows[2]["accuracy_all_inputs_within_1deg"], 0.25)

    def test_invalid_candidates_do_not_become_hits(self):
        rows = summarize_topk_errors(
            np.array([[np.nan, 0.25], [np.nan, np.nan]]),
            total_input_samples=2,
        )
        self.assertEqual(rows[0]["num_valid_predictions"], 0)
        self.assertEqual(rows[1]["num_valid_predictions"], 1)
        self.assertEqual(rows[1]["accuracy_all_inputs_within_1deg"], 0.5)


if __name__ == "__main__":
    unittest.main()
