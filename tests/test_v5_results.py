import unittest

import numpy as np

from src.v5_results import (
    aggregate_group_keys,
    aggregate_topk_error_blocks,
    group_label,
    parse_clean_c_condition_stem,
)


class V5ResultsTests(unittest.TestCase):
    def test_parse_counted_and_noiseless_conditions(self):
        counted = parse_clean_c_condition_stem(
            "dose300_noise_empad_g2_16frames_repeat4_py4dstem"
        )
        self.assertEqual(counted["dose_electrons"], 300)
        self.assertEqual(counted["noise"], "empad_g2_16frames")
        self.assertEqual(counted["repeat"], 4)
        self.assertEqual(counted["frames"], 16)

        single_frame = parse_clean_c_condition_stem(
            "dose100_noise_empad_g2_1frame_repeat0_py4dstem"
        )
        self.assertEqual(single_frame["noise"], "empad_g2_1frame")
        self.assertEqual(single_frame["frames"], 1)

        noiseless = parse_clean_c_condition_stem(
            "dose1000000_noise_noiseless_py4dstem"
        )
        self.assertEqual(noiseless["repeat"], None)

    def test_aggregate_keeps_missing_inputs_in_accuracy_denominator(self):
        first = np.array([[0.5, 0.4], [4.0, 1.0]])
        second = np.array([[3.0, 2.5]])
        rows = aggregate_topk_error_blocks([(first, 3), (second, 2)])
        self.assertEqual(rows[0]["num_input_samples"], 5)
        self.assertAlmostEqual(rows[0]["prediction_coverage"], 3 / 5)
        self.assertAlmostEqual(rows[0]["accuracy_all_inputs_within_2deg"], 1 / 5)
        self.assertAlmostEqual(rows[1]["accuracy_all_inputs_within_2deg"], 2 / 5)

    def test_clean_e_inputs_are_aggregated_separately(self):
        label = {"track": "Clean-E", "input": "oracle"}
        self.assertEqual(
            aggregate_group_keys(label),
            [("track", "Clean-E"), ("clean_e_input", "oracle")],
        )
        self.assertEqual(
            group_label("clean_e_input", "py4dstem"),
            {
                "group_by": "clean_e_input",
                "track": "Clean-E",
                "input": "py4dstem",
            },
        )


if __name__ == "__main__":
    unittest.main()
