from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "06_evaluate_submission.py"
SPEC = importlib.util.spec_from_file_location("evaluate_submission", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
EVALUATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVALUATOR)


class EvaluationContractTest(unittest.TestCase):
    def test_accepts_proper_rotation(self) -> None:
        matrix = EVALUATOR.validate_prediction_matrix("sample", np.eye(3))
        np.testing.assert_array_equal(matrix, np.eye(3))

    def test_rejects_bad_shape_nonfinite_reflection_and_shear(self) -> None:
        invalid = [
            np.eye(4),
            np.array([[np.nan, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
            np.diag([1.0, 1.0, -1.0]),
            np.array([[1.0, 0.1, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        ]
        for matrix in invalid:
            with self.subTest(matrix=matrix):
                with self.assertRaises(ValueError):
                    EVALUATOR.validate_prediction_matrix("sample", matrix)

    def test_rejects_duplicate_submission_ids(self) -> None:
        records = [{"sample_id": "same"}, {"sample_id": "same"}]
        with self.assertRaisesRegex(ValueError, "Duplicate sample_id"):
            EVALUATOR.unique_records_by_id(records, source="test")


if __name__ == "__main__":
    unittest.main()
