from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_evaluator_module():
    path = ROOT / "scripts" / "26_evaluate_v5_pyxem.py"
    spec = importlib.util.spec_from_file_location("evaluate_v5_pyxem", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PyxemEvaluationLabelsTests(unittest.TestCase):
    def test_clean_e_identifies_expectation_image_input(self) -> None:
        evaluator = load_evaluator_module()
        self.assertEqual(
            evaluator.labels_for_group(None, "clean_e"),
            [{"track": "Clean-E", "input": "expectation_image"}],
        )


if __name__ == "__main__":
    unittest.main()
