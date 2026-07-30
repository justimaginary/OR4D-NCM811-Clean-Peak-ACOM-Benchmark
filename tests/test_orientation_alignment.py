from __future__ import annotations

import sys
import unittest
import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from or4d_common import best_friedel_alignment  # noqa: E402

VISUALIZATION_PATH = ROOT / "scripts" / "12_write_coordinate_visualization.py"
VISUALIZATION_SPEC = importlib.util.spec_from_file_location(
    "coordinate_visualization",
    VISUALIZATION_PATH,
)
assert VISUALIZATION_SPEC is not None and VISUALIZATION_SPEC.loader is not None
VISUALIZATION = importlib.util.module_from_spec(VISUALIZATION_SPEC)
VISUALIZATION_SPEC.loader.exec_module(VISUALIZATION)


def rotation_z(degrees: float) -> np.ndarray:
    angle = np.deg2rad(degrees)
    cosine = np.cos(angle)
    sine = np.sin(angle)
    return np.array(
        [
            [cosine, -sine, 0.0],
            [sine, cosine, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


class OrientationAlignmentTest(unittest.TestCase):
    def test_recovers_crystal_symmetry_equivalent_representative(self) -> None:
        matrix_gt = rotation_z(17.0)
        symmetry = rotation_z(-120.0)
        small_error = rotation_z(0.022)
        matrix_predicted = symmetry @ matrix_gt @ small_error

        result = best_friedel_alignment(
            matrix_predicted,
            matrix_gt,
            [np.eye(3), rotation_z(120.0), symmetry],
        )

        self.assertAlmostEqual(result["raw_misorientation_deg"], 119.978, places=3)
        self.assertAlmostEqual(result["equivalent_misorientation_deg"], 0.022, places=3)
        self.assertFalse(result["friedel_used"])
        np.testing.assert_allclose(
            result["aligned_matrix"],
            matrix_gt @ small_error,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            result["symmetry_aligned_matrix"],
            result["aligned_matrix"],
            atol=1e-12,
        )
        np.testing.assert_allclose(result["crystal_symmetry"], symmetry, atol=1e-12)

    def test_v3_visualization_aligns_axes_and_maps_related_hkl(self) -> None:
        samples = VISUALIZATION.load_samples()
        sample = next(
            row
            for row in samples
            if row["sample_id"] == "clean_core_0970"
        )

        self.assertAlmostEqual(sample["orientation_error_deg"], 0.022073, places=5)
        self.assertAlmostEqual(sample["raw_misorientation_deg"], 120.010967, places=5)
        self.assertEqual(
            sample["best_crystal_symmetry_description"]["text"],
            "绕 crystal Z 轴 -120.000°",
        )
        self.assertFalse(sample["friedel_used"])
        np.testing.assert_allclose(
            sample["acom_aligned_matrix"],
            sample["standard_matrix"],
            atol=4e-4,
        )

        reflection = sample["observed"][0]
        self.assertEqual(reflection["hkl"], [0, -1, 4])
        self.assertEqual(reflection["acom_related_hkl"], [-1, 1, 4])
        np.testing.assert_allclose(
            reflection["q_acom_related_raw"],
            reflection["q_standard"],
            atol=2e-4,
        )
        np.testing.assert_allclose(
            reflection["q_acom_aligned_same_hkl"],
            reflection["q_standard"],
            atol=2e-4,
        )

        friedel = next(row for row in samples if row["label"] == "Friedel branch")
        self.assertEqual(friedel["sample_id"], "clean_core_0061")
        self.assertTrue(friedel["friedel_used"])
        self.assertGreater(friedel["strict_misorientation_deg"], 80.0)
        self.assertLess(friedel["orientation_error_deg"], 1.0)
        self.assertGreater(friedel["symmetry_step_misorientation_deg"], 170.0)
        self.assertFalse(
            np.allclose(
                friedel["acom_symmetry_aligned_matrix"],
                friedel["acom_aligned_matrix"],
                atol=1e-3,
            )
        )
        np.testing.assert_allclose(
            np.asarray(friedel["acom_aligned_matrix"])
            @ np.asarray(friedel["standard_matrix"]).T,
            np.eye(3),
            atol=1.2e-2,
        )


if __name__ == "__main__":
    unittest.main()
