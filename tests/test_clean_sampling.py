from __future__ import annotations

import copy
import sys
import unittest
from collections import Counter
from pathlib import Path

import numpy as np
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clean_sampling import build_clean_orientation_records  # noqa: E402
from or4d_common import (  # noqa: E402
    cif_path,
    load_config,
    proper_point_group_rotations,
    quaternion_wxyz_to_matrix,
    sobol_so3_quaternions,
)


class CleanSamplingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config()
        cls.structure = Structure.from_file(cif_path(cls.config))

    def test_point_group_filter_rejects_improper_operations(self) -> None:
        operations = SpacegroupAnalyzer(
            self.structure
        ).get_point_group_operations(cartesian=True)
        raw_proper = [
            operation
            for operation in operations
            if np.linalg.det(operation.rotation_matrix) > 0.0
        ]
        rotations = proper_point_group_rotations(self.structure)
        self.assertEqual(len(rotations), len(raw_proper))
        self.assertLess(len(rotations), len(operations))
        self.assertTrue(
            all(np.isclose(np.linalg.det(rotation), 1.0) for rotation in rotations)
        )

    def test_sobol_so3_is_deterministic_and_proper(self) -> None:
        first = sobol_so3_quaternions(16, scramble=True, seed=20260728)
        second = sobol_so3_quaternions(16, scramble=True, seed=20260728)
        np.testing.assert_allclose(first, second, atol=0.0, rtol=0.0)
        np.testing.assert_allclose(np.linalg.norm(first, axis=1), 1.0, atol=1e-12)
        for quaternion in first:
            matrix = quaternion_wxyz_to_matrix(quaternion)
            np.testing.assert_allclose(matrix.T @ matrix, np.eye(3), atol=1e-12)
            self.assertAlmostEqual(float(np.linalg.det(matrix)), 1.0, places=12)

    def test_small_manifest_has_expected_roles_and_unique_ids(self) -> None:
        config = copy.deepcopy(self.config)
        config["clean_sampling"]["headline_core"]["count"] = 8
        config["dataset"]["expected_sample_counts"]["headline_core"] = 8
        config["dataset"]["expected_num_orientations"] = 65
        records = build_clean_orientation_records(config, self.structure)

        counts = Counter(record["sample_role"] for record in records)
        self.assertEqual(
            counts,
            Counter(
                {
                    "legacy_smoke": 17,
                    "headline_core": 8,
                    "acom_grid_probe": 40,
                }
            ),
        )
        ids = [record["orientation_id"] for record in records]
        self.assertEqual(len(ids), len(set(ids)))


if __name__ == "__main__":
    unittest.main()
