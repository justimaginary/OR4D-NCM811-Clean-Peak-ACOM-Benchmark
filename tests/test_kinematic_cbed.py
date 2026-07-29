from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kinematic_cbed import (
    ProjectedReflectionSet,
    ReflectionLibrary,
    render_acom_matched_cbed,
    render_kinematic_cbed,
)


class KinematicCBEDTest(unittest.TestCase):
    def setUp(self) -> None:
        wavelength = 0.02
        k0 = 1.0 / wavelength
        gx = 0.4
        gz = np.sqrt(k0 * k0 - gx * gx) - k0
        self.library = ReflectionLibrary(
            g_crystal_Ainv=np.asarray(
                [[0.0, 0.0, 0.0], [gx, 0.0, gz], [-gx, 0.0, gz]],
                dtype=float,
            ),
            hkl=np.asarray([[0, 0, 0], [1, 0, 0], [-1, 0, 0]], dtype=np.int32),
            structure_factor=np.asarray([1.0 + 0j, 0.1 + 0.02j, 0.05 - 0.01j]),
            wavelength_A=wavelength,
        )
        self.config = {
            "gpts": [64, 64],
            "q_max_Ainv": 1.0,
            "convergence_semiangle_mrad": 1.0,
            "thickness_nm": 5.0,
            "oversampling": 2,
            "aperture_soft_edge_fraction": 0.1,
            "defocus_A": 0.0,
            "astigmatism_A": 0.0,
            "astigmatism_angle_deg": 0.0,
            "detector_psf_sigma_px": 0.0,
            "canonical_direct_beam_fraction": 0.9,
            "beam_center_offset_px": [0.0, 0.0],
            "physical_oracle_min_relative_intensity": 1e-6,
            "oracle_merge_distance_px": 1.0,
            "oracle_integration_radius_fraction": 0.9,
            "oracle_require_full_disk": True,
        }

    def test_rendered_centers_axes_and_probability_contract(self) -> None:
        result = render_kinematic_cbed(
            self.library,
            np.eye(3),
            self.config,
            k_max_Ainv=0.8,
        )
        self.assertEqual(result.expectation.shape, (64, 64))
        self.assertEqual(result.vacuum_probe.shape, (64, 64))
        np.testing.assert_allclose(result.oracle_qx_Ainv, [-0.4, 0.4], atol=1e-6)
        np.testing.assert_allclose(result.oracle_qy_Ainv, [0.0, 0.0], atol=1e-6)
        self.assertTrue(np.all(result.oracle_intensity_raw > 0.0))
        self.assertAlmostEqual(float(result.oracle_intensity_normalized.max()), 1.0)
        self.assertAlmostEqual(float(result.direct_expectation.sum()), 1.0, places=6)
        self.assertAlmostEqual(float(result.scattered_expectation.sum()), 1.0, places=6)
        self.assertAlmostEqual(float(result.expectation.sum()), 1.0, places=6)
        self.assertGreater(result.disk_radius_px, 1.0)
        self.assertTrue(np.all(np.diff(result.qx_axis_Ainv) > 0.0))
        self.assertTrue(np.all(np.diff(result.qy_axis_Ainv) < 0.0))
        self.assertEqual(result.oracle_candidate_reflection_count, 2)
        self.assertEqual(result.oracle_merged_disk_count, 2)
        self.assertEqual(result.oracle_rejected_edge_count, 0)

    def test_direct_beam_fraction_is_explicit_and_deterministic(self) -> None:
        first = render_kinematic_cbed(
            self.library,
            np.eye(3),
            self.config,
            k_max_Ainv=0.8,
            direct_beam_fraction=0.8,
        )
        second = render_kinematic_cbed(
            self.library,
            np.eye(3),
            self.config,
            k_max_Ainv=0.8,
            direct_beam_fraction=0.8,
        )
        np.testing.assert_array_equal(first.expectation, second.expectation)
        expected = 0.8 * first.direct_expectation + 0.2 * first.scattered_expectation
        np.testing.assert_allclose(first.expectation, expected, atol=2e-8)
        np.testing.assert_array_equal(
            first.oracle_intensity_raw, second.oracle_intensity_raw
        )

    def test_acom_matched_renderer_preserves_projected_support(self) -> None:
        reflections = ProjectedReflectionSet(
            qx_Ainv=np.asarray([-0.4, 0.4]),
            qy_Ainv=np.asarray([0.0, 0.0]),
            intensity=np.asarray([0.25, 1.0]),
            hkl=np.asarray([[-1, 0, 0], [1, 0, 0]], dtype=np.int32),
            wavelength_A=0.02,
        )
        result = render_acom_matched_cbed(
            reflections,
            self.config,
            k_max_Ainv=0.8,
        )
        np.testing.assert_allclose(result.oracle_qx_Ainv, [-0.4, 0.4], atol=1e-6)
        self.assertAlmostEqual(float(result.expectation.sum()), 1.0, places=6)
        self.assertEqual(result.oracle_candidate_reflection_count, 2)


if __name__ == "__main__":
    unittest.main()
