import unittest

import numpy as np

from src.pyxem_template_adapter import prepare_cartesian_patterns


class PyxemTemplateAdapterTests(unittest.TestCase):
    def test_even_detector_center_is_padded_without_interpolation(self):
        image = np.zeros((1, 512, 512), dtype=np.float32)
        image[0, 255:257, 255:257] = 1
        prepared = prepare_cartesian_patterns(
            image,
            q_pixel_size_Ainv=0.00625,
            central_beam_exclusion_Ainv=0,
        )
        self.assertEqual(prepared.shape, (1, 513, 513))
        np.testing.assert_array_equal(
            prepared[0, 256:258, 256:258], np.ones((2, 2))
        )

    def test_direct_beam_mask_uses_reciprocal_units(self):
        image = np.ones((1, 512, 512), dtype=np.float32)
        prepared = prepare_cartesian_patterns(
            image,
            q_pixel_size_Ainv=0.01,
            central_beam_exclusion_Ainv=0.03,
        )
        self.assertEqual(prepared[0, 256, 256], 0)
        self.assertEqual(prepared[0, 260, 256], 1)


if __name__ == "__main__":
    unittest.main()
