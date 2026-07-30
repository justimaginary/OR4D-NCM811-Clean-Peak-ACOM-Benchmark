from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from or4d_common import read_peak_h5, write_peak_h5  # noqa: E402


class PeakH5DiagnosticsTest(unittest.TestCase):
    def test_optional_diagnostics_preserve_pointlist_contract(self) -> None:
        samples = [
            {
                "sample_id": "sample_0",
                "qx": np.asarray([0.1, 0.2]),
                "qy": np.asarray([-0.2, 0.3]),
                "intensity": np.asarray([1.0, 0.5]),
                "peak_diagnostics": {
                    "initial_row_px": np.asarray([10.0, 20.0]),
                    "refined_row_px": np.asarray([10.1, 20.2]),
                },
                "sample_metadata": {
                    "detection_status": "ok",
                    "runtime_seconds": 0.25,
                },
            }
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "peaks.h5"
            write_peak_h5(path, samples, {"detector": "test"})
            loaded = read_peak_h5(path)
            np.testing.assert_allclose(loaded[0]["qx"], samples[0]["qx"])
            with h5py.File(path, "r") as h5:
                np.testing.assert_allclose(
                    h5["peaks/diagnostics/refined_row_px"][:],
                    [10.1, 20.2],
                )
                self.assertEqual(
                    h5["sample_diagnostics/detection_status"][0].decode(),
                    "ok",
                )


if __name__ == "__main__":
    unittest.main()
