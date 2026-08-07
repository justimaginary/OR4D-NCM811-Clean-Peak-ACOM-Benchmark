from __future__ import annotations

import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from or4d_common import load_config
import v6_runtime


def _config():
    return copy.deepcopy(load_config("config/benchmark_v6.yaml"))


def _nvidia_run_factory(*, processes: str):
    gpu_rows = "2, GPU-test, 944, 13\n"

    def fake_run(command, **_kwargs):
        if "--query-gpu=index,uuid,memory.used,utilization.gpu" in command:
            return SimpleNamespace(stdout=gpu_rows)
        if "--query-compute-apps=gpu_uuid,pid" in command:
            return SimpleNamespace(stdout=processes)
        raise AssertionError(command)

    return fake_run


class V6RuntimeGpuSharingTests(unittest.TestCase):
    def test_acom_allows_second_owned_matcher_process(self):
        config = _config()
        with (
            patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "2"}),
            patch.object(
                v6_runtime.subprocess,
                "run",
                _nvidia_run_factory(processes="GPU-test, 123\n"),
            ),
            patch.object(
                v6_runtime, "_process_owned_by_current_user", return_value=True
            ),
            patch.object(
                v6_runtime,
                "_process_command",
                return_value="python scripts/07_run_acom_baseline.py --cuda",
            ),
        ):
            self.assertEqual(
                v6_runtime.require_empty_bound_gpu(config, workload="acom"),
                "2",
            )

    def test_acom_rejects_third_process(self):
        config = _config()
        with (
            patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "2"}),
            patch.object(
                v6_runtime.subprocess,
                "run",
                _nvidia_run_factory(
                    processes="GPU-test, 123\nGPU-test, 456\n"
                ),
            ),
            patch.object(
                v6_runtime, "_process_owned_by_current_user", return_value=True
            ),
            patch.object(
                v6_runtime,
                "_process_command",
                return_value="python scripts/07_run_acom_baseline.py --cuda",
            ),
        ):
            with self.assertRaisesRegex(
                RuntimeError, "cannot accept another ACOM process"
            ):
                v6_runtime.require_empty_bound_gpu(config, workload="acom")

    def test_detection_still_requires_an_empty_gpu(self):
        config = _config()
        with (
            patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "2"}),
            patch.object(
                v6_runtime.subprocess,
                "run",
                _nvidia_run_factory(processes="GPU-test, 123\n"),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "is not empty"):
                v6_runtime.require_empty_bound_gpu(config)


if __name__ == "__main__":
    unittest.main()
