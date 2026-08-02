from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESOLVER = ROOT / "scripts" / "00_resolve_runtime_paths.py"


class RuntimePathTests(unittest.TestCase):
    def make_config(self, directory: Path) -> Path:
        path = directory / "runtime_paths.json"
        path.write_text(
            json.dumps(
                {
                    "schema": "or4d-runtime-paths-v1",
                    "clean_python": {
                        "environment_name": "or4d-clean",
                        "candidate_executables": [],
                    },
                    "reports": {
                        "v3": "artifacts/v3",
                        "v4": "artifacts/v4",
                        "v5": "artifacts/v5",
                    },
                    "v5": {"server_data_root": "/missing/server/data"},
                }
            ),
            encoding="utf-8",
        )
        return path

    def resolve(
        self,
        *arguments: str,
        config: Path,
        environment: dict[str, str] | None = None,
    ) -> str:
        result = subprocess.run(
            [sys.executable, str(RESOLVER), *arguments, "--config", str(config)],
            cwd=ROOT,
            env={**os.environ, **(environment or {})},
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    def test_report_directory_is_resolved_from_repository_root(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            config = self.make_config(Path(raw_directory))
            actual = self.resolve("report-dir", "--version", "v4", config=config)
        self.assertEqual(actual, str((ROOT / "artifacts/v4").resolve()))

    def test_clean_python_environment_override_is_authoritative(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            config = self.make_config(Path(raw_directory))
            actual = self.resolve(
                "clean-python",
                config=config,
                environment={"OR4D_CLEAN_PYTHON": sys.executable},
            )
        self.assertEqual(Path(actual), Path(sys.executable).resolve())

    def test_v5_data_root_can_be_overridden_for_another_machine(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            config = self.make_config(directory)
            actual = self.resolve(
                "v5-data-root",
                "--must-exist",
                config=config,
                environment={"OR4D_V5_DATA_ROOT": str(directory)},
            )
        self.assertEqual(Path(actual), directory.resolve())


if __name__ == "__main__":
    unittest.main()
