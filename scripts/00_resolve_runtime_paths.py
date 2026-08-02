#!/usr/bin/env python3
"""Resolve portable runtime paths without importing benchmark dependencies."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config/runtime_paths.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("clean-python", "report-dir", "v5-data-root", "show"),
    )
    parser.add_argument("--version", choices=("v3", "v4", "v5"))
    parser.add_argument("--must-exist", action="store_true")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(os.environ.get("OR4D_RUNTIME_PATHS_FILE", DEFAULT_CONFIG)),
    )
    return parser.parse_args()


def load_paths(path: Path) -> dict:
    payload = json.loads(path.resolve().read_text(encoding="utf-8"))
    if payload.get("schema") != "or4d-runtime-paths-v1":
        raise ValueError(f"unsupported runtime path schema in {path}")
    return payload


def conda_environment_python(environment_name: str) -> Path | None:
    conda = shutil.which("conda")
    if conda is None:
        return None
    try:
        payload = json.loads(
            subprocess.run(
                [conda, "info", "--json"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        )
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError):
        return None
    for raw_prefix in payload.get("envs", []):
        prefix = Path(raw_prefix)
        if prefix.name == environment_name:
            candidate = prefix / "bin/python"
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return candidate.resolve()
    return None


def clean_python(config: dict) -> Path:
    override = os.environ.get("OR4D_CLEAN_PYTHON", "").strip()
    candidates = []
    if override:
        candidates.append(Path(override))
    candidates.extend(
        Path(value)
        for value in config["clean_python"]["candidate_executables"]
    )
    active_prefix = os.environ.get("CONDA_PREFIX", "").strip()
    if active_prefix and Path(active_prefix).name == config["clean_python"]["environment_name"]:
        candidates.insert(0, Path(active_prefix) / "bin/python")
    for candidate in candidates:
        expanded = candidate.expanduser()
        if expanded.is_file() and os.access(expanded, os.X_OK):
            return expanded.resolve()
    discovered = conda_environment_python(
        str(config["clean_python"]["environment_name"])
    )
    if discovered is not None:
        return discovered
    checked = ", ".join(str(path) for path in candidates) or "none"
    raise FileNotFoundError(
        "or4d-clean Python was not found. Set OR4D_CLEAN_PYTHON to the "
        f"environment's executable. Checked: {checked}"
    )


def report_dir(config: dict, version: str | None) -> Path:
    if version is None:
        raise ValueError("report-dir requires --version")
    override = os.environ.get(f"OR4D_REPORT_{version.upper()}_DIR", "").strip()
    raw = Path(override or config["reports"][version]).expanduser()
    return (raw if raw.is_absolute() else ROOT / raw).resolve()


def v5_data_root(config: dict, must_exist: bool) -> Path:
    override = os.environ.get("OR4D_V5_DATA_ROOT", "").strip()
    path = Path(override or config["v5"]["server_data_root"]).expanduser().resolve()
    if must_exist and not path.is_dir():
        raise FileNotFoundError(
            f"V5 data root does not exist: {path}. Set OR4D_V5_DATA_ROOT."
        )
    return path


def main() -> None:
    args = parse_args()
    config = load_paths(args.config)
    if args.command == "clean-python":
        print(clean_python(config))
    elif args.command == "report-dir":
        print(report_dir(config, args.version))
    elif args.command == "v5-data-root":
        print(v5_data_root(config, args.must_exist))
    else:
        print(
            json.dumps(
                {
                    "config": str(args.config.resolve()),
                    "clean_python": str(clean_python(config)),
                    "reports": {
                        version: str(report_dir(config, version))
                        for version in ("v3", "v4", "v5")
                    },
                    "v5_data_root": str(v5_data_root(config, False)),
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
