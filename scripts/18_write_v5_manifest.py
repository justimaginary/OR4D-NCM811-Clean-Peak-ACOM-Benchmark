#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import h5py

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from or4d_common import load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write a compact reproducibility manifest for V5 artifacts."
    )
    parser.add_argument(
        "--artifact",
        action="append",
        required=True,
        help="Artifact mapping NAME=PATH; may be repeated.",
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_output(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def h5_metadata(path: Path) -> dict:
    if path.suffix.lower() not in {".h5", ".hdf5"}:
        return {}
    with h5py.File(path, "r") as h5:
        return {
            "hdf5_root_keys": sorted(h5.keys()),
            "track": str(h5.attrs.get("track", "")),
            "counting_model": str(h5.attrs.get("counting_model", "")),
            "git_commit_attribute": str(h5.attrs.get("git_commit", "")),
        }


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    artifacts = []
    for item in args.artifact:
        name, separator, raw_path = item.partition("=")
        if not separator or not name or not raw_path:
            raise ValueError(f"Artifact must be NAME=PATH: {item}")
        path = Path(raw_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        try:
            relative_path = str(path.relative_to(data_root))
        except ValueError:
            relative_path = str(path)
        artifacts.append(
            {
                "artifact_id": name,
                "server_relative_path": relative_path,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                **h5_metadata(path),
            }
        )

    overlay = os.environ.get("OR4D_CONFIG")
    config_paths = [ROOT / "config" / "benchmark.yaml"]
    if overlay:
        overlay_path = Path(overlay)
        if not overlay_path.is_absolute():
            overlay_path = ROOT / overlay_path
        if overlay_path.resolve() != config_paths[0].resolve():
            config_paths.append(overlay_path.resolve())
    config = load_config()
    manifest = {
        "schema_version": config.get("v5", {}).get(
            "schema_version", "unknown"
        ),
        "dataset_id": config["dataset"]["id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_branch": git_output("branch", "--show-current"),
        "git_status_porcelain": git_output("status", "--short"),
        "config_files": [
            {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256_file(path),
            }
            for path in config_paths
        ],
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "hostname": platform.node(),
            "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV", ""),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS", ""),
        },
        "data_root": str(data_root),
        "artifacts": artifacts,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Manifest: {output}")


if __name__ == "__main__":
    main()
