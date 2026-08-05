#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

import h5py

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from or4d_common import load_config  # noqa: E402
from v6_observations import (  # noqa: E402
    build_observation_conditions,
    logical_observation_count,
    write_observation_shard,
)
from v6_runtime import enforce_server_write_scope  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build or assemble the compressed V6 Poisson observation store. "
            "Deterministic and read-noise layers remain losslessly factorized."
        )
    )
    parser.add_argument(
        "--config", type=Path, default=ROOT / "config" / "benchmark_v6.yaml"
    )
    parser.add_argument("--expectation-file", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--shard-index", type=int, action="append")
    parser.add_argument(
        "--all-shards",
        action="store_true",
        help="Build every shard sequentially; full server runs normally assign shard indices per worker.",
    )
    parser.add_argument(
        "--assemble-manifest",
        action="store_true",
        help="Validate all shard fragments and write the final manifest.",
    )
    parser.add_argument(
        "--smoke-samples",
        type=int,
        help="Restrict shard zero to this many samples for a bounded smoke test.",
    )
    return parser.parse_args()


def sha256_file(path: Path, block_mib: int = 8) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_mib * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def configured_sha256(path: Path, config: dict) -> str:
    return sha256_file(
        path.resolve(),
        block_mib=int(
            config["v6"]["observation_store"]["sha256_read_block_MiB"]
        ),
    )


def directory_size(path: Path) -> int:
    return sum(
        entry.stat().st_size
        for entry in path.rglob("*")
        if entry.is_file() and not entry.is_symlink()
    )


def resolved_paths(
    args: argparse.Namespace, config: dict
) -> tuple[Path, Path, Path]:
    paths = config["v6"]["paths"]
    expectation = Path(args.expectation_file or paths["expectation"]).resolve()
    output_dir = enforce_server_write_scope(
        args.output_dir or paths["observation_shards"], config
    )
    manifest = enforce_server_write_scope(
        args.manifest or paths["observation_manifest"], config
    )
    return expectation, output_dir, manifest


def shard_range(
    shard_index: int, sample_count: int, shard_size: int
) -> tuple[int, int]:
    start = shard_index * shard_size
    stop = min(start + shard_size, sample_count)
    if start < 0 or start >= sample_count:
        raise ValueError(f"Shard index {shard_index} is outside the dataset")
    return start, stop


def validate_quota(output_dir: Path, config: dict) -> None:
    root = Path(config["v6"]["paths"]["server_root"]).resolve()
    if root.exists():
        used = directory_size(root)
        soft_limit = int(
            float(config["v6"]["observation_store"]["soft_quota_GB"])
            * 1000**3
        )
        if used >= soft_limit:
            raise RuntimeError(
                f"V6 storage uses {used / 1000**3:.2f} GB, at or above "
                f"the configured {soft_limit / 1000**3:.2f} GB soft limit"
            )
    free = shutil.disk_usage(output_dir).free
    minimum_free = float(
        config["v6"]["observation_store"]["minimum_free_GiB"]
    )
    if free < minimum_free * 1024**3:
        raise RuntimeError(
            f"Only {free / 1024**3:.2f} GiB free at {output_dir}; "
            f"configured minimum is {minimum_free:.2f} GiB"
        )


def fragment_path(output_dir: Path, shard_index: int) -> Path:
    return output_dir / f"observations_{shard_index:04d}.json"


def data_path(output_dir: Path, shard_index: int) -> Path:
    return output_dir / f"observations_{shard_index:04d}.h5"


def build_selected_shards(
    args: argparse.Namespace,
    config: dict,
    config_path: Path,
    expectation: Path,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(expectation, "r") as h5:
        sample_count = int(h5["expectation/intensity"].shape[0])
    shard_size = int(config["v6"]["observation_store"]["physical_sample_shard_size"])
    shard_count = (sample_count + shard_size - 1) // shard_size
    if args.smoke_samples is not None:
        if args.smoke_samples <= 0:
            raise ValueError("--smoke-samples must be positive")
        selected = [0]
    elif args.all_shards:
        selected = list(range(shard_count))
    else:
        selected = list(args.shard_index or [])
    if not selected:
        raise ValueError("Select --shard-index, --all-shards, or --assemble-manifest")
    resume = bool(config["v6"]["runtime"]["resume"])
    exact_config_sha = configured_sha256(config_path, config)
    validate_quota(output_dir, config)
    for index in selected:
        start, stop = shard_range(index, sample_count, shard_size)
        if args.smoke_samples is not None:
            stop = min(stop, start + args.smoke_samples)
        output = data_path(output_dir, index)
        fragment = fragment_path(output_dir, index)
        if resume and output.exists() and fragment.exists():
            saved = json.loads(fragment.read_text(encoding="utf-8"))
            if (
                saved.get("config_sha256") == exact_config_sha
                and saved.get("file_sha256") == configured_sha256(output, config)
                and int(saved.get("sample_start", -1)) == start
                and int(saved.get("sample_stop", -1)) == stop
            ):
                print(f"resume: shard {index:04d} already verified")
                continue
            raise RuntimeError(
                f"Existing shard {index:04d} failed resume validation; "
                "move it aside before rebuilding"
            )
        minimum_free = float(
            config["v6"]["observation_store"]["minimum_free_GiB"]
        )
        if shutil.disk_usage(output_dir).free < minimum_free * 1024**3:
            raise RuntimeError(
                f"Less than the configured {minimum_free:.2f} GiB remains "
                f"at {output_dir}"
            )
        report = write_observation_shard(
            expectation,
            output,
            config,
            sample_start=start,
            sample_stop=stop,
        )
        report.update(
            {
                "shard_index": index,
                "file_sha256": configured_sha256(output, config),
                "file_size_bytes": output.stat().st_size,
                "config_sha256": exact_config_sha,
            }
        )
        fragment.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(
            f"built shard {index:04d}: samples [{start}, {stop}), "
            f"{output.stat().st_size / 1024**2:.2f} MiB"
        )


def assemble_manifest(
    config: dict,
    config_path: Path,
    expectation: Path,
    output_dir: Path,
    manifest: Path,
) -> None:
    with h5py.File(expectation, "r") as h5:
        sample_count, ny, nx = (
            int(value) for value in h5["expectation/intensity"].shape
        )
    shard_size = int(config["v6"]["observation_store"]["physical_sample_shard_size"])
    shard_count = (sample_count + shard_size - 1) // shard_size
    exact_config_sha = configured_sha256(config_path, config)
    fragments = []
    expected_start = 0
    for index in range(shard_count):
        fragment_file = fragment_path(output_dir, index)
        if not fragment_file.exists():
            raise FileNotFoundError(f"Missing shard fragment: {fragment_file}")
        fragment = json.loads(fragment_file.read_text(encoding="utf-8"))
        output = data_path(output_dir, index)
        if fragment["config_sha256"] != exact_config_sha:
            raise RuntimeError(f"Config hash mismatch in shard {index:04d}")
        if int(fragment["sample_start"]) != expected_start:
            raise RuntimeError(f"Sample gap before shard {index:04d}")
        if fragment["file_sha256"] != configured_sha256(output, config):
            raise RuntimeError(f"File hash mismatch in shard {index:04d}")
        expected_start = int(fragment["sample_stop"])
        fragments.append(fragment)
    if expected_start != sample_count:
        raise RuntimeError(
            f"Shard coverage ends at {expected_start}, expected {sample_count}"
        )
    area_values = {
        round(float(row["effective_illumination_area_A2"]), 12)
        for row in fragments
    }
    if len(area_values) != 1:
        raise RuntimeError("Effective illumination area differs between shards")
    payload = {
        "schema_version": config["v6"]["schema_version"],
        "dataset_id": config["dataset"]["id"],
        "config_path": str(config_path.resolve()),
        "config_sha256": exact_config_sha,
        "expectation_file": str(expectation),
        "expectation_file_sha256": configured_sha256(expectation, config),
        "sample_count": sample_count,
        "image_shape": [ny, nx],
        "conditions_per_sample": len(build_observation_conditions(config)),
        "logical_observation_count": logical_observation_count(config, sample_count),
        "conditions": [
            asdict(value) for value in build_observation_conditions(config)
        ],
        "effective_illumination_area_A2": fragments[0][
            "effective_illumination_area_A2"
        ],
        "expected_total_electrons": fragments[0]["expected_total_electrons"],
        "storage": config["v6"]["observation_store"],
        "shards": fragments,
    }
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"manifest: {manifest}")


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
    expectation, output_dir, manifest = resolved_paths(args, config)
    if args.assemble_manifest:
        assemble_manifest(
            config, config_path, expectation, output_dir, manifest
        )
    else:
        build_selected_shards(
            args, config, config_path, expectation, output_dir
        )


if __name__ == "__main__":
    main()
