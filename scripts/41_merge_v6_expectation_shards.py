#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from or4d_common import (  # noqa: E402
    load_config,
    read_jsonl,
    read_peak_h5,
    write_peak_h5,
)
from v6_runtime import enforce_server_write_scope  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and merge V6 First-Born image/oracle shards."
    )
    parser.add_argument(
        "--config", type=Path, default=ROOT / "config" / "benchmark_v6.yaml"
    )
    parser.add_argument("--orientation-file", type=Path)
    parser.add_argument("--image-shard", type=Path, action="append", required=True)
    parser.add_argument("--oracle-shard", type=Path, action="append", required=True)
    parser.add_argument(
        "--reflection-shard", type=Path, action="append", required=True
    )
    parser.add_argument("--output-image", type=Path)
    parser.add_argument("--output-oracle", type=Path)
    parser.add_argument("--output-reflections", type=Path)
    parser.add_argument("--manifest", type=Path)
    return parser.parse_args()


def decode(values: np.ndarray) -> list[str]:
    return [
        value.decode() if isinstance(value, bytes) else str(value)
        for value in values
    ]


def sha256_file(path: Path, block_mib: int) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_mib * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_attrs(source: h5py.AttributeManager, target: h5py.AttributeManager) -> None:
    for name, value in source.items():
        target[name] = value


def validate_shard_lists(args: argparse.Namespace, expected_count: int) -> None:
    counts = {
        "image": len(args.image_shard),
        "oracle": len(args.oracle_shard),
        "reflection": len(args.reflection_shard),
    }
    if len(set(counts.values())) != 1:
        raise ValueError(f"V6 expectation shard list lengths differ: {counts}")
    if counts["image"] != expected_count:
        raise ValueError(
            f"Expected {expected_count} V6 expectation shards, got {counts['image']}"
        )


def merge_images(
    shards: list[Path],
    output_path: Path,
    expected_sample_ids: list[str],
    config: dict,
) -> list[str]:
    handles = [h5py.File(path, "r") for path in shards]
    try:
        shard_ids = [decode(handle["sample_id"][:]) for handle in handles]
        merged_ids = [sample_id for values in shard_ids for sample_id in values]
        if merged_ids != expected_sample_ids:
            raise ValueError("Image shard sample order differs from V6 orientations")
        image_shapes = {
            tuple(handle["expectation/intensity"].shape[1:]) for handle in handles
        }
        if len(image_shapes) != 1:
            raise ValueError(f"Image shard detector shapes differ: {image_shapes}")
        ny, nx = next(iter(image_shapes))
        expected_shape = tuple(int(value) for value in config["clean_image"]["gpts"])
        if (ny, nx) != expected_shape:
            raise ValueError(
                f"Image shard shape {(ny, nx)} differs from config {expected_shape}"
            )
        first = handles[0]
        for other in handles[1:]:
            for name in (
                "detector/qx_Ainv",
                "detector/qy_Ainv",
                "detector/vacuum_probe",
                "detector/valid_mask",
            ):
                if not np.array_equal(first[name][:], other[name][:]):
                    raise ValueError(f"Image shard detector field differs: {name}")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_suffix(output_path.suffix + ".partial")
        compression = str(config["clean_image"].get("compression", "gzip"))
        compression_level = int(config["clean_image"].get("compression_level", 4))
        with h5py.File(temporary, "w") as output:
            copy_attrs(first.attrs, output.attrs)
            output.attrs["merged_v6_shards"] = json.dumps(
                [str(path) for path in shards]
            )
            output.create_dataset(
                "sample_id",
                data=np.asarray(merged_ids, dtype=h5py.string_dtype("utf-8")),
            )
            first.copy("detector", output)
            orientation = output.create_group("orientation")
            for name in first["orientation"]:
                values = np.concatenate(
                    [handle[f"orientation/{name}"][:] for handle in handles], axis=0
                )
                orientation.create_dataset(name, data=values, compression="gzip")
            images = output.create_dataset(
                "expectation/intensity",
                shape=(len(merged_ids), ny, nx),
                dtype=np.float32,
                chunks=(1, ny, nx),
                compression=compression,
                compression_opts=compression_level,
            )
            chunk = int(config["v6"]["expectation_generation"]["merge_chunk_patterns"])
            cursor = 0
            for handle in handles:
                source = handle["expectation/intensity"]
                for start in range(0, len(source), chunk):
                    stop = min(start + chunk, len(source))
                    images[cursor + start : cursor + stop] = source[start:stop]
                cursor += len(source)
        temporary.replace(output_path)
        return merged_ids
    finally:
        for handle in handles:
            handle.close()


def merge_oracles(
    shards: list[Path], output_path: Path, merged_image_path: Path
) -> int:
    samples = []
    attributes = None
    for path in shards:
        with h5py.File(path, "r") as handle:
            current = dict(handle.attrs)
            current.pop("source_image_file", None)
            if attributes is None:
                attributes = current
            elif current != attributes:
                raise ValueError(f"Oracle shard attributes differ: {path}")
        samples.extend(read_peak_h5(path))
    assert attributes is not None
    attributes["source_image_file"] = str(merged_image_path)
    attributes["merged_v6_shards"] = [str(path) for path in shards]
    write_peak_h5(output_path, samples, attributes)
    return len(samples)


def merge_reflections(shards: list[Path], output_path: Path) -> int:
    handles = [h5py.File(path, "r") for path in shards]
    try:
        sample_counts = [len(handle["sample_id"]) for handle in handles]
        reflection_counts = [
            int(handle["reflections/offsets"][-1]) for handle in handles
        ]
        fields = [name for name in handles[0]["reflections"] if name != "offsets"]
        for handle in handles[1:]:
            if [name for name in handle["reflections"] if name != "offsets"] != fields:
                raise ValueError("Raw reflection shard fields differ")
            if not np.array_equal(
                handles[0]["crystallography/reciprocal_basis_B_Ainv"][:],
                handle["crystallography/reciprocal_basis_B_Ainv"][:],
            ):
                raise ValueError("Raw reflection reciprocal bases differ")
        total_samples = sum(sample_counts)
        total_reflections = sum(reflection_counts)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_suffix(output_path.suffix + ".partial")
        with h5py.File(temporary, "w") as output:
            copy_attrs(handles[0].attrs, output.attrs)
            output.attrs["merged_v6_shards"] = json.dumps(
                [str(path) for path in shards]
            )
            output.create_dataset(
                "sample_id",
                data=np.asarray(
                    [
                        sample_id
                        for handle in handles
                        for sample_id in decode(handle["sample_id"][:])
                    ],
                    dtype=h5py.string_dtype("utf-8"),
                ),
            )
            reflections = output.create_group("reflections")
            offsets = np.empty(total_samples + 1, dtype=np.int64)
            offsets[0] = 0
            sample_cursor = 0
            reflection_cursor = 0
            for handle, sample_count in zip(handles, sample_counts, strict=True):
                local = np.asarray(handle["reflections/offsets"][:], dtype=np.int64)
                offsets[sample_cursor + 1 : sample_cursor + sample_count + 1] = (
                    local[1:] + reflection_cursor
                )
                sample_cursor += sample_count
                reflection_cursor += int(local[-1])
            reflections.create_dataset("offsets", data=offsets)
            for name in fields:
                template = handles[0][f"reflections/{name}"]
                shape = (total_reflections, *template.shape[1:])
                target = reflections.create_dataset(
                    name,
                    shape=shape,
                    dtype=template.dtype,
                    chunks=True,
                    compression="gzip",
                )
                cursor = 0
                for handle, count in zip(handles, reflection_counts, strict=True):
                    target[cursor : cursor + count] = handle[f"reflections/{name}"][:]
                    cursor += count
            handles[0].copy("crystallography", output)
            diagnostics = output.create_group("diagnostics")
            for name in handles[0]["diagnostics"]:
                values = np.concatenate(
                    [handle[f"diagnostics/{name}"][:] for handle in handles]
                )
                diagnostics.create_dataset(name, data=values, compression="gzip")
        temporary.replace(output_path)
        return total_samples
    finally:
        for handle in handles:
            handle.close()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    paths = config["v6"]["paths"]
    expected_shards = int(
        config["v6"]["expectation_generation"]["expected_shard_count"]
    )
    validate_shard_lists(args, expected_shards)
    image_shards = [path.resolve() for path in args.image_shard]
    oracle_shards = [path.resolve() for path in args.oracle_shard]
    reflection_shards = [path.resolve() for path in args.reflection_shard]
    orientation_file = Path(args.orientation_file or paths["orientations"]).resolve()
    expected_sample_ids = [
        f"clean_{row['orientation_id']}" for row in read_jsonl(orientation_file)
    ]
    output_image = enforce_server_write_scope(
        args.output_image or paths["expectation"], config
    )
    output_oracle = enforce_server_write_scope(
        args.output_oracle or paths["physical_oracle_peaks"], config
    )
    output_reflections = enforce_server_write_scope(
        args.output_reflections or paths["physical_oracle_reflections"], config
    )
    manifest = enforce_server_write_scope(
        args.manifest or paths["expectation_merge_manifest"], config
    )
    merged_ids = merge_images(image_shards, output_image, expected_sample_ids, config)
    oracle_count = merge_oracles(oracle_shards, output_oracle, output_image)
    reflection_count = merge_reflections(reflection_shards, output_reflections)
    if oracle_count != len(merged_ids) or reflection_count != len(merged_ids):
        raise RuntimeError("Merged V6 image/oracle/reflection sample counts differ")
    block_mib = int(config["v6"]["observation_store"]["sha256_read_block_MiB"])
    payload = {
        "schema": "or4d-clean-v6-expectation-merge-v1",
        "config": str(args.config.resolve()),
        "orientation_file": str(orientation_file),
        "sample_count": len(merged_ids),
        "image_shape": [int(value) for value in config["clean_image"]["gpts"]],
        "inputs": {
            "image": [
                {"path": str(path), "sha256": sha256_file(path, block_mib)}
                for path in image_shards
            ],
            "oracle": [
                {"path": str(path), "sha256": sha256_file(path, block_mib)}
                for path in oracle_shards
            ],
            "reflections": [
                {"path": str(path), "sha256": sha256_file(path, block_mib)}
                for path in reflection_shards
            ],
        },
        "outputs": {
            "image": {
                "path": str(output_image),
                "sha256": sha256_file(output_image, block_mib),
            },
            "oracle": {
                "path": str(output_oracle),
                "sha256": sha256_file(output_oracle, block_mib),
            },
            "reflections": {
                "path": str(output_reflections),
                "sha256": sha256_file(output_reflections, block_mib),
            },
        },
    }
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
