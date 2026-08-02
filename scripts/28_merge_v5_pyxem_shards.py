#!/usr/bin/env python3
"""Merge disjoint V5 Pyxem condition shards into one verified HDF5."""

from __future__ import annotations

import argparse
import hashlib
import json
from contextlib import ExitStack
from pathlib import Path

import h5py
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-file", type=Path, required=True)
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--expected-shards", type=int, default=4)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decode(values: np.ndarray) -> list[str]:
    return [x.decode() if isinstance(x, bytes) else str(x) for x in values]


def copy_root_dataset(source: h5py.File, destination: h5py.File, name: str) -> None:
    if name in source:
        source.copy(name, destination)


def merge_condition_group(
    shards: list[h5py.File],
    destination: h5py.File,
    group_name: str,
) -> dict:
    template = shards[0][group_name]
    template.file.copy(group_name, destination)
    output = destination[group_name]
    complete_arrays = [
        np.asarray(shard[f"{group_name}/condition_complete"][:], dtype=bool)
        for shard in shards
    ]
    ownership = np.sum(np.stack(complete_arrays, axis=0), axis=0)
    if not np.all(ownership == 1):
        bad = np.argwhere(ownership != 1)
        raise RuntimeError(
            f"{group_name} conditions must have exactly one owner; "
            f"bad indices={bad[:10].tolist()}"
        )
    output["condition_complete"][:] = False
    output["condition_seconds"][:] = np.nan
    payload_names = [
        name
        for name in output
        if name not in {"condition_complete", "condition_seconds"}
    ]
    for condition in np.ndindex(ownership.shape):
        owner = int(
            np.flatnonzero([values[condition] for values in complete_arrays])[0]
        )
        source = shards[owner][group_name]
        for name in payload_names:
            output[name][condition] = source[name][condition]
        output["condition_seconds"][condition] = source["condition_seconds"][
            condition
        ]
        output["condition_complete"][condition] = True
    return {
        "group": group_name,
        "condition_shape": list(ownership.shape),
        "num_conditions": int(ownership.size),
        "owners_per_shard": [
            int(values.sum()) for values in complete_arrays
        ],
    }


def main() -> None:
    args = parse_args()
    base_path = args.base_file.resolve()
    shard_paths = sorted(args.shard_dir.resolve().glob("shard*of*.h5"))
    output_path = args.output_file.resolve()
    if len(shard_paths) != args.expected_shards:
        raise ValueError(
            f"expected {args.expected_shards} shards, found {len(shard_paths)}"
        )
    if output_path.exists():
        raise FileExistsError(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".partial")
    if temporary_path.exists():
        raise FileExistsError(temporary_path)

    with h5py.File(base_path, "r") as base, ExitStack() as stack:
        shards = [stack.enter_context(h5py.File(path, "r")) for path in shard_paths]
        expected_ids = decode(base["sample_id"][:])
        for path, shard in zip(shard_paths, shards, strict=True):
            if decode(shard["sample_id"][:]) != expected_ids:
                raise ValueError(f"sample IDs differ in {path}")
            for name in ("dose_electrons", "counted_noise_level_id"):
                if not np.array_equal(base[name][:], shard[name][:]):
                    raise ValueError(f"{name} differs in {path}")

        summaries: list[dict] = []
        with h5py.File(temporary_path, "w") as output:
            for name, value in base.attrs.items():
                output.attrs[name] = value
            output.attrs["merge_schema"] = "or4d-pyxem-topk-merged-v1"
            output.attrs["clean_e_execution_target"] = "cpu"
            output.attrs["clean_c_execution_target"] = "gpu"
            output.attrs["source_shards_json"] = json.dumps(
                [str(path) for path in shard_paths]
            )
            for name in (
                "sample_id",
                "dose_electrons",
                "counted_noise_level_id",
            ):
                copy_root_dataset(base, output, name)
            base.copy("clean_e", output)
            for group_name in ("clean_c_noiseless", "clean_c_counted"):
                summaries.append(
                    merge_condition_group(shards, output, group_name)
                )
            output.flush()
        temporary_path.replace(output_path)

    manifest = {
        "schema": "or4d-pyxem-topk-merge-manifest-v1",
        "output_file": str(output_path),
        "output_sha256": sha256_file(output_path),
        "base_file": str(base_path),
        "base_sha256": sha256_file(base_path),
        "shards": [
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in shard_paths
        ],
        "groups": summaries,
    }
    manifest_path = output_path.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Merged Pyxem result: {output_path}")
    print(f"Merge manifest: {manifest_path}")


if __name__ == "__main__":
    main()
