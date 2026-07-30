#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clean_counting import (  # noqa: E402
    deterministic_count_seed,
    multinomial_count_image,
    poisson_count_image,
)
from or4d_common import load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample fixed-total ideal electron-count images from Clean-E."
    )
    parser.add_argument(
        "--expectation-file",
        type=Path,
        default=ROOT / "public" / "clean_images.h5",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "public" / "clean_counted_images.h5",
    )
    parser.add_argument("--dose", type=int, action="append")
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--report-output", type=Path)
    parser.add_argument(
        "--minimum-free-gib",
        type=float,
        default=5.0,
        help="Abort before writing unless this much free space remains.",
    )
    return parser.parse_args()


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    config = load_config()
    counting = config["clean_image"]["counting"]
    model = str(counting["model"])
    if model not in ("multinomial_fixed_total", "poisson_expected_total"):
        raise ValueError(f"Unsupported Clean-C counting model: {model}")
    doses = np.asarray(
        args.dose or counting["doses_electrons"], dtype=np.int64
    )
    repeats = int(args.repeats or counting["repeats"])
    if np.any(doses <= 0) or repeats <= 0:
        raise ValueError("doses and repeats must be positive")

    source_path = args.expectation_file.resolve()
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(output_path.parent).free
    reserve_bytes = int(args.minimum_free_gib * 1024**3)
    if free_bytes <= reserve_bytes:
        raise RuntimeError(
            f"Only {free_bytes / 1024**3:.2f} GiB free at "
            f"{output_path.parent}; reserve is {args.minimum_free_gib:.2f} GiB."
        )
    temporary = output_path.with_suffix(output_path.suffix + ".partial")
    seed_base = int(counting["seed_base"])
    started = time.perf_counter()

    with h5py.File(source_path, "r") as source, h5py.File(temporary, "w") as target:
        expectation_source = source["expectation/intensity"]
        n, ny, nx = expectation_source.shape
        expectation = target.create_dataset(
            "expectation/intensity",
            shape=expectation_source.shape,
            dtype=np.float32,
            chunks=(1, ny, nx),
            compression="gzip",
            compression_opts=4,
        )
        counts = target.create_dataset(
            "images/counts",
            shape=(n, len(doses), repeats, ny, nx),
            dtype=np.uint32,
            chunks=(1, 1, 1, ny, nx),
            compression="gzip",
            compression_opts=4,
            shuffle=True,
        )
        seeds = target.create_dataset(
            "rng_seed",
            shape=(n, len(doses), repeats),
            dtype=np.uint64,
        )
        actual_totals = target.create_dataset(
            "diagnostics/actual_total_count",
            shape=(n, len(doses), repeats),
            dtype=np.uint64,
        )
        nonzero_pixels = target.create_dataset(
            "diagnostics/nonzero_pixel_count",
            shape=(n, len(doses), repeats),
            dtype=np.uint32,
        )
        maximum_pixel_count = target.create_dataset(
            "diagnostics/maximum_pixel_count",
            shape=(n, len(doses), repeats),
            dtype=np.uint32,
        )
        target.create_dataset("dose_electrons", data=doses)
        target.create_dataset("sample_id", data=source["sample_id"][:])
        sample_ids = [
            value.decode() if isinstance(value, bytes) else str(value)
            for value in source["sample_id"][:]
        ]
        detector = target.create_group("detector")
        for name in ("qx_Ainv", "qy_Ainv", "vacuum_probe", "valid_mask"):
            detector.create_dataset(name, data=source[f"detector/{name}"][:])

        for sample_index in range(n):
            probability = np.asarray(expectation_source[sample_index], dtype=np.float64)
            expectation[sample_index] = probability
            for dose_index, dose in enumerate(doses):
                for repeat in range(repeats):
                    seed = deterministic_count_seed(
                        seed_base,
                        sample_ids[sample_index],
                        int(dose),
                        repeat,
                    )
                    seeds[sample_index, dose_index, repeat] = seed
                    rng = np.random.default_rng(seed)
                    if model == "multinomial_fixed_total":
                        image = multinomial_count_image(
                            probability, int(dose), rng
                        )
                        if int(image.sum()) != int(dose):
                            raise RuntimeError(
                                "multinomial sampling violated fixed total"
                            )
                    else:
                        image = poisson_count_image(
                            probability, int(dose), rng
                        )
                    counts[sample_index, dose_index, repeat] = image
                    actual_totals[sample_index, dose_index, repeat] = int(
                        image.sum()
                    )
                    nonzero_pixels[sample_index, dose_index, repeat] = int(
                        np.count_nonzero(image)
                    )
                    maximum_pixel_count[
                        sample_index, dose_index, repeat
                    ] = int(image.max())
            target.flush()
            print(f"{sample_index + 1}/{n} {source['sample_id'][sample_index]!r}")

        for key, value in source.attrs.items():
            target.attrs[f"expectation_{key}"] = value
        target.attrs["track"] = "clean_counted"
        target.attrs["forward_model"] = source.attrs.get(
            "forward_model", "unknown"
        )
        target.attrs["counting_model"] = model
        target.attrs["detector_model"] = "ideal_counting"
        target.attrs["normalization"] = (
            "each count image sums exactly to dose_electrons"
            if model == "multinomial_fixed_total"
            else "dose_electrons is expected total; actual totals fluctuate"
        )
        target.attrs["source_expectation_file"] = str(source_path)
        target.attrs["seed_formula"] = (
            "uint64(blake2b('or4d-clean-v5|seed_base|sample_id|dose|repeat'))"
        )
        target.attrs["counting_config"] = json.dumps(counting, sort_keys=True)
        target.attrs["git_commit"] = git_commit()

    temporary.replace(output_path)
    report = {
        "expectation_file": str(source_path),
        "output": str(output_path),
        "samples": int(n),
        "shape": [int(n), int(len(doses)), repeats, ny, nx],
        "doses_electrons": doses.tolist(),
        "repeats": repeats,
        "total_seconds": time.perf_counter() - started,
        "counting_model": model,
        "fixed_total_verified": model == "multinomial_fixed_total",
        "git_commit": git_commit(),
        "source_sha256": sha256_file(source_path),
        "output_sha256": sha256_file(output_path),
        "seed_derivation": (
            "blake2b(seed_base, sample_id, dose_electrons, repeat)"
        ),
    }
    with h5py.File(source_path, "r") as source:
        forward_model = str(source.attrs.get("forward_model", "unknown"))
    model_suffix = "" if forward_model == "acom_matched" else "_first_born"
    smoke_suffix = "_smoke" if "smoke" in output_path.stem else ""
    report_name = (
        f"clean_counted_generation{model_suffix}{smoke_suffix}.json"
    )
    report_path = (
        args.report_output.resolve()
        if args.report_output is not None
        else ROOT / "reports" / report_name
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Counted images: {output_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
