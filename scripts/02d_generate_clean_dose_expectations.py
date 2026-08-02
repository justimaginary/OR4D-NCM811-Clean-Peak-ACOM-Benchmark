#!/usr/bin/env python3
"""Generate dose-scaled Clean-C images without random counting noise."""

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

from clean_counting import noiseless_expected_count_image  # noqa: E402
from or4d_common import load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expectation-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dose", type=int, action="append")
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--minimum-free-gib", type=float, default=5.0)
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
    doses = np.asarray(
        args.dose
        or config["clean_image"]["counting"]["doses_electrons"],
        dtype=np.int64,
    )
    if np.any(doses <= 0):
        raise ValueError("doses must be positive")

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
    started = time.perf_counter()

    with h5py.File(source_path, "r") as source, h5py.File(
        temporary, "w"
    ) as target:
        source_images = source["expectation/intensity"]
        num_samples, ny, nx = source_images.shape
        expected_counts = target.create_dataset(
            "images/expected_counts",
            shape=(num_samples, len(doses), ny, nx),
            dtype=np.float32,
            chunks=(1, 1, ny, nx),
            compression="gzip",
            compression_opts=4,
            shuffle=True,
        )
        expected_totals = target.create_dataset(
            "diagnostics/expected_total_count",
            shape=(num_samples, len(doses)),
            dtype=np.float64,
        )
        target.create_dataset("dose_electrons", data=doses)
        target.create_dataset("sample_id", data=source["sample_id"][:])
        detector = target.create_group("detector")
        for name in ("qx_Ainv", "qy_Ainv", "vacuum_probe", "valid_mask"):
            detector.create_dataset(name, data=source[f"detector/{name}"][:])

        for sample_index in range(num_samples):
            expectation = np.asarray(
                source_images[sample_index], dtype=np.float64
            )
            for dose_index, dose in enumerate(doses):
                image = noiseless_expected_count_image(
                    expectation, int(dose)
                )
                expected_counts[sample_index, dose_index] = image
                expected_totals[sample_index, dose_index] = float(image.sum())
            target.flush()
            print(
                f"{sample_index + 1}/{num_samples} "
                f"{source['sample_id'][sample_index]!r}"
            )

        for key, value in source.attrs.items():
            target.attrs[f"expectation_{key}"] = value
        target.attrs["track"] = "clean_counted_noiseless"
        target.attrs["forward_model"] = source.attrs.get(
            "forward_model", "unknown"
        )
        target.attrs["noise_model"] = "none"
        target.attrs["detector_model"] = "ideal_linear_count_scale"
        target.attrs["source_expectation_file"] = str(source_path)
        target.attrs["expected_count_formula"] = (
            "images/expected_counts[i,d] = "
            "dose_electrons[d] * normalized(expectation[i])"
        )
        target.attrs["git_commit"] = git_commit()

    temporary.replace(output_path)
    report = {
        "expectation_file": str(source_path),
        "output": str(output_path),
        "samples": int(num_samples),
        "shape": [
            int(num_samples),
            int(len(doses)),
            int(ny),
            int(nx),
        ],
        "doses_electrons": doses.tolist(),
        "noise_model": "none",
        "total_seconds": time.perf_counter() - started,
        "git_commit": git_commit(),
        "source_sha256": sha256_file(source_path),
        "output_sha256": sha256_file(output_path),
    }
    report_path = args.report_output.resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Noiseless dose images: {output_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
