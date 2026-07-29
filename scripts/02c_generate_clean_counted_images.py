#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clean_counting import multinomial_count_image  # noqa: E402
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config()
    counting = config["clean_image"]["counting"]
    if counting["model"] != "multinomial_fixed_total":
        raise ValueError("Only multinomial_fixed_total is implemented for Clean-C")
    doses = np.asarray(
        args.dose or counting["doses_electrons"], dtype=np.int64
    )
    repeats = int(args.repeats or counting["repeats"])
    if np.any(doses <= 0) or repeats <= 0:
        raise ValueError("doses and repeats must be positive")

    source_path = args.expectation_file.resolve()
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
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
        target.create_dataset("dose_electrons", data=doses)
        target.create_dataset("sample_id", data=source["sample_id"][:])
        detector = target.create_group("detector")
        for name in ("qx_Ainv", "qy_Ainv", "vacuum_probe", "valid_mask"):
            detector.create_dataset(name, data=source[f"detector/{name}"][:])

        for sample_index in range(n):
            probability = np.asarray(expectation_source[sample_index], dtype=np.float64)
            expectation[sample_index] = probability
            for dose_index, dose in enumerate(doses):
                for repeat in range(repeats):
                    seed = (
                        seed_base
                        + sample_index * 1_000_003
                        + dose_index * 10_007
                        + repeat
                    )
                    seeds[sample_index, dose_index, repeat] = seed
                    image = multinomial_count_image(
                        probability,
                        int(dose),
                        np.random.default_rng(seed),
                    )
                    if int(image.sum()) != int(dose):
                        raise RuntimeError("multinomial sampling violated fixed total")
                    counts[sample_index, dose_index, repeat] = image
            target.flush()
            print(f"{sample_index + 1}/{n} {source['sample_id'][sample_index]!r}")

        for key, value in source.attrs.items():
            target.attrs[f"expectation_{key}"] = value
        target.attrs["track"] = "clean_counted"
        target.attrs["counting_model"] = "multinomial_fixed_total"
        target.attrs["detector_model"] = "ideal_counting"
        target.attrs["normalization"] = "each count image sums exactly to dose_electrons"
        target.attrs["source_expectation_file"] = str(source_path)
        target.attrs["seed_formula"] = (
            "seed_base + sample_index*1000003 + dose_index*10007 + repeat"
        )
        target.attrs["counting_config"] = json.dumps(counting, sort_keys=True)

    temporary.replace(output_path)
    report = {
        "expectation_file": str(source_path),
        "output": str(output_path),
        "samples": int(n),
        "shape": [int(n), int(len(doses)), repeats, ny, nx],
        "doses_electrons": doses.tolist(),
        "repeats": repeats,
        "total_seconds": time.perf_counter() - started,
        "fixed_total_verified": True,
    }
    report_name = (
        "clean_counted_generation_smoke.json"
        if "smoke" in output_path.stem
        else "clean_counted_generation.json"
    )
    report_path = ROOT / "reports" / report_name
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Counted images: {output_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
