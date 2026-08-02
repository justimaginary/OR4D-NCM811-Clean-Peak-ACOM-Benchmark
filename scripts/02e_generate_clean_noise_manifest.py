#!/usr/bin/env python3
"""Write the independent V5 detector-noise axis and deterministic seeds."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clean_counting import deterministic_read_noise_seed  # noqa: E402
from or4d_common import load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    return parser.parse_args()


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main() -> None:
    args = parse_args()
    config = load_config()
    noise = config["clean_image"]["instrument_noise"]
    levels = list(noise["levels"])
    reference_sigma = float(
        noise["reference_read_noise_primary_e_rms_per_pixel"]
    )
    level_ids = [str(level["id"]) for level in levels]
    frame_counts = np.asarray(
        [int(level["summed_frame_count"]) for level in levels],
        dtype=np.int32,
    )
    poisson_flags = np.asarray(
        [bool(level["poisson_shot_noise"]) for level in levels],
        dtype=np.uint8,
    )
    read_sigmas = reference_sigma * np.sqrt(frame_counts)

    count_path = args.count_file.resolve()
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".partial")
    with h5py.File(count_path, "r") as source, h5py.File(
        temporary, "w"
    ) as target:
        sample_ids_raw = source["sample_id"][:]
        sample_ids = [
            value.decode() if isinstance(value, bytes) else str(value)
            for value in sample_ids_raw
        ]
        doses = np.asarray(source["dose_electrons"][:], dtype=np.int64)
        repeats = int(source["images/counts"].shape[2])
        seeds = target.create_dataset(
            "read_noise_seed",
            shape=(len(sample_ids), len(doses), len(levels), repeats),
            dtype=np.uint64,
        )
        seed_base = int(noise["seed_base"])
        for sample_index, sample_id in enumerate(sample_ids):
            for dose_index, dose in enumerate(doses):
                for level_index, level_id in enumerate(level_ids):
                    for repeat in range(repeats):
                        seeds[
                            sample_index,
                            dose_index,
                            level_index,
                            repeat,
                        ] = deterministic_read_noise_seed(
                            seed_base,
                            sample_id,
                            int(dose),
                            level_id,
                            repeat,
                        )
        target.create_dataset("sample_id", data=sample_ids_raw)
        target.create_dataset("dose_electrons", data=doses)
        target.create_dataset(
            "noise_level_id",
            data=np.asarray(
                level_ids, dtype=h5py.string_dtype(encoding="utf-8")
            ),
        )
        target.create_dataset("summed_frame_count", data=frame_counts)
        target.create_dataset(
            "poisson_shot_noise_enabled", data=poisson_flags
        )
        target.create_dataset(
            "read_noise_sigma_primary_e_rms_per_pixel",
            data=read_sigmas,
        )
        target.attrs["track"] = "clean_instrument_noise_manifest"
        target.attrs["source_count_file"] = str(count_path)
        target.attrs["noise_model"] = str(noise["model"])
        target.attrs["noise_config"] = json.dumps(noise, sort_keys=True)
        target.attrs["read_noise_formula"] = (
            "sigma_level = reference_sigma * sqrt(summed_frame_count)"
        )
        target.attrs["paired_design"] = (
            "all noisy levels at one sample/dose/repeat reuse the same "
            "Poisson count image; only read noise changes"
        )
        target.attrs["git_commit"] = git_commit()
    temporary.replace(output_path)

    report = {
        "count_file": str(count_path),
        "output": str(output_path),
        "samples": len(sample_ids),
        "doses_electrons": doses.tolist(),
        "repeats": repeats,
        "noise_model": noise["model"],
        "reference_detector": noise["reference_detector"],
        "source_doi": noise["source_doi"],
        "levels": [
            {
                "id": level_id,
                "poisson_shot_noise": bool(poisson_flags[index]),
                "summed_frame_count": int(frame_counts[index]),
                "read_noise_sigma_primary_e_rms_per_pixel": float(
                    read_sigmas[index]
                ),
            }
            for index, level_id in enumerate(level_ids)
        ],
        "seed_shape": [
            len(sample_ids),
            len(doses),
            len(levels),
            repeats,
        ],
        "git_commit": git_commit(),
    }
    report_path = args.report_output.resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Noise manifest: {output_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
