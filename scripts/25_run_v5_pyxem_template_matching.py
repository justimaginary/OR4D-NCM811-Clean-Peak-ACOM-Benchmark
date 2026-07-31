#!/usr/bin/env python3
"""Run Pyxem accelerated template matching on full V5 Clean-E/Clean-C data."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clean_counting import add_gaussian_read_noise  # noqa: E402
from or4d_common import cif_path, load_config  # noqa: E402
from pyxem_template_adapter import (  # noqa: E402
    build_template_library,
    euler_to_sample_to_crystal,
    match_prepared_batch,
    prepare_cartesian_patterns,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument(
        "--track",
        choices=("expectation", "clean_c", "all"),
        default="all",
    )
    parser.add_argument("--target", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Validation-only prefix; omit for the formal full run.",
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def decode(values: np.ndarray) -> list[str]:
    return [value.decode() if isinstance(value, bytes) else str(value) for value in values]


def require_one_empty_gpu() -> str:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible.isdigit():
        raise RuntimeError(
            "GPU matching requires CUDA_VISIBLE_DEVICES to contain exactly one "
            "physical GPU index"
        )
    rows = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    status = {
        fields[0]: (int(fields[1]), int(fields[2]))
        for line in rows.splitlines()
        if len(fields := [part.strip() for part in line.split(",")]) == 3
    }
    memory_mib, utilization = status[visible]
    if memory_mib > 100 or utilization > 5:
        raise RuntimeError(
            f"GPU {visible} is not empty: {memory_mib} MiB, {utilization}%"
        )
    return visible


def create_result_group(
    parent: h5py.Group,
    name: str,
    condition_shape: tuple[int, ...],
    n: int,
    n_best: int,
) -> h5py.Group:
    group = parent.require_group(name)
    specifications = {
        "orientation_matrix_sample_to_crystal": (
            condition_shape + (n, n_best, 3, 3),
            np.float64,
            np.nan,
        ),
        "euler_bunge_lab_to_crystal_deg": (
            condition_shape + (n, n_best, 3),
            np.float64,
            np.nan,
        ),
        "correlation": (condition_shape + (n, n_best), np.float32, np.nan),
        "mirrored_template": (
            condition_shape + (n, n_best),
            np.bool_,
            False,
        ),
        "template_index": (
            condition_shape + (n, n_best),
            np.int32,
            -1,
        ),
    }
    for dataset_name, (shape, dtype, fillvalue) in specifications.items():
        if dataset_name not in group:
            group.create_dataset(
                dataset_name,
                shape=shape,
                dtype=dtype,
                fillvalue=fillvalue,
                chunks=True,
                compression="gzip",
                compression_opts=4,
            )
    if "condition_complete" not in group:
        group.create_dataset(
            "condition_complete",
            shape=condition_shape or (1,),
            dtype=np.bool_,
            fillvalue=False,
        )
    if "condition_seconds" not in group:
        group.create_dataset(
            "condition_seconds",
            shape=condition_shape or (1,),
            dtype=np.float64,
            fillvalue=np.nan,
        )
    return group


def condition_index(condition: tuple[int, ...]) -> tuple[int, ...]:
    return condition or (0,)


def run_condition(
    *,
    source,
    sample_count: int,
    batch_size: int,
    q_pixel_size: float,
    exclusion_radius: float,
    library,
    settings: dict,
    target: str,
    destination: h5py.Group,
    condition: tuple[int, ...],
    image_reader,
) -> float:
    started = time.perf_counter()
    for begin in range(0, sample_count, batch_size):
        end = min(begin + batch_size, sample_count)
        images = image_reader(source, begin, end)
        prepared = prepare_cartesian_patterns(
            images,
            q_pixel_size_Ainv=q_pixel_size,
            central_beam_exclusion_Ainv=exclusion_radius,
        )
        matched = match_prepared_batch(
            prepared,
            library=library,
            q_pixel_size_Ainv=q_pixel_size,
            settings=settings,
            target=target,
        )
        key = condition + (slice(begin, end),)
        euler = matched["euler_deg"]
        destination["euler_bunge_lab_to_crystal_deg"][key] = euler
        destination["orientation_matrix_sample_to_crystal"][key] = (
            euler_to_sample_to_crystal(euler)
        )
        destination["correlation"][key] = matched["correlation"]
        destination["mirrored_template"][key] = matched["mirrored"]
        destination["template_index"][key] = matched["template_index"]
        destination.file.flush()
        print(
            f"condition={condition or ('expectation',)} "
            f"samples={end}/{sample_count}",
            flush=True,
        )
    seconds = time.perf_counter() - started
    destination["condition_seconds"][condition_index(condition)] = seconds
    destination["condition_complete"][condition_index(condition)] = True
    destination.file.flush()
    return seconds


def expectation_reader(dataset, begin: int, end: int) -> np.ndarray:
    return np.asarray(dataset[begin:end], dtype=np.float32)


def noiseless_reader(dose_index: int):
    def read(dataset, begin: int, end: int) -> np.ndarray:
        return np.asarray(dataset[begin:end, dose_index], dtype=np.float32)

    return read


def counted_reader(
    *,
    dose_index: int,
    repeat: int,
    level_index: int,
    read_sigma: float,
    seeds,
):
    def read(dataset, begin: int, end: int) -> np.ndarray:
        images = np.asarray(
            dataset[begin:end, dose_index, repeat], dtype=np.float32
        )
        if read_sigma == 0:
            return images
        for local, sample_index in enumerate(range(begin, end)):
            images[local] = add_gaussian_read_noise(
                images[local],
                read_sigma,
                np.random.default_rng(
                    int(seeds[sample_index, dose_index, level_index, repeat])
                ),
            )
        return images

    return read


def main() -> None:
    args = parse_args()
    config = load_config()
    settings = dict(config["clean_image"]["pyxem_template_matching"])
    n_best = int(settings["n_best"])
    if n_best != 5:
        raise ValueError(f"V5 Top-5 run requires n_best=5, got {n_best}")
    batch_size = int(args.batch_size or settings["batch_size"])
    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    physical_gpu = require_one_empty_gpu() if args.target == "gpu" else None
    data_root = args.data_root.resolve()
    expectation_path = (
        data_root / "datasets" / "clean_v5_first_born_expectation_2048.h5"
    )
    counted_path = (
        data_root / "datasets" / "clean_v5_first_born_counted_2048.h5"
    )
    noiseless_path = (
        data_root / "datasets" / "clean_v5_first_born_dose_noiseless_2048.h5"
    )
    noise_path = (
        data_root / "manifests" / "clean_v5_instrument_noise_2048.h5"
    )
    for path in (expectation_path, counted_path, noiseless_path, noise_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    with h5py.File(expectation_path, "r") as source:
        sample_ids = decode(source["sample_id"][:])
        qx = np.asarray(source["detector/qx_Ainv"][:], dtype=float)
        q_pixel_size = float(np.median(np.diff(qx)))
        if not np.isclose(q_pixel_size, 0.00625, atol=1e-10):
            raise ValueError(f"unexpected q pixel size {q_pixel_size}")
    sample_count = len(sample_ids)
    if args.max_samples is not None:
        if not 1 <= args.max_samples <= sample_count:
            raise ValueError("--max-samples is out of range")
        sample_count = args.max_samples
        sample_ids = sample_ids[:sample_count]

    cache_path = (
        data_root
        / "intermediates"
        / "pyxem_templates"
        / "ncm811_s2_2deg_kmax1p5_v2.pickle"
    )
    library, library_metadata = build_template_library(
        cif_path=cif_path(config),
        cache_path=cache_path,
        voltage_kV=float(config["common"]["accelerating_voltage_V"]) / 1000,
        q_pixel_size_Ainv=q_pixel_size,
        settings=settings,
    )

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume else "w"
    with h5py.File(args.output_file, mode) as output:
        output.attrs["method"] = "pyxem_accelerated_template_matching"
        output.attrs["pyxem_version"] = "0.21.0"
        output.attrs["target"] = args.target
        output.attrs["physical_gpu_index"] = physical_gpu or ""
        output.attrs["settings_json"] = json.dumps(settings, sort_keys=True)
        output.attrs["template_metadata_json"] = json.dumps(
            library_metadata, sort_keys=True
        )
        output.attrs["sample_count"] = sample_count
        if "sample_id" not in output:
            output.create_dataset(
                "sample_id",
                data=np.asarray(sample_ids, dtype=h5py.string_dtype("utf-8")),
            )
        elif decode(output["sample_id"][:]) != sample_ids:
            raise ValueError("resume output sample IDs differ")

        if args.track in {"expectation", "all"}:
            group = create_result_group(
                output, "clean_e", (), sample_count, n_best
            )
            if not args.resume or not bool(group["condition_complete"][0]):
                with h5py.File(expectation_path, "r") as source:
                    run_condition(
                        source=source,
                        sample_count=sample_count,
                        batch_size=batch_size,
                        q_pixel_size=q_pixel_size,
                        exclusion_radius=float(
                            settings["central_beam_exclusion_Ainv"]
                        ),
                        library=library,
                        settings=settings,
                        target=args.target,
                        destination=group,
                        condition=(),
                        image_reader=lambda h5, begin, end: expectation_reader(
                            h5["expectation/intensity"], begin, end
                        ),
                    )

        if args.track in {"clean_c", "all"}:
            with h5py.File(counted_path, "r") as counted, h5py.File(
                noiseless_path, "r"
            ) as noiseless, h5py.File(noise_path, "r") as noise:
                if decode(counted["sample_id"][:])[:sample_count] != sample_ids:
                    raise ValueError("counted sample IDs differ")
                doses = np.asarray(counted["dose_electrons"][:], dtype=np.int64)
                repeats = int(counted["images/counts"].shape[2])
                level_ids = decode(noise["noise_level_id"][:])
                counted_level_indices = [
                    index
                    for index, level in enumerate(level_ids)
                    if level != "noiseless"
                ]
                read_sigma = np.asarray(
                    noise["read_noise_sigma_primary_e_rms_per_pixel"][:],
                    dtype=float,
                )
                seeds = noise["read_noise_seed"]
                if "dose_electrons" not in output:
                    output.create_dataset("dose_electrons", data=doses)
                    output.create_dataset(
                        "counted_noise_level_id",
                        data=np.asarray(
                            [level_ids[i] for i in counted_level_indices],
                            dtype=h5py.string_dtype("utf-8"),
                        ),
                    )
                    output.attrs["counted_repeats"] = repeats
                noiseless_group = create_result_group(
                    output,
                    "clean_c_noiseless",
                    (len(doses),),
                    sample_count,
                    n_best,
                )
                counted_group = create_result_group(
                    output,
                    "clean_c_counted",
                    (len(doses), len(counted_level_indices), repeats),
                    sample_count,
                    n_best,
                )
                for dose_index, dose in enumerate(doses):
                    condition = (dose_index,)
                    if not args.resume or not bool(
                        noiseless_group["condition_complete"][condition]
                    ):
                        print(f"Clean-C noiseless dose={dose}", flush=True)
                        run_condition(
                            source=noiseless,
                            sample_count=sample_count,
                            batch_size=batch_size,
                            q_pixel_size=q_pixel_size,
                            exclusion_radius=float(
                                settings["central_beam_exclusion_Ainv"]
                            ),
                            library=library,
                            settings=settings,
                            target=args.target,
                            destination=noiseless_group,
                            condition=condition,
                            image_reader=lambda h5, begin, end, d=dose_index: (
                                noiseless_reader(d)(
                                    h5["images/expected_counts"], begin, end
                                )
                            ),
                        )
                    for compact_level, level_index in enumerate(
                        counted_level_indices
                    ):
                        for repeat in range(repeats):
                            condition = (dose_index, compact_level, repeat)
                            if args.resume and bool(
                                counted_group["condition_complete"][condition]
                            ):
                                continue
                            print(
                                f"Clean-C dose={dose} "
                                f"noise={level_ids[level_index]} repeat={repeat}",
                                flush=True,
                            )
                            run_condition(
                                source=counted,
                                sample_count=sample_count,
                                batch_size=batch_size,
                                q_pixel_size=q_pixel_size,
                                exclusion_radius=float(
                                    settings["central_beam_exclusion_Ainv"]
                                ),
                                library=library,
                                settings=settings,
                                target=args.target,
                                destination=counted_group,
                                condition=condition,
                                image_reader=lambda h5, begin, end, d=dose_index, r=repeat, li=level_index, sigma=float(
                                    read_sigma[level_index]
                                ): counted_reader(
                                    dose_index=d,
                                    repeat=r,
                                    level_index=li,
                                    read_sigma=sigma,
                                    seeds=seeds,
                                )(h5["images/counts"], begin, end),
                            )
    print(f"Output: {args.output_file}", flush=True)


if __name__ == "__main__":
    main()
