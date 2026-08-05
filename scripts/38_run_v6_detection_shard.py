#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from or4d_common import load_config  # noqa: E402
from v6_detection import (  # noqa: E402
    UBraggXInference,
    V6ObservationShardLoader,
    detect_cpu_batch,
    detect_py4dstem_batch,
    detect_ubragg_batch,
    detector_geometry,
    failed_detection_record,
    successful_detection_record,
    write_v6_peak_h5,
)
from v6_runtime import enforce_server_write_scope  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one V6 detector on one compressed observation shard."
    )
    parser.add_argument(
        "--config", type=Path, default=ROOT / "config" / "benchmark_v6.yaml"
    )
    parser.add_argument("--expectation-file", type=Path)
    parser.add_argument("--observation-shard", type=Path, required=True)
    parser.add_argument(
        "--detector",
        choices=("autodisk", "dog_rgm", "py4dstem", "ubragg_x"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--condition-index", type=int, action="append")
    parser.add_argument("--sample-limit", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument(
        "--compute-backend", choices=("cpu", "cuda"), required=True
    )
    return parser.parse_args()


def sha256_file(path: Path, block_mib: int) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_mib * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_empty_bound_gpu(config: dict) -> str:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible.isdigit():
        raise RuntimeError(
            "CUDA V6 runs require CUDA_VISIBLE_DEVICES to expose one physical GPU"
        )
    rows = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    status = {}
    for row in rows.splitlines():
        fields = [value.strip() for value in row.split(",")]
        if len(fields) == 3:
            status[fields[0]] = (int(fields[1]), int(fields[2]))
    if visible not in status:
        raise RuntimeError(f"Physical GPU {visible} is absent from nvidia-smi")
    memory, utilization = status[visible]
    runtime = config["v6"]["runtime"]
    if memory > int(runtime["empty_gpu_max_memory_MiB"]) or utilization > int(
        runtime["empty_gpu_max_utilization_percent"]
    ):
        raise RuntimeError(
            f"Physical GPU {visible} is not empty: {memory} MiB, {utilization}%"
        )
    return visible


def selected_conditions(args: argparse.Namespace, count: int) -> list[int]:
    selected = args.condition_index or list(range(count))
    if len(set(selected)) != len(selected):
        raise ValueError("condition indices must be unique")
    if any(value < 0 or value >= count for value in selected):
        raise IndexError("condition index is outside the V6 condition grid")
    return selected


def detector_batch_size(args: argparse.Namespace, config: dict) -> int:
    if args.batch_size is not None:
        value = args.batch_size
    elif args.detector == "py4dstem":
        value = int(config["v6"]["detection"]["py4dstem_batch_size"])
    elif args.detector == "ubragg_x":
        value = max(
            int(item) for item in config["v6"]["ubragg_x"]["batch_size_candidates"]
        )
    else:
        value = int(config["v6"]["detection"]["cpu_sample_workers"])
    if value <= 0:
        raise ValueError("detection batch size must be positive")
    return value


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    configured_detectors = set(config["v6"]["detection"]["detectors"])
    if args.detector not in configured_detectors:
        raise ValueError(f"Detector {args.detector} is disabled by the V6 config")
    cuda_detectors = {"py4dstem", "ubragg_x"}
    if args.detector in cuda_detectors and args.compute_backend != "cuda":
        raise ValueError(f"V6 {args.detector} formal inference requires CUDA")
    if args.detector not in cuda_detectors and args.compute_backend != "cpu":
        raise ValueError(
            f"V6 {args.detector} remains CPU until its CuPy equivalence audit passes"
        )
    physical_gpu = require_empty_bound_gpu(config) if args.compute_backend == "cuda" else None
    paths = config["v6"]["paths"]
    expectation = Path(args.expectation_file or paths["expectation"]).resolve()
    output = enforce_server_write_scope(args.output, config)
    report_output = enforce_server_write_scope(args.report_output, config)
    observation_shard = args.observation_shard.resolve()
    batch_size = detector_batch_size(args, config)
    inference = UBraggXInference(config) if args.detector == "ubragg_x" else None
    if inference is not None:
        inference.torch.set_num_threads(
            int(config["v6"]["runtime"]["max_cpu_threads_per_gpu_at_four_gpus"])
        )

    records: list[dict] = []
    failures = 0
    empty = 0
    started_run = time.perf_counter()
    with V6ObservationShardLoader(expectation, observation_shard, config) as loader:
        geometry = detector_geometry(loader.expectation_h5, config)
        condition_indices = selected_conditions(args, len(loader.conditions))
        sample_count = len(loader.sample_ids)
        if args.sample_limit is not None:
            if args.sample_limit <= 0:
                raise ValueError("--sample-limit must be positive")
            sample_count = min(sample_count, args.sample_limit)
        total_observations = sample_count * len(condition_indices)
        completed = 0
        for condition_index in condition_indices:
            for batch_start in range(0, sample_count, batch_size):
                batch_stop = min(batch_start + batch_size, sample_count)
                local_indices = list(range(batch_start, batch_stop))
                images = [loader.image(index, condition_index) for index in local_indices]
                expected_totals: list[float] = []
                sigmas: list[float] = []
                for local_index, image in zip(local_indices, images, strict=True):
                    metadata = loader.metadata(local_index, condition_index)
                    if metadata["expected_total_electrons"] is None:
                        reference = float(
                            config["v6"]["ubragg_x"][
                                "clean_e_inference_reference_total_electrons"
                            ]
                        )
                        expected_totals.append(reference)
                        if args.detector == "ubragg_x":
                            image *= reference
                    else:
                        expected_totals.append(
                            float(metadata["expected_total_electrons"])
                        )
                    sigmas.append(
                        float(
                            metadata["condition"][
                                "read_noise_sigma_e_per_pixel"
                            ]
                        )
                    )
                batch_started = time.perf_counter()
                try:
                    if args.detector in {"autodisk", "dog_rgm"}:
                        results = detect_cpu_batch(
                            args.detector, images, geometry, config
                        )
                    elif args.detector == "py4dstem":
                        results = detect_py4dstem_batch(
                            images, geometry, config, cuda=True
                        )
                    else:
                        assert inference is not None
                        results = detect_ubragg_batch(
                            inference,
                            images,
                            expected_totals,
                            sigmas,
                            geometry,
                        )
                except Exception as error:
                    results = [error] * len(local_indices)
                elapsed = time.perf_counter() - batch_started
                per_observation = elapsed / len(local_indices)
                for local_index, result in zip(local_indices, results, strict=True):
                    if isinstance(result, Exception):
                        record = failed_detection_record(
                            loader,
                            local_index,
                            condition_index,
                            result,
                            per_observation,
                            config,
                        )
                        failures += 1
                    else:
                        record = successful_detection_record(
                            loader,
                            local_index,
                            condition_index,
                            result,
                            per_observation,
                        )
                        empty += int(record["peak_count"] == 0)
                    records.append(record)
                    completed += 1
                progress_every = int(
                    config["v6"]["detection"]["progress_every_observations"]
                )
                if completed % progress_every < len(local_indices) or completed == total_observations:
                    print(
                        f"{args.detector}: {completed}/{total_observations}, "
                        f"failures={failures}, empty={empty}",
                        flush=True,
                    )

    attrs = {
        "schema": "or4d-clean-v6-peaks-v1",
        "dataset_id": config["dataset"]["id"],
        "detector": args.detector,
        "compute_backend": args.compute_backend,
        "physical_gpu": physical_gpu or "not_applicable",
        "source_expectation_file": str(expectation),
        "source_observation_shard": str(observation_shard),
        "condition_indices": condition_indices,
        "score_threshold": (
            float(config["v6"]["ubragg_x"]["score_threshold"])
            if args.detector == "ubragg_x"
            else "detector_configured_candidate_threshold"
        ),
        "k_max_Ainv": float(config["common"]["k_max_Ainv"]),
        "central_exclusion_Ainv": geometry["central_exclusion_Ainv"],
        "batch_size": batch_size,
    }
    if inference is not None:
        attrs["ubragg_x_artifacts"] = inference.artifacts
    write_v6_peak_h5(output, records, attrs, config)
    hash_block = int(
        config["v6"]["observation_store"]["sha256_read_block_MiB"]
    )
    report = {
        **attrs,
        "output": str(output),
        "output_sha256": sha256_file(output, hash_block),
        "output_size_bytes": output.stat().st_size,
        "observation_count": len(records),
        "failure_count": failures,
        "empty_peak_count": empty,
        "peak_count": int(sum(record["peak_count"] for record in records)),
        "wall_seconds": time.perf_counter() - started_run,
    }
    report_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
