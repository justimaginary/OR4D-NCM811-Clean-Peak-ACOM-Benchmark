#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import h5py

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from or4d_common import load_config, read_jsonl, write_jsonl  # noqa: E402
from v6_runtime import (  # noqa: E402
    enforce_server_write_scope,
    require_empty_bound_gpu,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the unchanged V5 py4DSTEM ACOM baseline on one V6 peak shard."
        )
    )
    parser.add_argument(
        "--config", type=Path, default=ROOT / "config" / "benchmark_v6.yaml"
    )
    parser.add_argument("--peak-file", type=Path, required=True)
    parser.add_argument("--orientation-file", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    return parser.parse_args()


def decode(values) -> list[str]:
    return [
        value.decode() if isinstance(value, bytes) else str(value)
        for value in values
    ]


def expanded_ground_truth(
    peak_file: Path, orientation_file: Path
) -> list[dict]:
    orientations = read_jsonl(orientation_file)
    by_sample_id = {
        f"clean_{record['orientation_id']}": record for record in orientations
    }
    with h5py.File(peak_file, "r") as h5:
        observation_ids = decode(h5["sample_id"][:])
        source_ids = decode(h5["sample/source_sample_id"][:])
    if len(observation_ids) != len(source_ids):
        raise ValueError("V6 peak sample/source_sample_id lengths differ")
    records = []
    for observation_id, source_id in zip(observation_ids, source_ids, strict=True):
        if source_id not in by_sample_id:
            raise KeyError(
                f"Peak source sample {source_id} is absent from {orientation_file}"
            )
        record = dict(by_sample_id[source_id])
        record["sample_id"] = observation_id
        record["source_sample_id"] = source_id
        records.append(record)
    if len({record["sample_id"] for record in records}) != len(records):
        raise ValueError("V6 peak shard contains duplicate logical observation IDs")
    return records


def main() -> None:
    args = parse_args()
    if not args.job_id.replace("_", "").isalnum():
        raise ValueError("--job-id may contain only letters, numbers and underscores")
    config_path = args.config.resolve()
    config = load_config(config_path)
    physical_gpu = require_empty_bound_gpu(config, workload="acom")
    peak_file = args.peak_file.resolve()
    orientation_file = Path(
        args.orientation_file or config["v6"]["paths"]["orientations"]
    ).resolve()
    output_dir = enforce_server_write_scope(args.output_dir, config)
    output_dir.mkdir(parents=True, exist_ok=True)
    expanded_gt = enforce_server_write_scope(
        output_dir / f"{args.job_id}_expanded_ground_truth.jsonl", config
    )
    prediction = enforce_server_write_scope(
        output_dir / f"{args.job_id}_predictions.jsonl", config
    )
    details = enforce_server_write_scope(
        output_dir / f"{args.job_id}_details.json", config
    )
    candidates = enforce_server_write_scope(
        output_dir / f"{args.job_id}_candidates.h5", config
    )
    audit = enforce_server_write_scope(
        output_dir / f"{args.job_id}_plan_audit.json", config
    )
    run_record = enforce_server_write_scope(
        output_dir / f"{args.job_id}_run.json", config
    )
    polar_h5 = enforce_server_write_scope(
        output_dir / f"{args.job_id}_polar.h5", config
    )
    polar_json = enforce_server_write_scope(
        output_dir / f"{args.job_id}_polar.json", config
    )
    ground_truth = expanded_ground_truth(peak_file, orientation_file)
    write_jsonl(expanded_gt, ground_truth)
    command = [
        sys.executable,
        str(ROOT / "scripts" / "07_run_acom_baseline.py"),
        "--peak-file",
        str(peak_file),
        "--ground-truth-file",
        str(expanded_gt),
        "--orientation-file",
        str(orientation_file),
        "--prediction-file",
        str(prediction),
        "--details-file",
        str(details),
        "--candidates-file",
        str(candidates),
        "--audit-file",
        str(audit),
        "--output-tag",
        args.job_id,
        "--cuda",
        "--insufficient-peaks-policy",
        str(config["v6"]["acom"]["insufficient_peaks_policy"]),
    ]
    environment = os.environ.copy()
    environment["OR4D_CONFIG"] = str(config_path)
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=ROOT, env=environment, check=False)
    polar_command = [
        sys.executable,
        str(ROOT / "scripts" / "40_evaluate_v6_acom_polar.py"),
        "--config",
        str(config_path),
        "--candidate-file",
        str(candidates),
        "--ground-truth-file",
        str(expanded_gt),
        "--output-h5",
        str(polar_h5),
        "--output-json",
        str(polar_json),
    ]
    polar_completed = None
    if completed.returncode == 0:
        polar_completed = subprocess.run(
            polar_command,
            cwd=ROOT,
            env=environment,
            check=False,
        )
    report = {
        "schema": "or4d-clean-v6-acom-shard-run-v1",
        "job_id": args.job_id,
        "physical_gpu": physical_gpu,
        "peak_file": str(peak_file),
        "orientation_file": str(orientation_file),
        "expanded_ground_truth": str(expanded_gt),
        "logical_observation_count": len(ground_truth),
        "command": command,
        "returncode": completed.returncode,
        "polar_command": polar_command,
        "polar_returncode": (
            polar_completed.returncode if polar_completed is not None else None
        ),
        "wall_seconds": time.perf_counter() - started,
        "outputs": {
            "prediction": str(prediction),
            "details": str(details),
            "candidates": str(candidates),
            "audit": str(audit),
            "polar_h5": str(polar_h5),
            "polar_json": str(polar_json),
        },
    }
    run_record.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(completed.returncode, command)
    if polar_completed is not None and polar_completed.returncode != 0:
        raise subprocess.CalledProcessError(
            polar_completed.returncode, polar_command
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
