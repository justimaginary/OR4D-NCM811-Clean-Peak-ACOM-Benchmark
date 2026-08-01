#!/usr/bin/env python3
"""Finish the 512-sample V5 [001] py4DSTEM/ACOM/Pyxem experiment.

This runner is intentionally sequential: it exposes one already verified
physical GPU, limits numerical-library threads, records every command, and
never retries or waits for a busy GPU.
"""

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
EXPECTED_CLEAN_C_CONDITIONS = 234


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--acom-workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def write_record(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.partial")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def gpu_status() -> tuple[dict[int, tuple[int, int, str]], list[tuple[str, int]]]:
    rows = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,utilization.gpu,uuid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    status: dict[int, tuple[int, int, str]] = {}
    for line in rows.splitlines():
        fields = [part.strip() for part in line.split(",")]
        if len(fields) == 4:
            status[int(fields[0])] = (int(fields[1]), int(fields[2]), fields[3])
    process_text = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    processes: list[tuple[str, int]] = []
    for line in process_text.splitlines():
        fields = [part.strip() for part in line.split(",")]
        if len(fields) == 2 and fields[1].isdigit():
            processes.append((fields[0], int(fields[1])))
    return status, processes


def require_empty_gpu(index: int) -> dict[str, int | str]:
    status, processes = gpu_status()
    if index not in status:
        raise RuntimeError(f"GPU {index} is not present")
    memory_mib, utilization, uuid = status[index]
    owners = [pid for process_uuid, pid in processes if process_uuid == uuid]
    if owners or memory_mib > 100 or utilization > 5:
        raise RuntimeError(
            f"GPU {index} is not empty: {memory_mib} MiB, "
            f"{utilization}%, process PIDs={owners}"
        )
    return {
        "physical_gpu": index,
        "uuid": uuid,
        "memory_used_MiB": memory_mib,
        "utilization_percent": utilization,
    }


def child_env(gpu: int) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["NUMBA_CUDA_USE_NVIDIA_BINDING"] = "1"
    env["OR4D_CONFIG"] = str(ROOT / "config" / "benchmark_v5.yaml")
    env["PYTHONPATH"] = str(ROOT / "src")
    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        env[variable] = "1"
    return env


def run_stage(
    *,
    name: str,
    command: list[str],
    log_path: Path,
    record_path: Path,
    payload: dict,
    env: dict[str, str],
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "stage": name,
        "status": "running",
        "started_unix": time.time(),
        "command": command,
        "log": str(log_path),
    }
    payload["stages"].append(row)
    write_record(record_path, payload)
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        try:
            subprocess.run(
                command,
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        except BaseException as error:
            row.update(
                {
                    "status": "failed",
                    "finished_unix": time.time(),
                    "seconds": time.perf_counter() - started,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            write_record(record_path, payload)
            raise
    row.update(
        {
            "status": "completed",
            "finished_unix": time.time(),
            "seconds": time.perf_counter() - started,
        }
    )
    write_record(record_path, payload)


def detection_complete(report: Path, expected_runs: int) -> bool:
    if not report.is_file():
        return False
    payload = json.loads(report.read_text(encoding="utf-8"))
    return len(payload.get("runs", [])) == expected_runs


def pyxem_complete(path: Path) -> bool:
    if not path.is_file():
        return False
    with h5py.File(path, "r") as h5:
        groups = ("clean_e", "clean_c_noiseless", "clean_c_counted")
        if any(group not in h5 for group in groups):
            return False
        return all(
            bool(np.all(np.asarray(h5[group]["condition_complete"][:], dtype=bool)))
            for group in groups
        )


def main() -> None:
    args = parse_args()
    if not 1 <= args.acom_workers <= 16:
        raise ValueError("--acom-workers must be in [1, 16]")
    data_root = args.data_root.resolve()
    ground_truth = data_root / "manifests/clean_v5_001_orientations.jsonl"
    if not ground_truth.is_file():
        raise FileNotFoundError(ground_truth)
    gpu_check = require_empty_gpu(args.gpu)
    record_path = data_root / "run_records/v5_001_py4_pyxem_full.json"
    payload = {
        "schema": "or4d-v5-001-py4-pyxem-full-v1",
        "started_unix": time.time(),
        "repo": str(ROOT),
        "data_root": str(data_root),
        "gpu_preflight": gpu_check,
        "cpu_worker_limit": 16,
        "acom_workers": args.acom_workers,
        "stages": [],
    }
    write_record(record_path, payload)
    env = child_env(args.gpu)
    logs = data_root / "logs/v5_001/py4_pyxem"
    peak_dir = data_root / "intermediates/detected_001_clean_c_full_py4dstem"
    detection_report_dir = data_root / "reports/v5_001_full_detectors"
    noiseless_report = detection_report_dir / "py4dstem_noiseless_detection.json"
    counted_report = detection_report_dir / "py4dstem_counted_detection.json"

    stages = [
        (
            "py4dstem_noiseless_detection",
            detection_complete(noiseless_report, 9),
            [
                sys.executable,
                str(ROOT / "scripts/03_extract_clean_disks.py"),
                "--image-file",
                str(data_root / "datasets/clean_v5_001_first_born_dose_noiseless_512.h5"),
                "--track",
                "dose_noiseless",
                "--detector",
                "py4dstem",
                "--compute-backend",
                "cuda",
                "--output-dir",
                str(peak_dir),
                "--report-output",
                str(noiseless_report),
                "--progress-every",
                "256",
            ],
        ),
        (
            "py4dstem_counted_detection",
            detection_complete(counted_report, 225),
            [
                sys.executable,
                str(ROOT / "scripts/03_extract_clean_disks.py"),
                "--image-file",
                str(data_root / "datasets/clean_v5_001_first_born_counted_512.h5"),
                "--track",
                "counted",
                "--detector",
                "py4dstem",
                "--compute-backend",
                "cuda",
                "--noise-manifest",
                str(data_root / "manifests/clean_v5_001_instrument_noise_512.h5"),
                "--output-dir",
                str(peak_dir),
                "--report-output",
                str(counted_report),
                "--progress-every",
                "256",
            ],
        ),
    ]
    for name, complete, command in stages:
        if args.resume and complete:
            payload["stages"].append({"stage": name, "status": "reused"})
            write_record(record_path, payload)
        else:
            run_stage(
                name=name,
                command=command,
                log_path=logs / f"{name}.log",
                record_path=record_path,
                payload=payload,
                env=env,
            )

    clean_c_results = data_root / "results/acom_top5/clean_001_c"
    py4_details = list(clean_c_results.glob("*_py4dstem_details.json"))
    if not (args.resume and len(py4_details) == EXPECTED_CLEAN_C_CONDITIONS):
        run_stage(
            name="py4dstem_acom_top5",
            command=[
                sys.executable,
                str(ROOT / "scripts/27_run_v5_clean_c_acom_full.py"),
                "--peak-dir",
                str(peak_dir),
                "--ground-truth-file",
                str(ground_truth),
                "--output-dir",
                str(clean_c_results),
                "--candidate-dir",
                str(data_root / "results/acom_top5_candidates/clean_001_c"),
                "--log-dir",
                str(data_root / "logs/v5_001/py4_acom"),
                "--headline-sample-role",
                "study_001",
                "--max-workers",
                str(args.acom_workers),
                "--cuda-visible-device",
                str(args.gpu),
                "--resume",
            ],
            log_path=logs / "py4dstem_acom_top5.log",
            record_path=record_path,
            payload=payload,
            env=env,
        )
    else:
        payload["stages"].append(
            {"stage": "py4dstem_acom_top5", "status": "reused"}
        )
        write_record(record_path, payload)

    pyxem_result = data_root / "results/pyxem_001_top5.h5"
    if not (args.resume and pyxem_complete(pyxem_result)):
        require_empty_gpu(args.gpu)
        run_stage(
            name="pyxem_top5_all_tracks",
            command=[
                sys.executable,
                str(ROOT / "scripts/25_run_v5_pyxem_template_matching.py"),
                "--data-root",
                str(data_root),
                "--output-file",
                str(pyxem_result),
                "--study",
                "001",
                "--track",
                "all",
                "--target",
                "gpu",
                "--resume",
            ],
            log_path=logs / "pyxem_top5_all_tracks.log",
            record_path=record_path,
            payload=payload,
            env=env,
        )
    else:
        payload["stages"].append(
            {"stage": "pyxem_top5_all_tracks", "status": "reused"}
        )
        write_record(record_path, payload)

    topk_dir = data_root / "reports/v5_001/topk"
    pyxem_summary = topk_dir / "V5_001_PYXEM_TOP5_FULL_SUMMARY.json"
    run_stage(
        name="evaluate_pyxem_top5",
        command=[
            sys.executable,
            str(ROOT / "scripts/26_evaluate_v5_pyxem.py"),
            "--result-file",
            str(pyxem_result),
            "--ground-truth-file",
            str(ground_truth),
            "--summary-output",
            str(pyxem_summary),
            "--clean-e-details-output",
            str(topk_dir / "V5_001_PYXEM_CLEAN_E_DETAILS.jsonl"),
        ],
        log_path=logs / "evaluate_pyxem_top5.log",
        record_path=record_path,
        payload=payload,
        env=env,
    )
    run_stage(
        name="finalize_001_top5",
        command=[
            sys.executable,
            str(ROOT / "scripts/29_finalize_v5_top5.py"),
            "--acom-clean-e-details-dir",
            str(data_root / "results/v5_001_suite/acom_001"),
            "--acom-clean-e-candidates-dir",
            str(data_root / "results/v5_001_suite/acom_001"),
            "--acom-clean-c-details-dir",
            str(clean_c_results),
            "--acom-clean-c-candidates-dir",
            str(data_root / "results/acom_top5_candidates/clean_001_c"),
            "--pyxem-summary",
            str(pyxem_summary),
            "--output-dir",
            str(topk_dir),
            "--artifact-prefix",
            "V5_001",
            "--dataset-label",
            "V5 independent [001] study",
            "--sample-count",
            "512",
        ],
        log_path=logs / "finalize_001_top5.log",
        record_path=record_path,
        payload=payload,
        env=env,
    )
    payload["finished_unix"] = time.time()
    payload["status"] = "completed"
    write_record(record_path, payload)
    print(f"Run record: {record_path}")


if __name__ == "__main__":
    main()
