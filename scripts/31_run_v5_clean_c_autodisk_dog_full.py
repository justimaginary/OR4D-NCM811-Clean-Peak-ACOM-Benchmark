#!/usr/bin/env python3
"""Run the missing full Clean-C AutoDisk/DoG-RGM -> ACOM Top-5 paths.

The task graph is finite and deterministic. CPU-only disk detection and the
single-GPU ACOM stage overlap, while every child is limited to one numerical
library thread. Re-running with ``--resume`` reuses only complete artifacts.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Condition:
    detector: str
    track: str
    dose_index: int
    dose_electrons: int
    repeat: int | None
    noise_level: str

    @property
    def condition_name(self) -> str:
        base = f"dose{self.dose_electrons}_noise_{self.noise_level}"
        return base if self.repeat is None else f"{base}_repeat{self.repeat}"

    @property
    def stem(self) -> str:
        return f"{self.condition_name}_{self.detector}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--ground-truth-file", type=Path, required=True)
    parser.add_argument("--study", choices=("main", "001"), default="main")
    parser.add_argument(
        "--cuda-visible-device",
        required=True,
        help="One or two comma-separated physical GPU indices, for example 0 or 0,1.",
    )
    parser.add_argument("--detection-workers", type=int, default=8)
    parser.add_argument("--acom-workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def child_env(cuda_device: str | None) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cuda_device or ""
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


def study_files(data_root: Path, study: str) -> dict[str, Path | str]:
    if study == "001":
        return {
            "counted": data_root / "datasets/clean_v5_001_first_born_counted_512.h5",
            "noiseless": data_root / "datasets/clean_v5_001_first_born_dose_noiseless_512.h5",
            "noise": data_root / "manifests/clean_v5_001_instrument_noise_512.h5",
            "detected_prefix": "detected_001_clean_c_full",
            "result_track": "clean_001_c",
            "sample_role": "study_001",
            "tag_prefix": "v5_001_c",
        }
    return {
        "counted": data_root / "datasets/clean_v5_first_born_counted_2048.h5",
        "noiseless": data_root / "datasets/clean_v5_first_born_dose_noiseless_2048.h5",
        "noise": data_root / "manifests/clean_v5_instrument_noise_2048.h5",
        "detected_prefix": "detected_clean_c_full",
        "result_track": "clean_c",
        "sample_role": "headline_core",
        "tag_prefix": "v5_c",
    }


def load_conditions(data_root: Path, study: str) -> list[Condition]:
    files = study_files(data_root, study)
    counted = Path(files["counted"])
    noise = Path(files["noise"])
    with h5py.File(counted, "r") as h5:
        doses = [int(value) for value in h5["dose_electrons"][:]]
        repeats = int(h5["images/counts"].shape[2])
    with h5py.File(noise, "r") as h5:
        levels = [
            value.decode() if isinstance(value, bytes) else str(value)
            for value in h5["noise_level_id"][:]
        ]
    noisy_levels = [level for level in levels if level != "noiseless"]
    conditions: list[Condition] = []
    for detector in ("autodisk", "dog_rgm"):
        for dose_index, dose in enumerate(doses):
            conditions.append(
                Condition(detector, "dose_noiseless", dose_index, dose, None, "noiseless")
            )
            for level in noisy_levels:
                for repeat in range(repeats):
                    conditions.append(
                        Condition(detector, "counted", dose_index, dose, repeat, level)
                    )
    expected = 2 * (len(doses) + len(doses) * len(noisy_levels) * repeats)
    if len(conditions) != expected:
        raise RuntimeError("condition construction is inconsistent")
    return conditions


def artifact_paths(data_root: Path, condition: Condition, study: str) -> dict[str, Path]:
    files = study_files(data_root, study)
    peak_dir = data_root / "intermediates" / f"{files['detected_prefix']}_{condition.detector}"
    peak = peak_dir / f"clean_{condition.condition_name}_{condition.detector}_peaks_first_born.h5"
    result_dir = data_root / "results" / "acom_top5" / str(files["result_track"])
    candidate_dir = data_root / "results" / "acom_top5_candidates" / str(files["result_track"])
    report_stem = "v5_001_full_detectors" if study == "001" else "v5_full_detectors"
    return {
        "peak": peak,
        "detection_report": data_root / "reports" / report_stem / f"{condition.stem}_detection.json",
        "detection_log": data_root / "logs" / report_stem / "detection" / f"{condition.stem}.log",
        "predictions": result_dir / f"{condition.stem}_predictions.jsonl",
        "details": result_dir / f"{condition.stem}_details.json",
        "audit": result_dir / f"{condition.stem}_audit.json",
        "evaluation": result_dir / f"{condition.stem}_evaluation.json",
        "candidates": candidate_dir / f"{condition.stem}_candidates.h5",
        "acom_log": data_root / "logs" / report_stem / "acom" / f"{condition.stem}.log",
    }


def complete_file(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def run_detection(
    data_root: Path,
    condition: Condition,
    *,
    study: str,
    resume: bool,
) -> dict:
    files = study_files(data_root, study)
    paths = artifact_paths(data_root, condition, study)
    if resume and complete_file(paths["peak"]) and complete_file(paths["detection_report"]):
        return {"status": "reused", "seconds": 0.0, "paths": paths}
    for key in ("peak", "detection_report", "detection_log"):
        paths[key].parent.mkdir(parents=True, exist_ok=True)
    image_file = Path(files["noiseless"] if condition.track == "dose_noiseless" else files["counted"])
    command = [
        sys.executable,
        str(ROOT / "scripts" / "03_extract_clean_disks.py"),
        "--image-file",
        str(image_file),
        "--track",
        condition.track,
        "--detector",
        condition.detector,
        "--dose-index",
        str(condition.dose_index),
        "--output-dir",
        str(paths["peak"].parent),
        "--report-output",
        str(paths["detection_report"]),
        "--progress-every",
        "256",
    ]
    if condition.repeat is not None:
        command.extend(["--repeat", str(condition.repeat)])
    if condition.track == "counted":
        command.extend(
            [
                "--noise-manifest",
                str(files["noise"]),
                "--noise-level",
                condition.noise_level,
            ]
        )
    started = time.perf_counter()
    with paths["detection_log"].open("w", encoding="utf-8") as log:
        subprocess.run(
            command,
            cwd=ROOT,
            env=child_env(None),
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    if not complete_file(paths["peak"]):
        raise RuntimeError(f"detection did not create {paths['peak']}")
    report = json.loads(paths["detection_report"].read_text(encoding="utf-8"))
    if len(report.get("runs", [])) != 1:
        raise RuntimeError("single-condition detection report must contain one run")
    return {
        "status": "completed",
        "seconds": time.perf_counter() - started,
        "num_detection_failures": int(report["runs"][0]["num_failures"]),
        "paths": paths,
    }


def run_acom(
    data_root: Path,
    ground_truth: Path,
    condition: Condition,
    *,
    study: str,
    cuda_device: str,
    resume: bool,
) -> dict:
    files = study_files(data_root, study)
    paths = artifact_paths(data_root, condition, study)
    required = ("predictions", "details", "audit", "evaluation", "candidates")
    if resume and all(complete_file(paths[key]) for key in required):
        return {"status": "reused", "seconds": 0.0, "paths": paths}
    for key in required + ("acom_log",):
        paths[key].parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with paths["acom_log"].open("w", encoding="utf-8") as log:
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "07_run_acom_baseline.py"),
                "--peak-file",
                str(paths["peak"]),
                "--ground-truth-file",
                str(ground_truth),
                "--ground-truth-id-prefix",
                "clean_",
                "--output-tag",
                f"{files['tag_prefix']}_{condition.stem}",
                "--prediction-file",
                str(paths["predictions"]),
                "--details-file",
                str(paths["details"]),
                "--audit-file",
                str(paths["audit"]),
                "--candidates-file",
                str(paths["candidates"]),
                "--cuda",
                "--insufficient-peaks-policy",
                "skip",
            ],
            cwd=ROOT,
            env=child_env(cuda_device),
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "06_evaluate_submission.py"),
                str(paths["predictions"]),
                "--track",
                "clean",
                "--ground-truth-file",
                str(ground_truth),
                "--ground-truth-id-prefix",
                "clean_",
                "--headline-sample-role",
                str(files["sample_role"]),
                "--allow-subset",
                "--output",
                str(paths["evaluation"]),
            ],
            cwd=ROOT,
            env=child_env(cuda_device),
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    if not all(complete_file(paths[key]) for key in required):
        raise RuntimeError(f"ACOM outputs incomplete for {condition.stem}")
    return {
        "status": "completed",
        "seconds": time.perf_counter() - started,
        "paths": paths,
    }


def serializable_result(condition: Condition, stage: str, result: dict) -> dict:
    return {
        "condition": asdict(condition),
        "stage": stage,
        **{
            key: ({name: str(path) for name, path in value.items()} if key == "paths" else value)
            for key, value in result.items()
        },
    }


def write_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.partial")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    cuda_devices = [value.strip() for value in args.cuda_visible_device.split(",")]
    if not cuda_devices or any(not value.isdigit() for value in cuda_devices):
        raise ValueError("--cuda-visible-device must contain physical GPU indices")
    if len(cuda_devices) > 2 or len(set(cuda_devices)) != len(cuda_devices):
        raise ValueError("at most two distinct physical GPU indices are allowed")
    if not 1 <= args.detection_workers <= 24:
        raise ValueError("--detection-workers must be in [1, 24]")
    if not 1 <= args.acom_workers <= 8:
        raise ValueError("--acom-workers must be in [1, 8]")
    worker_limit = 16 if len(cuda_devices) == 1 else 32
    if args.detection_workers + args.acom_workers > worker_limit:
        raise ValueError(
            f"combined worker count must not exceed {worker_limit} for "
            f"{len(cuda_devices)} visible physical GPU(s)"
        )
    data_root = args.data_root.resolve()
    ground_truth = args.ground_truth_file.resolve()
    conditions = load_conditions(data_root, args.study)
    manifest_name = (
        "v5_001_clean_c_autodisk_dog_full.json"
        if args.study == "001"
        else "v5_clean_c_autodisk_dog_full.json"
    )
    manifest_path = data_root / "run_records" / manifest_name
    payload = {
        "schema": "or4d-v5-clean-c-autodisk-dog-full-v1",
        "started_unix": time.time(),
        "repo": str(ROOT),
        "data_root": str(data_root),
        "ground_truth": str(ground_truth),
        "study": args.study,
        "cuda_visible_devices": cuda_devices,
        "detection_workers": args.detection_workers,
        "acom_workers": args.acom_workers,
        "num_conditions": len(conditions),
        "records": [],
    }
    write_manifest(manifest_path, payload)
    acom_futures: dict[concurrent.futures.Future, Condition] = {}
    failures: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.detection_workers
    ) as detection_pool, concurrent.futures.ThreadPoolExecutor(
        max_workers=args.acom_workers
    ) as acom_pool:
        detection_futures = {
            detection_pool.submit(
                run_detection, data_root, condition, study=args.study, resume=args.resume
            ): condition
            for condition in conditions
        }
        detected = 0
        for future in concurrent.futures.as_completed(detection_futures):
            condition = detection_futures[future]
            detected += 1
            try:
                result = future.result()
                payload["records"].append(serializable_result(condition, "detection", result))
                cuda_device = cuda_devices[(detected - 1) % len(cuda_devices)]
                acom_future = acom_pool.submit(
                    run_acom,
                    data_root,
                    ground_truth,
                    condition,
                    study=args.study,
                    cuda_device=cuda_device,
                    resume=args.resume,
                )
                acom_futures[acom_future] = condition
                print(
                    f"detect {detected}/{len(conditions)} {condition.stem}: {result['status']}",
                    flush=True,
                )
            except Exception as error:
                failure = {
                    "condition": asdict(condition),
                    "stage": "detection",
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
                payload["records"].append(failure)
                failures.append(failure)
            write_manifest(manifest_path, payload)
        indexed = 0
        for future in concurrent.futures.as_completed(acom_futures):
            condition = acom_futures[future]
            indexed += 1
            try:
                result = future.result()
                payload["records"].append(serializable_result(condition, "acom", result))
                print(
                    f"acom {indexed}/{len(acom_futures)} {condition.stem}: {result['status']}",
                    flush=True,
                )
            except Exception as error:
                failure = {
                    "condition": asdict(condition),
                    "stage": "acom",
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
                payload["records"].append(failure)
                failures.append(failure)
            write_manifest(manifest_path, payload)
    payload["finished_unix"] = time.time()
    payload["num_failures"] = len(failures)
    write_manifest(manifest_path, payload)
    print(f"Manifest: {manifest_path}")
    if failures:
        raise RuntimeError(f"{len(failures)} program-stage failures")


if __name__ == "__main__":
    main()
