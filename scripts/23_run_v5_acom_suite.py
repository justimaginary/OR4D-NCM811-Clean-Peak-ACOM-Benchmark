#!/usr/bin/env python3
"""Run the frozen V5 Clean ACOM comparisons from external server artifacts."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="V5 data root containing manifests/ and intermediates/.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "reports" / "v5",
        help="Directory receiving acom_full/ and acom_001/ result folders.",
    )
    parser.add_argument(
        "--study",
        choices=("main", "001", "all"),
        default="all",
    )
    parser.add_argument(
        "--cuda",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use py4DSTEM CUDA plan construction. Set CUDA_VISIBLE_DEVICES to "
            "exactly one empty physical GPU before using this flag."
        ),
    )
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=8,
        help="Thread limit propagated to OpenMP/BLAS libraries (maximum 16).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip a method only when all four of its result files already exist.",
    )
    return parser.parse_args()


def require_one_empty_gpu() -> str:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    parts = [part.strip() for part in visible.split(",") if part.strip()]
    if len(parts) != 1 or not parts[0].isdigit():
        raise RuntimeError(
            "--cuda requires CUDA_VISIBLE_DEVICES to contain exactly one "
            "numeric physical GPU index"
        )
    gpu_index = parts[0]
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    rows = {
        fields[0]: (int(fields[1]), int(fields[2]))
        for line in output.splitlines()
        if len(fields := [part.strip() for part in line.split(",")]) == 3
    }
    if gpu_index not in rows:
        raise RuntimeError(f"GPU {gpu_index} is absent from nvidia-smi output")
    memory_mib, utilization = rows[gpu_index]
    if memory_mib > 100 or utilization > 5:
        raise RuntimeError(
            f"GPU {gpu_index} is not empty: memory={memory_mib} MiB, "
            f"utilization={utilization}%"
        )
    return gpu_index


def study_definition(data_root: Path, study: str) -> dict:
    if study == "main":
        return {
            "output_dir": "acom_full",
            "manifest": data_root / "manifests" / "clean_v5_orientations.jsonl",
            "headline_role": "headline_core",
            "count": 2048,
            "peaks": {
                "oracle": (
                    data_root
                    / "intermediates"
                    / "clean_v5_first_born_oracle_2048.h5"
                ),
                **{
                    detector: (
                        data_root
                        / "intermediates"
                        / "detected_expectation_2048"
                        / f"clean_expectation_{detector}_peaks_first_born.h5"
                    )
                    for detector in ("autodisk", "py4dstem", "dog_rgm")
                },
            },
        }
    return {
        "output_dir": "acom_001",
        "manifest": data_root / "manifests" / "clean_v5_001_orientations.jsonl",
        "headline_role": "study_001",
        "count": 512,
        "peaks": {
            "oracle": (
                data_root
                / "intermediates"
                / "clean_v5_001_first_born_oracle_512.h5"
            ),
            **{
                detector: (
                    data_root
                    / "intermediates"
                    / "detected_001_expectation_512"
                    / f"clean_expectation_{detector}_peaks_first_born.h5"
                )
                for detector in ("autodisk", "py4dstem", "dog_rgm")
            },
        },
    }


def run_command(command: list[str], env: dict[str, str]) -> float:
    print("+", " ".join(command), flush=True)
    start = time.perf_counter()
    subprocess.run(command, cwd=ROOT, env=env, check=True)
    return time.perf_counter() - start


def main() -> None:
    args = parse_args()
    if not 1 <= args.cpu_threads <= 16:
        raise ValueError("--cpu-threads must be between 1 and 16")
    data_root = args.data_root.resolve()
    output_root = args.output_root.resolve()
    studies = ("main", "001") if args.study == "all" else (args.study,)
    env = os.environ.copy()
    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        env[variable] = str(args.cpu_threads)

    gpu_index = require_one_empty_gpu() if args.cuda else None
    run_records: list[dict] = []
    for study in studies:
        definition = study_definition(data_root, study)
        manifest = definition["manifest"]
        required = [manifest, *definition["peaks"].values()]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing V5 ACOM inputs: {missing}")
        result_dir = output_root / definition["output_dir"]
        result_dir.mkdir(parents=True, exist_ok=True)

        for method, peak_file in definition["peaks"].items():
            paths = {
                "prediction": result_dir / f"{method}_predictions.jsonl",
                "details": result_dir / f"{method}_details.json",
                "audit": result_dir / f"{method}_audit.json",
                "evaluation": result_dir / f"{method}_evaluation.json",
            }
            if args.resume and all(path.is_file() for path in paths.values()):
                print(f"Skipping completed {study}/{method}", flush=True)
                continue
            if args.cuda:
                gpu_index = require_one_empty_gpu()
            tag = f"v5_{'001_' if study == '001' else ''}{method}_{definition['count']}"
            acom_command = [
                sys.executable,
                str(ROOT / "scripts" / "07_run_acom_baseline.py"),
                "--peak-file",
                str(peak_file),
                "--ground-truth-file",
                str(manifest),
                "--ground-truth-id-prefix",
                "clean_",
                "--output-tag",
                tag,
                "--prediction-file",
                str(paths["prediction"]),
                "--details-file",
                str(paths["details"]),
                "--audit-file",
                str(paths["audit"]),
                "--cuda" if args.cuda else "--no-cuda",
            ]
            evaluation_command = [
                sys.executable,
                str(ROOT / "scripts" / "06_evaluate_submission.py"),
                str(paths["prediction"]),
                "--track",
                "clean",
                "--ground-truth-file",
                str(manifest),
                "--ground-truth-id-prefix",
                "clean_",
                "--headline-sample-role",
                definition["headline_role"],
                "--output",
                str(paths["evaluation"]),
            ]
            acom_seconds = run_command(acom_command, env)
            evaluation_seconds = run_command(evaluation_command, env)
            run_records.append(
                {
                    "study": study,
                    "method": method,
                    "num_samples": definition["count"],
                    "source_peaks": str(peak_file),
                    "manifest": str(manifest),
                    "cuda": bool(args.cuda),
                    "physical_gpu_index": gpu_index,
                    "cpu_threads": args.cpu_threads,
                    "acom_seconds": acom_seconds,
                    "evaluation_seconds": evaluation_seconds,
                    "outputs": {key: str(value) for key, value in paths.items()},
                }
            )

    manifest_path = output_root / "v5_acom_suite_run.json"
    manifest_path.write_text(
        json.dumps({"runs": run_records}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Run manifest: {manifest_path}")


if __name__ == "__main__":
    main()
