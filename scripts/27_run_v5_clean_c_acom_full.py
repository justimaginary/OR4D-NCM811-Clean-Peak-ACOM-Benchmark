#!/usr/bin/env python3
"""Run the frozen ACOM evaluator on every full Clean-C peak file."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAME = re.compile(
    r"clean_(?P<condition>.+)_(?P<detector>"
    r"autodisk|py4dstem|dog_rgm)_peaks_first_born\.h5"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--peak-dir", type=Path, required=True)
    parser.add_argument("--ground-truth-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument(
        "--cuda-visible-device",
        help=(
            "One already-verified empty physical GPU ID. Omit for CPU. "
            "The runner exposes only this device to every child."
        ),
    )
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def run_one(
    peak_file: Path,
    *,
    ground_truth: Path,
    output_dir: Path,
    candidate_dir: Path,
    log_dir: Path,
    resume: bool,
    cuda_visible_device: str | None,
) -> dict:
    match = NAME.fullmatch(peak_file.name)
    if match is None:
        raise ValueError(f"unrecognized Clean-C peak filename: {peak_file.name}")
    condition = match.group("condition")
    detector = match.group("detector")
    stem = f"{condition}_{detector}"
    outputs = {
        "predictions": output_dir / f"{stem}_predictions.jsonl",
        "details": output_dir / f"{stem}_details.json",
        "audit": output_dir / f"{stem}_audit.json",
        "evaluation": output_dir / f"{stem}_evaluation.json",
        "candidates": candidate_dir / f"{stem}_candidates.h5",
    }
    if resume and all(path.is_file() for path in outputs.values()):
        return {
            "condition": condition,
            "detector": detector,
            "peak_file": str(peak_file),
            "status": "reused",
            "outputs": {key: str(value) for key, value in outputs.items()},
        }
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cuda_visible_device or ""
    env["OR4D_CONFIG"] = "config/benchmark_v5.yaml"
    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        env[variable] = "1"
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{stem}.log"
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "07_run_acom_baseline.py"),
                "--peak-file",
                str(peak_file),
                "--ground-truth-file",
                str(ground_truth),
                "--ground-truth-id-prefix",
                "clean_",
                "--output-tag",
                f"v5_c_{stem}",
                "--prediction-file",
                str(outputs["predictions"]),
                "--details-file",
                str(outputs["details"]),
                "--audit-file",
                str(outputs["audit"]),
                "--candidates-file",
                str(outputs["candidates"]),
                "--cuda" if cuda_visible_device is not None else "--no-cuda",
                "--insufficient-peaks-policy",
                "skip",
            ],
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "06_evaluate_submission.py"),
                str(outputs["predictions"]),
                "--track",
                "clean",
                "--ground-truth-file",
                str(ground_truth),
                "--ground-truth-id-prefix",
                "clean_",
                "--headline-sample-role",
                "headline_core",
                "--allow-subset",
                "--output",
                str(outputs["evaluation"]),
            ],
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    return {
        "condition": condition,
        "detector": detector,
        "peak_file": str(peak_file),
        "status": "completed",
        "seconds": time.perf_counter() - started,
        "log": str(log_path),
        "outputs": {key: str(value) for key, value in outputs.items()},
    }


def main() -> None:
    args = parse_args()
    if not 1 <= args.max_workers <= 16:
        raise ValueError("--max-workers must be between 1 and 16")
    if args.shard_count < 1:
        raise ValueError("--shard-count must be positive")
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("--shard-index must be in [0, shard-count)")
    if args.cuda_visible_device is not None and not args.cuda_visible_device.isdigit():
        raise ValueError("--cuda-visible-device must be one physical GPU index")
    peak_dir = args.peak_dir.resolve()
    ground_truth = args.ground_truth_file.resolve()
    peak_files = sorted(peak_dir.glob("*.h5"))
    if not peak_files:
        raise FileNotFoundError(f"no peak files under {peak_dir}")
    unknown = [path.name for path in peak_files if NAME.fullmatch(path.name) is None]
    if unknown:
        raise ValueError(f"unrecognized peak files: {unknown}")
    peak_files = [
        path
        for index, path in enumerate(peak_files)
        if index % args.shard_count == args.shard_index
    ]
    if not peak_files:
        raise ValueError("selected shard contains no peak files")

    records: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.max_workers
    ) as executor:
        futures = {
            executor.submit(
                run_one,
                peak_file,
                ground_truth=ground_truth,
                output_dir=args.output_dir.resolve(),
                candidate_dir=args.candidate_dir.resolve(),
                log_dir=args.log_dir.resolve(),
                resume=args.resume,
                cuda_visible_device=args.cuda_visible_device,
            ): peak_file
            for peak_file in peak_files
        }
        for future in concurrent.futures.as_completed(futures):
            peak_file = futures[future]
            try:
                record = future.result()
            except Exception as error:
                record = {
                    "peak_file": str(peak_file),
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            records.append(record)
            print(
                f"{len(records)}/{len(peak_files)} "
                f"{peak_file.name}: {record['status']}",
                flush=True,
            )
    records.sort(key=lambda row: row["peak_file"])
    manifest = {
        "num_peak_files": len(peak_files),
        "num_completed": sum(row["status"] != "failed" for row in records),
        "num_failed": sum(row["status"] == "failed" for row in records),
        "max_workers": args.max_workers,
        "cuda_visible_device": args.cuda_visible_device,
        "shard_count": args.shard_count,
        "shard_index": args.shard_index,
        "runs": records,
    }
    manifest_path = args.output_dir.resolve() / (
        f"run_manifest_shard{args.shard_index}of{args.shard_count}.json"
    )
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Manifest: {manifest_path}")
    if manifest["num_failed"]:
        raise RuntimeError(f"{manifest['num_failed']} ACOM conditions failed")


if __name__ == "__main__":
    main()
