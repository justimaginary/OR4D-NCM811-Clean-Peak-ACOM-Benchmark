#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports" / "v4"
RUN_DIR = REPORT_DIR / "runs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run identical ACOM settings on every Clean-C detector output."
    )
    parser.add_argument("--detection-report", type=Path, required=True)
    parser.add_argument("--baseline-details", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPORT_DIR / "clean_counted_acom_comparison.json",
    )
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--allow-subset", action="store_true")
    parser.add_argument(
        "--reuse-details",
        action="store_true",
        help="Skip ACOM and rebuild only the comparison from existing details.",
    )
    return parser.parse_args()


def run_one(run: dict, allow_subset: bool) -> tuple[str, Path]:
    detector = str(run["detector"])
    variant = str(run["variant"])
    tag = f"{variant}_{detector}"
    peak_file = Path(run["output"]).resolve()
    prediction = ROOT / "submissions" / f"acom_clean_{tag}.jsonl"
    details = RUN_DIR / f"acom_clean_details_{tag}.json"
    audit = RUN_DIR / f"acom_plan_audit_{tag}.json"
    candidates = RUN_DIR / f"acom_candidates_{tag}.h5"
    command = [
        sys.executable,
        str(ROOT / "scripts" / "07_run_acom_baseline.py"),
        "--peak-file",
        str(peak_file),
        "--output-tag",
        tag,
        "--prediction-file",
        str(prediction),
        "--details-file",
        str(details),
        "--audit-file",
        str(audit),
        "--candidates-file",
        str(candidates),
    ]
    if allow_subset:
        command.append("--allow-subset")
    subprocess.run(command, cwd=ROOT, check=True)
    return tag, details


def main() -> None:
    args = parse_args()
    if args.max_workers <= 0:
        raise ValueError("--max-workers must be positive")
    report = json.loads(args.detection_report.read_text(encoding="utf-8"))
    runs = report["runs"]
    if args.reuse_details:
        completed = [
            (
                f"{run['variant']}_{run['detector']}",
                RUN_DIR
                / f"acom_clean_details_{run['variant']}_{run['detector']}.json",
            )
            for run in runs
        ]
        missing = [str(path) for _, path in completed if not path.exists()]
        if missing:
            raise FileNotFoundError(f"missing ACOM details: {missing}")
    else:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.max_workers
        ) as executor:
            futures = [
                executor.submit(run_one, run, args.allow_subset) for run in runs
            ]
            completed = [future.result() for future in futures]

    command = [
        sys.executable,
        str(ROOT / "scripts" / "14_compare_clean_acom.py"),
        "--baseline-details",
        str(args.baseline_details.resolve()),
        "--output",
        str(args.output.resolve()),
    ]
    for label, details in completed:
        command.extend(["--candidate", f"{label}={details}"])
    subprocess.run(command, cwd=ROOT, check=True)
    print(f"Counted ACOM comparison: {args.output.resolve()}")


if __name__ == "__main__":
    main()
