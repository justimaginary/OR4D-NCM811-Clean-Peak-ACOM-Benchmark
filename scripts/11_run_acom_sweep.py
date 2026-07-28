#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from or4d_common import load_config  # noqa: E402


def step_tag(step: float) -> str:
    text = f"{step:g}".replace(".", "p")
    return f"angle_{text}deg"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run isolated ACOM baselines for multiple angular steps."
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        type=float,
        help="Override config acom.sweep_angle_steps_deg.",
    )
    return parser.parse_args()


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    args = parse_args()
    config = load_config()
    canonical = float(config["acom"]["angle_step_zone_axis_deg"])
    canonical_in_plane = float(config["acom"]["angle_step_in_plane_deg"])
    if canonical != canonical_in_plane:
        raise ValueError("Sweep runner requires equal zone-axis and in-plane steps")
    steps = (
        [float(value) for value in args.steps]
        if args.steps
        else [float(value) for value in config["acom"]["sweep_angle_steps_deg"]]
    )
    for step in steps:
        is_canonical = step == canonical
        tag = "" if is_canonical else step_tag(step)
        suffix = f"_{tag}" if tag else ""
        baseline = [
            sys.executable,
            str(ROOT / "scripts" / "07_run_acom_baseline.py"),
            "--angle-step-deg",
            f"{step:g}",
        ]
        if tag:
            baseline.extend(["--output-tag", tag])
        run(baseline)
        run(
            [
                sys.executable,
                str(ROOT / "scripts" / "06_evaluate_submission.py"),
                str(
                    ROOT
                    / "submissions"
                    / f"acom_clean_predictions{suffix}.jsonl"
                ),
                "--track",
                "clean",
                "--output",
                str(
                    ROOT
                    / "reports"
                    / f"acom_clean_evaluation{suffix}.json"
                ),
            ]
        )

    print("ACOM sweep finished.")


if __name__ == "__main__":
    main()
