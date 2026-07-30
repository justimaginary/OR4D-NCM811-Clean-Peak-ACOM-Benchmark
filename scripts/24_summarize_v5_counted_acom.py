#!/usr/bin/env python3
"""Aggregate V5 Clean-C smoke ACOM JSON files without discarding failures."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DETECTOR_PATTERN = r"(?P<detector>autodisk|py4dstem|dog_rgm)"
DOSE_PATTERN = re.compile(
    r"clean_dose(?P<dose>\d+)_noise_poisson_only_repeat(?P<repeat>\d+)_"
    + DETECTOR_PATTERN
    + r"_peaks_first_born_smoke_evaluation\.json"
)
NOISE_PATTERN = re.compile(
    r"clean_dose(?P<dose>\d+)_noise_empad_g2_(?P<frames>\d+)frames?_"
    r"repeat(?P<repeat>\d+)_"
    + DETECTOR_PATTERN
    + r"_peaks_first_born_smoke_evaluation\.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=ROOT / "reports" / "v5" / "acom_counted_smoke",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports" / "v5" / "V5_ACOM_COUNTED_SUMMARY.json",
    )
    return parser.parse_args()


def row_from_files(
    evaluation_path: Path,
    match: re.Match[str],
) -> dict:
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    details_path = evaluation_path.with_name(
        evaluation_path.name.replace("_evaluation.json", "_details.json")
    )
    details = json.loads(details_path.read_text(encoding="utf-8"))
    metrics = evaluation["metrics"]
    row = {
        key: int(value) if key != "detector" else value
        for key, value in match.groupdict().items()
        if value is not None
    }
    row.update(
        {
            "num_input_samples": int(details["num_input_samples"]),
            "num_matched_samples": int(details["num_matched_samples"]),
            "num_indexing_failures": int(details["num_indexing_failures"]),
            "prediction_coverage": float(
                evaluation["coverage"]["prediction_coverage"]
            ),
            "median_misorientation_deg": float(
                metrics["median_misorientation_deg"]
            ),
            "p95_misorientation_deg": float(metrics["p95_misorientation_deg"]),
            "max_misorientation_deg": float(metrics["max_misorientation_deg"]),
            "accuracy_within_1deg": float(metrics["accuracy_within_1deg"]),
            "accuracy_within_2deg": float(metrics["accuracy_within_2deg"]),
            "accuracy_within_5deg": float(metrics["accuracy_within_5deg"]),
            "evaluation_file": evaluation_path.name,
            "details_file": details_path.name,
        }
    )
    return row


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    dose_rows: list[dict] = []
    noise_rows: list[dict] = []
    unmatched: list[str] = []
    for evaluation_path in sorted(input_dir.glob("*_evaluation.json")):
        dose_match = DOSE_PATTERN.fullmatch(evaluation_path.name)
        noise_match = NOISE_PATTERN.fullmatch(evaluation_path.name)
        if dose_match:
            dose_rows.append(row_from_files(evaluation_path, dose_match))
        elif noise_match:
            noise_rows.append(row_from_files(evaluation_path, noise_match))
        else:
            unmatched.append(evaluation_path.name)
    if unmatched:
        raise ValueError(f"Unrecognized counted ACOM result names: {unmatched}")
    if len(dose_rows) != 27 or len(noise_rows) != 12:
        raise ValueError(
            f"Expected 27 dose and 12 noise rows, got "
            f"{len(dose_rows)} and {len(noise_rows)}"
        )
    dose_rows.sort(key=lambda row: (row["dose"], row["detector"]))
    noise_rows.sort(key=lambda row: (row["frames"], row["detector"]))
    output = {
        "scope": {
            "track": "V5 Clean-C counted smoke",
            "num_fixed_orientations": 8,
            "repeat": 0,
            "dose_electrons": sorted({row["dose"] for row in dose_rows}),
            "instrument_noise_dose_electrons": 10000,
            "instrument_noise_frames": sorted(
                {row["frames"] for row in noise_rows}
            ),
            "detectors": sorted({row["detector"] for row in dose_rows}),
            "interpretation": (
                "Diagnostic smoke trend only; percentages are not the "
                "2048-orientation headline metric."
            ),
        },
        "dose_ladder": dose_rows,
        "instrument_noise_ladder": noise_rows,
    }
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Counted ACOM summary: {output_path}")


if __name__ == "__main__":
    main()
