#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from pymatgen.core import Structure

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = Path(
    os.environ.get("OR4D_REPORT_V4_DIR", ROOT / "reports" / "v4")
).resolve()
sys.path.insert(0, str(ROOT / "src"))

from or4d_common import (  # noqa: E402
    cif_path,
    friedel_aware_misorientation_deg,
    load_config,
    proper_point_group_rotations,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Clean ACOM predictions with physical-oracle ACOM."
    )
    parser.add_argument("--baseline-details", type=Path, required=True)
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        metavar="LABEL=DETAILS_JSON",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPORT_DIR / "clean_acom_comparison.json",
    )
    return parser.parse_args()


def load_samples(path: Path) -> dict[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(row["sample_id"]): row for row in payload["samples"]}


def metric_summary(values: np.ndarray) -> dict:
    return {
        "mean_deg": float(values.mean()),
        "median_deg": float(np.median(values)),
        "p95_deg": float(np.percentile(values, 95)),
        "max_deg": float(values.max()),
        "acc_at_1deg": float(np.mean(values <= 1.0)),
        "acc_at_2deg": float(np.mean(values <= 2.0)),
        "acc_at_5deg": float(np.mean(values <= 5.0)),
        "catastrophic_gt_5deg": int(np.count_nonzero(values > 5.0)),
    }


def main() -> None:
    args = parse_args()
    config = load_config()
    symmetries = proper_point_group_rotations(
        Structure.from_file(cif_path(config))
    )
    baseline_path = args.baseline_details.resolve()
    baseline = load_samples(baseline_path)
    baseline_gt = np.asarray(
        [
            baseline[sample_id]["friedel_equivalent_misorientation_deg"]
            for sample_id in sorted(baseline)
        ],
        dtype=float,
    )
    candidates = []

    for specification in args.candidate:
        if "=" not in specification:
            raise ValueError("--candidate must be LABEL=DETAILS_JSON")
        label, path_text = specification.split("=", 1)
        path = Path(path_text).resolve()
        candidate = load_samples(path)
        if set(candidate) != set(baseline):
            raise ValueError(f"sample IDs differ for {label}")
        sample_ids = sorted(baseline)
        delta = np.asarray(
            [
                friedel_aware_misorientation_deg(
                    np.asarray(
                        candidate[sample_id][
                            "predicted_orientation_matrix_sample_to_crystal"
                        ],
                        dtype=float,
                    ),
                    np.asarray(
                        baseline[sample_id][
                            "predicted_orientation_matrix_sample_to_crystal"
                        ],
                        dtype=float,
                    ),
                    symmetries,
                )
                for sample_id in sample_ids
            ],
            dtype=float,
        )
        candidate_gt = np.asarray(
            [
                candidate[sample_id]["friedel_equivalent_misorientation_deg"]
                for sample_id in sample_ids
            ],
            dtype=float,
        )
        candidates.append(
            {
                "label": label,
                "details_file": str(path),
                "versus_physical_oracle_acom": metric_summary(delta),
                "versus_physical_oracle_acom_on_baseline_gt_le5": (
                    metric_summary(delta[baseline_gt <= 5.0])
                ),
                "versus_ground_truth": metric_summary(candidate_gt),
                "acc_at_2deg_change_from_physical_oracle": float(
                    np.mean(candidate_gt <= 2.0) - np.mean(baseline_gt <= 2.0)
                ),
                "new_gt_5deg_failures": [
                    sample_id
                    for sample_id, base_error, candidate_error in zip(
                        sample_ids, baseline_gt, candidate_gt
                    )
                    if base_error <= 5.0 and candidate_error > 5.0
                ],
                "per_sample": [
                    {
                        "sample_id": sample_id,
                        "delta_from_physical_oracle_acom_deg": float(delta_value),
                        "ground_truth_error_deg": float(candidate_error),
                    }
                    for sample_id, delta_value, candidate_error in zip(
                        sample_ids, delta, candidate_gt
                    )
                ],
            }
        )

    grouped: dict[tuple[str, int], list[dict]] = {}
    for candidate in candidates:
        label = str(candidate["label"])
        if not label.startswith("counted_dose") or "_repeat" not in label:
            continue
        prefix, detector = label.rsplit("_", 1)
        dose_text = prefix.split("_repeat", 1)[0].replace("counted_dose", "")
        if dose_text.isdigit():
            grouped.setdefault((detector, int(dose_text)), []).append(candidate)
    dose_summary = []
    for (detector, dose), group in sorted(grouped.items()):
        row = {
            "detector": detector,
            "dose_electrons": dose,
            "repeats": len(group),
        }
        metric_paths = {
            "delta_median_deg": (
                "versus_physical_oracle_acom",
                "median_deg",
            ),
            "delta_p95_deg": ("versus_physical_oracle_acom", "p95_deg"),
            "stable_delta_median_deg": (
                "versus_physical_oracle_acom_on_baseline_gt_le5",
                "median_deg",
            ),
            "stable_delta_p95_deg": (
                "versus_physical_oracle_acom_on_baseline_gt_le5",
                "p95_deg",
            ),
            "ground_truth_acc_at_2deg": ("versus_ground_truth", "acc_at_2deg"),
            "ground_truth_acc_at_5deg": ("versus_ground_truth", "acc_at_5deg"),
            "ground_truth_catastrophic_gt_5deg": (
                "versus_ground_truth",
                "catastrophic_gt_5deg",
            ),
        }
        for output_name, (section, metric) in metric_paths.items():
            values = np.asarray(
                [candidate[section][metric] for candidate in group], dtype=float
            )
            row[f"{output_name}_mean"] = float(values.mean())
            row[f"{output_name}_std"] = (
                float(values.std(ddof=1)) if len(values) > 1 else 0.0
            )
        dose_summary.append(row)

    report = {
        "scope": "Clean only",
        "physical_oracle_details": str(baseline_path),
        "physical_oracle_versus_ground_truth": metric_summary(baseline_gt),
        "candidates": candidates,
        "dose_summary": dose_summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Comparison: {args.output.resolve()}")


if __name__ == "__main__":
    main()
