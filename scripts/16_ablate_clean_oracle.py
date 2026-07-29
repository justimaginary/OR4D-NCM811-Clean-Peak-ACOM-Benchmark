#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from or4d_common import read_peak_h5, write_peak_h5  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ablate Clean physical-oracle coordinates and intensities."
    )
    parser.add_argument(
        "--acom-reference",
        type=Path,
        default=ROOT / "private" / "clean_oracle_peaks.h5",
    )
    parser.add_argument("--physical-oracle", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "diagnostics"
    )
    parser.add_argument("--match-tolerance-Ainv", type=float, default=0.01)
    return parser.parse_args()


def by_id(path: Path) -> dict[str, dict]:
    return {str(row["sample_id"]): row for row in read_peak_h5(path)}


def main() -> None:
    args = parse_args()
    reference = by_id(args.acom_reference.resolve())
    physical = by_id(args.physical_oracle.resolve())
    if set(reference) != set(physical):
        if not set(physical) <= set(reference):
            raise ValueError("physical oracle contains IDs absent from reference")
        reference = {sample_id: reference[sample_id] for sample_id in physical}
    args.output_dir.mkdir(parents=True, exist_ok=True)

    variants: dict[str, list[dict]] = {
        "reference_uniform": [],
        "physical_uniform": [],
        "matched_reference_physical_intensity": [],
    }
    sample_reports = []
    for sample_id in sorted(physical):
        ref = reference[sample_id]
        phy = physical[sample_id]
        ref_xy = np.column_stack((ref["qx"], ref["qy"])).astype(float)
        phy_xy = np.column_stack((phy["qx"], phy["qy"])).astype(float)
        distances = np.linalg.norm(
            ref_xy[:, None, :] - phy_xy[None, :, :], axis=2
        )
        ref_index, phy_index = linear_sum_assignment(distances)
        matched_distance = distances[ref_index, phy_index]
        keep = matched_distance <= args.match_tolerance_Ainv
        ref_index = ref_index[keep]
        phy_index = phy_index[keep]
        matched_distance = matched_distance[keep]
        physical_intensity = np.asarray(
            phy["intensity"], dtype=np.float32
        )[phy_index]
        physical_intensity /= physical_intensity.max()

        variants["reference_uniform"].append(
            {
                "sample_id": sample_id,
                "qx": ref["qx"],
                "qy": ref["qy"],
                "intensity": np.ones(len(ref_xy), dtype=np.float32),
            }
        )
        variants["physical_uniform"].append(
            {
                "sample_id": sample_id,
                "qx": phy["qx"],
                "qy": phy["qy"],
                "intensity": np.ones(len(phy_xy), dtype=np.float32),
            }
        )
        variants["matched_reference_physical_intensity"].append(
            {
                "sample_id": sample_id,
                "qx": np.asarray(ref["qx"])[ref_index],
                "qy": np.asarray(ref["qy"])[ref_index],
                "intensity": physical_intensity,
            }
        )
        sample_reports.append(
            {
                "sample_id": sample_id,
                "reference_peaks": len(ref_xy),
                "physical_peaks": len(phy_xy),
                "matched_reference_peaks": len(ref_index),
                "match_rmse_Ainv": float(
                    np.sqrt(np.mean(matched_distance**2))
                ),
                "match_max_Ainv": float(matched_distance.max()),
            }
        )

    outputs = {}
    for name, samples in variants.items():
        output = args.output_dir / f"clean_oracle_ablation_{name}.h5"
        write_peak_h5(
            output,
            samples,
            {
                "track": "clean_oracle_ablation",
                "variant": name,
                "coordinate_units": "1/angstrom",
                "acom_reference": str(args.acom_reference.resolve()),
                "physical_oracle": str(args.physical_oracle.resolve()),
            },
        )
        outputs[name] = str(output.resolve())

    report = {
        "match_tolerance_Ainv": args.match_tolerance_Ainv,
        "outputs": outputs,
        "samples": sample_reports,
    }
    report_path = ROOT / "reports" / "clean_oracle_ablation_smoke.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Ablation report: {report_path}")


if __name__ == "__main__":
    main()
