#!/usr/bin/env python3
"""Summarize all saved V5 Clean-C disk detections against the oracle.

This script does not rerun image formation, disk detection, ACOM, or Pyxem.
It only matches the already-saved detected peak lists to the physical oracle
with the frozen 1-pixel detector-space rule and writes compact condition-level
and controlled-variable aggregates.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from or4d_common import load_config, read_peak_h5  # noqa: E402
from v5_disk_recovery import match_peak_arrays, summarize_matches  # noqa: E402


_ORACLE: dict[str, dict] = {}
_Q_PIXEL_AINV = 0.0
_TOLERANCE_AINV = 0.0
_HIGH_ANGLE_AINV = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--run-record", type=Path, required=True)
    parser.add_argument(
        "--py4dstem-detection-report", type=Path, action="append", default=[]
    )
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports/v5/pipeline/clean_c_disk_recovery_full.json",
    )
    return parser.parse_args()


def decoded(values: np.ndarray) -> list[str]:
    return [value.decode() if isinstance(value, bytes) else str(value) for value in values]


def init_worker(
    oracle_path: str,
    q_pixel_Ainv: float,
    tolerance_Ainv: float,
    high_angle_Ainv: float,
) -> None:
    global _ORACLE, _Q_PIXEL_AINV, _TOLERANCE_AINV, _HIGH_ANGLE_AINV
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[variable] = "1"
    _ORACLE = {str(row["sample_id"]): row for row in read_peak_h5(Path(oracle_path))}
    _Q_PIXEL_AINV = float(q_pixel_Ainv)
    _TOLERANCE_AINV = float(tolerance_Ainv)
    _HIGH_ANGLE_AINV = float(high_angle_Ainv)


def summarize_condition(condition: dict) -> dict:
    detected_rows = read_peak_h5(Path(condition["peak_file"]))
    detected = {str(row["sample_id"]): row for row in detected_rows}
    if set(detected) != set(_ORACLE):
        missing = sorted(set(_ORACLE) - set(detected))[:5]
        extra = sorted(set(detected) - set(_ORACLE))[:5]
        raise ValueError(f"sample IDs differ; missing={missing}, extra={extra}")
    matches = []
    for sample_id, oracle in _ORACLE.items():
        found = detected[sample_id]
        oracle_xy = np.column_stack((oracle["qx"], oracle["qy"]))
        detected_xy = np.column_stack((found["qx"], found["qy"]))
        matches.append(
            match_peak_arrays(
                oracle_xy,
                detected_xy,
                tolerance_Ainv=_TOLERANCE_AINV,
                high_angle_Ainv=_HIGH_ANGLE_AINV,
            )
        )
    return {**condition, **summarize_matches(matches, q_pixel_Ainv=_Q_PIXEL_AINV)}


def collect_conditions(args: argparse.Namespace) -> list[dict]:
    run_record = json.loads(args.run_record.read_text(encoding="utf-8"))
    rows: dict[tuple, dict] = {}
    for record in run_record["records"]:
        if record.get("stage") != "detection" or record.get("status") not in {"completed", "reused"}:
            continue
        condition = record["condition"]
        key = (
            str(condition["detector"]),
            int(condition["dose_electrons"]),
            str(condition["noise_level"]),
            condition.get("repeat"),
        )
        rows[key] = {
            "detector": key[0],
            "dose_electrons": key[1],
            "noise_level_id": key[2],
            "repeat": key[3],
            "peak_file": record["paths"]["peak"],
        }

    for report_path in args.py4dstem_detection_report:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        for run in report["runs"]:
            noise = str(run["noise_level_id"])
            variant = str(run["variant"])
            repeat = None
            if "_repeat" in variant:
                repeat = int(variant.rsplit("_repeat", 1)[1])
            key = ("py4dstem", int(run["dose_electrons"]), noise, repeat)
            rows[key] = {
                "detector": key[0],
                "dose_electrons": key[1],
                "noise_level_id": key[2],
                "repeat": key[3],
                "peak_file": run["output"],
            }
    conditions = sorted(
        rows.values(),
        key=lambda row: (
            row["detector"],
            row["dose_electrons"],
            row["noise_level_id"],
            -1 if row["repeat"] is None else row["repeat"],
        ),
    )
    counts = Counter(row["detector"] for row in conditions)
    if counts != Counter({"autodisk": 234, "dog_rgm": 234, "py4dstem": 234}):
        raise RuntimeError(f"expected 234 conditions per detector, got {counts}")
    return conditions


def aggregate_conditions(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[
            (row["detector"], row["dose_electrons"], row["noise_level_id"])
        ].append(row)
    metrics = (
        "precision",
        "recall",
        "position_rmse_px",
        "position_p95_px",
        "high_angle_recall",
        "sample_detection_coverage",
        "detected_peaks_per_sample",
        "false_positive_per_sample",
        "false_negative_per_sample",
    )
    result = []
    for (detector, dose, noise), group in sorted(grouped.items()):
        record = {
            "detector": detector,
            "dose_electrons": dose,
            "noise_level_id": noise,
            "num_conditions": len(group),
            "num_samples_total": sum(row["num_samples"] for row in group),
            "oracle_peaks_total": sum(row["oracle_peaks"] for row in group),
            "detected_peaks_total": sum(row["detected_peaks"] for row in group),
            "true_positive_total": sum(row["true_positive"] for row in group),
            "ambiguous_sample_count": sum(row["ambiguous_sample_count"] for row in group),
        }
        for metric in metrics:
            values = np.asarray([row[metric] for row in group], dtype=float)
            record[f"{metric}_mean"] = float(values.mean())
            record[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        result.append(record)
    return result


def main() -> None:
    args = parse_args()
    if not 1 <= args.workers <= 16:
        raise ValueError("--workers must be in [1, 16]")
    data_root = args.data_root.resolve()
    oracle_path = data_root / "intermediates/clean_v5_first_born_oracle_2048.h5"
    expectation_path = data_root / "datasets/clean_v5_first_born_expectation_2048.h5"
    with h5py.File(expectation_path, "r") as handle:
        qx = np.asarray(handle["detector/qx_Ainv"][:], dtype=float)
        q_pixel = float(np.median(np.diff(qx)))
        sample_count = len(handle["sample_id"])
    config = load_config()
    tolerance_px = float(config["clean_image"]["acceptance"]["match_tolerance_px"])
    tolerance_Ainv = tolerance_px * q_pixel
    high_angle_Ainv = 0.75 * float(config["common"]["k_max_Ainv"])
    conditions = collect_conditions(args)
    completed = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=init_worker,
        initargs=(str(oracle_path), q_pixel, tolerance_Ainv, high_angle_Ainv),
    ) as pool:
        futures = {pool.submit(summarize_condition, row): row for row in conditions}
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            completed.append(future.result())
            if index == 1 or index % 10 == 0 or index == len(futures):
                print(f"disk recovery {index}/{len(futures)}", flush=True)
    completed.sort(
        key=lambda row: (
            row["detector"], row["dose_electrons"], row["noise_level_id"],
            -1 if row["repeat"] is None else row["repeat"],
        )
    )
    report = {
        "schema": "or4d-v5-clean-c-disk-recovery-v1",
        "scope": "Clean-C saved peak lists; no detector or indexer rerun",
        "num_samples_per_condition": sample_count,
        "num_conditions": len(completed),
        "conditions_by_detector": dict(Counter(row["detector"] for row in completed)),
        "matching": {
            "method": "distance-ordered one-to-one nearest neighbour",
            "match_tolerance_px": tolerance_px,
            "match_tolerance_Ainv": tolerance_Ainv,
            "high_angle_threshold_Ainv": high_angle_Ainv,
            "q_pixel_size_Ainv": q_pixel,
        },
        "conditions": completed,
        "aggregates": aggregate_conditions(completed),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
