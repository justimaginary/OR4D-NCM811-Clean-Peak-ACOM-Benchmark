#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import py4DSTEM

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from or4d_common import read_jsonl, read_peak_h5  # noqa: E402


def validate_track(track: str) -> dict:
    peak_path = ROOT / "public" / f"{track}_peaks.h5"
    gt_path = ROOT / "private" / f"{track}_ground_truth.jsonl"
    samples = read_peak_h5(peak_path)
    gt = {x["sample_id"]: x for x in read_jsonl(gt_path)}

    counts = []
    for sample in samples:
        sample_id = sample["sample_id"]
        if sample_id not in gt:
            raise ValueError(f"Missing ground truth for {sample_id}")
        if len(sample["qx"]) < 3:
            raise ValueError(f"{sample_id} has fewer than three peaks")
        for field in ("qx", "qy", "intensity"):
            if not np.all(np.isfinite(sample[field])):
                raise ValueError(f"{sample_id} contains non-finite {field}")

        dtype = np.dtype([("qx", "f4"), ("qy", "f4"), ("intensity", "f4")])
        data = np.empty(len(sample["qx"]), dtype=dtype)
        data["qx"] = sample["qx"]
        data["qy"] = sample["qy"]
        data["intensity"] = sample["intensity"]
        point_list = py4DSTEM.PointList(data=data, name=sample_id)
        if len(point_list.data) != len(sample["qx"]):
            raise RuntimeError("PointList conversion failed")

        R = np.asarray(gt[sample_id]["orientation_matrix_sample_to_crystal"], dtype=float)
        if not np.allclose(R.T @ R, np.eye(3), atol=1e-6):
            raise ValueError(f"{sample_id} ground-truth matrix is not orthonormal")
        if not np.isclose(np.linalg.det(R), 1.0, atol=1e-6):
            raise ValueError(f"{sample_id} ground-truth determinant is not +1")
        counts.append(len(sample["qx"]))

    return {
        "track": track,
        "num_samples": len(samples),
        "peak_count_min": int(np.min(counts)),
        "peak_count_max": int(np.max(counts)),
        "peak_count_mean": float(np.mean(counts)),
        "baseline_input_dtype": ["qx", "qy", "intensity"],
        "baseline_output_field": "orientation_matrix_sample_to_crystal",
    }


def main() -> None:
    results = []
    for track in ("clean", "dynamical"):
        path = ROOT / "public" / f"{track}_peaks.h5"
        if path.exists():
            results.append(validate_track(track))
    if not results:
        raise FileNotFoundError("No generated track found under public/")

    output = ROOT / "reports" / "dataset_validation.json"
    with output.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"Validation report: {output}")


if __name__ == "__main__":
    main()
