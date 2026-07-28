#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import numpy as np
import py4DSTEM

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from or4d_common import load_config, read_jsonl, read_peak_h5  # noqa: E402


def unique_records_by_id(records: list[dict], *, source: str) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for record in records:
        sample_id = str(record["sample_id"])
        if sample_id in result:
            raise ValueError(f"Duplicate sample_id in {source}: {sample_id}")
        result[sample_id] = record
    return result


def validate_track(track: str) -> dict:
    config = load_config()
    peak_path = ROOT / "public" / f"{track}_peaks.h5"
    gt_path = ROOT / "private" / f"{track}_ground_truth.jsonl"
    samples = read_peak_h5(peak_path)
    if not samples:
        raise ValueError(f"{track} contains no samples")
    samples_by_id = unique_records_by_id(samples, source=str(peak_path))
    gt = unique_records_by_id(read_jsonl(gt_path), source=str(gt_path))
    if set(samples_by_id) != set(gt):
        missing_gt = sorted(set(samples_by_id) - set(gt))
        extra_gt = sorted(set(gt) - set(samples_by_id))
        raise ValueError(
            f"{track} public/ground-truth ID mismatch: "
            f"missing_gt={missing_gt}, extra_gt={extra_gt}"
        )

    with h5py.File(peak_path, "r") as h5:
        expected_attrs = {
            "dataset_id": config["dataset"]["id"],
            "track": track,
            "coordinate_units": "1/angstrom",
        }
        for name, expected in expected_attrs.items():
            actual = h5.attrs.get(name)
            if actual != expected:
                raise ValueError(
                    f"{peak_path} attribute {name!r}: "
                    f"actual={actual!r}, expected={expected!r}"
                )
        offsets = np.asarray(h5["peaks/offsets"][:], dtype=np.int64)
        total_peaks = len(h5["peaks/qx"])
        if len(offsets) != len(samples) + 1:
            raise ValueError("Peak offsets length does not equal num_samples + 1")
        if offsets[0] != 0 or offsets[-1] != total_peaks:
            raise ValueError("Peak offsets do not span the flattened peak arrays")
        if np.any(np.diff(offsets) < 0):
            raise ValueError("Peak offsets are not monotonic")

    counts = []
    role_counts: Counter[str] = Counter()
    peak_counts_by_role: defaultdict[str, list[int]] = defaultdict(list)
    k_max = float(config["common"]["k_max_Ainv"])
    central_exclusion = float(config["common"]["central_beam_exclusion_Ainv"])
    for sample in samples:
        sample_id = sample["sample_id"]
        lengths = [len(sample[field]) for field in ("qx", "qy", "intensity")]
        if len(set(lengths)) != 1:
            raise ValueError(f"{sample_id} peak arrays have different lengths")
        if lengths[0] < 3:
            raise ValueError(f"{sample_id} has fewer than three peaks")
        for field in ("qx", "qy", "intensity"):
            if not np.all(np.isfinite(sample[field])):
                raise ValueError(f"{sample_id} contains non-finite {field}")
        intensity = np.asarray(sample["intensity"], dtype=float)
        if np.any(intensity <= 0.0):
            raise ValueError(f"{sample_id} contains non-positive intensity")
        if not np.isclose(float(intensity.max()), 1.0, atol=1e-6):
            raise ValueError(f"{sample_id} intensity is not max-normalized")
        radius = np.hypot(sample["qx"], sample["qy"])
        if np.any(radius < central_exclusion - 1e-6):
            raise ValueError(f"{sample_id} contains a peak inside central exclusion")
        if np.any(radius > k_max + 1e-6):
            raise ValueError(f"{sample_id} contains a peak beyond k_max")

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
        count = len(sample["qx"])
        counts.append(count)
        role = str(gt[sample_id].get("sample_role", "unspecified"))
        role_counts[role] += 1
        peak_counts_by_role[role].append(count)

    if track == "clean":
        expected_role_counts = {
            str(role): int(count)
            for role, count in config["dataset"]["expected_sample_counts"].items()
        }
        if dict(role_counts) != expected_role_counts:
            raise ValueError(
                f"Clean role counts differ from config: "
                f"actual={dict(role_counts)}, expected={expected_role_counts}"
            )
        expected_total = int(config["dataset"]["expected_num_orientations"])
        if len(samples) != expected_total:
            raise ValueError(
                f"Clean sample count is {len(samples)}, expected {expected_total}"
            )

    return {
        "track": track,
        "num_samples": len(samples),
        "peak_count_min": int(np.min(counts)),
        "peak_count_max": int(np.max(counts)),
        "peak_count_mean": float(np.mean(counts)),
        "sample_counts_by_role": dict(role_counts),
        "peak_counts_by_role": {
            role: {
                "min": int(np.min(role_counts_values)),
                "max": int(np.max(role_counts_values)),
                "mean": float(np.mean(role_counts_values)),
            }
            for role, role_counts_values in peak_counts_by_role.items()
        },
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
