#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from or4d_common import load_config, read_peak_h5  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Clean image disk recovery against the private oracle."
    )
    parser.add_argument("--image-file", type=Path, required=True)
    parser.add_argument("--oracle-file", type=Path, required=True)
    parser.add_argument("--detected-file", type=Path, action="append", required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports" / "clean_image_pipeline_evaluation.json",
    )
    parser.add_argument(
        "--overlay-dir",
        type=Path,
        default=ROOT / "diagnostics" / "clean_image_overlays",
    )
    parser.add_argument("--overlay-count", type=int, default=4)
    return parser.parse_args()


def records_by_id(path: Path) -> dict[str, dict]:
    return {str(row["sample_id"]): row for row in read_peak_h5(path)}


def match_peaks(
    oracle: dict,
    detected: dict,
    tolerance_Ainv: float,
    *,
    high_angle_Ainv: float,
) -> dict:
    oracle_xy = np.column_stack((oracle["qx"], oracle["qy"])).astype(float)
    detected_xy = np.column_stack((detected["qx"], detected["qy"])).astype(float)
    if len(oracle_xy) and len(detected_xy):
        distances = np.linalg.norm(
            oracle_xy[:, None, :] - detected_xy[None, :, :], axis=2
        )
        oracle_index, detected_index = linear_sum_assignment(distances)
        matched_distance = distances[oracle_index, detected_index]
        keep = matched_distance <= tolerance_Ainv
        oracle_index = oracle_index[keep]
        detected_index = detected_index[keep]
        matched_distance = matched_distance[keep]
    else:
        oracle_index = np.empty(0, dtype=int)
        detected_index = np.empty(0, dtype=int)
        matched_distance = np.empty(0, dtype=float)
    high_oracle = np.hypot(oracle_xy[:, 0], oracle_xy[:, 1]) >= high_angle_Ainv
    matched_high = high_oracle[oracle_index].sum()
    return {
        "oracle_count": len(oracle_xy),
        "detected_count": len(detected_xy),
        "true_positive": len(matched_distance),
        "precision": float(len(matched_distance) / len(detected_xy))
        if len(detected_xy)
        else 0.0,
        "recall": float(len(matched_distance) / len(oracle_xy))
        if len(oracle_xy)
        else 1.0,
        "high_angle_oracle_count": int(high_oracle.sum()),
        "high_angle_recall": float(matched_high / high_oracle.sum())
        if high_oracle.any()
        else 1.0,
        "matched_distance_Ainv": matched_distance,
        "oracle_match_index": oracle_index,
        "detected_match_index": detected_index,
    }


def aggregate(rows: list[dict], q_pixel_Ainv: float) -> dict:
    oracle_count = sum(row["oracle_count"] for row in rows)
    detected_count = sum(row["detected_count"] for row in rows)
    true_positive = sum(row["true_positive"] for row in rows)
    distances = np.concatenate(
        [row["matched_distance_Ainv"] for row in rows]
    ) if rows else np.empty(0)
    high_count = sum(row["high_angle_oracle_count"] for row in rows)
    high_matched = sum(
        round(row["high_angle_recall"] * row["high_angle_oracle_count"])
        for row in rows
    )
    error_px = distances / q_pixel_Ainv
    return {
        "samples": len(rows),
        "oracle_peaks": oracle_count,
        "detected_peaks": detected_count,
        "true_positive": true_positive,
        "precision": float(true_positive / detected_count) if detected_count else 0.0,
        "recall": float(true_positive / oracle_count) if oracle_count else 1.0,
        "position_rmse_px": float(np.sqrt(np.mean(error_px**2)))
        if len(error_px)
        else None,
        "position_p95_px": float(np.percentile(error_px, 95))
        if len(error_px)
        else None,
        "high_angle_recall": float(high_matched / high_count)
        if high_count
        else 1.0,
        "sample_precision_mean": float(np.mean([row["precision"] for row in rows])),
        "sample_recall_mean": float(np.mean([row["recall"] for row in rows])),
    }


def make_overlay(
    image: np.ndarray,
    qx_axis: np.ndarray,
    qy_axis: np.ndarray,
    oracle: dict,
    detected: dict,
    match: dict,
    title: str,
    output: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    scale = np.log10(np.asarray(image, dtype=float) + 1e-10)
    ax.imshow(
        scale,
        cmap="gray",
        extent=(qx_axis[0], qx_axis[-1], qy_axis[-1], qy_axis[0]),
        origin="upper",
    )
    ax.scatter(
        oracle["qx"],
        oracle["qy"],
        s=34,
        facecolors="none",
        edgecolors="#16a34a",
        linewidths=1.2,
        label="physical oracle",
    )
    ax.scatter(
        detected["qx"],
        detected["qy"],
        s=22,
        marker="x",
        color="#dc2626",
        linewidths=1.1,
        label="detected",
    )
    for oi, di in zip(
        match["oracle_match_index"], match["detected_match_index"]
    ):
        ax.plot(
            [oracle["qx"][oi], detected["qx"][di]],
            [oracle["qy"][oi], detected["qy"][di]],
            color="#2563eb",
            alpha=0.55,
            linewidth=0.7,
        )
    ax.set_title(title)
    ax.set_xlabel(r"$q_x$ ($\AA^{-1}$)")
    ax.set_ylabel(r"$q_y$ ($\AA^{-1}$)")
    ax.set_aspect("equal")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    config = load_config()
    acceptance = config["clean_image"]["acceptance"]
    image_path = args.image_file.resolve()
    oracle = records_by_id(args.oracle_file.resolve())
    args.overlay_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results = []

    with h5py.File(image_path, "r") as h5:
        qx_axis = np.asarray(h5["detector/qx_Ainv"][:], dtype=float)
        qy_axis = np.asarray(h5["detector/qy_Ainv"][:], dtype=float)
        q_pixel = float(np.median(np.diff(qx_axis)))
        sample_ids = [
            value.decode() if isinstance(value, bytes) else str(value)
            for value in h5["sample_id"][:]
        ]
        images = h5["expectation/intensity"]
        tolerance = float(acceptance["match_tolerance_px"]) * q_pixel
        high_angle = 0.75 * float(config["common"]["k_max_Ainv"])

        for detected_path_arg in args.detected_file:
            detected_path = detected_path_arg.resolve()
            detected = records_by_id(detected_path)
            if set(detected) != set(sample_ids):
                raise ValueError(
                    f"sample IDs differ between image and {detected_path}"
                )
            rows = []
            for index, sample_id in enumerate(sample_ids):
                row = match_peaks(
                    oracle[sample_id],
                    detected[sample_id],
                    tolerance,
                    high_angle_Ainv=high_angle,
                )
                row["sample_id"] = sample_id
                rows.append(row)
                if index < args.overlay_count:
                    make_overlay(
                        images[index],
                        qx_axis,
                        qy_axis,
                        oracle[sample_id],
                        detected[sample_id],
                        row,
                        f"{sample_id}: {detected_path.stem}",
                        args.overlay_dir / f"{sample_id}_{detected_path.stem}.png",
                    )
            summary = aggregate(rows, q_pixel)
            summary["detected_file"] = str(detected_path)
            summary["per_sample"] = [
                {
                    key: value
                    for key, value in row.items()
                    if key
                    not in {
                        "matched_distance_Ainv",
                        "oracle_match_index",
                        "detected_match_index",
                    }
                }
                for row in rows
            ]
            results.append(summary)

    report = {
        "scope": "Clean only",
        "image_file": str(image_path),
        "oracle_file": str(args.oracle_file.resolve()),
        "match_tolerance_px": float(acceptance["match_tolerance_px"]),
        "q_pixel_size_Ainv": q_pixel,
        "detectors": results,
    }
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Evaluation: {args.output.resolve()}")


if __name__ == "__main__":
    main()
