#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clean_counting import add_gaussian_read_noise  # noqa: E402
from or4d_common import load_config, read_peak_h5  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Clean image disk recovery against the private oracle."
    )
    parser.add_argument("--image-file", type=Path, required=True)
    parser.add_argument("--oracle-file", type=Path, required=True)
    parser.add_argument("--detected-file", type=Path, action="append")
    parser.add_argument(
        "--detection-report",
        type=Path,
        help="Use every detector output listed by scripts/03_extract_clean_disks.py.",
    )
    parser.add_argument(
        "--noise-manifest",
        type=Path,
        help="Required for overlays of independent read-noise levels.",
    )
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
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Write the full JSON report without echoing it to stdout.",
    )
    return parser.parse_args()


def records_by_id(path: Path) -> dict[str, dict]:
    return {str(row["sample_id"]): row for row in read_peak_h5(path)}


def decoded(values: np.ndarray) -> list[str]:
    return [
        value.decode() if isinstance(value, bytes) else str(value)
        for value in values
    ]


def load_noise_manifest(path: Path | None) -> dict | None:
    if path is None:
        return None
    with h5py.File(path.resolve(), "r") as h5:
        return {
            "path": str(path.resolve()),
            "sample_id": decoded(h5["sample_id"][:]),
            "dose_electrons": np.asarray(
                h5["dose_electrons"][:], dtype=np.int64
            ),
            "noise_level_id": decoded(h5["noise_level_id"][:]),
            "read_noise_sigma": np.asarray(
                h5["read_noise_sigma_primary_e_rms_per_pixel"][:],
                dtype=float,
            ),
            "read_noise_seed": np.asarray(
                h5["read_noise_seed"][:], dtype=np.uint64
            ),
        }


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
    scale = np.log10(
        np.maximum(np.asarray(image, dtype=float), 0.0) + 1e-10
    )
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
    noise_manifest = load_noise_manifest(args.noise_manifest)
    args.overlay_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results = []
    detected_paths = list(args.detected_file or [])
    run_metadata: dict[str, dict] = {}
    if args.detection_report:
        detection_report = json.loads(
            args.detection_report.read_text(encoding="utf-8")
        )
        for run in detection_report["runs"]:
            path = str(Path(run["output"]).resolve())
            run_metadata[path] = run
            detected_paths.append(Path(path))
    if not detected_paths:
        raise ValueError("provide --detected-file or --detection-report")

    with h5py.File(image_path, "r") as h5:
        qx_axis = np.asarray(h5["detector/qx_Ainv"][:], dtype=float)
        qy_axis = np.asarray(h5["detector/qy_Ainv"][:], dtype=float)
        q_pixel = float(np.median(np.diff(qx_axis)))
        sample_ids = decoded(h5["sample_id"][:])
        if noise_manifest is not None:
            if sample_ids != noise_manifest["sample_id"]:
                raise ValueError(
                    "noise manifest sample IDs do not match images"
                )
            if not np.array_equal(
                h5["dose_electrons"][:],
                noise_manifest["dose_electrons"],
            ):
                raise ValueError("noise manifest doses do not match images")
        tolerance = float(acceptance["match_tolerance_px"]) * q_pixel
        high_angle = 0.75 * float(config["common"]["k_max_Ainv"])

        for detected_path_arg in detected_paths:
            detected_path = detected_path_arg.resolve()
            detected = records_by_id(detected_path)
            if set(detected) != set(sample_ids):
                raise ValueError(
                    f"sample IDs differ between image and {detected_path}"
                )
            rows = []
            variant_name = str(
                run_metadata.get(str(detected_path), {}).get(
                    "variant", detected_path.stem
                )
            )
            counted_match = re.fullmatch(
                r"counted_dose(\d+)_repeat(\d+)", variant_name
            )
            independent_match = re.fullmatch(
                r"dose(\d+)_noise_(.+?)(?:_repeat(\d+))?",
                variant_name,
            )
            if counted_match:
                doses = np.asarray(h5["dose_electrons"][:], dtype=np.int64)
                dose = int(counted_match.group(1))
                repeat = int(counted_match.group(2))
                matching_dose = np.flatnonzero(doses == dose)
                if len(matching_dose) != 1:
                    raise ValueError(
                        f"Could not resolve dose {dose} in {image_path}"
                    )
                image_dataset = h5["images/counts"]
                image_selector = {
                    "dose_index": int(matching_dose[0]),
                    "repeat": repeat,
                    "noise_level_id": "poisson_only",
                }
            elif independent_match:
                doses = np.asarray(h5["dose_electrons"][:], dtype=np.int64)
                dose = int(independent_match.group(1))
                noise_level_id = independent_match.group(2)
                repeat_text = independent_match.group(3)
                matching_dose = np.flatnonzero(doses == dose)
                if len(matching_dose) != 1:
                    raise ValueError(
                        f"Could not resolve dose {dose} in {image_path}"
                    )
                if noise_level_id == "noiseless":
                    image_dataset = h5["images/expected_counts"]
                    image_selector = {
                        "dose_index": int(matching_dose[0]),
                        "repeat": None,
                        "noise_level_id": noise_level_id,
                    }
                else:
                    if repeat_text is None:
                        raise ValueError(
                            f"{variant_name} is missing a repeat index"
                        )
                    image_dataset = h5["images/counts"]
                    image_selector = {
                        "dose_index": int(matching_dose[0]),
                        "repeat": int(repeat_text),
                        "noise_level_id": noise_level_id,
                    }
            else:
                image_dataset = h5["expectation/intensity"]
                image_selector = None
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
                    image = (
                        image_dataset[index]
                        if image_selector is None
                        else (
                            image_dataset[
                                index, image_selector["dose_index"]
                            ]
                            if image_selector["repeat"] is None
                            else image_dataset[
                                index,
                                image_selector["dose_index"],
                                image_selector["repeat"],
                            ]
                        )
                    )
                    if (
                        image_selector is not None
                        and image_selector["repeat"] is not None
                        and image_selector["noise_level_id"]
                        not in ("poisson_only", "noiseless")
                    ):
                        if noise_manifest is None:
                            raise ValueError(
                                "read-noise overlays require "
                                "--noise-manifest"
                            )
                        level_index = noise_manifest[
                            "noise_level_id"
                        ].index(image_selector["noise_level_id"])
                        sigma = float(
                            noise_manifest["read_noise_sigma"][level_index]
                        )
                        seed = int(
                            noise_manifest["read_noise_seed"][
                                index,
                                image_selector["dose_index"],
                                level_index,
                                image_selector["repeat"],
                            ]
                        )
                        image = add_gaussian_read_noise(
                            image, sigma, np.random.default_rng(seed)
                        )
                    make_overlay(
                        image,
                        qx_axis,
                        qy_axis,
                        oracle[sample_id],
                        detected[sample_id],
                        row,
                        f"{sample_id}: {detected_path.stem}",
                        args.overlay_dir / f"{sample_id}_{detected_path.stem}.png",
                    )
            summary = aggregate(rows, q_pixel)
            summary["acceptance"] = {
                "precision": summary["precision"]
                >= float(acceptance["precision_min"]),
                "recall": summary["recall"]
                >= float(acceptance["recall_min"]),
                "position_rmse_px": (
                    summary["position_rmse_px"] is not None
                    and summary["position_rmse_px"]
                    <= float(acceptance["position_rmse_px_max"])
                ),
                "position_p95_px": (
                    summary["position_p95_px"] is not None
                    and summary["position_p95_px"]
                    <= float(acceptance["position_p95_px_max"])
                ),
                "high_angle_recall": summary["high_angle_recall"]
                >= float(acceptance["high_angle_recall_min"]),
            }
            summary["acceptance"]["all"] = all(
                summary["acceptance"].values()
            )
            summary["detected_file"] = str(detected_path)
            metadata = run_metadata.get(str(detected_path), {})
            summary["variant"] = metadata.get("variant", detected_path.stem)
            summary["detector"] = metadata.get("detector", "unknown")
            summary["dose_electrons"] = metadata.get(
                "dose_electrons",
                dose if counted_match or independent_match else None,
            )
            summary["noise_level_id"] = metadata.get(
                "noise_level_id",
                (
                    image_selector["noise_level_id"]
                    if image_selector is not None
                    else "none"
                ),
            )
            summary["runtime"] = {
                key: metadata[key]
                for key in (
                    "total_seconds",
                    "mean_seconds",
                    "p95_seconds",
                    "throughput_patterns_per_second",
                    "num_failures",
                )
                if key in metadata
            }
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

    grouped: dict[tuple[str, str, str], list[dict]] = {}
    for result in results:
        variant = str(result["variant"])
        dose_value = result.get("dose_electrons")
        dose = str(dose_value) if dose_value is not None else variant
        noise_level_id = str(result.get("noise_level_id", "none"))
        grouped.setdefault(
            (str(result["detector"]), dose, noise_level_id), []
        ).append(result)
    noise_summary = []
    ordered_groups = sorted(
        grouped.items(),
        key=lambda item: (
            item[0][0],
            0 if item[0][1].isdigit() else 1,
            int(item[0][1]) if item[0][1].isdigit() else item[0][1],
            item[0][2],
        )
    )
    for (detector, dose, noise_level_id), group in ordered_groups:
        row = {
            "detector": detector,
            "dose_electrons": int(dose) if dose.isdigit() else dose,
            "noise_level_id": noise_level_id,
            "repeats": len(group),
            "accepted_repeats": sum(
                bool(item["acceptance"]["all"]) for item in group
            ),
        }
        row["acceptance_fraction"] = row["accepted_repeats"] / len(group)
        for metric in (
            "precision",
            "recall",
            "position_rmse_px",
            "position_p95_px",
            "high_angle_recall",
        ):
            values = np.asarray(
                [item[metric] for item in group if item[metric] is not None],
                dtype=float,
            )
            row[f"{metric}_mean"] = float(values.mean()) if len(values) else None
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        timings = [
            item["runtime"].get("mean_seconds")
            for item in group
            if item["runtime"].get("mean_seconds") is not None
        ]
        row["runtime_mean_seconds_per_pattern"] = (
            float(np.mean(timings)) if timings else None
        )
        noise_summary.append(row)

    report = {
        "scope": "Clean only",
        "image_file": str(image_path),
        "oracle_file": str(args.oracle_file.resolve()),
        "match_tolerance_px": float(acceptance["match_tolerance_px"]),
        "q_pixel_size_Ainv": q_pixel,
        "detectors": results,
        "noise_summary": noise_summary,
        "dose_summary": noise_summary,
        "noise_manifest": (
            noise_manifest["path"] if noise_manifest is not None else None
        ),
    }
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not args.quiet:
        print(json.dumps(report, indent=2))
    print(f"Evaluation: {args.output.resolve()}")


if __name__ == "__main__":
    main()
