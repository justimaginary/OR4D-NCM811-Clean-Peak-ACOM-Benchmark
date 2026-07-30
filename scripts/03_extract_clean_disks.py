#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from autodisk_adapter import detect_autodisk_peaks  # noqa: E402
from or4d_common import load_config, write_peak_h5  # noqa: E402
from py4dstem_disk_adapter import (  # noqa: E402
    detect_py4dstem_bragg_disks,
    detect_py4dstem_bragg_disks_batch,
)
from dog_rgm_disk_adapter import detect_dog_rgm_peaks  # noqa: E402
from cuda_xcorr_disk_adapter import detect_cuda_xcorr_poly_batch  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract Clean diffraction disks with AutoDisk and find_Bragg_disks."
    )
    parser.add_argument("--image-file", type=Path, required=True)
    parser.add_argument(
        "--track", choices=("expectation", "counted"), required=True
    )
    parser.add_argument(
        "--detector",
        action="append",
        choices=("autodisk", "py4dstem", "dog_rgm", "cuda_xcorr_poly"),
        help="Detector to run; default runs both.",
    )
    parser.add_argument("--dose-index", type=int, action="append")
    parser.add_argument("--repeat", type=int, action="append")
    parser.add_argument(
        "--compute-backend",
        choices=("cpu", "cuda"),
        default="cpu",
        help="CUDA uses py4DSTEM's batched GPU implementation and requires --detector py4dstem.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "diagnostics"
    )
    parser.add_argument("--report-output", type=Path)
    return parser.parse_args()


def decode_ids(values: np.ndarray) -> list[str]:
    return [value.decode() if isinstance(value, bytes) else str(value) for value in values]


def variants(h5: h5py.File, args: argparse.Namespace) -> list[dict]:
    if args.track == "expectation":
        return [
            {
                "name": "expectation",
                "dataset": h5["expectation/intensity"],
                "dose_electrons": None,
                "dose_index": None,
                "repeat": None,
            }
        ]
    doses = np.asarray(h5["dose_electrons"][:], dtype=np.int64)
    dataset = h5["images/counts"]
    dose_indices = args.dose_index or list(range(len(doses)))
    repeats = args.repeat or list(range(dataset.shape[2]))
    result = []
    for dose_index in dose_indices:
        if not 0 <= dose_index < len(doses):
            raise IndexError(f"dose index {dose_index} is out of range")
        for repeat in repeats:
            if not 0 <= repeat < dataset.shape[2]:
                raise IndexError(f"repeat {repeat} is out of range")
            result.append(
                {
                    "name": f"counted_dose{int(doses[dose_index])}_repeat{repeat}",
                    "dataset": dataset,
                    "dose_electrons": int(doses[dose_index]),
                    "dose_index": dose_index,
                    "repeat": repeat,
                }
            )
    return result


def read_image(variant: dict, sample_index: int) -> np.ndarray:
    if variant["dose_index"] is None:
        return np.asarray(variant["dataset"][sample_index], dtype=np.float32)
    return np.asarray(
        variant["dataset"][
            sample_index, variant["dose_index"], variant["repeat"]
        ],
        dtype=np.float32,
    )


def main() -> None:
    args = parse_args()
    config = load_config()
    image_cfg = config["clean_image"]
    detectors = args.detector or ["autodisk", "py4dstem", "dog_rgm"]
    if args.compute_backend == "cuda" and detectors not in (
        ["py4dstem"],
        ["cuda_xcorr_poly"],
    ):
        raise ValueError(
            "--compute-backend cuda requires exactly one CUDA detector; "
            "AutoDisk and DoG-RGM remain CPU methods"
        )
    if "cuda_xcorr_poly" in detectors and args.compute_backend != "cuda":
        raise ValueError("cuda_xcorr_poly requires --compute-backend cuda")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_path = args.image_file.resolve()
    report_rows: list[dict] = []

    with h5py.File(source_path, "r") as h5:
        forward_model = str(
            h5.attrs.get(
                "forward_model",
                h5.attrs.get("expectation_forward_model", "unknown"),
            )
        )
        model_suffix = (
            "" if forward_model == "acom_matched" else "_first_born"
        )
        sample_ids = decode_ids(h5["sample_id"][:])
        qx_axis = np.asarray(h5["detector/qx_Ainv"][:], dtype=float)
        qy_axis = np.asarray(h5["detector/qy_Ainv"][:], dtype=float)
        vacuum_probe = np.asarray(h5["detector/vacuum_probe"][:], dtype=float)
        q_pixel = float(np.median(np.diff(qx_axis)))
        disk_radius_px = float(
            h5.attrs.get(
                "disk_radius_px",
                np.sqrt(np.count_nonzero(vacuum_probe >= 0.5) / np.pi),
            )
        )
        central_exclusion = (
            disk_radius_px
            * q_pixel
            * float(
                image_cfg["detector_central_exclusion_radius_fraction"]
            )
        )

        for variant in variants(h5, args):
            for detector_name in detectors:
                samples: list[dict] = []
                failures: list[dict] = []
                timings: list[float] = []
                batch_results = None
                batch_seconds = None
                if (
                    detector_name in ("py4dstem", "cuda_xcorr_poly")
                    and args.compute_backend == "cuda"
                ):
                    images = np.stack(
                        [
                            read_image(variant, sample_index)
                            for sample_index in range(len(sample_ids))
                        ]
                    )
                    batch_started = time.perf_counter()
                    batch_results = (
                        detect_py4dstem_bragg_disks_batch(
                            images,
                            qx_axis,
                            qy_axis,
                            vacuum_probe,
                            image_cfg["py4dstem_find_bragg_disks"],
                            k_max_Ainv=float(config["common"]["k_max_Ainv"]),
                            central_exclusion_Ainv=central_exclusion,
                            cuda=True,
                        )
                        if detector_name == "py4dstem"
                        else detect_cuda_xcorr_poly_batch(
                            images,
                            qx_axis,
                            qy_axis,
                            vacuum_probe,
                            image_cfg["cuda_xcorr_poly"],
                            k_max_Ainv=float(config["common"]["k_max_Ainv"]),
                            central_exclusion_Ainv=central_exclusion,
                        )
                    )
                    batch_seconds = time.perf_counter() - batch_started
                for sample_index, sample_id in enumerate(sample_ids):
                    image = read_image(variant, sample_index)
                    started = time.perf_counter()
                    try:
                        if detector_name == "autodisk":
                            result = detect_autodisk_peaks(
                                image,
                                qx_axis,
                                qy_axis,
                                vacuum_probe,
                                image_cfg["autodisk"],
                                k_max_Ainv=float(config["common"]["k_max_Ainv"]),
                                central_exclusion_Ainv=central_exclusion,
                            )
                            peak_diagnostics = {
                                "initial_row_px": result.initial_row_px,
                                "initial_col_px": result.initial_col_px,
                                "refined_row_px": result.refined_row_px,
                                "refined_col_px": result.refined_col_px,
                                "correlation_score": result.correlation_score,
                                "rgm_score": result.rgm_score,
                            }
                        elif detector_name in ("py4dstem", "cuda_xcorr_poly"):
                            result = (
                                batch_results[sample_index]
                                if batch_results is not None
                                else detect_py4dstem_bragg_disks(
                                    image,
                                    qx_axis,
                                    qy_axis,
                                    vacuum_probe,
                                    image_cfg["py4dstem_find_bragg_disks"],
                                    k_max_Ainv=float(
                                        config["common"]["k_max_Ainv"]
                                    ),
                                    central_exclusion_Ainv=central_exclusion,
                                )
                            )
                            peak_diagnostics = {
                                "refined_row_px": result.row_px,
                                "refined_col_px": result.col_px,
                                "correlation_score": result.correlation_intensity,
                            }
                        else:
                            result = detect_dog_rgm_peaks(
                                image,
                                qx_axis,
                                qy_axis,
                                vacuum_probe,
                                image_cfg["dog_rgm"],
                                k_max_Ainv=float(config["common"]["k_max_Ainv"]),
                                central_exclusion_Ainv=central_exclusion,
                            )
                            peak_diagnostics = {
                                "initial_row_px": result.initial_row_px,
                                "initial_col_px": result.initial_col_px,
                                "refined_row_px": result.refined_row_px,
                                "refined_col_px": result.refined_col_px,
                                "dog_score": result.dog_score,
                                "rgm_score": result.rgm_score,
                            }
                        sample = {
                            "sample_id": sample_id,
                            "qx": result.qx_Ainv,
                            "qy": result.qy_Ainv,
                            "intensity": result.intensity,
                            "peak_diagnostics": peak_diagnostics,
                            "sample_metadata": {
                                "detection_status": "ok",
                                "failure_type": "",
                                "failure_message": "",
                                "runtime_seconds": 0.0,
                            },
                        }
                    except Exception as error:
                        failures.append(
                            {
                                "sample_id": sample_id,
                                "error_type": type(error).__name__,
                                "error": str(error),
                            }
                        )
                        sample = {
                            "sample_id": sample_id,
                            "qx": np.empty(0, dtype=np.float32),
                            "qy": np.empty(0, dtype=np.float32),
                            "intensity": np.empty(0, dtype=np.float32),
                            "peak_diagnostics": {},
                            "sample_metadata": {
                                "detection_status": "failed",
                                "failure_type": type(error).__name__,
                                "failure_message": str(error),
                                "runtime_seconds": 0.0,
                            },
                        }
                    elapsed = time.perf_counter() - started
                    if batch_seconds is not None:
                        elapsed += batch_seconds / len(sample_ids)
                    sample["sample_metadata"]["runtime_seconds"] = elapsed
                    timings.append(elapsed)
                    samples.append(sample)
                    print(
                        f"{variant['name']} {detector_name} "
                        f"{sample_index + 1}/{len(sample_ids)} {sample_id}: "
                        f"peaks={len(sample['qx'])}, seconds={elapsed:.3f}"
                    )

                suffix = "_smoke" if "smoke" in source_path.stem else ""
                output = (
                    args.output_dir
                    / (
                        f"clean_{variant['name']}_{detector_name}_peaks"
                        f"{model_suffix}{suffix}.h5"
                    )
                )
                attrs = {
                    "track": f"clean_{variant['name']}",
                    "detector": detector_name,
                    "compute_backend": args.compute_backend,
                    "source_image_file": str(source_path),
                    "dose_electrons": variant["dose_electrons"]
                    if variant["dose_electrons"] is not None
                    else "expectation",
                    "repeat": variant["repeat"]
                    if variant["repeat"] is not None
                    else "not_applicable",
                    "coordinate_units": "1/angstrom",
                    "forward_model": forward_model,
                    "central_exclusion_Ainv": central_exclusion,
                    "detector_config": image_cfg[
                        "autodisk"
                        if detector_name == "autodisk"
                        else (
                            "py4dstem_find_bragg_disks"
                            if detector_name == "py4dstem"
                            else (
                                "dog_rgm"
                                if detector_name == "dog_rgm"
                                else "cuda_xcorr_poly"
                            )
                        )
                    ],
                }
                write_peak_h5(output, samples, attrs)
                timing_values = np.asarray(timings, dtype=float)
                report_rows.append(
                    {
                        "variant": variant["name"],
                        "detector": detector_name,
                        "compute_backend": args.compute_backend,
                        "output": str(output),
                        "num_samples": len(samples),
                        "num_failures": len(failures),
                        "failures": failures,
                        "total_seconds": float(timing_values.sum()),
                        "mean_seconds": float(timing_values.mean()),
                        "p95_seconds": float(np.percentile(timing_values, 95)),
                        "throughput_patterns_per_second": float(
                            len(timing_values) / timing_values.sum()
                        ),
                    }
                )

    report = {
        "source_image_file": str(source_path),
        "track": args.track,
        "detectors": detectors,
        "compute_backend": args.compute_backend,
        "runs": report_rows,
    }
    suffix = "_smoke" if "smoke" in source_path.stem else ""
    report_path = (
        args.report_output.resolve()
        if args.report_output is not None
        else ROOT
        / "reports"
        / f"clean_disk_detection_{args.track}{model_suffix}{suffix}.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Detection report: {report_path}")


if __name__ == "__main__":
    main()
