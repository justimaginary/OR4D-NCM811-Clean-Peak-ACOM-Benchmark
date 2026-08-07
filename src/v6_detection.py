from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

import h5py
import numpy as np

from autodisk_adapter import detect_autodisk_peaks, measure_vacuum_probe
from dog_rgm_disk_adapter import detect_dog_rgm_peaks
from py4dstem_disk_adapter import detect_py4dstem_bragg_disks_batch
from v6_observations import V6ObservationShardLoader
from v6_ubragg_adapter import UBraggXInference, UBraggXResult


def _compression(config: dict[str, Any]) -> dict[str, Any]:
    settings = config["v6"]["detection"]
    codec = str(settings["output_compression"])
    level = int(settings["output_compression_level"])
    shuffle = str(settings["output_shuffle"])
    if codec == "gzip":
        return {"compression": "gzip", "compression_opts": level, "shuffle": True}
    if codec != "zstd":
        raise ValueError(f"Unsupported V6 peak compression: {codec}")
    try:
        import hdf5plugin
    except ImportError as error:
        raise RuntimeError("V6 zstd peak storage requires hdf5plugin") from error
    if shuffle == "bitshuffle":
        return dict(hdf5plugin.Bitshuffle(nelems=0, cname="zstd", clevel=level))
    if shuffle == "none":
        return dict(hdf5plugin.Zstd(clevel=level))
    raise ValueError(f"Unsupported V6 peak shuffle: {shuffle}")


def _normalized_score(values: np.ndarray) -> np.ndarray:
    score = np.asarray(values, dtype=np.float32)
    maximum = float(np.max(score)) if len(score) else 0.0
    if maximum > 0.0:
        score = score / maximum
    return score


def _empty_peak_record() -> dict[str, np.ndarray]:
    empty = np.empty(0, dtype=np.float32)
    return {"qx": empty, "qy": empty, "intensity": empty, "score": empty}


def _autodisk_record(result: Any) -> dict[str, np.ndarray]:
    return {
        "qx": result.qx_Ainv,
        "qy": result.qy_Ainv,
        "intensity": result.intensity,
        "score": _normalized_score(result.correlation_score),
        "initial_row_px": result.initial_row_px,
        "initial_col_px": result.initial_col_px,
        "refined_row_px": result.refined_row_px,
        "refined_col_px": result.refined_col_px,
        "correlation_score": result.correlation_score,
        "rgm_score": result.rgm_score,
    }


def _dog_record(result: Any) -> dict[str, np.ndarray]:
    return {
        "qx": result.qx_Ainv,
        "qy": result.qy_Ainv,
        "intensity": result.intensity,
        "score": _normalized_score(result.dog_score),
        "initial_row_px": result.initial_row_px,
        "initial_col_px": result.initial_col_px,
        "refined_row_px": result.refined_row_px,
        "refined_col_px": result.refined_col_px,
        "dog_score": result.dog_score,
        "rgm_score": result.rgm_score,
    }


def _py4dstem_record(result: Any) -> dict[str, np.ndarray]:
    return {
        "qx": result.qx_Ainv,
        "qy": result.qy_Ainv,
        "intensity": result.intensity,
        "score": _normalized_score(result.correlation_intensity),
        "refined_row_px": result.row_px,
        "refined_col_px": result.col_px,
        "correlation_score": result.correlation_intensity,
    }


def _ubragg_record(
    result: UBraggXResult,
    *,
    k_max_Ainv: float,
    central_exclusion_Ainv: float,
) -> dict[str, np.ndarray]:
    radius = np.hypot(result.qx_Ainv, result.qy_Ainv)
    keep = (
        np.isfinite(result.qx_Ainv)
        & np.isfinite(result.qy_Ainv)
        & np.isfinite(result.intensity)
        & (radius >= central_exclusion_Ainv)
        & (radius <= k_max_Ainv)
    )
    return {
        "qx": result.qx_Ainv[keep],
        "qy": result.qy_Ainv[keep],
        "intensity": result.intensity[keep],
        "score": result.score[keep],
        "objectness": result.objectness[keep],
        "quality": result.quality[keep],
        "refined_row_px": result.row_px[keep],
        "refined_col_px": result.col_px[keep],
        "cov_xx": result.cov_xx_Ainv2[keep],
        "cov_xy": result.cov_xy_Ainv2[keep],
        "cov_yy": result.cov_yy_Ainv2[keep],
    }


def write_v6_peak_h5(
    path: Path,
    records: list[dict[str, Any]],
    attrs: dict[str, Any],
    config: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    field_names = sorted(
        {
            field
            for record in records
            for field in record["peaks"]
            if field not in {"qx", "qy", "intensity"}
        }
    )
    offsets = [0]
    fields: dict[str, list[np.ndarray]] = {
        "qx": [],
        "qy": [],
        "intensity": [],
        **{name: [] for name in field_names},
    }
    for record in records:
        peaks = record["peaks"]
        count = len(peaks["qx"])
        for required in ("qy", "intensity"):
            if len(peaks[required]) != count:
                raise ValueError(
                    f"Peak length mismatch for {record['observation_id']}: {required}"
                )
        for name in fields:
            if name in peaks:
                values = np.asarray(peaks[name], dtype=np.float32)
                if len(values) != count:
                    raise ValueError(
                        f"Peak length mismatch for {record['observation_id']}: {name}"
                    )
            else:
                values = np.full(count, np.nan, dtype=np.float32)
            fields[name].append(values)
        offsets.append(offsets[-1] + count)
    compression = _compression(config)
    with h5py.File(temporary, "w") as h5:
        h5.create_dataset(
            "sample_id",
            data=np.asarray(
                [record["observation_id"] for record in records],
                dtype=h5py.string_dtype("utf-8"),
            ),
        )
        peaks_group = h5.create_group("peaks")
        for name, parts in fields.items():
            values = np.concatenate(parts) if parts else np.empty(0, np.float32)
            peaks_group.create_dataset(name, data=values, **compression)
        peaks_group.create_dataset("offsets", data=np.asarray(offsets, dtype=np.int64))
        sample = h5.create_group("sample")
        for name in (
            "source_sample_id",
            "condition_id",
            "detection_status",
            "failure_type",
            "failure_message",
        ):
            sample.create_dataset(
                name,
                data=np.asarray(
                    [str(record[name]) for record in records],
                    dtype=h5py.string_dtype("utf-8"),
                ),
            )
        for name, dtype in (
            ("global_sample_index", np.int64),
            ("condition_index", np.int16),
            ("peak_count", np.int32),
            ("seed", np.uint64),
            ("runtime_seconds", np.float64),
            ("expected_total_electrons", np.float64),
            ("actual_total_electrons", np.float64),
            ("read_noise_sigma_e_per_pixel", np.float64),
        ):
            sample.create_dataset(
                name,
                data=np.asarray([record[name] for record in records], dtype=dtype),
            )
        sample.create_dataset(
            "observation_validation_sha256",
            data=np.asarray(
                [
                    np.frombuffer(
                        bytes.fromhex(record["validation_sha256"]), dtype=np.uint8
                    )
                    for record in records
                ],
                dtype=np.uint8,
            ),
        )
        for key, value in attrs.items():
            h5.attrs[key] = (
                json.dumps(value, sort_keys=True)
                if isinstance(value, (dict, list, tuple))
                else value
            )
    temporary.replace(path)


def failed_detection_record(
    loader: V6ObservationShardLoader,
    local_index: int,
    condition_index: int,
    error: Exception,
    runtime_seconds: float,
    config: dict[str, Any],
) -> dict[str, Any]:
    metadata = loader.metadata(local_index, condition_index)
    condition = metadata["condition"]
    maximum = int(config["v6"]["detection"]["failure_message_max_characters"])
    return {
        "observation_id": (
            f"{metadata['sample_id']}::condition_{condition_index:03d}"
        ),
        "source_sample_id": metadata["sample_id"],
        "global_sample_index": metadata["global_sample_index"],
        "condition_index": condition_index,
        "condition_id": condition["condition_id"],
        "seed": metadata["seed"],
        "validation_sha256": metadata["validation_sha256"],
        "expected_total_electrons": (
            np.nan
            if metadata["expected_total_electrons"] is None
            else metadata["expected_total_electrons"]
        ),
        "actual_total_electrons": metadata["actual_total_electrons"],
        "read_noise_sigma_e_per_pixel": condition[
            "read_noise_sigma_e_per_pixel"
        ],
        "detection_status": "failed",
        "failure_type": type(error).__name__,
        "failure_message": str(error)[:maximum],
        "runtime_seconds": runtime_seconds,
        "peak_count": 0,
        "peaks": _empty_peak_record(),
    }


def successful_detection_record(
    loader: V6ObservationShardLoader,
    local_index: int,
    condition_index: int,
    peaks: dict[str, np.ndarray],
    runtime_seconds: float,
) -> dict[str, Any]:
    metadata = loader.metadata(local_index, condition_index)
    condition = metadata["condition"]
    return {
        "observation_id": (
            f"{metadata['sample_id']}::condition_{condition_index:03d}"
        ),
        "source_sample_id": metadata["sample_id"],
        "global_sample_index": metadata["global_sample_index"],
        "condition_index": condition_index,
        "condition_id": condition["condition_id"],
        "seed": metadata["seed"],
        "validation_sha256": metadata["validation_sha256"],
        "expected_total_electrons": (
            np.nan
            if metadata["expected_total_electrons"] is None
            else metadata["expected_total_electrons"]
        ),
        "actual_total_electrons": metadata["actual_total_electrons"],
        "read_noise_sigma_e_per_pixel": condition[
            "read_noise_sigma_e_per_pixel"
        ],
        "detection_status": "ok" if len(peaks["qx"]) else "ok_empty",
        "failure_type": "",
        "failure_message": "",
        "runtime_seconds": runtime_seconds,
        "peak_count": len(peaks["qx"]),
        "peaks": peaks,
    }


def detector_geometry(
    expectation_h5: h5py.File, config: dict[str, Any]
) -> dict[str, Any]:
    qx = np.asarray(expectation_h5["detector/qx_Ainv"][:], dtype=np.float32)
    qy = np.asarray(expectation_h5["detector/qy_Ainv"][:], dtype=np.float32)
    probe = np.asarray(expectation_h5["detector/vacuum_probe"][:], dtype=np.float32)
    valid = np.asarray(expectation_h5["detector/valid_mask"][:], dtype=np.float32)
    _, _, radius_px = measure_vacuum_probe(probe)
    q_pixel = float(np.median(np.abs(np.diff(qx))))
    central = (
        radius_px
        * q_pixel
        * float(config["clean_image"]["detector_central_exclusion_radius_fraction"])
    )
    return {
        "qx": qx,
        "qy": qy,
        "probe": probe,
        "valid": valid,
        "disk_radius_px": radius_px,
        "central_exclusion_Ainv": central,
        "k_max_Ainv": float(config["common"]["k_max_Ainv"]),
    }


def detect_cpu_batch(
    detector: str,
    images: list[np.ndarray],
    geometry: dict[str, Any],
    config: dict[str, Any],
) -> list[dict[str, np.ndarray] | Exception]:
    image_config = config["clean_image"]
    if detector == "autodisk":
        function: Callable[[np.ndarray], dict[str, np.ndarray]] = lambda image: _autodisk_record(
            detect_autodisk_peaks(
                image,
                geometry["qx"],
                geometry["qy"],
                geometry["probe"],
                image_config["autodisk"],
                k_max_Ainv=geometry["k_max_Ainv"],
                central_exclusion_Ainv=geometry["central_exclusion_Ainv"],
            )
        )
    elif detector == "dog_rgm":
        function = lambda image: _dog_record(
            detect_dog_rgm_peaks(
                image,
                geometry["qx"],
                geometry["qy"],
                geometry["probe"],
                image_config["dog_rgm"],
                k_max_Ainv=geometry["k_max_Ainv"],
                central_exclusion_Ainv=geometry["central_exclusion_Ainv"],
            )
        )
    else:
        raise ValueError(f"Unsupported CPU detector: {detector}")

    def guarded(image: np.ndarray) -> dict[str, np.ndarray] | Exception:
        try:
            return function(image)
        except Exception as error:
            return error

    workers = int(config["v6"]["detection"]["cpu_sample_workers"])
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(guarded, images))


def detect_py4dstem_batch(
    images: list[np.ndarray],
    geometry: dict[str, Any],
    config: dict[str, Any],
    *,
    cuda: bool,
) -> list[dict[str, np.ndarray] | Exception]:
    results = detect_py4dstem_bragg_disks_batch(
        np.stack(images),
        geometry["qx"],
        geometry["qy"],
        geometry["probe"],
        config["clean_image"]["py4dstem_find_bragg_disks"],
        k_max_Ainv=geometry["k_max_Ainv"],
        central_exclusion_Ainv=geometry["central_exclusion_Ainv"],
        cuda=cuda,
    )
    return [value if isinstance(value, Exception) else _py4dstem_record(value) for value in results]


def detect_ubragg_batch(
    inference: UBraggXInference,
    images: list[np.ndarray],
    expected_totals: list[float],
    sigmas: list[float],
    geometry: dict[str, Any],
) -> list[dict[str, np.ndarray]]:
    results = inference.infer_batch(
        np.stack(images),
        np.asarray(expected_totals, dtype=np.float32),
        np.asarray(sigmas, dtype=np.float32),
        geometry["probe"],
        geometry["valid"],
        geometry["qx"],
        geometry["qy"],
    )
    return [
        _ubragg_record(
            result,
            k_max_Ainv=geometry["k_max_Ainv"],
            central_exclusion_Ainv=geometry["central_exclusion_Ainv"],
        )
        for result in results
    ]
