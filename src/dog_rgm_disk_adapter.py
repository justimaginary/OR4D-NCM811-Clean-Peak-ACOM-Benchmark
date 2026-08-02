from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.ndimage import map_coordinates, spline_filter
from scipy.signal import fftconvolve
from skimage.feature import blob_dog

from autodisk_adapter import _radial_kernel, _sample_cubic, measure_vacuum_probe


@dataclass(frozen=True)
class DoGRGMResult:
    qx_Ainv: np.ndarray
    qy_Ainv: np.ndarray
    intensity: np.ndarray
    initial_row_px: np.ndarray
    initial_col_px: np.ndarray
    refined_row_px: np.ndarray
    refined_col_px: np.ndarray
    dog_score: np.ndarray
    rgm_score: np.ndarray


def _sample(image: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    return map_coordinates(
        image,
        [rows, cols],
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )


def detect_dog_rgm_peaks(
    image: np.ndarray,
    qx_axis_Ainv: np.ndarray,
    qy_axis_Ainv: np.ndarray,
    vacuum_probe: np.ndarray,
    config: dict[str, Any],
    *,
    k_max_Ainv: float,
    central_exclusion_Ainv: float,
) -> DoGRGMResult:
    """Independent scale-space DoG initialization with modified-RGM refinement."""
    raw = np.maximum(np.asarray(image, dtype=float), 0.0)
    qx_axis = np.asarray(qx_axis_Ainv, dtype=float)
    qy_axis = np.asarray(qy_axis_Ainv, dtype=float)
    if raw.ndim != 2 or raw.shape != (len(qy_axis), len(qx_axis)):
        raise ValueError("image shape does not match q axes")
    if not np.all(np.isfinite(raw)) or raw.max() <= 0.0:
        raise ValueError("diffraction image must contain finite positive intensity")

    beam_row, beam_col, disk_radius = measure_vacuum_probe(vacuum_probe)
    processed = np.sqrt(raw)
    normalized = processed / processed.max()
    blobs = blob_dog(
        normalized,
        min_sigma=float(config["dog_min_sigma_px"]),
        max_sigma=float(config["dog_max_sigma_px"]),
        sigma_ratio=float(config["dog_sigma_ratio"]),
        threshold=float(config["threshold_abs"]),
        overlap=float(config["overlap"]),
        exclude_border=False,
    )
    candidates: list[tuple[float, float, float]] = []
    for row, col, _ in blobs:
        score = float(
            _sample(
                normalized,
                np.asarray([row], dtype=float),
                np.asarray([col], dtype=float),
            )[0]
        )
        candidates.append((float(row), float(col), score))
    candidates.sort(key=lambda value: value[2], reverse=True)
    deduplicated: list[tuple[float, float, float]] = []
    spacing = float(config["min_peak_spacing_px"])
    max_num_peaks = int(config["max_num_peaks"])
    for candidate in candidates:
        if all(
            np.hypot(candidate[0] - kept[0], candidate[1] - kept[1])
            >= spacing
            for kept in deduplicated
        ):
            deduplicated.append(candidate)
            # Score ordering makes later candidates ineligible for the top-N
            # output once N spatially distinct candidates have been retained.
            if len(deduplicated) == max_num_peaks:
                break
    candidates = deduplicated
    if not candidates:
        raise RuntimeError("DoG scale space found no diffraction disks")

    rgm_kernel = _radial_kernel(
        disk_radius,
        inner_fraction=float(config["rgm_inner_radius_fraction"]),
        split_fraction=float(config["rgm_split_radius_fraction"]),
        outer_fraction=float(config["rgm_outer_radius_fraction"]),
    )
    rgm_map = fftconvolve(processed, rgm_kernel[::-1, ::-1], mode="same")
    rgm_spline = spline_filter(rgm_map, order=3)
    offsets = np.arange(
        -float(config["rgm_search_radius_px"]),
        float(config["rgm_search_radius_px"])
        + 0.5 * float(config["rgm_search_step_px"]),
        float(config["rgm_search_step_px"]),
    )
    integration_radius = (
        disk_radius * float(config["integration_radius_fraction"])
    )
    patch_half = int(np.ceil(integration_radius)) + 1
    rows_axis = np.arange(raw.shape[0], dtype=float)
    cols_axis = np.arange(raw.shape[1], dtype=float)
    retained: list[tuple[float, ...]] = []
    for initial_row, initial_col, dog_score in candidates:
        dy, dx = np.meshgrid(offsets, offsets, indexing="ij")
        search_rows = initial_row + dy.ravel()
        search_cols = initial_col + dx.ravel()
        scores = _sample_cubic(
            rgm_spline, search_rows, search_cols, prefilter=False
        )
        best = int(np.argmax(scores))
        row = float(search_rows[best])
        col = float(search_cols[best])
        qx = float(np.interp(col, cols_axis, qx_axis))
        qy = float(np.interp(row, rows_axis, qy_axis))
        q_radius = float(np.hypot(qx, qy))
        if not central_exclusion_Ainv <= q_radius <= k_max_Ainv:
            continue
        r0 = max(0, int(np.floor(row)) - patch_half)
        r1 = min(raw.shape[0], int(np.floor(row)) + patch_half + 2)
        c0 = max(0, int(np.floor(col)) - patch_half)
        c1 = min(raw.shape[1], int(np.floor(col)) + patch_half + 2)
        yy, xx = np.mgrid[r0:r1, c0:c1]
        mask = np.hypot(yy - row, xx - col) <= integration_radius
        integrated = float(raw[r0:r1, c0:c1][mask].sum())
        if integrated > 0.0:
            retained.append(
                (
                    qx,
                    qy,
                    integrated,
                    initial_row,
                    initial_col,
                    row,
                    col,
                    dog_score,
                    float(scores[best]),
                )
            )
    if not retained:
        raise RuntimeError("DoG-RGM filtering removed every diffraction disk")
    values = np.asarray(retained, dtype=float)
    values[:, 2] /= values[:, 2].max()
    values = values[
        values[:, 2] >= float(config["min_integrated_intensity_relative"])
    ]
    if not len(values):
        raise RuntimeError("DoG-RGM intensity threshold removed every disk")
    order = np.lexsort((values[:, 1], values[:, 0], np.hypot(values[:, 0], values[:, 1])))
    values = values[order]
    return DoGRGMResult(
        qx_Ainv=values[:, 0].astype(np.float32),
        qy_Ainv=values[:, 1].astype(np.float32),
        intensity=values[:, 2].astype(np.float32),
        initial_row_px=values[:, 3].astype(np.float32),
        initial_col_px=values[:, 4].astype(np.float32),
        refined_row_px=values[:, 5].astype(np.float32),
        refined_col_px=values[:, 6].astype(np.float32),
        dog_score=values[:, 7].astype(np.float32),
        rgm_score=values[:, 8].astype(np.float32),
    )
