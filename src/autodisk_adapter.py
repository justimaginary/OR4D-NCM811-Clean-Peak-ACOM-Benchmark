from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.ndimage import map_coordinates, spline_filter
from scipy.signal import fftconvolve
from skimage.feature import blob_log


@dataclass(frozen=True)
class AutoDiskResult:
    qx_Ainv: np.ndarray
    qy_Ainv: np.ndarray
    intensity: np.ndarray
    initial_row_px: np.ndarray
    initial_col_px: np.ndarray
    refined_row_px: np.ndarray
    refined_col_px: np.ndarray
    correlation_score: np.ndarray
    rgm_score: np.ndarray
    measured_beam_center_px: tuple[float, float]
    measured_disk_radius_px: float


def measure_vacuum_probe(vacuum_probe: np.ndarray) -> tuple[float, float, float]:
    probe = np.maximum(np.asarray(vacuum_probe, dtype=float), 0.0)
    if probe.ndim != 2 or probe.max() <= 0.0:
        raise ValueError("vacuum_probe must be a positive 2D image")
    normalized = probe / probe.max()
    mask = normalized >= 0.5
    if not np.any(mask):
        raise ValueError("vacuum probe has no pixels above half maximum")
    rows, cols = np.indices(probe.shape, dtype=float)
    weights = normalized * mask
    total = weights.sum()
    row = float((rows * weights).sum() / total)
    col = float((cols * weights).sum() / total)
    radius = float(np.sqrt(mask.sum() / np.pi))
    return row, col, radius


def _radial_kernel(
    radius_px: float,
    *,
    inner_fraction: float,
    split_fraction: float,
    outer_fraction: float,
) -> np.ndarray:
    if not 0.0 <= inner_fraction < split_fraction < outer_fraction:
        raise ValueError("radial kernel fractions must be strictly increasing")
    half = int(np.ceil(radius_px * outer_fraction)) + 1
    y, x = np.mgrid[-half : half + 1, -half : half + 1]
    r = np.hypot(x, y)
    inner = (r >= inner_fraction * radius_px) & (r < split_fraction * radius_px)
    outer = (r >= split_fraction * radius_px) & (r <= outer_fraction * radius_px)
    kernel = np.zeros_like(r, dtype=float)
    if not np.any(inner) or not np.any(outer):
        raise ValueError("disk radius is too small for the requested radial kernel")
    kernel[inner] = 1.0 / inner.sum()
    kernel[outer] = -1.0 / outer.sum()
    return kernel


def _ring_kernel(radius_px: float, width_px: float) -> np.ndarray:
    if radius_px <= 0.0 or width_px <= 0.0:
        raise ValueError("ring radius and width must be positive")
    half = int(np.ceil(radius_px + 2.0 * width_px)) + 1
    y, x = np.mgrid[-half : half + 1, -half : half + 1]
    r = np.hypot(x, y)
    ring = np.exp(-0.5 * ((r - radius_px) / width_px) ** 2)
    ring -= ring.mean()
    norm = np.linalg.norm(ring)
    if norm <= 0.0:
        raise ValueError("ring filter has zero norm")
    return ring / norm


def _sample_bilinear(image: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    return map_coordinates(
        image,
        [rows, cols],
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )


def _sample_cubic(
    image: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    *,
    prefilter: bool = True,
) -> np.ndarray:
    """Sample a smooth response map without pinning maxima to integer pixels."""
    return map_coordinates(
        image,
        [rows, cols],
        order=3,
        mode="constant",
        cval=0.0,
        prefilter=prefilter,
    )


def _deduplicate_candidates(
    candidates: list[tuple[float, float, float]],
    min_spacing_px: float,
    max_num_peaks: int | None = None,
) -> list[tuple[float, float, float]]:
    if max_num_peaks is not None and max_num_peaks <= 0:
        raise ValueError("max_num_peaks must be positive")
    kept: list[tuple[float, float, float]] = []
    for candidate in sorted(candidates, key=lambda row: row[2], reverse=True):
        if all(
            np.hypot(candidate[0] - row[0], candidate[1] - row[1])
            >= min_spacing_px
            for row in kept
        ):
            kept.append(candidate)
            # Candidates are processed from highest to lowest score. Once the
            # requested output size is reached, no remaining candidate can
            # enter the retained top-N set, so continuing the O(N^2) spacing
            # scan cannot change the result.
            if max_num_peaks is not None and len(kept) == max_num_peaks:
                break
    return kept


def detect_autodisk_peaks(
    image: np.ndarray,
    qx_axis_Ainv: np.ndarray,
    qy_axis_Ainv: np.ndarray,
    vacuum_probe: np.ndarray,
    config: dict[str, Any],
    *,
    k_max_Ainv: float,
    central_exclusion_Ainv: float,
) -> AutoDiskResult:
    """Detect CBED disks using the AutoDisk ring/LoG/modified-RGM chain."""
    raw = np.maximum(np.asarray(image, dtype=float), 0.0)
    qx_axis = np.asarray(qx_axis_Ainv, dtype=float)
    qy_axis = np.asarray(qy_axis_Ainv, dtype=float)
    if raw.ndim != 2 or raw.shape != (len(qy_axis), len(qx_axis)):
        raise ValueError("image shape does not match q axes")
    if not np.all(np.isfinite(raw)) or raw.max() <= 0.0:
        raise ValueError("diffraction image must contain finite positive intensity")

    beam_row, beam_col, measured_radius = measure_vacuum_probe(vacuum_probe)
    processed = np.sqrt(raw)
    ring_radius = measured_radius * float(config["ring_radius_fraction"])
    ring = _ring_kernel(ring_radius, float(config["ring_width_px"]))
    correlation = fftconvolve(processed, ring[::-1, ::-1], mode="same")
    correlation -= np.median(correlation)
    correlation = np.maximum(correlation, 0.0)
    exclusion = np.hypot(
        *(
            coordinate - center
            for coordinate, center in zip(
                np.indices(raw.shape, dtype=float), (beam_row, beam_col)
            )
        )
    ) <= measured_radius * 1.5
    correlation[exclusion] = 0.0
    maximum = float(correlation.max())
    if maximum <= 0.0:
        raise RuntimeError("AutoDisk ring correlation contains no positive peak")
    normalized_correlation = correlation / maximum

    blobs = blob_log(
        normalized_correlation,
        min_sigma=float(config["log_min_sigma_px"]),
        max_sigma=float(config["log_max_sigma_px"]),
        num_sigma=int(config["log_num_sigma"]),
        threshold=float(config["correlation_threshold_rel"]),
        overlap=0.5,
        exclude_border=False,
    )
    candidates: list[tuple[float, float, float]] = []
    for row, col, _ in blobs:
        score = float(_sample_bilinear(correlation, np.asarray([row]), np.asarray([col]))[0])
        candidates.append((float(row), float(col), score))
    candidates = _deduplicate_candidates(
        candidates,
        float(config["min_peak_spacing_px"]),
        int(config["max_num_peaks"]),
    )
    if not candidates:
        raise RuntimeError("AutoDisk LoG stage found no diffraction disks")

    rgm_kernel = _radial_kernel(
        measured_radius,
        inner_fraction=float(config["rgm_inner_radius_fraction"]),
        split_fraction=float(config["rgm_split_radius_fraction"]),
        outer_fraction=float(config["rgm_outer_radius_fraction"]),
    )
    rgm_map = fftconvolve(processed, rgm_kernel[::-1, ::-1], mode="same")
    rgm_spline = spline_filter(rgm_map, order=3)
    search_radius = float(config["rgm_search_radius_px"])
    search_step = float(config["rgm_search_step_px"])
    offsets = np.arange(-search_radius, search_radius + 0.5 * search_step, search_step)

    initial_rows: list[float] = []
    initial_cols: list[float] = []
    refined_rows: list[float] = []
    refined_cols: list[float] = []
    corr_scores: list[float] = []
    rgm_scores: list[float] = []
    intensities: list[float] = []
    qx_values: list[float] = []
    qy_values: list[float] = []
    integration_radius = measured_radius * float(config["integration_radius_fraction"])
    patch_half = int(np.ceil(integration_radius)) + 1

    for initial_row, initial_col, corr_score in candidates:
        dy, dx = np.meshgrid(offsets, offsets, indexing="ij")
        sample_rows = initial_row + dy.ravel()
        sample_cols = initial_col + dx.ravel()
        scores = _sample_cubic(
            rgm_spline, sample_rows, sample_cols, prefilter=False
        )
        best = int(np.argmax(scores))
        row = float(sample_rows[best])
        col = float(sample_cols[best])
        qx = float(np.interp(col, np.arange(len(qx_axis)), qx_axis))
        qy = float(np.interp(row, np.arange(len(qy_axis)), qy_axis))
        q_radius = float(np.hypot(qx, qy))
        if q_radius < central_exclusion_Ainv or q_radius > k_max_Ainv:
            continue

        r0 = max(0, int(np.floor(row)) - patch_half)
        r1 = min(raw.shape[0], int(np.floor(row)) + patch_half + 2)
        c0 = max(0, int(np.floor(col)) - patch_half)
        c1 = min(raw.shape[1], int(np.floor(col)) + patch_half + 2)
        yy, xx = np.mgrid[r0:r1, c0:c1]
        mask = np.hypot(yy - row, xx - col) <= integration_radius
        integrated = float(raw[r0:r1, c0:c1][mask].sum())
        if integrated <= 0.0:
            continue

        initial_rows.append(initial_row)
        initial_cols.append(initial_col)
        refined_rows.append(row)
        refined_cols.append(col)
        corr_scores.append(corr_score)
        rgm_scores.append(float(scores[best]))
        intensities.append(integrated)
        qx_values.append(qx)
        qy_values.append(qy)

    if not intensities:
        raise RuntimeError("AutoDisk filtering removed every diffraction disk")
    intensities_array = np.asarray(intensities, dtype=float)
    intensities_array /= intensities_array.max()
    keep_intensity = intensities_array >= float(
        config["min_integrated_intensity_relative"]
    )
    intensities_array = intensities_array[keep_intensity]
    initial_rows = np.asarray(initial_rows, dtype=float)[keep_intensity]
    initial_cols = np.asarray(initial_cols, dtype=float)[keep_intensity]
    refined_rows = np.asarray(refined_rows, dtype=float)[keep_intensity]
    refined_cols = np.asarray(refined_cols, dtype=float)[keep_intensity]
    corr_scores = np.asarray(corr_scores, dtype=float)[keep_intensity]
    rgm_scores = np.asarray(rgm_scores, dtype=float)[keep_intensity]
    qx_values = np.asarray(qx_values, dtype=float)[keep_intensity]
    qy_values = np.asarray(qy_values, dtype=float)[keep_intensity]
    if not len(intensities_array):
        raise RuntimeError("AutoDisk intensity threshold removed every disk")
    qx_array = np.asarray(qx_values, dtype=float)
    qy_array = np.asarray(qy_values, dtype=float)
    order = np.lexsort((qy_array, qx_array, np.hypot(qx_array, qy_array)))

    return AutoDiskResult(
        qx_Ainv=qx_array[order].astype(np.float32),
        qy_Ainv=qy_array[order].astype(np.float32),
        intensity=intensities_array[order].astype(np.float32),
        initial_row_px=np.asarray(initial_rows, dtype=np.float32)[order],
        initial_col_px=np.asarray(initial_cols, dtype=np.float32)[order],
        refined_row_px=np.asarray(refined_rows, dtype=np.float32)[order],
        refined_col_px=np.asarray(refined_cols, dtype=np.float32)[order],
        correlation_score=np.asarray(corr_scores, dtype=np.float32)[order],
        rgm_score=np.asarray(rgm_scores, dtype=np.float32)[order],
        measured_beam_center_px=(beam_row, beam_col),
        measured_disk_radius_px=measured_radius,
    )
