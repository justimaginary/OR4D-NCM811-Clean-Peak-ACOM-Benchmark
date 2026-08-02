from __future__ import annotations

from typing import Any

import numpy as np

from py4dstem_disk_adapter import (
    Py4DSTEMDiskResult,
    _normalize_detection_images,
    _postprocess_py4dstem_peaks,
)


def _deduplicate_by_spacing(
    rows: np.ndarray,
    cols: np.ndarray,
    scores: np.ndarray,
    *,
    min_spacing_px: float,
    max_num_peaks: int,
) -> np.ndarray:
    order = np.argsort(scores)[::-1]
    kept: list[int] = []
    for index in order:
        if all(
            np.hypot(rows[index] - rows[other], cols[index] - cols[other])
            >= min_spacing_px
            for other in kept
        ):
            kept.append(int(index))
            if len(kept) >= max_num_peaks:
                break
    return np.asarray(kept, dtype=int)


def detect_cuda_xcorr_poly_batch(
    images: np.ndarray,
    qx_axis_Ainv: np.ndarray,
    qy_axis_Ainv: np.ndarray,
    vacuum_probe: np.ndarray,
    config: dict[str, Any],
    *,
    k_max_Ainv: float,
    central_exclusion_Ainv: float,
) -> list[Py4DSTEMDiskResult]:
    """Vectorized CUDA cross-correlation with polynomial subpixel refinement.

    This is a distinct detector implementation. It does not claim to be
    py4DSTEM ``find_Bragg_disks`` because the latter performs multicorr DFT
    upsampling and serial per-pattern suppression.
    """
    try:
        import cupy as cp
        from cupyx.scipy.ndimage import gaussian_filter, maximum_filter
    except ImportError as error:
        raise RuntimeError(
            "cuda_xcorr_poly requires CuPy in the active environment"
        ) from error

    raw = np.maximum(np.asarray(images, dtype=np.float32), 0.0)
    qx_axis = np.asarray(qx_axis_Ainv, dtype=float)
    qy_axis = np.asarray(qy_axis_Ainv, dtype=float)
    probe = np.maximum(np.asarray(vacuum_probe, dtype=np.float32), 0.0)
    if raw.ndim != 3 or raw.shape[1:] != probe.shape:
        raise ValueError("images must be [N, row, col] and match vacuum_probe")
    detection = _normalize_detection_images(raw, config)
    template = np.fft.ifftshift(probe)
    template_ft = cp.conj(cp.fft.fft2(cp.asarray(template))).astype(cp.complex64)
    batch_size = int(config["batch_size"])
    min_absolute = float(config["min_absolute_intensity"])
    min_relative = float(config["min_relative_intensity"])
    min_spacing = float(config["min_peak_spacing_px"])
    max_num_peaks = int(config["max_num_peaks"])
    sigma = float(config["sigma_cc"])
    edge = int(config["edge_boundary_px"])
    peak_rows: list[np.ndarray] = []
    peak_cols: list[np.ndarray] = []
    peak_scores: list[np.ndarray] = []

    for start in range(0, len(detection), batch_size):
        stop = min(start + batch_size, len(detection))
        gpu_images = cp.asarray(detection[start:stop], dtype=cp.float32)
        spectrum = cp.fft.fft2(gpu_images, axes=(-2, -1))
        correlation = cp.maximum(
            cp.fft.ifft2(
                spectrum * template_ft[None, :, :], axes=(-2, -1)
            ).real,
            0.0,
        )
        if sigma > 0.0:
            correlation = gaussian_filter(
                correlation, sigma=(0.0, sigma, sigma)
            )
        local_max = maximum_filter(correlation, size=(1, 3, 3))
        threshold = cp.maximum(
            min_absolute,
            correlation.max(axis=(-2, -1))[:, None, None] * min_relative,
        )
        maxima = (correlation == local_max) & (correlation >= threshold)
        if edge > 0:
            maxima[:, :edge, :] = False
            maxima[:, -edge:, :] = False
            maxima[:, :, :edge] = False
            maxima[:, :, -edge:] = False
        else:
            maxima[:, :1, :] = False
            maxima[:, -1:, :] = False
            maxima[:, :, :1] = False
            maxima[:, :, -1:] = False

        for local_index in range(stop - start):
            rows, cols = cp.nonzero(maxima[local_index])
            scores = correlation[local_index, rows, cols]
            if not len(rows):
                peak_rows.append(np.empty(0, dtype=float))
                peak_cols.append(np.empty(0, dtype=float))
                peak_scores.append(np.empty(0, dtype=float))
                continue
            center = scores
            row_minus = correlation[local_index, rows - 1, cols]
            row_plus = correlation[local_index, rows + 1, cols]
            col_minus = correlation[local_index, rows, cols - 1]
            col_plus = correlation[local_index, rows, cols + 1]
            row_denominator = 4 * center - 2 * row_plus - 2 * row_minus
            col_denominator = 4 * center - 2 * col_plus - 2 * col_minus
            row_delta = cp.where(
                cp.abs(row_denominator) > 1e-20,
                (row_plus - row_minus) / row_denominator,
                0.0,
            )
            col_delta = cp.where(
                cp.abs(col_denominator) > 1e-20,
                (col_plus - col_minus) / col_denominator,
                0.0,
            )
            row_delta = cp.where(cp.abs(row_delta) <= 1.0, row_delta, 0.0)
            col_delta = cp.where(cp.abs(col_delta) <= 1.0, col_delta, 0.0)
            rows_cpu = cp.asnumpy(rows.astype(cp.float32) + row_delta)
            cols_cpu = cp.asnumpy(cols.astype(cp.float32) + col_delta)
            scores_cpu = cp.asnumpy(scores)
            keep = _deduplicate_by_spacing(
                rows_cpu,
                cols_cpu,
                scores_cpu,
                min_spacing_px=min_spacing,
                max_num_peaks=max_num_peaks,
            )
            peak_rows.append(rows_cpu[keep])
            peak_cols.append(cols_cpu[keep])
            peak_scores.append(scores_cpu[keep])
        del gpu_images, spectrum, correlation, local_max, maxima

    results = []
    peak_dtype = np.dtype(
        [("qx", np.float64), ("qy", np.float64), ("intensity", np.float64)]
    )
    for index, (rows, cols, scores) in enumerate(
        zip(peak_rows, peak_cols, peak_scores)
    ):
        data = np.empty(len(rows), dtype=peak_dtype)
        data["qx"] = rows
        data["qy"] = cols
        data["intensity"] = scores
        results.append(
            _postprocess_py4dstem_peaks(
                raw[index],
                data,
                qx_axis,
                qy_axis,
                probe,
                config,
                k_max_Ainv=k_max_Ainv,
                central_exclusion_Ainv=central_exclusion_Ainv,
            )
        )
    cp.cuda.get_current_stream().synchronize()
    return results
