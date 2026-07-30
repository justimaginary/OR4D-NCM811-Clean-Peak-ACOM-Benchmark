from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from py4DSTEM import DataCube
from py4DSTEM.braggvectors import find_Bragg_disks

from autodisk_adapter import measure_vacuum_probe


@dataclass(frozen=True)
class Py4DSTEMDiskResult:
    qx_Ainv: np.ndarray
    qy_Ainv: np.ndarray
    intensity: np.ndarray
    row_px: np.ndarray
    col_px: np.ndarray
    correlation_intensity: np.ndarray


def _postprocess_py4dstem_peaks(
    raw: np.ndarray,
    peak_data: np.ndarray,
    qx_axis: np.ndarray,
    qy_axis: np.ndarray,
    vacuum_probe: np.ndarray,
    config: dict[str, Any],
    *,
    k_max_Ainv: float,
    central_exclusion_Ainv: float,
) -> Py4DSTEMDiskResult:
    row = np.asarray(peak_data["qx"], dtype=float)
    col = np.asarray(peak_data["qy"], dtype=float)
    physical_zero_col = float(
        np.interp(0.0, qx_axis, np.arange(len(qx_axis), dtype=float))
    )
    physical_zero_row = float(
        np.interp(
            0.0,
            qy_axis[::-1],
            np.arange(len(qy_axis), dtype=float)[::-1],
        )
    )
    col -= raw.shape[1] // 2 - physical_zero_col
    row -= raw.shape[0] // 2 - physical_zero_row
    correlation = np.asarray(peak_data["intensity"], dtype=float)
    qx = np.interp(col, np.arange(len(qx_axis), dtype=float), qx_axis)
    qy = np.interp(row, np.arange(len(qy_axis), dtype=float), qy_axis)
    radius = np.hypot(qx, qy)
    keep = (
        np.isfinite(qx)
        & np.isfinite(qy)
        & np.isfinite(correlation)
        & (correlation > 0.0)
        & (radius >= float(central_exclusion_Ainv))
        & (radius <= float(k_max_Ainv))
    )
    if not np.any(keep):
        raise RuntimeError("find_Bragg_disks returned no usable non-central disk")

    row = row[keep]
    col = col[keep]
    qx = qx[keep]
    qy = qy[keep]
    correlation = correlation[keep]
    _, _, disk_radius_px = measure_vacuum_probe(vacuum_probe)
    integration_radius = (
        disk_radius_px * float(config["integration_radius_fraction"])
    )
    patch_half = int(np.ceil(integration_radius)) + 1
    integrated = []
    for peak_row, peak_col in zip(row, col):
        r0 = max(0, int(np.floor(peak_row)) - patch_half)
        r1 = min(raw.shape[0], int(np.floor(peak_row)) + patch_half + 2)
        c0 = max(0, int(np.floor(peak_col)) - patch_half)
        c1 = min(raw.shape[1], int(np.floor(peak_col)) + patch_half + 2)
        yy, xx = np.mgrid[r0:r1, c0:c1]
        mask = np.hypot(yy - peak_row, xx - peak_col) <= integration_radius
        integrated.append(float(raw[r0:r1, c0:c1][mask].sum()))
    integrated = np.asarray(integrated, dtype=float)
    normalized = integrated / integrated.max()
    keep_intensity = normalized >= float(
        config["min_integrated_intensity_relative"]
    )
    row = row[keep_intensity]
    col = col[keep_intensity]
    qx = qx[keep_intensity]
    qy = qy[keep_intensity]
    correlation = correlation[keep_intensity]
    normalized = normalized[keep_intensity]
    if not len(normalized):
        raise RuntimeError("find_Bragg_disks intensity threshold removed every disk")
    order = np.lexsort((qy, qx, np.hypot(qx, qy)))
    return Py4DSTEMDiskResult(
        qx_Ainv=qx[order].astype(np.float32),
        qy_Ainv=qy[order].astype(np.float32),
        intensity=normalized[order].astype(np.float32),
        row_px=row[order].astype(np.float32),
        col_px=col[order].astype(np.float32),
        correlation_intensity=correlation[order].astype(np.float32),
    )


def detect_py4dstem_bragg_disks(
    image: np.ndarray,
    qx_axis_Ainv: np.ndarray,
    qy_axis_Ainv: np.ndarray,
    vacuum_probe: np.ndarray,
    config: dict[str, Any],
    *,
    k_max_Ainv: float,
    central_exclusion_Ainv: float,
) -> Py4DSTEMDiskResult:
    """Run py4DSTEM's public ``find_Bragg_disks`` on one diffraction image.

    py4DSTEM calls detector-array axis 0 ``qx`` and axis 1 ``qy``. This
    benchmark instead calls the horizontal physical coordinate qx and the
    upward vertical coordinate qy, so the returned pixel fields are mapped
    explicitly rather than copied by name.
    """
    raw = np.maximum(np.asarray(image, dtype=float), 0.0)
    # py4DSTEM's FFT correlation expects the template origin at array index 0.
    template = np.fft.ifftshift(
        np.maximum(np.asarray(vacuum_probe, dtype=float), 0.0)
    )
    qx_axis = np.asarray(qx_axis_Ainv, dtype=float)
    qy_axis = np.asarray(qy_axis_Ainv, dtype=float)
    if raw.ndim != 2 or raw.shape != template.shape:
        raise ValueError("image and vacuum_probe must be same-shape 2D arrays")
    if raw.shape != (len(qy_axis), len(qx_axis)):
        raise ValueError("image shape does not match q axes")

    peaks = find_Bragg_disks(
        data=raw,
        template=template,
        corrPower=float(config["corr_power"]),
        sigma_cc=float(config["sigma_cc"]),
        subpixel=str(config["subpixel"]),
        upsample_factor=int(config["upsample_factor"]),
        minRelativeIntensity=float(config["min_relative_intensity"]),
        minPeakSpacing=float(config["min_peak_spacing_px"]),
        edgeBoundary=int(config["edge_boundary_px"]),
        maxNumPeaks=int(config["max_num_peaks"]),
    )
    return _postprocess_py4dstem_peaks(
        raw,
        peaks.data,
        qx_axis,
        qy_axis,
        vacuum_probe,
        config,
        k_max_Ainv=k_max_Ainv,
        central_exclusion_Ainv=central_exclusion_Ainv,
    )


def detect_py4dstem_bragg_disks_batch(
    images: np.ndarray,
    qx_axis_Ainv: np.ndarray,
    qy_axis_Ainv: np.ndarray,
    vacuum_probe: np.ndarray,
    config: dict[str, Any],
    *,
    k_max_Ainv: float,
    central_exclusion_Ainv: float,
    cuda: bool,
) -> list[Py4DSTEMDiskResult]:
    """Run py4DSTEM on an image stack, using its batched CUDA path when requested."""
    raw = np.maximum(np.asarray(images, dtype=np.float32), 0.0)
    qx_axis = np.asarray(qx_axis_Ainv, dtype=float)
    qy_axis = np.asarray(qy_axis_Ainv, dtype=float)
    probe = np.maximum(np.asarray(vacuum_probe, dtype=float), 0.0)
    if raw.ndim != 3 or raw.shape[1:] != probe.shape:
        raise ValueError("images must be [N, row, col] and match vacuum_probe")
    if raw.shape[1:] != (len(qy_axis), len(qx_axis)):
        raise ValueError("image shape does not match q axes")
    template = np.fft.ifftshift(probe)
    datacube = DataCube(data=raw[:, None, :, :])
    vectors = find_Bragg_disks(
        data=datacube,
        template=template,
        corrPower=float(config["corr_power"]),
        sigma_cc=float(config["sigma_cc"]),
        subpixel=str(config["subpixel"]),
        upsample_factor=int(config["upsample_factor"]),
        minRelativeIntensity=float(config["min_relative_intensity"]),
        minPeakSpacing=float(config["min_peak_spacing_px"]),
        edgeBoundary=int(config["edge_boundary_px"]),
        maxNumPeaks=int(config["max_num_peaks"]),
        CUDA=bool(cuda),
        CUDA_batched=bool(cuda),
    )
    return [
        _postprocess_py4dstem_peaks(
            raw[index],
            vectors.raw[index, 0].data,
            qx_axis,
            qy_axis,
            probe,
            config,
            k_max_Ainv=k_max_Ainv,
            central_exclusion_Ainv=central_exclusion_Ainv,
        )
        for index in range(len(raw))
    ]
