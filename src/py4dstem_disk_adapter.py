from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from py4DSTEM.braggvectors import find_Bragg_disks


@dataclass(frozen=True)
class Py4DSTEMDiskResult:
    qx_Ainv: np.ndarray
    qy_Ainv: np.ndarray
    intensity: np.ndarray
    row_px: np.ndarray
    col_px: np.ndarray
    correlation_intensity: np.ndarray


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
    data = peaks.data
    row = np.asarray(data["qx"], dtype=float)
    col = np.asarray(data["qy"], dtype=float)
    # FFT peak coordinates are referenced to floor(N/2), whereas this
    # benchmark places the beam at (N-1)/2 for even-sized arrays.
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
    correlation = np.asarray(data["intensity"], dtype=float)
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
    normalized = correlation / correlation.max()
    order = np.lexsort((qy, qx, np.hypot(qx, qy)))
    return Py4DSTEMDiskResult(
        qx_Ainv=qx[order].astype(np.float32),
        qy_Ainv=qy[order].astype(np.float32),
        intensity=normalized[order].astype(np.float32),
        row_px=row[order].astype(np.float32),
        col_px=col[order].astype(np.float32),
        correlation_intensity=correlation[order].astype(np.float32),
    )
