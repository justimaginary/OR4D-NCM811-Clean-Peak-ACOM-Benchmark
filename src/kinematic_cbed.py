from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter


@dataclass(frozen=True)
class ReflectionLibrary:
    """Crystal-frame reciprocal vectors and complex structure factors."""

    g_crystal_Ainv: np.ndarray
    hkl: np.ndarray
    structure_factor: np.ndarray
    wavelength_A: float


@dataclass(frozen=True)
class ProjectedReflectionSet:
    """Projected kinematical reflections from the matched ACOM model."""

    qx_Ainv: np.ndarray
    qy_Ainv: np.ndarray
    intensity: np.ndarray
    hkl: np.ndarray
    wavelength_A: float


@dataclass(frozen=True)
class RenderedPattern:
    expectation: np.ndarray
    direct_expectation: np.ndarray
    scattered_expectation: np.ndarray
    vacuum_probe: np.ndarray
    qx_axis_Ainv: np.ndarray
    qy_axis_Ainv: np.ndarray
    beam_center_px: tuple[float, float]
    disk_radius_px: float
    oracle_qx_Ainv: np.ndarray
    oracle_qy_Ainv: np.ndarray
    oracle_intensity_raw: np.ndarray
    oracle_intensity_normalized: np.ndarray
    oracle_hkl: np.ndarray
    oracle_candidate_reflection_count: int
    oracle_merged_disk_count: int
    oracle_rejected_edge_count: int
    oracle_rejected_low_intensity_count: int


def _soft_circular_aperture_amplitude(
    radius_Ainv: np.ndarray,
    cutoff_Ainv: float,
    soft_edge_fraction: float,
) -> np.ndarray:
    if not 0.0 <= soft_edge_fraction < 1.0:
        raise ValueError("aperture_soft_edge_fraction must lie in [0, 1)")
    inner = cutoff_Ainv * (1.0 - soft_edge_fraction)
    result = np.zeros_like(radius_Ainv, dtype=float)
    result[radius_Ainv <= inner] = 1.0
    edge = (radius_Ainv > inner) & (radius_Ainv < cutoff_Ainv)
    if np.any(edge):
        phase = (radius_Ainv[edge] - inner) / (cutoff_Ainv - inner)
        result[edge] = 0.5 * (1.0 + np.cos(np.pi * phase))
    return result


def _probe_phase(
    kappa_x_Ainv: np.ndarray,
    kappa_y_Ainv: np.ndarray,
    wavelength_A: float,
    config: dict[str, Any],
) -> np.ndarray:
    """Return the coherent probe phase.

    Pure phase aberrations affect the intensity only where coherent scattering
    amplitudes overlap. They are retained here for a well-defined First-Born
    wave model, while the canonical Clean configuration keeps them at zero.
    """
    defocus = float(config.get("defocus_A", 0.0))
    astigmatism = float(config.get("astigmatism_A", 0.0))
    angle = np.deg2rad(float(config.get("astigmatism_angle_deg", 0.0)))
    k2 = kappa_x_Ainv**2 + kappa_y_Ainv**2
    theta = np.arctan2(kappa_y_Ainv, kappa_x_Ainv)
    effective_defocus = defocus + astigmatism * np.cos(2.0 * (theta - angle))
    chi = np.pi * wavelength_A * effective_defocus * k2
    return np.exp(-1j * chi)


def _downsample_mean(image: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 0:
        raise ValueError("oversampling must be positive")
    if factor == 1:
        return image
    ny, nx = image.shape
    if ny % factor or nx % factor:
        raise ValueError("oversampled dimensions must be divisible by oversampling")
    return image.reshape(ny // factor, factor, nx // factor, factor).mean(
        axis=(1, 3)
    )


def _normalize_probability(image: np.ndarray, name: str) -> np.ndarray:
    image = np.maximum(np.asarray(image, dtype=float), 0.0)
    total = float(image.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise RuntimeError(f"{name} contains no finite positive intensity")
    return image / total


def _merge_oracle_disks(
    qxy: np.ndarray,
    intensity: np.ndarray,
    hkl: np.ndarray,
    tolerance_Ainv: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(qxy) == 0:
        return qxy, intensity, hkl
    groups: list[dict[str, Any]] = []
    for index in np.argsort(intensity)[::-1]:
        point = qxy[index]
        group = next(
            (
                candidate
                for candidate in groups
                if np.linalg.norm(point - candidate["qxy"]) <= tolerance_Ainv
            ),
            None,
        )
        if group is None:
            groups.append(
                {
                    "qxy": point.copy(),
                    "weighted_qxy": point * intensity[index],
                    "intensity": float(intensity[index]),
                    "hkl": hkl[index].copy(),
                    "dominant": float(intensity[index]),
                }
            )
        else:
            group["weighted_qxy"] += point * intensity[index]
            group["intensity"] += float(intensity[index])
            if intensity[index] > group["dominant"]:
                group["hkl"] = hkl[index].copy()
                group["dominant"] = float(intensity[index])
    return (
        np.stack(
            [group["weighted_qxy"] / group["intensity"] for group in groups]
        ),
        np.asarray([group["intensity"] for group in groups]),
        np.stack([group["hkl"] for group in groups]).astype(np.int32),
    )


def _integrate_image_matched_oracle(
    image: np.ndarray,
    qxy: np.ndarray,
    hkl: np.ndarray,
    qx_axis_Ainv: np.ndarray,
    qy_axis_Ainv: np.ndarray,
    integration_radius_px: float,
    *,
    require_full_disk: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Integrate the final coherent scattered image at theoretical disk centers."""
    rows_axis = np.arange(image.shape[0], dtype=float)
    cols_axis = np.arange(image.shape[1], dtype=float)
    patch_half = int(np.ceil(integration_radius_px)) + 1
    retained_qxy: list[np.ndarray] = []
    retained_hkl: list[np.ndarray] = []
    integrated: list[float] = []
    rejected_edge = 0
    for point, reflection_hkl in zip(qxy, hkl):
        col = float(np.interp(point[0], qx_axis_Ainv, cols_axis))
        row = float(
            np.interp(point[1], qy_axis_Ainv[::-1], rows_axis[::-1])
        )
        if require_full_disk and (
            row - integration_radius_px < 0.0
            or row + integration_radius_px > image.shape[0] - 1
            or col - integration_radius_px < 0.0
            or col + integration_radius_px > image.shape[1] - 1
        ):
            rejected_edge += 1
            continue
        r0 = max(0, int(np.floor(row)) - patch_half)
        r1 = min(image.shape[0], int(np.floor(row)) + patch_half + 2)
        c0 = max(0, int(np.floor(col)) - patch_half)
        c1 = min(image.shape[1], int(np.floor(col)) + patch_half + 2)
        yy, xx = np.mgrid[r0:r1, c0:c1]
        mask = np.hypot(yy - row, xx - col) <= integration_radius_px
        value = float(image[r0:r1, c0:c1][mask].sum())
        if value <= 0.0 or not np.isfinite(value):
            continue
        retained_qxy.append(point)
        retained_hkl.append(reflection_hkl)
        integrated.append(value)
    if not integrated:
        raise RuntimeError("image-matched oracle contains no positive disk")
    return (
        np.stack(retained_qxy),
        np.asarray(integrated, dtype=float),
        np.stack(retained_hkl).astype(np.int32),
        rejected_edge,
    )


def render_acom_matched_cbed(
    reflections: ProjectedReflectionSet,
    config: dict[str, Any],
    *,
    k_max_Ainv: float,
    direct_beam_fraction: float | None = None,
) -> RenderedPattern:
    """Form finite convergent-beam disks from py4DSTEM-matched reflections.

    Reflection support and integrated kinematical intensity come directly from
    ``Crystal.generate_diffraction_pattern`` using the same excitation-error
    contract as the ACOM orientation plan. Image formation remains separate:
    complex aperture amplitudes are placed at subpixel coordinates, coherently
    accumulated where disks overlap, and detector pixels integrate the result.
    """
    fraction = float(
        config["canonical_direct_beam_fraction"]
        if direct_beam_fraction is None
        else direct_beam_fraction
    )
    if not 0.0 < fraction < 1.0:
        raise ValueError("direct_beam_fraction must lie strictly between 0 and 1")
    ny, nx = (int(value) for value in config["gpts"])
    oversampling = int(config["oversampling"])
    q_max = float(config["q_max_Ainv"])
    center_offset_y, center_offset_x = (
        float(value) for value in config.get("beam_center_offset_px", (0.0, 0.0))
    )
    center_y = (ny - 1) / 2.0 + center_offset_y
    center_x = (nx - 1) / 2.0 + center_offset_x
    q_pixel = 2.0 * q_max / max(nx, ny)
    qx_axis = (np.arange(nx, dtype=float) - center_x) * q_pixel
    qy_axis = (center_y - np.arange(ny, dtype=float)) * q_pixel
    high_ny, high_nx = ny * oversampling, nx * oversampling
    high_center_y = (high_ny - 1) / 2.0 + center_offset_y * oversampling
    high_center_x = (high_nx - 1) / 2.0 + center_offset_x * oversampling
    high_q_pixel = q_pixel / oversampling

    wavelength_A = float(reflections.wavelength_A)
    disk_radius_Ainv = (
        float(config["convergence_semiangle_mrad"]) * 1e-3 / wavelength_A
    )
    disk_radius_px = disk_radius_Ainv / q_pixel
    soft_edge = float(config["aperture_soft_edge_fraction"])
    patch_half_width = int(np.ceil(disk_radius_Ainv / high_q_pixel)) + 2
    scattered_amplitude = np.zeros((high_ny, high_nx), dtype=np.complex128)
    direct_high = np.zeros((high_ny, high_nx), dtype=np.float64)

    qx_input = np.asarray(reflections.qx_Ainv, dtype=float)
    qy_input = np.asarray(reflections.qy_Ainv, dtype=float)
    intensity_input = np.asarray(reflections.intensity, dtype=float)
    hkl_input = np.asarray(reflections.hkl, dtype=np.int32)
    radius = np.hypot(qx_input, qy_input)
    keep = (
        np.isfinite(qx_input)
        & np.isfinite(qy_input)
        & np.isfinite(intensity_input)
        & (intensity_input > 0.0)
        & (radius > disk_radius_Ainv)
        & (radius <= float(k_max_Ainv) + 1e-12)
    )
    qxy_input = np.column_stack((qx_input[keep], qy_input[keep]))
    intensity_input = intensity_input[keep]
    hkl_input = hkl_input[keep]

    for (gx, gy), intensity in zip(qxy_input, intensity_input):
        col_center = high_center_x + gx / high_q_pixel
        row_center = high_center_y - gy / high_q_pixel
        c0 = max(0, int(np.floor(col_center)) - patch_half_width)
        c1 = min(high_nx, int(np.floor(col_center)) + patch_half_width + 2)
        r0 = max(0, int(np.floor(row_center)) - patch_half_width)
        r1 = min(high_ny, int(np.floor(row_center)) + patch_half_width + 2)
        if c0 >= c1 or r0 >= r1:
            continue
        cols = np.arange(c0, c1, dtype=float)[None, :]
        rows = np.arange(r0, r1, dtype=float)[:, None]
        qx = (cols - high_center_x) * high_q_pixel
        qy = (high_center_y - rows) * high_q_pixel
        kappa_x = qx - gx
        kappa_y = qy - gy
        aperture = _soft_circular_aperture_amplitude(
            np.hypot(kappa_x, kappa_y), disk_radius_Ainv, soft_edge
        )
        component = (
            np.sqrt(intensity)
            * aperture
            * _probe_phase(kappa_x, kappa_y, wavelength_A, config)
        )
        scattered_amplitude[r0:r1, c0:c1] += component

    c0 = max(0, int(np.floor(high_center_x)) - patch_half_width)
    c1 = min(high_nx, int(np.floor(high_center_x)) + patch_half_width + 2)
    r0 = max(0, int(np.floor(high_center_y)) - patch_half_width)
    r1 = min(high_ny, int(np.floor(high_center_y)) + patch_half_width + 2)
    cols = np.arange(c0, c1, dtype=float)[None, :]
    rows = np.arange(r0, r1, dtype=float)[:, None]
    qx = (cols - high_center_x) * high_q_pixel
    qy = (high_center_y - rows) * high_q_pixel
    direct_aperture = _soft_circular_aperture_amplitude(
        np.hypot(qx, qy), disk_radius_Ainv, soft_edge
    )
    direct_high[r0:r1, c0:c1] = direct_aperture**2

    direct = _downsample_mean(direct_high, oversampling)
    scattered = _downsample_mean(np.abs(scattered_amplitude) ** 2, oversampling)
    psf_sigma = float(config.get("detector_psf_sigma_px", 0.0))
    if psf_sigma > 0.0:
        direct = gaussian_filter(direct, sigma=psf_sigma, mode="constant")
        scattered = gaussian_filter(scattered, sigma=psf_sigma, mode="constant")
    direct_probability = _normalize_probability(direct, "direct expectation")
    scattered_probability = _normalize_probability(
        scattered, "scattered expectation"
    )
    expectation = (
        fraction * direct_probability + (1.0 - fraction) * scattered_probability
    )
    expectation /= expectation.sum()
    vacuum_probe = direct / direct.max()

    merged_qxy, _, merged_hkl = _merge_oracle_disks(
        qxy_input,
        intensity_input,
        hkl_input,
        tolerance_Ainv=float(config["oracle_merge_distance_px"]) * q_pixel,
    )
    merged_count = len(merged_qxy)
    merged_qxy, integrated, merged_hkl, rejected_edge = (
        _integrate_image_matched_oracle(
            scattered,
            merged_qxy,
            merged_hkl,
            qx_axis,
            qy_axis,
            integration_radius_px=(
                float(config["oracle_integration_radius_fraction"])
                * disk_radius_px
            ),
            require_full_disk=bool(config["oracle_require_full_disk"]),
        )
    )
    relative = integrated / integrated.max()
    retain = relative >= float(config["physical_oracle_min_relative_intensity"])
    rejected_low = int(np.count_nonzero(~retain))
    merged_qxy = merged_qxy[retain]
    integrated = integrated[retain]
    merged_hkl = merged_hkl[retain]
    relative = relative[retain]
    order = np.lexsort(
        (
            merged_qxy[:, 1],
            merged_qxy[:, 0],
            np.linalg.norm(merged_qxy, axis=1),
        )
    )
    return RenderedPattern(
        expectation=expectation.astype(np.float32),
        direct_expectation=direct_probability.astype(np.float32),
        scattered_expectation=scattered_probability.astype(np.float32),
        vacuum_probe=vacuum_probe.astype(np.float32),
        qx_axis_Ainv=qx_axis.astype(np.float32),
        qy_axis_Ainv=qy_axis.astype(np.float32),
        beam_center_px=(float(center_y), float(center_x)),
        disk_radius_px=float(disk_radius_px),
        oracle_qx_Ainv=merged_qxy[order, 0].astype(np.float32),
        oracle_qy_Ainv=merged_qxy[order, 1].astype(np.float32),
        oracle_intensity_raw=integrated[order].astype(np.float32),
        oracle_intensity_normalized=relative[order].astype(np.float32),
        oracle_hkl=merged_hkl[order],
        oracle_candidate_reflection_count=len(qxy_input),
        oracle_merged_disk_count=merged_count,
        oracle_rejected_edge_count=rejected_edge,
        oracle_rejected_low_intensity_count=rejected_low,
    )


def render_kinematic_cbed(
    library: ReflectionLibrary,
    orientation_matrix_sample_to_crystal: np.ndarray,
    config: dict[str, Any],
    *,
    k_max_Ainv: float,
    direct_beam_fraction: float | None = None,
) -> RenderedPattern:
    """Render one coherent First-Born/kinematical convergent-beam pattern.

    Reciprocal coordinates use cycles/angstrom (no 2*pi). Scattered complex
    amplitudes are summed before taking their magnitude squared. The direct and
    total scattered expectations are then independently normalized and mixed by
    an explicit benchmark-level direct-beam fraction.
    """
    R = np.asarray(orientation_matrix_sample_to_crystal, dtype=float)
    if R.shape != (3, 3):
        raise ValueError(f"orientation matrix must be 3x3, got {R.shape}")
    if not np.allclose(R.T @ R, np.eye(3), atol=1e-7):
        raise ValueError("orientation matrix is not orthonormal")
    if not np.isclose(np.linalg.det(R), 1.0, atol=1e-7):
        raise ValueError("orientation matrix determinant is not +1")

    fraction = float(
        config["canonical_direct_beam_fraction"]
        if direct_beam_fraction is None
        else direct_beam_fraction
    )
    if not 0.0 < fraction < 1.0:
        raise ValueError("direct_beam_fraction must lie strictly between 0 and 1")

    ny, nx = (int(value) for value in config["gpts"])
    oversampling = int(config["oversampling"])
    q_max = float(config["q_max_Ainv"])
    center_offset_y, center_offset_x = (
        float(value) for value in config.get("beam_center_offset_px", (0.0, 0.0))
    )
    center_y = (ny - 1) / 2.0 + center_offset_y
    center_x = (nx - 1) / 2.0 + center_offset_x
    q_pixel = 2.0 * q_max / max(nx, ny)
    qx_axis = (np.arange(nx, dtype=float) - center_x) * q_pixel
    qy_axis = (center_y - np.arange(ny, dtype=float)) * q_pixel

    high_ny, high_nx = ny * oversampling, nx * oversampling
    high_center_y = (high_ny - 1) / 2.0 + center_offset_y * oversampling
    high_center_x = (high_nx - 1) / 2.0 + center_offset_x * oversampling
    high_q_pixel = q_pixel / oversampling
    scattered_amplitude = np.zeros((high_ny, high_nx), dtype=np.complex128)
    direct_high = np.zeros((high_ny, high_nx), dtype=np.float64)

    wavelength_A = float(library.wavelength_A)
    k0_Ainv = 1.0 / wavelength_A
    disk_radius_Ainv = (
        float(config["convergence_semiangle_mrad"]) * 1e-3 / wavelength_A
    )
    disk_radius_px = disk_radius_Ainv / q_pixel
    thickness_A = float(config["thickness_nm"]) * 10.0
    soft_edge = float(config["aperture_soft_edge_fraction"])

    g_sample = (R.T @ np.asarray(library.g_crystal_Ainv, dtype=float).T).T
    hkl = np.asarray(library.hkl, dtype=np.int32)
    structure_factor = np.asarray(library.structure_factor, dtype=np.complex128)
    is_center = np.linalg.norm(g_sample, axis=1) < 1e-10
    projected_radius = np.linalg.norm(g_sample[:, :2], axis=1)
    candidates = np.flatnonzero(
        (~is_center)
        & np.isfinite(structure_factor)
        & (np.abs(structure_factor) > 0.0)
        & (projected_radius > disk_radius_Ainv)
        & (projected_radius <= float(k_max_Ainv) + 1e-12)
    )
    patch_half_width = int(np.ceil(disk_radius_Ainv / high_q_pixel)) + 2
    oracle_qxy: list[np.ndarray] = []
    oracle_intensity: list[float] = []
    oracle_hkl: list[np.ndarray] = []

    for index in candidates:
        gx, gy, gz = g_sample[index]
        col_center = high_center_x + gx / high_q_pixel
        row_center = high_center_y - gy / high_q_pixel
        c0 = max(0, int(np.floor(col_center)) - patch_half_width)
        c1 = min(high_nx, int(np.floor(col_center)) + patch_half_width + 2)
        r0 = max(0, int(np.floor(row_center)) - patch_half_width)
        r1 = min(high_ny, int(np.floor(row_center)) + patch_half_width + 2)
        if c0 >= c1 or r0 >= r1:
            continue

        cols = np.arange(c0, c1, dtype=float)[None, :]
        rows = np.arange(r0, r1, dtype=float)[:, None]
        qx = (cols - high_center_x) * high_q_pixel
        qy = (high_center_y - rows) * high_q_pixel
        kappa_x = qx - gx
        kappa_y = qy - gy
        aperture = _soft_circular_aperture_amplitude(
            np.hypot(kappa_x, kappa_y), disk_radius_Ainv, soft_edge
        )
        if not np.any(aperture):
            continue
        kz_incident = np.sqrt(
            np.maximum(k0_Ainv**2 - kappa_x**2 - kappa_y**2, 0.0)
        )
        kz_outgoing = np.sqrt(np.maximum(k0_Ainv**2 - qx**2 - qy**2, 0.0))
        delta_kz = kz_outgoing - kz_incident - gz
        thickness_amplitude = (
            thickness_A
            * np.sinc(thickness_A * delta_kz)
            * np.exp(1j * np.pi * thickness_A * delta_kz)
        )
        component = (
            structure_factor[index]
            * aperture
            * _probe_phase(kappa_x, kappa_y, wavelength_A, config)
            * thickness_amplitude
        )
        scattered_amplitude[r0:r1, c0:c1] += component
        component_intensity = float(np.sum(np.abs(component) ** 2))
        if component_intensity > 0.0:
            oracle_qxy.append(np.asarray([gx, gy], dtype=float))
            oracle_intensity.append(component_intensity)
            oracle_hkl.append(hkl[index])

    # Independent vacuum/direct probe. F_000 is not used because direct-beam
    # depletion is outside the First-Born model and is controlled explicitly.
    c0 = max(0, int(np.floor(high_center_x)) - patch_half_width)
    c1 = min(high_nx, int(np.floor(high_center_x)) + patch_half_width + 2)
    r0 = max(0, int(np.floor(high_center_y)) - patch_half_width)
    r1 = min(high_ny, int(np.floor(high_center_y)) + patch_half_width + 2)
    cols = np.arange(c0, c1, dtype=float)[None, :]
    rows = np.arange(r0, r1, dtype=float)[:, None]
    qx = (cols - high_center_x) * high_q_pixel
    qy = (high_center_y - rows) * high_q_pixel
    direct_aperture = _soft_circular_aperture_amplitude(
        np.hypot(qx, qy), disk_radius_Ainv, soft_edge
    )
    direct_high[r0:r1, c0:c1] = direct_aperture**2

    direct = _downsample_mean(direct_high, oversampling)
    scattered = _downsample_mean(np.abs(scattered_amplitude) ** 2, oversampling)
    psf_sigma = float(config.get("detector_psf_sigma_px", 0.0))
    if psf_sigma > 0.0:
        direct = gaussian_filter(direct, sigma=psf_sigma, mode="constant")
        scattered = gaussian_filter(scattered, sigma=psf_sigma, mode="constant")
    direct_probability = _normalize_probability(direct, "direct expectation")
    scattered_probability = _normalize_probability(
        scattered, "scattered expectation"
    )
    expectation = (
        fraction * direct_probability + (1.0 - fraction) * scattered_probability
    )
    expectation /= expectation.sum()
    vacuum_probe = direct / direct.max()

    candidate_count = len(oracle_qxy)
    merged_qxy, _, merged_hkl = _merge_oracle_disks(
        np.stack(oracle_qxy),
        np.asarray(oracle_intensity, dtype=float),
        np.stack(oracle_hkl),
        tolerance_Ainv=float(config["oracle_merge_distance_px"]) * q_pixel,
    )
    merged_count = len(merged_qxy)
    merged_qxy, merged_intensity, merged_hkl, rejected_edge = (
        _integrate_image_matched_oracle(
            scattered,
            merged_qxy,
            merged_hkl,
            qx_axis,
            qy_axis,
            integration_radius_px=(
                float(config["oracle_integration_radius_fraction"])
                * disk_radius_px
            ),
            require_full_disk=bool(config["oracle_require_full_disk"]),
        )
    )
    relative = merged_intensity / merged_intensity.max()
    keep = relative >= float(config["physical_oracle_min_relative_intensity"])
    rejected_low_intensity = int(np.count_nonzero(~keep))
    merged_qxy = merged_qxy[keep]
    merged_intensity = merged_intensity[keep]
    merged_hkl = merged_hkl[keep]
    relative = relative[keep]
    order = np.lexsort(
        (
            merged_qxy[:, 1],
            merged_qxy[:, 0],
            np.linalg.norm(merged_qxy, axis=1),
        )
    )

    return RenderedPattern(
        expectation=expectation.astype(np.float32),
        direct_expectation=direct_probability.astype(np.float32),
        scattered_expectation=scattered_probability.astype(np.float32),
        vacuum_probe=vacuum_probe.astype(np.float32),
        qx_axis_Ainv=qx_axis.astype(np.float32),
        qy_axis_Ainv=qy_axis.astype(np.float32),
        beam_center_px=(float(center_y), float(center_x)),
        disk_radius_px=float(disk_radius_px),
        oracle_qx_Ainv=merged_qxy[order, 0].astype(np.float32),
        oracle_qy_Ainv=merged_qxy[order, 1].astype(np.float32),
        oracle_intensity_raw=merged_intensity[order].astype(np.float32),
        oracle_intensity_normalized=relative[order].astype(np.float32),
        oracle_hkl=merged_hkl[order],
        oracle_candidate_reflection_count=candidate_count,
        oracle_merged_disk_count=merged_count,
        oracle_rejected_edge_count=rejected_edge,
        oracle_rejected_low_intensity_count=rejected_low_intensity,
    )
