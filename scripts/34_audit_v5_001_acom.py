#!/usr/bin/env python3
"""Audit the exact-[001] ACOM failure without overwriting V5 results.

This script exposes the complete ACOM correlation surface and runs controlled
peak-list variants.  Large correlation arrays stay under the server data root;
the compact JSON/JSONL outputs are suitable for syncing into the repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

# Keep shared-server CPU use bounded even when the caller forgot a thread limit.
for variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(variable, "16")

import h5py
import numpy as np
import py4DSTEM
from pymatgen.core import Structure
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from or4d_common import (  # noqa: E402
    ACOM_MIRROR_SAMPLE_ROTATION,
    FRIEDEL_SAMPLE_ROTATION,
    cif_path,
    friedel_aware_misorientation_deg,
    load_config,
    nearest_rotation,
    proper_point_group_rotations,
    read_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/mnt/data/xietianhong/or4d-clean-v5"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Large server-only audit directory (default: DATA_ROOT/results/v5_001_audit).",
    )
    parser.add_argument(
        "--compact-output-dir",
        type=Path,
        default=ROOT / "reports" / "v5" / "study_001" / "acom_audit",
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        default=["exact_001", "transition_001"],
    )
    parser.add_argument(
        "--limit-per-group",
        type=int,
        default=0,
        help="0 runs every selected sample; positive values provide a bounded smoke run.",
    )
    parser.add_argument("--top-raw", type=int, default=100)
    parser.add_argument(
        "--save-surface-groups",
        nargs="*",
        default=["exact_001"],
    )
    parser.add_argument("--cuda", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decode_strings(values: np.ndarray) -> list[str]:
    return [value.decode() if isinstance(value, bytes) else str(value) for value in values]


def in_plane_rotation_matrix(phi: float) -> np.ndarray:
    c = float(np.cos(phi))
    s = float(np.sin(phi))
    return np.asarray([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])


def enumerate_scored_plan_matrices(crystal, inversion_symmetry: bool) -> np.ndarray:
    """Return matrices in the same flattened order as the saved correlations."""
    base = np.asarray(crystal.orientation_rotation_matrices, dtype=float)
    gamma = np.asarray(crystal.orientation_gamma, dtype=float)
    normal_rotations = np.stack(
        [in_plane_rotation_matrix(phi) for phi in gamma], axis=0
    )
    normal = (base[:, None] @ normal_rotations[None]).reshape(-1, 3, 3)
    if not inversion_symmetry:
        return normal

    # match_single_pattern adds pi to the inverse-branch angle before applying
    # the right-acting sample-x 180-degree rotation.
    inverse_rotations = np.stack(
        [in_plane_rotation_matrix(phi + np.pi) for phi in gamma], axis=0
    )
    inverse = (
        base[:, None]
        @ inverse_rotations[None]
        @ ACOM_MIRROR_SAMPLE_ROTATION
    ).reshape(-1, 3, 3)
    return np.concatenate([normal, inverse], axis=0)


def make_point_list(sample_id: str, qx: np.ndarray, qy: np.ndarray, intensity: np.ndarray):
    dtype = np.dtype([("qx", "f4"), ("qy", "f4"), ("intensity", "f4")])
    data = np.empty(len(qx), dtype=dtype)
    data["qx"] = np.asarray(qx, dtype=np.float32)
    data["qy"] = np.asarray(qy, dtype=np.float32)
    data["intensity"] = np.asarray(intensity, dtype=np.float32)
    return py4DSTEM.PointList(data=data, name=sample_id)


def normalize_peaks(qx: np.ndarray, qy: np.ndarray, intensity: np.ndarray, kmax: float, central: float):
    qx = np.asarray(qx, dtype=float)
    qy = np.asarray(qy, dtype=float)
    intensity = np.asarray(intensity, dtype=float)
    radius = np.hypot(qx, qy)
    keep = (
        np.isfinite(qx)
        & np.isfinite(qy)
        & np.isfinite(intensity)
        & (intensity > 0.0)
        & (radius >= central)
        & (radius <= kmax + 1e-8)
    )
    qx, qy, intensity = qx[keep], qy[keep], intensity[keep]
    if not len(qx):
        raise RuntimeError("peak variant contains no positive non-central peaks")
    intensity = intensity / float(np.max(intensity))
    order = np.lexsort((qy, qx, np.hypot(qx, qy)))
    return qx[order], qy[order], intensity[order]


def simulate_acom_peaks(crystal, matrix: np.ndarray, config: dict[str, Any]):
    clean = config["clean"]
    common = config["common"]
    pattern = crystal.generate_diffraction_pattern(
        orientation_matrix=np.asarray(matrix, dtype=float),
        sigma_excitation_error=float(clean["sigma_excitation_error_Ainv"]),
        tol_excitation_error_mult=float(clean["tol_excitation_error_mult"]),
        tol_intensity=float(clean["tol_intensity"]),
        k_max=float(common["k_max_Ainv"]),
    )
    data = pattern.data
    return normalize_peaks(
        data["qx"],
        data["qy"],
        data["intensity"],
        float(common["k_max_Ainv"]),
        float(common["central_beam_exclusion_Ainv"]),
    )


def polar_image(crystal, qx: np.ndarray, qy: np.ndarray, intensity: np.ndarray) -> np.ndarray:
    """Faithful copy of py4DSTEM 0.14.18's single-pattern polar transform."""
    qr = np.hypot(qx, qy)
    qphi = np.arctan2(qy, qx)
    gamma = np.asarray(crystal.orientation_gamma, dtype=float)
    radii = np.asarray(crystal.orientation_shell_radii, dtype=float)
    kernel = float(crystal.orientation_kernel_size)
    power = float(crystal.orientation_power_intensity_experiment)
    result = np.zeros((len(radii), len(gamma)), dtype=float)
    for index, radius in enumerate(radii):
        dqr = np.abs(qr - radius)
        selected = dqr < kernel
        if not np.any(selected):
            continue
        angular = (
            np.mod(gamma[None, :] - qphi[selected, None] + np.pi, 2.0 * np.pi)
            - np.pi
        ) * radius
        result[index] = np.sum(
            np.maximum(intensity[selected, None], 0.0) ** power
            * np.exp(
                (dqr[selected, None] ** 2 + angular**2) / (-2.0 * kernel**2)
            ),
            axis=0,
        )
    return result


def full_correlations(crystal, qx: np.ndarray, qy: np.ndarray, intensity: np.ndarray):
    polar = polar_image(crystal, qx, qy, intensity)
    if crystal.CUDA:
        import cupy as cp

        polar_fft = cp.fft.fft(cp.asarray(polar))
        normal = cp.maximum(
            cp.sum(
                cp.real(cp.fft.ifft(crystal.orientation_ref * polar_fft[None, :, :])),
                axis=1,
            ),
            0,
        )
        inverse = cp.maximum(
            cp.sum(
                cp.real(
                    cp.fft.ifft(
                        crystal.orientation_ref * cp.conj(polar_fft)[None, :, :]
                    )
                ),
                axis=1,
            ),
            0,
        )
        return polar.astype(np.float32), cp.asnumpy(normal).astype(np.float32), cp.asnumpy(inverse).astype(np.float32)

    polar_fft = np.fft.fft(polar)
    normal = np.maximum(
        np.sum(
            np.real(np.fft.ifft(crystal.orientation_ref * polar_fft[None, :, :])),
            axis=1,
        ),
        0,
    )
    inverse = np.maximum(
        np.sum(
            np.real(
                np.fft.ifft(crystal.orientation_ref * np.conj(polar_fft)[None, :, :])
            ),
            axis=1,
        ),
        0,
    )
    return polar.astype(np.float32), normal.astype(np.float32), inverse.astype(np.float32)


def rotation_chord_radius(angle_deg: float) -> float:
    return float(2.0 * np.sqrt(2.0) * np.sin(np.deg2rad(angle_deg) / 2.0))


def equivalent_gt_matrices(matrix_gt: np.ndarray, symmetries: list[np.ndarray]):
    for symmetry in symmetries:
        for friedel in (np.eye(3), FRIEDEL_SAMPLE_ROTATION):
            yield nearest_rotation(symmetry @ matrix_gt @ friedel)


def correct_seed_indices(
    matrix_gt: np.ndarray,
    symmetries: list[np.ndarray],
    seed_tree: cKDTree,
    tolerance_deg: float,
) -> np.ndarray:
    indices: set[int] = set()
    radius = rotation_chord_radius(tolerance_deg + 1e-8)
    for equivalent in equivalent_gt_matrices(matrix_gt, symmetries):
        indices.update(seed_tree.query_ball_point(equivalent.reshape(-1), r=radius))
    return np.asarray(sorted(indices), dtype=np.int64)


def nearest_seed_index(
    matrix_gt: np.ndarray,
    symmetries: list[np.ndarray],
    seed_tree: cKDTree,
) -> tuple[int, float]:
    best_index = -1
    best_chord = np.inf
    for equivalent in equivalent_gt_matrices(matrix_gt, symmetries):
        chord, index = seed_tree.query(equivalent.reshape(-1), k=1)
        if float(chord) < best_chord:
            best_chord = float(chord)
            best_index = int(index)
    angle = np.degrees(
        2.0
        * np.arcsin(np.clip(best_chord / (2.0 * np.sqrt(2.0)), 0.0, 1.0))
    )
    return best_index, float(angle)


def top_indices(values: np.ndarray, count: int) -> np.ndarray:
    count = min(int(count), len(values))
    if count == len(values):
        return np.argsort(values)[::-1]
    selected = np.argpartition(values, -count)[-count:]
    return selected[np.argsort(values[selected])[::-1]]


def collision_diagnostics(
    crystal,
    matrix: np.ndarray,
    config: dict[str, Any],
    exact_candidate_count: int,
    exact_merged_count: int,
) -> dict[str, Any]:
    g_crystal = np.asarray(crystal.g_vec_all, dtype=float).T
    hkl = np.rint(np.asarray(crystal.hkl, dtype=float).T).astype(np.int32)
    structure_factor = np.asarray(crystal.struct_factors)
    g_sample = (np.asarray(matrix, dtype=float).T @ g_crystal.T).T
    qxy = g_sample[:, :2]
    projected_radius = np.linalg.norm(qxy, axis=1)
    wavelength = float(crystal.wavelength)
    disk_radius = (
        float(config["clean_image"]["convergence_semiangle_mrad"])
        * 1e-3
        / wavelength
    )
    keep = (
        (np.linalg.norm(g_sample, axis=1) >= 1e-10)
        & np.isfinite(structure_factor)
        & (np.abs(structure_factor) > 0.0)
        & (projected_radius > disk_radius)
        & (projected_radius <= float(config["common"]["k_max_Ainv"]) + 1e-12)
    )
    qxy = qxy[keep]
    selected_hkl = hkl[keep]
    q_pixel = (
        2.0
        * float(config["clean_image"]["q_max_Ainv"])
        / max(int(value) for value in config["clean_image"]["gpts"])
    )
    tolerance = float(config["clean_image"]["oracle_merge_distance_px"]) * q_pixel

    parent = np.arange(len(qxy), dtype=np.int64)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    if len(qxy):
        for a, b in cKDTree(qxy).query_pairs(tolerance):
            union(int(a), int(b))
    components: dict[int, list[int]] = defaultdict(list)
    for index in range(len(qxy)):
        components[find(index)].append(index)
    ordered = sorted(components.values(), key=lambda group: (-len(group), group[0]))
    examples = []
    for group in ordered[:10]:
        examples.append(
            {
                "multiplicity": len(group),
                "qxy_centroid_Ainv": np.mean(qxy[group], axis=0).tolist(),
                "max_pair_distance_Ainv": float(
                    np.max(
                        np.linalg.norm(
                            qxy[group, None, :] - qxy[None, group, :], axis=2
                        )
                    )
                ),
                "hkl": selected_hkl[group].tolist(),
            }
        )
    multiplicities = np.asarray([len(group) for group in ordered], dtype=int)
    return {
        "renderer_candidate_reflection_count": int(exact_candidate_count),
        "renderer_merged_disk_count": int(exact_merged_count),
        "renderer_merge_fraction": float(
            1.0 - exact_merged_count / max(exact_candidate_count, 1)
        ),
        "geometric_candidate_count": int(len(qxy)),
        "geometric_component_count": int(len(ordered)),
        "geometric_collision_fraction": float(1.0 - len(ordered) / max(len(qxy), 1)),
        "geometric_mean_multiplicity": float(multiplicities.mean()) if len(multiplicities) else 0.0,
        "geometric_max_multiplicity": int(multiplicities.max()) if len(multiplicities) else 0,
        "merge_tolerance_Ainv": tolerance,
        "largest_collision_groups": examples,
    }


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for (group, variant), selected in sorted(
        defaultdict(list, {
            key: [row for row in rows if (row["study_group"], row["variant"]) == key]
            for key in {(row["study_group"], row["variant"]) for row in rows}
        }).items()
    ):
        native = np.asarray([row["native_error_deg"] for row in selected], dtype=float)
        raw1 = np.asarray([row["raw_top1_error_deg"] for row in selected], dtype=float)
        raw5 = np.asarray([row["raw_top5_min_error_deg"] for row in selected], dtype=float)
        ranks = np.asarray([row["best_correct_seed_rank"] for row in selected], dtype=float)
        ratios = np.asarray([row["best_correct_score_ratio"] for row in selected], dtype=float)
        result[f"{group}/{variant}"] = {
            "sample_count": len(selected),
            "native_acc_at_2deg": float(np.mean(native <= 2.0)),
            "native_median_error_deg": float(np.median(native)),
            "raw_cell_acc_at_2deg_top1": float(np.mean(raw1 <= 2.0)),
            "raw_cell_acc_at_2deg_top5": float(np.mean(raw5 <= 2.0)),
            "best_correct_seed_rank_median": float(np.median(ranks)),
            "best_correct_seed_rank_p95": float(np.percentile(ranks, 95)),
            "best_correct_seed_in_top1": float(np.mean(ranks <= 1)),
            "best_correct_seed_in_top5": float(np.mean(ranks <= 5)),
            "best_correct_seed_in_top100": float(np.mean(ranks <= 100)),
            "best_correct_score_ratio_median": float(np.median(ratios)),
        }
    return result


def main() -> None:
    args = parse_args()
    started = time.time()
    data_root = args.data_root.resolve()
    output_dir = (args.output_dir or data_root / "results" / "v5_001_audit").resolve()
    compact_dir = args.compact_output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    compact_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(ROOT / "config" / "benchmark_v5.yaml")
    common = config["common"]
    clean = config["clean"]
    acom = config["acom"]
    input_paths = {
        "expectation": data_root / "datasets" / "clean_v5_001_first_born_expectation_512.h5",
        "oracle": data_root / "intermediates" / "clean_v5_001_first_born_oracle_512.h5",
        "trace": data_root / "intermediates" / "clean_v5_001_first_born_trace_512.h5",
        "manifest": data_root / "manifests" / "clean_v5_001_orientations.jsonl",
    }
    for path in input_paths.values():
        if not path.exists():
            raise FileNotFoundError(path)

    manifest = {row["sample_id"]: row for row in read_jsonl(input_paths["manifest"])}
    with h5py.File(input_paths["expectation"], "r") as handle:
        sample_ids = decode_strings(handle["sample_id"][:])
        matrices = np.asarray(handle["orientation/canonical_matrix_sample_to_crystal"][:], dtype=float)
    with h5py.File(input_paths["oracle"], "r") as handle:
        oracle_ids = decode_strings(handle["sample_id"][:])
        offsets = np.asarray(handle["peaks/offsets"][:], dtype=np.int64)
        all_qx = np.asarray(handle["peaks/qx"][:], dtype=float)
        all_qy = np.asarray(handle["peaks/qy"][:], dtype=float)
        all_intensity = np.asarray(handle["peaks/intensity"][:], dtype=float)
    if sample_ids != oracle_ids:
        raise RuntimeError("expectation and oracle sample order differs")
    with h5py.File(input_paths["trace"], "r") as handle:
        trace_ids = decode_strings(handle["sample_id"][:])
        candidate_counts = np.asarray(handle["diagnostics/candidate_reflection_count"][:])
        merged_counts = np.asarray(handle["diagnostics/merged_disk_count"][:])
    if sample_ids != trace_ids:
        raise RuntimeError("expectation and trace sample order differs")

    selected_indices: list[int] = []
    group_counts: dict[str, int] = defaultdict(int)
    selected_group_set = set(args.groups)
    for index, sample_id in enumerate(sample_ids):
        group = str(manifest[sample_id]["study_group"])
        if group not in selected_group_set:
            continue
        if args.limit_per_group and group_counts[group] >= args.limit_per_group:
            continue
        selected_indices.append(index)
        group_counts[group] += 1
    if not selected_indices:
        raise RuntimeError("no samples matched --groups")

    structure = Structure.from_file(cif_path(config))
    symmetries = proper_point_group_rotations(structure)
    crystal = py4DSTEM.process.diffraction.Crystal.from_pymatgen_structure(
        structure=structure,
        conventional_standard_structure=False,
    )
    voltage = float(common["accelerating_voltage_V"])
    kmax = float(common["k_max_Ainv"])
    crystal.setup_diffraction(accelerating_voltage=voltage)
    crystal.calculate_structure_factors(
        k_max=max(kmax, float(config["clean_image"]["q_max_Ainv"]) + 0.15),
        tol_structure_factor=float(clean["tol_structure_factor"]),
    )
    plan_started = time.perf_counter()
    crystal.orientation_plan(
        zone_axis_range=acom["zone_axis_range"],
        angle_step_zone_axis=float(acom["angle_step_zone_axis_deg"]),
        angle_step_in_plane=float(acom["angle_step_in_plane_deg"]),
        accel_voltage=voltage,
        corr_kernel_size=float(acom["corr_kernel_size_Ainv"]),
        sigma_excitation_error=float(acom["sigma_excitation_error_Ainv"]),
        power_radial=float(acom["power_radial"]),
        power_intensity=float(acom["power_intensity_simulated"]),
        power_intensity_experiment=float(acom["power_intensity_experiment"]),
        tol_distance=float(acom["tol_distance_Ainv"]),
        CUDA=bool(args.cuda),
        progress_bar=False,
    )
    plan_seconds = time.perf_counter() - plan_started
    inversion = bool(acom["inversion_symmetry"])
    seed_matrices = enumerate_scored_plan_matrices(crystal, inversion)
    seed_tree = cKDTree(seed_matrices.reshape(-1, 9))
    nzone = int(crystal.orientation_num_zones)
    nphi = int(crystal.orientation_in_plane_steps)
    if len(seed_matrices) != nzone * nphi * (2 if inversion else 1):
        raise RuntimeError("search-seed matrix count does not match correlation shape")

    surface_path = output_dir / "correlation_surfaces.h5"
    rows: list[dict[str, Any]] = []
    sample_diagnostics: list[dict[str, Any]] = []
    with h5py.File(surface_path, "w") as surfaces:
        surfaces.attrs["schema"] = "or4d-v5-001-acom-correlation-audit-v1"
        surfaces.attrs["normal_shape"] = (nzone, nphi)
        surfaces.attrs["flatten_order"] = "normal[zone,phi] then inverse[zone,phi]"
        surfaces.attrs["kmax_Ainv"] = kmax
        for sequence, index in enumerate(selected_indices, start=1):
            sample_id = sample_ids[index]
            metadata = manifest[sample_id]
            group = str(metadata["study_group"])
            matrix_gt = matrices[index]
            start, end = int(offsets[index]), int(offsets[index + 1])
            physical = normalize_peaks(
                all_qx[start:end],
                all_qy[start:end],
                all_intensity[start:end],
                kmax,
                float(common["central_beam_exclusion_Ainv"]),
            )
            nearest_index, nearest_distance = nearest_seed_index(
                matrix_gt, symmetries, seed_tree
            )
            variants = {
                "physical_oracle": physical,
                "physical_uniform_intensity": (
                    physical[0],
                    physical[1],
                    np.ones_like(physical[2]),
                ),
                "acom_exact_gt": simulate_acom_peaks(crystal, matrix_gt, config),
                "acom_discrete_seed": simulate_acom_peaks(
                    crystal, seed_matrices[nearest_index], config
                ),
            }
            correct_indices = correct_seed_indices(
                matrix_gt, symmetries, seed_tree, tolerance_deg=2.0
            )
            if not len(correct_indices):
                raise RuntimeError(f"{sample_id} has no discrete ACOM seed within 2 degrees")

            collision = collision_diagnostics(
                crystal,
                matrix_gt,
                config,
                int(candidate_counts[index]),
                int(merged_counts[index]),
            )
            sample_diagnostics.append(
                {
                    "sample_id": sample_id,
                    "study_group": group,
                    "tilt_deg": float(metadata["tilt_deg"]),
                    "tilt_azimuth_deg": float(metadata["tilt_azimuth_deg"]),
                    "in_plane_rotation_deg": float(metadata["in_plane_rotation_deg"]),
                    "nearest_discrete_seed_index": nearest_index,
                    "nearest_discrete_seed_distance_deg": nearest_distance,
                    "collision": collision,
                }
            )

            for variant_name, (qx, qy, intensity) in variants.items():
                variant_started = time.perf_counter()
                polar, normal, inverse_corr = full_correlations(
                    crystal, qx, qy, intensity
                )
                scores = np.concatenate([normal.reshape(-1), inverse_corr.reshape(-1)])
                raw_top = top_indices(scores, args.top_raw)
                raw_errors = np.asarray(
                    [
                        friedel_aware_misorientation_deg(
                            seed_matrices[candidate], matrix_gt, symmetries
                        )
                        for candidate in raw_top
                    ],
                    dtype=float,
                )
                correct_scores = scores[correct_indices]
                best_local = int(np.argmax(correct_scores))
                best_correct_index = int(correct_indices[best_local])
                best_correct_score = float(correct_scores[best_local])
                best_correct_rank = 1 + int(np.count_nonzero(scores > best_correct_score))
                top_score = float(scores[raw_top[0]])

                native = crystal.match_single_pattern(
                    bragg_peaks=make_point_list(sample_id, qx, qy, intensity),
                    num_matches_return=1,
                    min_number_peaks=int(acom["min_number_peaks"]),
                    inversion_symmetry=inversion,
                    plot_polar=False,
                    plot_corr=False,
                    verbose=False,
                )
                native_matrix = nearest_rotation(np.asarray(native.matrix[0], dtype=float))
                native_error = friedel_aware_misorientation_deg(
                    native_matrix, matrix_gt, symmetries
                )
                row = {
                    "sample_id": sample_id,
                    "study_group": group,
                    "tilt_deg": float(metadata["tilt_deg"]),
                    "tilt_azimuth_deg": float(metadata["tilt_azimuth_deg"]),
                    "in_plane_rotation_deg": float(metadata["in_plane_rotation_deg"]),
                    "variant": variant_name,
                    "num_input_peaks": int(len(qx)),
                    "native_error_deg": float(native_error),
                    "native_correlation": float(native.corr[0]),
                    "native_zone_index": int(native.inds[0, 0]),
                    "native_phi_index": int(native.inds[0, 1]),
                    "native_mirror": bool(native.mirror[0]),
                    "native_matrix_sample_to_crystal": native_matrix.tolist(),
                    "raw_top1_seed_index": int(raw_top[0]),
                    "raw_top1_error_deg": float(raw_errors[0]),
                    "raw_top5_min_error_deg": float(np.min(raw_errors[:5])),
                    "raw_top100_min_error_deg": float(np.min(raw_errors)),
                    "raw_top_seed_indices": raw_top.tolist(),
                    "raw_top_scores": scores[raw_top].astype(float).tolist(),
                    "raw_top_errors_deg": raw_errors.tolist(),
                    "correct_seed_count_at_2deg": int(len(correct_indices)),
                    "best_correct_seed_index": best_correct_index,
                    "best_correct_seed_error_deg": float(
                        friedel_aware_misorientation_deg(
                            seed_matrices[best_correct_index], matrix_gt, symmetries
                        )
                    ),
                    "best_correct_seed_score": best_correct_score,
                    "best_correct_seed_rank": best_correct_rank,
                    "best_correct_score_ratio": float(
                        best_correct_score / top_score if top_score > 0 else 0.0
                    ),
                    "nearest_discrete_seed_index": nearest_index,
                    "nearest_discrete_seed_distance_deg": nearest_distance,
                    "seconds": float(time.perf_counter() - variant_started),
                    "surface_saved": group in set(args.save_surface_groups),
                }
                rows.append(row)

                if group in set(args.save_surface_groups):
                    sample_group = surfaces.require_group(sample_id)
                    variant_group = sample_group.create_group(variant_name)
                    variant_group.create_dataset("polar", data=polar, compression="lzf")
                    variant_group.create_dataset("normal", data=normal, compression="lzf")
                    variant_group.create_dataset("inverse", data=inverse_corr, compression="lzf")
                    variant_group.attrs["best_correct_seed_rank"] = best_correct_rank
                    variant_group.attrs["best_correct_seed_index"] = best_correct_index
                    variant_group.attrs["raw_top1_seed_index"] = int(raw_top[0])

            print(
                f"[{sequence}/{len(selected_indices)}] {sample_id} "
                f"group={group} peaks={len(physical[0])} "
                f"merged={candidate_counts[index]}->{merged_counts[index]}",
                flush=True,
            )

    rows_path = compact_dir / "acom_001_correlation_audit.jsonl"
    with rows_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(jsonable(row), ensure_ascii=False) + "\n")
    diagnostics_path = compact_dir / "acom_001_projection_diagnostics.jsonl"
    with diagnostics_path.open("w", encoding="utf-8") as handle:
        for row in sample_diagnostics:
            handle.write(json.dumps(jsonable(row), ensure_ascii=False) + "\n")

    summary = {
        "schema": "or4d-v5-001-acom-audit-summary-v1",
        "purpose": "Root-cause audit of exact-[001] ACOM failures; no formal V5 result was overwritten.",
        "created_unix": time.time(),
        "duration_seconds": time.time() - started,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "pid": os.getpid(),
        "command": [sys.executable, *sys.argv],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cpu_thread_limits": {
            key: os.environ.get(key)
            for key in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "py4DSTEM": py4DSTEM.__version__,
        },
        "inputs": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in input_paths.items()
        },
        "selection": {
            "groups": args.groups,
            "limit_per_group": args.limit_per_group,
            "sample_counts": dict(group_counts),
        },
        "acom": {
            "kmax_Ainv": kmax,
            "angle_step_zone_axis_deg": float(acom["angle_step_zone_axis_deg"]),
            "angle_step_in_plane_deg": float(acom["angle_step_in_plane_deg"]),
            "correlation_kernel_Ainv": float(acom["corr_kernel_size_Ainv"]),
            "power_intensity_simulated": float(acom["power_intensity_simulated"]),
            "power_intensity_experiment": float(acom["power_intensity_experiment"]),
            "inversion_symmetry": inversion,
            "num_zone_axes": nzone,
            "num_in_plane_steps": nphi,
            "num_scored_cells": int(len(seed_matrices)),
            "plan_seconds": plan_seconds,
        },
        "variants": {
            "physical_oracle": "V5 image-matched physical oracle positions and intensities.",
            "physical_uniform_intensity": "Same physical positions with every peak intensity set to one.",
            "acom_exact_gt": "py4DSTEM kinematical peaks generated at the exact GT matrix.",
            "acom_discrete_seed": "py4DSTEM kinematical peaks generated at the nearest searched ACOM seed.",
        },
        "metrics": {
            "orientation_equivalence": "proper crystal point-group symmetry plus detector-plane Friedel equivalence",
            "correct_seed_tolerance_deg": 2.0,
            "rank": "global rank over every zone, in-plane bin, and normal/inverse correlation cell before zone-axis Top-K suppression",
        },
        "aggregate": summarize(rows),
        "artifacts": {
            "compact_rows": str(rows_path),
            "projection_diagnostics": str(diagnostics_path),
            "large_correlation_surfaces": str(surface_path),
        },
    }
    summary_path = compact_dir / "acom_001_audit_summary.json"
    summary_path.write_text(
        json.dumps(jsonable(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"summary": str(summary_path), "surfaces": str(surface_path)}, indent=2))


if __name__ == "__main__":
    main()
