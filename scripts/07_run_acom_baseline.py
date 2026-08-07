#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import py4DSTEM
from pymatgen.core import Structure
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
V3_REPORT_DIR = Path(
    os.environ.get("OR4D_REPORT_V3_DIR", ROOT / "reports" / "v3")
).resolve()
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
    read_peak_h5,
    symmetry_aware_misorientation_deg,
    write_jsonl,
)
from topk_evaluation import summarize_topk_errors  # noqa: E402


def in_plane_rotation_matrix(phi: float) -> np.ndarray:
    """Return the exact m3z convention used by py4DSTEM ACOM."""
    c = float(np.cos(phi))
    s = float(np.sin(phi))
    return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])


def enumerate_normal_plan_matrices(crystal) -> np.ndarray:
    base = np.asarray(crystal.orientation_rotation_matrices, dtype=float)
    gamma = np.asarray(crystal.orientation_gamma, dtype=float)
    rotations = np.stack([in_plane_rotation_matrix(phi) for phi in gamma], axis=0)
    plan = base[:, None, :, :] @ rotations[None, :, :, :]
    return plan.reshape(-1, 3, 3)


def enumerate_discrete_search_seed_matrices(
    crystal,
    inversion_symmetry: bool,
) -> np.ndarray:
    normal = enumerate_normal_plan_matrices(crystal)
    if not inversion_symmetry:
        return normal
    mirror = normal @ ACOM_MIRROR_SAMPLE_ROTATION
    return np.concatenate([normal, mirror], axis=0)


def min_discrete_seed_distance_deg(
    matrix_gt: np.ndarray,
    search_seed_tree: cKDTree,
    symmetries: list[np.ndarray],
) -> float:
    """Distance to the nearest discrete seed in Clean observable space.

    This is not an error lower bound: ``match_single_pattern`` performs a
    parabolic sub-grid fit of the in-plane correlation peak.
    """
    best_chord = np.inf
    for symmetry in symmetries:
        crystal_equivalent = symmetry @ matrix_gt
        for sample_branch in (np.eye(3), FRIEDEL_SAMPLE_ROTATION):
            equivalent = crystal_equivalent @ sample_branch
            chord, _ = search_seed_tree.query(
                equivalent.reshape(-1),
                k=1,
            )
            best_chord = min(best_chord, float(chord))
    # For proper rotations, ||R1-R2||_F = 2*sqrt(2)*sin(theta/2).
    sine_half_angle = np.clip(
        best_chord / (2.0 * np.sqrt(2.0)),
        0.0,
        1.0,
    )
    return float(np.degrees(2.0 * np.arcsin(sine_half_angle)))


def min_zone_axis_node_distance_deg(
    matrix_gt: np.ndarray,
    crystal,
    symmetries: list[np.ndarray],
    inversion_symmetry: bool,
) -> float:
    """Distance from the true beam direction to the nearest searched zone axis."""
    plan_zone_axes = np.asarray(
        crystal.orientation_rotation_matrices,
        dtype=float,
    )[:, :, 2]
    if inversion_symmetry:
        plan_zone_axes = np.concatenate([plan_zone_axes, -plan_zone_axes], axis=0)

    best = np.inf
    beam_gt = np.asarray(matrix_gt, dtype=float)[:, 2]
    for symmetry in symmetries:
        equivalent_beam = symmetry @ beam_gt
        cosines = np.clip(plan_zone_axes @ equivalent_beam, -1.0, 1.0)
        best = min(best, float(np.degrees(np.arccos(cosines)).min()))
    return float(best)


def make_point_list(sample: dict) -> py4DSTEM.PointList:
    dtype = np.dtype([("qx", "f4"), ("qy", "f4"), ("intensity", "f4")])
    data = np.empty(len(sample["qx"]), dtype=dtype)
    data["qx"] = sample["qx"]
    data["qy"] = sample["qy"]
    data["intensity"] = sample["intensity"]
    return py4DSTEM.PointList(data=data, name=sample["sample_id"])


def unique_records_by_id(records: list[dict], *, source: str) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for record in records:
        record_id = record.get("sample_id", record.get("orientation_id"))
        if record_id is None:
            raise ValueError(
                f"Record in {source} has neither sample_id nor orientation_id"
            )
        sample_id = str(record_id)
        if sample_id in result:
            raise ValueError(f"Duplicate sample_id in {source}: {sample_id}")
        normalized = dict(record)
        normalized["sample_id"] = sample_id
        result[sample_id] = normalized
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def runtime_summary(seconds: list[float]) -> dict:
    values = np.asarray(seconds, dtype=float)
    return {
        "total_seconds": float(values.sum()),
        "mean_seconds": float(values.mean()),
        "p50_seconds": float(np.percentile(values, 50)),
        "p90_seconds": float(np.percentile(values, 90)),
        "p99_seconds": float(np.percentile(values, 99)),
        "throughput_samples_per_second": float(len(values) / values.sum()),
    }


def match_topk_distinct_zone_axes(
    crystal,
    point_list,
    *,
    n_best: int,
    min_number_peaks: int,
    inversion_symmetry: bool,
    min_zone_separation_deg: float,
) -> dict[str, np.ndarray]:
    """Return ranked ACOM hypotheses from the unchanged input peak list.

    ``Crystal.match_single_pattern(num_matches_return > 1)`` removes peaks
    explained by each preceding match.  Those outputs are a multi-orientation
    decomposition, not classification Top-K candidates.  For benchmark Top-K,
    run the native matcher repeatedly on the same peaks and mask only the
    previously selected zone-axis neighbourhood in the orientation plan.

    py4DSTEM reduces every zone-axis row to its best in-plane angle and
    normal/Friedel branch, so these are explicitly *zone-axis-distinct*
    candidates.  Rank 1 is byte-for-byte the normal native single match.
    """
    if n_best < 1:
        raise ValueError("n_best must be positive")
    if min_zone_separation_deg <= 0:
        raise ValueError("min_zone_separation_deg must be positive")

    orientation_ref = crystal.orientation_ref
    orientation_vecs = np.asarray(crystal.orientation_vecs, dtype=float)
    excluded = np.zeros(orientation_vecs.shape[0], dtype=bool)
    restored_rows: list[tuple[np.ndarray, object]] = []
    matrices: list[np.ndarray] = []
    correlations: list[float] = []
    indices: list[np.ndarray] = []
    mirrors: list[bool] = []
    angles: list[np.ndarray] = []
    cosine_limit = float(np.cos(np.deg2rad(min_zone_separation_deg)))

    try:
        for _ in range(n_best):
            orientation = crystal.match_single_pattern(
                bragg_peaks=point_list,
                num_matches_return=1,
                min_number_peaks=min_number_peaks,
                inversion_symmetry=inversion_symmetry,
                plot_polar=False,
                plot_corr=False,
                verbose=False,
            )
            correlation = float(np.asarray(orientation.corr)[0])
            if not np.isfinite(correlation) or correlation <= 0:
                raise RuntimeError("ACOM exhausted positive Top-K hypotheses")
            zone_index = int(np.asarray(orientation.inds)[0, 0])
            matrices.append(np.asarray(orientation.matrix, dtype=float)[0])
            correlations.append(correlation)
            indices.append(np.asarray(orientation.inds, dtype=np.int32)[0])
            mirrors.append(bool(np.asarray(orientation.mirror)[0]))
            angles.append(np.asarray(orientation.angles, dtype=float)[0])

            cosine = orientation_vecs @ orientation_vecs[zone_index]
            new_mask = (cosine > cosine_limit) & ~excluded
            new_indices = np.flatnonzero(new_mask)
            if new_indices.size == 0:
                raise RuntimeError(
                    f"ACOM Top-K failed to exclude zone index {zone_index}"
                )
            saved = orientation_ref[new_indices, :, :].copy()
            restored_rows.append((new_indices, saved))
            orientation_ref[new_indices, :, :] = 0
            excluded[new_indices] = True
    finally:
        for row_indices, saved in restored_rows:
            orientation_ref[row_indices, :, :] = saved

    correlations_array = np.asarray(correlations, dtype=float)
    if np.any(np.diff(correlations_array) > 1e-8):
        raise RuntimeError(
            "ACOM Top-K correlations are not monotonically non-increasing"
        )
    return {
        "matrix": np.stack(matrices, axis=0),
        "correlation": correlations_array,
        "indices": np.stack(indices, axis=0),
        "mirror": np.asarray(mirrors, dtype=bool),
        "angles": np.stack(angles, axis=0),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--angle-step-deg",
        type=float,
        help="Override both zone-axis and in-plane ACOM angle steps.",
    )
    parser.add_argument(
        "--power-intensity-experiment",
        type=float,
        help="Override the ACOM exponent applied to measured peak intensity.",
    )
    parser.add_argument(
        "--power-intensity-simulated",
        type=float,
        help="Override the ACOM exponent applied to simulated template intensity.",
    )
    parser.add_argument(
        "--output-tag",
        "--report-tag",
        default="",
        help="Suffix output filenames so sweep runs do not overwrite each other.",
    )
    parser.add_argument(
        "--peak-file",
        type=Path,
        default=ROOT / "public" / "clean_peaks.h5",
        help="Peak HDF5 input; all paths use the same frozen ACOM configuration.",
    )
    parser.add_argument(
        "--prediction-file",
        type=Path,
        help="Exact prediction JSONL output path.",
    )
    parser.add_argument(
        "--ground-truth-file",
        type=Path,
        default=ROOT / "private" / "clean_ground_truth.jsonl",
        help=(
            "Ground-truth JSONL. Records may use sample_id (V3) or "
            "orientation_id (V5)."
        ),
    )
    parser.add_argument(
        "--ground-truth-id-prefix",
        default="",
        help=(
            "Explicit prefix ensured on ground-truth IDs before matching peak IDs "
            "(V5 image/peak files use 'clean_' before orientation_id)."
        ),
    )
    parser.add_argument(
        "--orientation-file",
        type=Path,
        help=(
            "Orientation-manifest file recorded in the provenance hash. "
            "Defaults to private/orientations.jsonl for V3 or the ground-truth "
            "file for external datasets."
        ),
    )
    parser.add_argument(
        "--details-file",
        type=Path,
        help="Exact ACOM per-sample details JSON output path.",
    )
    parser.add_argument(
        "--candidates-file",
        type=Path,
        help=(
            "HDF5 output containing every returned candidate. Full Top-5 "
            "runs should place this large artifact under /mnt/data."
        ),
    )
    parser.add_argument(
        "--audit-file",
        type=Path,
        help="Exact orientation-plan audit JSON output path.",
    )
    parser.add_argument(
        "--cuda",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override config acom.cuda for orientation-plan FFT operations.",
    )
    parser.add_argument(
        "--allow-subset",
        action="store_true",
        help="Allow a strict sample-ID subset for legacy smoke runs.",
    )
    parser.add_argument(
        "--insufficient-peaks-policy",
        choices=("error", "skip"),
        default="error",
        help=(
            "Behavior when a pattern has fewer than acom.min_number_peaks. "
            "'skip' records a deterministic indexing failure and omits only "
            "that prediction."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config()
    acom = config["acom"]
    clean = config["clean"]
    common = config["common"]
    angle_step_zone_axis = float(
        args.angle_step_deg
        if args.angle_step_deg is not None
        else acom["angle_step_zone_axis_deg"]
    )
    angle_step_in_plane = float(
        args.angle_step_deg
        if args.angle_step_deg is not None
        else acom["angle_step_in_plane_deg"]
    )
    power_intensity_experiment = float(
        args.power_intensity_experiment
        if args.power_intensity_experiment is not None
        else acom["power_intensity_experiment"]
    )
    power_intensity_simulated = float(
        args.power_intensity_simulated
        if args.power_intensity_simulated is not None
        else acom["power_intensity_simulated"]
    )
    if angle_step_zone_axis <= 0.0 or angle_step_in_plane <= 0.0:
        raise ValueError("ACOM angle steps must be positive")
    if power_intensity_experiment < 0.0 or power_intensity_simulated < 0.0:
        raise ValueError("intensity exponents must be non-negative")
    num_matches_return = int(acom["num_matches_return"])
    if num_matches_return != 5:
        raise ValueError(
            "The V5 candidate contract requires acom.num_matches_return=5, "
            f"got {num_matches_return}"
        )
    topk_min_zone_separation_deg = float(
        acom.get("topk_min_zone_separation_deg", 0.01)
    )
    if topk_min_zone_separation_deg <= 0:
        raise ValueError("acom.topk_min_zone_separation_deg must be positive")
    if args.output_tag and not args.output_tag.replace("_", "").isalnum():
        raise ValueError(
            "output tag may contain only letters, numbers, and underscores"
        )
    output_suffix = f"_{args.output_tag}" if args.output_tag else ""

    peak_path = args.peak_file.resolve()
    gt_path = args.ground_truth_file.resolve()
    default_gt_path = (ROOT / "private" / "clean_ground_truth.jsonl").resolve()
    manifest_path = (
        args.orientation_file.resolve()
        if args.orientation_file
        else (
            (ROOT / "private" / "orientations.jsonl").resolve()
            if gt_path == default_gt_path
            else gt_path
        )
    )
    use_cuda = bool(acom["cuda"]) if args.cuda is None else bool(args.cuda)
    samples = read_peak_h5(peak_path)
    ground_truth_unprefixed = unique_records_by_id(
        read_jsonl(gt_path),
        source=str(gt_path),
    )
    ground_truth: dict[str, dict] = {}
    for ground_truth_id, record in ground_truth_unprefixed.items():
        sample_id = (
            ground_truth_id
            if (
                not args.ground_truth_id_prefix
                or ground_truth_id.startswith(args.ground_truth_id_prefix)
            )
            else f"{args.ground_truth_id_prefix}{ground_truth_id}"
        )
        normalized = dict(record)
        normalized["sample_id"] = sample_id
        normalized["ground_truth_source_id"] = ground_truth_id
        ground_truth[sample_id] = normalized
    sample_ids = [str(sample["sample_id"]) for sample in samples]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("public/clean_peaks.h5 contains duplicate sample IDs")
    sample_id_set = set(sample_ids)
    ground_truth_id_set = set(ground_truth)
    if args.allow_subset:
        if not sample_id_set <= ground_truth_id_set:
            raise ValueError("Peak input includes IDs absent from ground truth")
    elif sample_id_set != ground_truth_id_set:
        raise ValueError("Clean peak input and ground-truth sample IDs differ")
    min_number_peaks = int(acom["min_number_peaks"])
    insufficient_peak_rows = [
        {
            "sample_id": str(sample["sample_id"]),
            "num_peaks": int(len(sample["qx"])),
            "required_min_number_peaks": min_number_peaks,
            "failure_reason": "insufficient_detected_peaks",
        }
        for sample in samples
        if len(sample["qx"]) < min_number_peaks
    ]
    if insufficient_peak_rows and args.insufficient_peaks_policy == "error":
        first = insufficient_peak_rows[0]
        raise ValueError(
            f"{first['sample_id']} has only {first['num_peaks']} peaks; "
            f"ACOM requires {min_number_peaks}. "
            "Use --insufficient-peaks-policy skip to record indexing failures."
        )
    insufficient_ids = {
        str(row["sample_id"]) for row in insufficient_peak_rows
    }
    matched_samples = [
        sample
        for sample in samples
        if str(sample["sample_id"]) not in insufficient_ids
    ]
    if not matched_samples:
        raise ValueError("No samples have enough peaks for ACOM matching")
    for sample in matched_samples:
        if len(sample["qx"]) < min_number_peaks:
            raise ValueError(
                f"{sample['sample_id']} has only {len(sample['qx'])} peaks; "
                f"ACOM requires {min_number_peaks}"
            )

    structure = Structure.from_file(cif_path(config))
    symmetries = proper_point_group_rotations(structure)
    crystal = py4DSTEM.process.diffraction.Crystal.from_pymatgen_structure(
        structure=structure,
        conventional_standard_structure=False,
    )
    voltage = float(common["accelerating_voltage_V"])
    k_max = float(common["k_max_Ainv"])
    crystal.setup_diffraction(accelerating_voltage=voltage)
    crystal.calculate_structure_factors(
        k_max=k_max,
        tol_structure_factor=float(clean["tol_structure_factor"]),
    )

    plan_start = time.perf_counter()
    crystal.orientation_plan(
        zone_axis_range=acom["zone_axis_range"],
        angle_step_zone_axis=angle_step_zone_axis,
        angle_step_in_plane=angle_step_in_plane,
        accel_voltage=voltage,
        corr_kernel_size=float(acom["corr_kernel_size_Ainv"]),
        sigma_excitation_error=float(acom["sigma_excitation_error_Ainv"]),
        power_radial=float(acom["power_radial"]),
        power_intensity=power_intensity_simulated,
        power_intensity_experiment=power_intensity_experiment,
        tol_distance=float(acom["tol_distance_Ainv"]),
        CUDA=use_cuda,
        progress_bar=bool(acom["progress_bar"]),
    )
    plan_seconds = time.perf_counter() - plan_start

    inversion_symmetry = bool(acom["inversion_symmetry"])
    normal_plan_matrices = enumerate_normal_plan_matrices(crystal)
    search_seed_matrices = enumerate_discrete_search_seed_matrices(
        crystal,
        inversion_symmetry=inversion_symmetry,
    )
    search_seed_tree = cKDTree(search_seed_matrices.reshape(-1, 9))

    audit_rows: list[dict] = []
    audit_distance_cache: dict[tuple[str, bytes], tuple[float, float]] = {}
    probe_threshold = float(
        acom["min_probe_discrete_seed_misorientation_deg"]
    )
    failed: list[str] = []
    allowed_policies = {"report_only", "require_discrete_seed"}
    for sample in samples:
        sample_id = sample["sample_id"]
        gt = ground_truth[sample_id]
        matrix_gt = np.asarray(
            gt["orientation_matrix_sample_to_crystal"],
            dtype=float,
        )
        source_id = str(
            gt.get(
                "source_sample_id",
                gt.get("orientation_id", gt.get("ground_truth_source_id", "")),
            )
        )
        distance_key = (source_id, matrix_gt.astype(np.float64).tobytes())
        distances = audit_distance_cache.get(distance_key)
        if distances is None:
            distances = (
                min_discrete_seed_distance_deg(
                    matrix_gt,
                    search_seed_tree,
                    symmetries,
                ),
                min_zone_axis_node_distance_deg(
                    matrix_gt,
                    crystal,
                    symmetries,
                    inversion_symmetry,
                ),
            )
            audit_distance_cache[distance_key] = distances
        discrete_distance, zone_distance = distances
        policy = str(gt.get("acom_offgrid_policy", "report_only"))
        if policy not in allowed_policies:
            raise ValueError(f"{sample_id} has unknown ACOM off-grid policy {policy}")
        passed = (
            policy != "require_discrete_seed"
            or discrete_distance >= probe_threshold
        )
        audit_rows.append(
            {
                "sample_id": sample_id,
                "sampling_type": gt.get("sampling_type"),
                "sample_role": gt.get("sample_role"),
                "acom_offgrid_policy": policy,
                "nearest_discrete_search_seed_misorientation_deg": discrete_distance,
                "nearest_zone_axis_node_misorientation_deg": zone_distance,
                "probe_discrete_seed_threshold_deg": probe_threshold,
                "acom_eligible": sample_id not in insufficient_ids,
                "passed": passed,
            }
        )
        if not passed:
            failed.append(sample_id)

    audit_output = {
        "dataset_id": config["dataset"]["id"],
        "py4DSTEM_version": py4DSTEM.__version__,
        "orientation_plan": {
            "zone_axis_range": acom["zone_axis_range"],
            "angle_step_zone_axis_deg": angle_step_zone_axis,
            "angle_step_in_plane_deg": angle_step_in_plane,
            "inversion_symmetry": inversion_symmetry,
            "in_plane_subgrid_interpolation": "parabolic correlation-peak fit",
            "num_zone_axes": int(crystal.orientation_num_zones),
            "num_in_plane_steps": int(crystal.orientation_in_plane_steps),
            "num_normal_discrete_seeds": int(normal_plan_matrices.shape[0]),
            "num_discrete_seeds_including_mirror": int(
                search_seed_matrices.shape[0]
            ),
            "build_seconds": plan_seconds,
        },
        "distance_interpretation": {
            "nearest_discrete_search_seed_misorientation_deg": (
                "Diagnostic distance to a discrete correlation seed; not an error "
                "lower bound because ACOM refines the in-plane angle."
            ),
            "nearest_zone_axis_node_misorientation_deg": (
                "Distance to the nearest searched zone-axis direction."
            ),
        },
        "evaluation_equivalence": {
            "crystal_symmetry": "proper point-group rotations only",
            "friedel_sample_rotation": FRIEDEL_SAMPLE_ROTATION.tolist(),
        },
        "sample_counts_by_role": dict(
            Counter(row["sample_role"] for row in audit_rows)
        ),
        "unique_orientation_distance_evaluations": len(audit_distance_cache),
        "samples": audit_rows,
    }
    audit_path = (
        args.audit_file.resolve()
        if args.audit_file
        else V3_REPORT_DIR / f"acom_plan_audit{output_suffix}.json"
    )
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(
        json.dumps(audit_output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    audit_by_id = {row["sample_id"]: row for row in audit_rows}
    if failed:
        raise RuntimeError(
            "ACOM grid probes are too close to a discrete seed: "
            f"{failed}. See {audit_path}."
        )

    predictions: list[dict] = []
    details: list[dict] = []
    prediction_seconds: list[float] = []
    candidate_matrices: list[np.ndarray] = []
    candidate_correlations: list[np.ndarray] = []
    candidate_indices: list[np.ndarray] = []
    candidate_mirrors: list[np.ndarray] = []
    candidate_angles_deg: list[np.ndarray] = []
    candidate_strict_errors: list[np.ndarray] = []
    candidate_friedel_errors: list[np.ndarray] = []
    candidate_sample_ids: list[str] = []
    for sample_index, sample in enumerate(matched_samples):
        point_list = make_point_list(sample)
        start = time.perf_counter()
        candidates = match_topk_distinct_zone_axes(
            crystal,
            point_list,
            n_best=num_matches_return,
            min_number_peaks=min_number_peaks,
            inversion_symmetry=inversion_symmetry,
            min_zone_separation_deg=topk_min_zone_separation_deg,
        )
        seconds = time.perf_counter() - start
        prediction_seconds.append(seconds)

        sample_id = sample["sample_id"]
        gt = ground_truth[sample_id]
        matrices = np.asarray(candidates["matrix"], dtype=float)
        if matrices.shape != (num_matches_return, 3, 3):
            raise RuntimeError(
                f"{sample_id} returned candidate shape {matrices.shape}; "
                f"expected {(num_matches_return, 3, 3)}"
            )
        matrices = np.stack(
            [nearest_rotation(matrix) for matrix in matrices], axis=0
        )
        correlations = np.asarray(candidates["correlation"], dtype=float)
        indices = np.asarray(candidates["indices"], dtype=np.int32)
        mirrors = np.asarray(candidates["mirror"], dtype=bool)
        angles_deg = np.degrees(
            np.asarray(candidates["angles"], dtype=float)
        )
        if (
            correlations.shape != (num_matches_return,)
            or indices.shape != (num_matches_return, 2)
            or mirrors.shape != (num_matches_return,)
            or angles_deg.shape != (num_matches_return, 3)
        ):
            raise RuntimeError(f"{sample_id} returned inconsistent Top-5 arrays")
        matrix_gt = np.asarray(
            gt["orientation_matrix_sample_to_crystal"],
            dtype=float,
        )
        strict_errors = np.asarray(
            [
                symmetry_aware_misorientation_deg(
                    matrix, matrix_gt, symmetries
                )
                for matrix in matrices
            ],
            dtype=float,
        )
        friedel_errors = np.asarray(
            [
                friedel_aware_misorientation_deg(
                    matrix, matrix_gt, symmetries
                )
                for matrix in matrices
            ],
            dtype=float,
        )
        matrix_pred = matrices[0]
        strict_error = float(strict_errors[0])
        friedel_error = float(friedel_errors[0])
        candidate_sample_ids.append(sample_id)
        candidate_matrices.append(matrices)
        candidate_correlations.append(correlations)
        candidate_indices.append(indices)
        candidate_mirrors.append(mirrors)
        candidate_angles_deg.append(angles_deg)
        candidate_strict_errors.append(strict_errors)
        candidate_friedel_errors.append(friedel_errors)
        predictions.append(
            {
                "sample_id": sample_id,
                "orientation_matrix_sample_to_crystal": matrix_pred.tolist(),
            }
        )
        audit = audit_by_id[sample_id]
        details.append(
            {
                "sample_id": sample_id,
                "sampling_type": gt.get("sampling_type"),
                "sample_role": gt.get("sample_role"),
                "probe_axis_id": gt.get("probe_axis_id"),
                "probe_offset_deg": gt.get("probe_offset_deg"),
                "num_peaks": int(len(sample["qx"])),
                "correlation_score": float(correlations[0]),
                "zone_axis_plan_index": int(indices[0, 0]),
                "in_plane_plan_index": int(indices[0, 1]),
                "mirror_match": bool(mirrors[0]),
                "euler_angles_deg": angles_deg[0].tolist(),
                "prediction_seconds": seconds,
                "strict_misorientation_deg": float(strict_error),
                "friedel_equivalent_misorientation_deg": float(friedel_error),
                "misorientation_deg": float(friedel_error),
                "nearest_discrete_search_seed_misorientation_deg": float(
                    audit["nearest_discrete_search_seed_misorientation_deg"]
                ),
                "nearest_zone_axis_node_misorientation_deg": float(
                    audit["nearest_zone_axis_node_misorientation_deg"]
                ),
                "predicted_orientation_matrix_sample_to_crystal": (
                    matrix_pred.tolist()
                ),
            }
        )
        if (
            (sample_index + 1) % 50 == 0
            or sample_index + 1 == len(matched_samples)
        ):
            print(
                f"Matched {sample_index + 1}/{len(matched_samples)} samples; "
                f"latest Friedel error={friedel_error:.3f}°"
            )

    submission_path = (
        args.prediction_file.resolve()
        if args.prediction_file
        else ROOT / "submissions" / f"acom_clean_predictions{output_suffix}.jsonl"
    )
    write_jsonl(submission_path, predictions)

    candidates_path = (
        args.candidates_file.resolve()
        if args.candidates_file
        else V3_REPORT_DIR / f"acom_candidates{output_suffix}.h5"
    )
    candidates_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_arrays = {
        "orientation_matrix_sample_to_crystal": np.asarray(
            candidate_matrices, dtype=np.float64
        ),
        "correlation": np.asarray(candidate_correlations, dtype=np.float32),
        "plan_index": np.asarray(candidate_indices, dtype=np.int32),
        "mirror_match": np.asarray(candidate_mirrors, dtype=np.bool_),
        "euler_angles_deg": np.asarray(
            candidate_angles_deg, dtype=np.float64
        ),
        "strict_misorientation_deg": np.asarray(
            candidate_strict_errors, dtype=np.float64
        ),
        "friedel_equivalent_misorientation_deg": np.asarray(
            candidate_friedel_errors, dtype=np.float64
        ),
    }
    with h5py.File(candidates_path, "w") as output:
        output.attrs["schema"] = "or4d-acom-topk-v1"
        output.attrs["n_best"] = num_matches_return
        output.attrs["rank_order"] = "algorithm_correlation_order"
        output.create_dataset(
            "sample_id",
            data=np.asarray(
                candidate_sample_ids, dtype=h5py.string_dtype("utf-8")
            ),
        )
        output.create_dataset(
            "rank",
            data=np.arange(1, num_matches_return + 1, dtype=np.int8),
        )
        for name, values in candidate_arrays.items():
            output.create_dataset(
                name,
                data=values,
                chunks=True,
                compression="gzip",
                compression_opts=4,
            )

    friedel_array = candidate_arrays[
        "friedel_equivalent_misorientation_deg"
    ]
    top_k_metrics = summarize_topk_errors(
        friedel_array,
        total_input_samples=len(samples),
    )

    config_path = ROOT / "config" / "benchmark.yaml"
    details_output = {
        "dataset_id": config["dataset"]["id"],
        "output_tag": args.output_tag or "canonical",
        "acom_angle_step_zone_axis_deg": angle_step_zone_axis,
        "acom_angle_step_in_plane_deg": angle_step_in_plane,
        "acom_power_intensity_experiment": power_intensity_experiment,
        "acom_power_intensity_simulated": power_intensity_simulated,
        "acom_cuda": use_cuda,
        "k_max_Ainv": k_max,
        "source_peak_file": str(peak_path),
        "ground_truth_file": str(gt_path),
        "ground_truth_id_prefix": args.ground_truth_id_prefix,
        "orientation_manifest_file": str(manifest_path),
        "insufficient_peaks_policy": args.insufficient_peaks_policy,
        "num_input_samples": len(samples),
        "num_matched_samples": len(matched_samples),
        "num_indexing_failures": len(insufficient_peak_rows),
        "num_matches_return": num_matches_return,
        "candidate_definition": (
            "Ranked zone-axis-distinct ACOM hypotheses. Every rank is matched "
            "against the unchanged full peak list; only previously selected "
            "zone-axis neighbourhoods are masked."
        ),
        "topk_min_zone_separation_deg": topk_min_zone_separation_deg,
        "candidate_file": str(candidates_path),
        "top_k_metrics": top_k_metrics,
        "indexing_failures": insufficient_peak_rows,
        "source_git_revision": git_revision(),
        "primary_metric": "friedel_equivalent_misorientation_deg",
        "headline_sample_role": config["evaluation"]["headline_sample_role"],
        "sample_counts_by_role": dict(
            Counter(row["sample_role"] for row in details)
        ),
        "runtime": {
            "plan_build_seconds": plan_seconds,
            "matching": runtime_summary(prediction_seconds),
        },
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "py4DSTEM": py4DSTEM.__version__,
            "pymatgen": importlib.metadata.version("pymatgen"),
            "h5py": importlib.metadata.version("h5py"),
        },
        "system": {
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
        "sha256": {
            "config": sha256_file(config_path),
            "cif": sha256_file(cif_path(config)),
            "orientation_manifest": sha256_file(manifest_path),
            "source_peaks": sha256_file(peak_path),
            "ground_truth": sha256_file(gt_path),
        },
        "matched_model_limitation": (
            "Clean inputs and ACOM templates use the same CIF and py4DSTEM "
            "kinematical model; this measures self-consistency, not real-data "
            "or cross-simulator generalization."
        ),
        "samples": details,
    }
    details_path = (
        args.details_file.resolve()
        if args.details_file
        else V3_REPORT_DIR / f"acom_clean_details{output_suffix}.json"
    )
    details_path.parent.mkdir(parents=True, exist_ok=True)
    details_path.write_text(
        json.dumps(details_output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Predictions: {submission_path}")
    print(f"Top-{num_matches_return} candidates: {candidates_path}")
    print(f"Plan audit: {audit_path}")
    print(f"ACOM details: {details_path}")


if __name__ == "__main__":
    main()
