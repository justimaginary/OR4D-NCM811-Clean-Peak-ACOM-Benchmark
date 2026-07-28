#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import py4DSTEM
from pymatgen.core import Structure

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
    read_peak_h5,
    symmetry_aware_misorientation_deg,
    write_jsonl,
)


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
    search_seed_matrices: np.ndarray,
    symmetries: list[np.ndarray],
) -> float:
    """Distance to the nearest discrete seed in Clean observable space.

    This is not an error lower bound: ``match_single_pattern`` performs a
    parabolic sub-grid fit of the in-plane correlation peak.
    """
    best = np.inf
    for symmetry in symmetries:
        crystal_equivalent = symmetry @ matrix_gt
        for sample_branch in (np.eye(3), FRIEDEL_SAMPLE_ROTATION):
            equivalent = crystal_equivalent @ sample_branch
            relative = search_seed_matrices @ equivalent.T
            traces = np.trace(relative, axis1=1, axis2=2)
            cosines = np.clip((traces - 1.0) / 2.0, -1.0, 1.0)
            best = min(
                best,
                float(np.degrees(np.arccos(cosines)).min()),
            )
    return float(best)


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
        sample_id = str(record["sample_id"])
        if sample_id in result:
            raise ValueError(f"Duplicate sample_id in {source}: {sample_id}")
        result[sample_id] = record
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


def main() -> None:
    config = load_config()
    acom = config["acom"]
    clean = config["clean"]
    common = config["common"]

    peak_path = ROOT / "public" / "clean_peaks.h5"
    gt_path = ROOT / "private" / "clean_ground_truth.jsonl"
    manifest_path = ROOT / "private" / "orientations.jsonl"
    samples = read_peak_h5(peak_path)
    ground_truth = unique_records_by_id(
        read_jsonl(gt_path),
        source=str(gt_path),
    )
    sample_ids = [str(sample["sample_id"]) for sample in samples]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("public/clean_peaks.h5 contains duplicate sample IDs")
    if set(sample_ids) != set(ground_truth):
        raise ValueError("Clean public and ground-truth sample IDs differ")

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
        angle_step_zone_axis=float(acom["angle_step_zone_axis_deg"]),
        angle_step_in_plane=float(acom["angle_step_in_plane_deg"]),
        accel_voltage=voltage,
        corr_kernel_size=float(acom["corr_kernel_size_Ainv"]),
        sigma_excitation_error=float(acom["sigma_excitation_error_Ainv"]),
        power_radial=float(acom["power_radial"]),
        power_intensity=float(acom["power_intensity_simulated"]),
        power_intensity_experiment=float(acom["power_intensity_experiment"]),
        tol_distance=float(acom["tol_distance_Ainv"]),
        CUDA=bool(acom["cuda"]),
        progress_bar=bool(acom["progress_bar"]),
    )
    plan_seconds = time.perf_counter() - plan_start

    inversion_symmetry = bool(acom["inversion_symmetry"])
    normal_plan_matrices = enumerate_normal_plan_matrices(crystal)
    search_seed_matrices = enumerate_discrete_search_seed_matrices(
        crystal,
        inversion_symmetry=inversion_symmetry,
    )

    audit_rows: list[dict] = []
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
        discrete_distance = min_discrete_seed_distance_deg(
            matrix_gt,
            search_seed_matrices,
            symmetries,
        )
        zone_distance = min_zone_axis_node_distance_deg(
            matrix_gt,
            crystal,
            symmetries,
            inversion_symmetry,
        )
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
            "angle_step_zone_axis_deg": float(acom["angle_step_zone_axis_deg"]),
            "angle_step_in_plane_deg": float(acom["angle_step_in_plane_deg"]),
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
        "samples": audit_rows,
    }
    audit_path = ROOT / "reports" / "acom_plan_audit.json"
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
    for sample_index, sample in enumerate(samples):
        point_list = make_point_list(sample)
        start = time.perf_counter()
        orientation = crystal.match_single_pattern(
            bragg_peaks=point_list,
            num_matches_return=int(acom["num_matches_return"]),
            min_number_peaks=int(acom["min_number_peaks"]),
            inversion_symmetry=inversion_symmetry,
            plot_polar=False,
            plot_corr=False,
            verbose=False,
        )
        seconds = time.perf_counter() - start
        prediction_seconds.append(seconds)

        sample_id = sample["sample_id"]
        gt = ground_truth[sample_id]
        matrix_pred = nearest_rotation(
            np.asarray(orientation.matrix[0], dtype=float)
        )
        matrix_gt = np.asarray(
            gt["orientation_matrix_sample_to_crystal"],
            dtype=float,
        )
        strict_error = symmetry_aware_misorientation_deg(
            matrix_pred,
            matrix_gt,
            symmetries,
        )
        friedel_error = friedel_aware_misorientation_deg(
            matrix_pred,
            matrix_gt,
            symmetries,
        )
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
                "correlation_score": float(orientation.corr[0]),
                "zone_axis_plan_index": int(orientation.inds[0, 0]),
                "in_plane_plan_index": int(orientation.inds[0, 1]),
                "mirror_match": bool(orientation.mirror[0]),
                "euler_angles_deg": np.degrees(orientation.angles[0]).tolist(),
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
        if (sample_index + 1) % 50 == 0 or sample_index + 1 == len(samples):
            print(
                f"Matched {sample_index + 1}/{len(samples)} samples; "
                f"latest Friedel error={friedel_error:.3f}°"
            )

    submission_path = ROOT / "submissions" / "acom_clean_predictions.jsonl"
    write_jsonl(submission_path, predictions)

    config_path = ROOT / "config" / "benchmark.yaml"
    details_output = {
        "dataset_id": config["dataset"]["id"],
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
            "public_peaks": sha256_file(peak_path),
            "ground_truth": sha256_file(gt_path),
        },
        "matched_model_limitation": (
            "Clean inputs and ACOM templates use the same CIF and py4DSTEM "
            "kinematical model; this measures self-consistency, not real-data "
            "or cross-simulator generalization."
        ),
        "samples": details,
    }
    details_path = ROOT / "reports" / "acom_clean_details.json"
    details_path.write_text(
        json.dumps(details_output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Predictions: {submission_path}")
    print(f"Plan audit: {audit_path}")
    print(f"ACOM details: {details_path}")


if __name__ == "__main__":
    main()
