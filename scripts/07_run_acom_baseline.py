#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import py4DSTEM
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from or4d_common import (  # noqa: E402
    ACOM_MIRROR_SAMPLE_ROTATION,
    FRIEDEL_SAMPLE_ROTATION,
    cif_path,
    friedel_aware_misorientation_deg,
    load_config,
    nearest_rotation,
    read_jsonl,
    read_peak_h5,
    rotation_angle_deg,
    symmetry_aware_misorientation_deg,
    write_jsonl,
)


def proper_point_group_rotations(structure: Structure) -> list[np.ndarray]:
    """Return only determinant +1 crystal point-group operations.

    The determinant is checked before orthogonalization. This avoids turning
    inversion or mirror operations into arbitrary proper rotations.
    """
    rotations: list[np.ndarray] = []
    operations = SpacegroupAnalyzer(structure).get_point_group_operations(
        cartesian=True
    )
    for operation in operations:
        raw = np.asarray(operation.rotation_matrix, dtype=float)
        if np.linalg.det(raw) < 0.0:
            continue
        matrix = nearest_rotation(raw)
        if not any(np.allclose(matrix, existing, atol=1e-8) for existing in rotations):
            rotations.append(matrix)
    if not rotations:
        raise RuntimeError("No proper crystal point-group rotations were found.")
    return rotations


def in_plane_rotation_matrix(phi: float) -> np.ndarray:
    # Exact m3z convention used by py4DSTEM ACOM.
    c = float(np.cos(phi))
    s = float(np.sin(phi))
    return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])


def enumerate_normal_plan_matrices(crystal) -> np.ndarray:
    base = np.asarray(crystal.orientation_rotation_matrices, dtype=float)
    gamma = np.asarray(crystal.orientation_gamma, dtype=float)
    rotations = np.stack([in_plane_rotation_matrix(phi) for phi in gamma], axis=0)
    plan = base[:, None, :, :] @ rotations[None, :, :, :]
    return plan.reshape(-1, 3, 3)


def enumerate_search_plan_matrices(crystal, inversion_symmetry: bool) -> np.ndarray:
    """Enumerate matrices reachable by the configured ACOM search.

    py4DSTEM's mirror branch negates matrix columns 1 and 2. Because the
    in-plane grid covers the full 360 degrees, right-multiplication by the
    ACOM mirror rotation enumerates the complete discrete mirror branch.
    """
    normal = enumerate_normal_plan_matrices(crystal)
    if not inversion_symmetry:
        return normal
    mirror = normal @ ACOM_MIRROR_SAMPLE_ROTATION
    return np.concatenate([normal, mirror], axis=0)


def min_observable_plan_distance_deg(
    R_gt: np.ndarray,
    search_plan_matrices: np.ndarray,
    symmetries: list[np.ndarray],
) -> float:
    """Distance to the nearest actually searched, Clean-Peak-equivalent node."""
    best = np.inf
    for symmetry in symmetries:
        crystal_equivalent = symmetry @ R_gt
        for sample_branch in (np.eye(3), FRIEDEL_SAMPLE_ROTATION):
            equivalent = crystal_equivalent @ sample_branch
            relative = search_plan_matrices @ equivalent.T
            traces = np.trace(relative, axis1=1, axis2=2)
            cosines = np.clip((traces - 1.0) / 2.0, -1.0, 1.0)
            angle = float(np.degrees(np.arccos(cosines)).min())
            best = min(best, angle)
    return float(best)


def make_point_list(sample: dict) -> py4DSTEM.PointList:
    dtype = np.dtype([("qx", "f4"), ("qy", "f4"), ("intensity", "f4")])
    data = np.empty(len(sample["qx"]), dtype=dtype)
    data["qx"] = sample["qx"]
    data["qy"] = sample["qy"]
    data["intensity"] = sample["intensity"]
    return py4DSTEM.PointList(data=data, name=sample["sample_id"])


def summarize(errors: np.ndarray) -> dict:
    return {
        "num_samples": int(errors.size),
        "mean_misorientation_deg": float(errors.mean()),
        "median_misorientation_deg": float(np.median(errors)),
        "p90_misorientation_deg": float(np.percentile(errors, 90)),
        "accuracy_within_1deg": float(np.mean(errors <= 1.0)),
        "accuracy_within_2deg": float(np.mean(errors <= 2.0)),
        "accuracy_within_5deg": float(np.mean(errors <= 5.0)),
    }


def main() -> None:
    config = load_config()
    acom = config["acom"]
    clean = config["clean"]
    common = config["common"]

    samples = read_peak_h5(ROOT / "public" / "clean_peaks.h5")
    ground_truth = {
        record["sample_id"]: record
        for record in read_jsonl(ROOT / "private" / "clean_ground_truth.jsonl")
    }
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
    search_plan_matrices = enumerate_search_plan_matrices(
        crystal, inversion_symmetry=inversion_symmetry
    )

    audit_rows: list[dict] = []
    required_threshold = float(acom["min_offgrid_misorientation_deg"])
    failed: list[str] = []
    for sample in samples:
        sample_id = sample["sample_id"]
        gt = ground_truth[sample_id]
        R_gt = np.asarray(gt["orientation_matrix_sample_to_crystal"], dtype=float)
        distance = min_observable_plan_distance_deg(
            R_gt, search_plan_matrices, symmetries
        )
        require_offgrid = gt.get("sampling_type") in {
            "nonzero_in_plane",
            "so3_low_discrepancy",
        }
        passed = (not require_offgrid) or distance >= required_threshold
        audit_rows.append(
            {
                "sample_id": sample_id,
                "sampling_type": gt.get("sampling_type"),
                "require_offgrid": require_offgrid,
                "nearest_search_node_friedel_equivalent_misorientation_deg": distance,
                "offgrid_threshold_deg": required_threshold,
                "passed": passed,
            }
        )
        if not passed:
            failed.append(sample_id)

    audit_output = {
        "py4DSTEM_version": py4DSTEM.__version__,
        "orientation_plan": {
            "zone_axis_range": acom["zone_axis_range"],
            "angle_step_zone_axis_deg": float(acom["angle_step_zone_axis_deg"]),
            "angle_step_in_plane_deg": float(acom["angle_step_in_plane_deg"]),
            "inversion_symmetry": inversion_symmetry,
            "num_zone_axes": int(crystal.orientation_num_zones),
            "num_in_plane_steps": int(crystal.orientation_in_plane_steps),
            "num_normal_plan_nodes": int(normal_plan_matrices.shape[0]),
            "num_search_nodes_including_mirror": int(search_plan_matrices.shape[0]),
            "build_seconds": plan_seconds,
        },
        "evaluation_equivalence": {
            "crystal_symmetry": "proper point-group rotations only",
            "friedel_sample_rotation": FRIEDEL_SAMPLE_ROTATION.tolist(),
        },
        "samples": audit_rows,
    }
    audit_path = ROOT / "reports" / "acom_plan_audit.json"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(
        json.dumps(audit_output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    audit_by_id = {row["sample_id"]: row for row in audit_rows}

    if failed:
        raise RuntimeError(
            "Required off-grid samples are too close to an actually searched "
            f"ACOM node in Clean-Peak equivalence space: {failed}. See {audit_path}."
        )

    predictions: list[dict] = []
    details: list[dict] = []
    prediction_start = time.perf_counter()
    for sample in samples:
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
        R_pred = nearest_rotation(np.asarray(orientation.matrix[0], dtype=float))
        R_gt = np.asarray(
            ground_truth[sample["sample_id"]]["orientation_matrix_sample_to_crystal"],
            dtype=float,
        )
        strict_error = symmetry_aware_misorientation_deg(
            R_pred, R_gt, symmetries
        )
        friedel_error = friedel_aware_misorientation_deg(
            R_pred, R_gt, symmetries
        )
        predictions.append(
            {
                "sample_id": sample["sample_id"],
                "orientation_matrix_sample_to_crystal": R_pred.tolist(),
            }
        )
        details.append(
            {
                "sample_id": sample["sample_id"],
                "sampling_type": ground_truth[sample["sample_id"]].get(
                    "sampling_type"
                ),
                "num_peaks": int(len(sample["qx"])),
                "correlation_score": float(orientation.corr[0]),
                "zone_axis_plan_index": int(orientation.inds[0, 0]),
                "in_plane_plan_index": int(orientation.inds[0, 1]),
                "mirror_match": bool(orientation.mirror[0]),
                "euler_angles_deg": np.degrees(orientation.angles[0]).tolist(),
                "prediction_seconds": seconds,
                "strict_misorientation_deg": float(strict_error),
                "friedel_equivalent_misorientation_deg": float(friedel_error),
                # The Clean-Peak primary metric is Friedel-aware because the
                # public input contains centrosymmetric +/-q peak pairs.
                "misorientation_deg": float(friedel_error),
                "nearest_search_node_friedel_equivalent_misorientation_deg": float(
                    audit_by_id[sample["sample_id"]][
                        "nearest_search_node_friedel_equivalent_misorientation_deg"
                    ]
                ),
                "predicted_orientation_matrix_sample_to_crystal": R_pred.tolist(),
            }
        )
        print(
            f"{sample['sample_id']}: peaks={len(sample['qx'])}, "
            f"corr={orientation.corr[0]:.6f}, "
            f"strict={strict_error:.4f} deg, "
            f"friedel={friedel_error:.4f} deg, "
            f"mirror={bool(orientation.mirror[0])}, time={seconds:.3f}s"
        )

    total_prediction_seconds = time.perf_counter() - prediction_start
    submission_path = ROOT / "submissions" / "acom_clean_predictions.jsonl"
    write_jsonl(submission_path, predictions)

    strict_errors = np.asarray(
        [row["strict_misorientation_deg"] for row in details], dtype=float
    )
    friedel_errors = np.asarray(
        [row["friedel_equivalent_misorientation_deg"] for row in details],
        dtype=float,
    )
    details_output = {
        "py4DSTEM_version": py4DSTEM.__version__,
        "plan_build_seconds": plan_seconds,
        "total_prediction_seconds": total_prediction_seconds,
        "average_prediction_seconds": total_prediction_seconds / max(len(samples), 1),
        "primary_metric": "friedel_equivalent_misorientation_deg",
        "metrics": summarize(friedel_errors),
        "metrics_strict": summarize(strict_errors),
        "metrics_friedel_equivalent": summarize(friedel_errors),
        "samples": details,
    }
    details_path = ROOT / "reports" / "acom_clean_details.json"
    details_path.write_text(
        json.dumps(details_output, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Predictions: {submission_path}")
    print(f"Plan audit: {audit_path}")
    print(f"ACOM details: {details_path}")


if __name__ == "__main__":
    main()
