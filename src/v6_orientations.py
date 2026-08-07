from __future__ import annotations

from typing import Any, Iterable

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from or4d_common import (
    FRIEDEL_SAMPLE_ROTATION,
    canonicalize_clean_orientation,
    matrix_to_quaternion_wxyz,
    nearest_rotation,
    proper_point_group_rotations,
    quaternion_wxyz_to_matrix,
    shoemake_so3_quaternion,
)


def _validated_sampling_config(config: dict[str, Any]) -> dict[str, Any]:
    sampling = dict(config["clean_sampling"]["headline_core"])
    if sampling.get("method") != "haar_uniform_so3":
        raise ValueError("V6 orientation sampling method must be haar_uniform_so3")
    if not bool(sampling.get("canonicalize_crystal_symmetry", False)):
        raise ValueError("V6 requires crystal-symmetry canonicalization")
    if not bool(sampling.get("canonicalize_friedel", False)):
        raise ValueError("V6 requires Friedel canonicalization")
    if int(sampling["count"]) <= 0:
        raise ValueError("V6 orientation count must be positive")
    if float(sampling["duplicate_tolerance_deg"]) < 0.0:
        raise ValueError("duplicate_tolerance_deg must be non-negative")
    if int(sampling["dedup_query_initial_neighbors"]) < 2:
        raise ValueError("dedup_query_initial_neighbors must be at least two")
    return sampling


def _equivalent_quaternions(
    matrix: np.ndarray,
    symmetries: Iterable[np.ndarray],
) -> np.ndarray:
    values: list[np.ndarray] = []
    for symmetry in symmetries:
        for friedel in (np.eye(3), FRIEDEL_SAMPLE_ROTATION):
            quaternion = matrix_to_quaternion_wxyz(
                nearest_rotation(np.asarray(symmetry) @ matrix @ friedel)
            )
            values.extend((quaternion, -quaternion))
    return np.asarray(values, dtype=np.float64)


def audit_symmetry_unique_orientations(
    matrices: np.ndarray,
    symmetries: Iterable[np.ndarray],
    *,
    duplicate_tolerance_deg: float,
    initial_neighbors: int,
) -> dict[str, Any]:
    """Audit nearest distinct orientation over symmetry and Friedel branches."""
    matrices = np.asarray(matrices, dtype=np.float64)
    symmetry_list = [np.asarray(value, dtype=np.float64) for value in symmetries]
    if matrices.ndim != 3 or matrices.shape[1:] != (3, 3):
        raise ValueError("matrices must have shape [N,3,3]")
    if len(matrices) < 2:
        return {
            "minimum_equivalent_misorientation_deg": None,
            "nearest_equivalent_misorientation_deg": [None] * len(matrices),
            "nearest_distinct_index": [None] * len(matrices),
        }

    variants: list[np.ndarray] = []
    owners: list[np.ndarray] = []
    for owner, matrix in enumerate(matrices):
        equivalent = _equivalent_quaternions(matrix, symmetry_list)
        variants.append(equivalent)
        owners.append(np.full(len(equivalent), owner, dtype=np.int32))
    points = np.concatenate(variants, axis=0)
    point_owners = np.concatenate(owners, axis=0)
    tree = cKDTree(points)
    queries = np.stack(
        [matrix_to_quaternion_wxyz(matrix) for matrix in matrices], axis=0
    )

    nearest_distances = np.full(len(matrices), np.inf, dtype=np.float64)
    nearest_indices = np.full(len(matrices), -1, dtype=np.int64)
    unresolved = np.arange(len(matrices), dtype=np.int64)
    neighbors = min(max(int(initial_neighbors), 2), len(points))
    while len(unresolved):
        distances, indices = tree.query(queries[unresolved], k=neighbors)
        distances = np.atleast_2d(distances)
        indices = np.atleast_2d(indices)
        next_unresolved: list[int] = []
        for local, sample_index in enumerate(unresolved):
            other = point_owners[indices[local]] != sample_index
            if np.any(other):
                first = int(np.flatnonzero(other)[0])
                nearest_distances[sample_index] = float(distances[local, first])
                nearest_indices[sample_index] = int(point_owners[indices[local, first]])
            elif neighbors < len(points):
                next_unresolved.append(int(sample_index))
        if not next_unresolved:
            break
        if neighbors == len(points):
            raise RuntimeError("Could not find a distinct orientation during audit")
        unresolved = np.asarray(next_unresolved, dtype=np.int64)
        neighbors = min(neighbors * 2, len(points))

    angles = np.degrees(
        4.0 * np.arcsin(np.clip(nearest_distances / 2.0, 0.0, 1.0))
    )
    minimum_index = int(np.argmin(angles))
    minimum = float(angles[minimum_index])
    if minimum <= float(duplicate_tolerance_deg):
        raise ValueError(
            "Symmetry/Friedel-equivalent orientations are duplicated: "
            f"indices {minimum_index} and {nearest_indices[minimum_index]}, "
            f"misorientation={minimum:.12g} deg, "
            f"tolerance={duplicate_tolerance_deg:.12g} deg"
        )
    return {
        "minimum_equivalent_misorientation_deg": minimum,
        "minimum_pair_indices": [
            minimum_index,
            int(nearest_indices[minimum_index]),
        ],
        "nearest_equivalent_misorientation_deg": angles.tolist(),
        "nearest_distinct_index": nearest_indices.tolist(),
    }


def orientation_distribution_summary(
    matrices: np.ndarray,
    *,
    euler_sequence: str,
) -> dict[str, Any]:
    matrices = np.asarray(matrices, dtype=np.float64)
    beam = matrices[:, :, 2]
    tilt = np.degrees(np.arccos(np.clip(beam[:, 2], -1.0, 1.0)))
    azimuth = np.degrees(np.arctan2(beam[:, 1], beam[:, 0]))
    euler = Rotation.from_matrix(matrices).as_euler(euler_sequence, degrees=True)

    def stats(values: np.ndarray) -> dict[str, float]:
        return {
            "min": float(np.min(values)),
            "p05": float(np.percentile(values, 5)),
            "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95)),
            "max": float(np.max(values)),
        }

    return {
        "beam_tilt_deg": stats(tilt),
        "beam_azimuth_deg": stats(azimuth),
        "euler_sequence": euler_sequence,
        "euler_component_deg": [stats(euler[:, index]) for index in range(3)],
        "mean_beam_direction_crystal": np.mean(beam, axis=0).tolist(),
    }


def build_v6_orientation_records(
    config: dict[str, Any],
    structure: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sampling = _validated_sampling_config(config)
    count = int(sampling["count"])
    seed = int(sampling["seed"])
    decimals = int(sampling["canonical_quaternion_decimals"])
    prefix = str(sampling["orientation_id_prefix"])
    symmetries = proper_point_group_rotations(structure)
    rng = np.random.default_rng(seed)

    records: list[dict[str, Any]] = []
    seen_classes: set[str] = set()
    rejected_exact_classes = 0
    candidate_index = 0
    while len(records) < count:
        unit_cube = rng.random(3)
        raw_quaternion = shoemake_so3_quaternion(unit_cube)
        raw_matrix = quaternion_wxyz_to_matrix(raw_quaternion)
        canonical = canonicalize_clean_orientation(
            raw_matrix,
            symmetries,
            decimals=decimals,
        )
        class_id = str(canonical["orientation_class_key"])
        if class_id in seen_classes:
            rejected_exact_classes += 1
            candidate_index += 1
            continue
        seen_classes.add(class_id)
        matrix = np.asarray(canonical["canonical_matrix"], dtype=np.float64)
        output_index = len(records)
        records.append(
            {
                "orientation_id": f"{prefix}{output_index:05d}",
                "sampling_type": "haar_uniform_so3",
                "sample_role": "headline_core",
                "acom_offgrid_policy": "report_only",
                "orientation_matrix_sample_to_crystal": matrix.tolist(),
                "raw_orientation_matrix_sample_to_crystal": raw_matrix.tolist(),
                "raw_quaternion_wxyz": raw_quaternion.tolist(),
                "canonical_quaternion_wxyz": np.asarray(
                    canonical["canonical_quaternion_wxyz"]
                ).tolist(),
                "orientation_class_id": class_id,
                "canonical_crystal_symmetry_index": int(
                    canonical["crystal_symmetry_index"]
                ),
                "canonical_friedel_branch_index": int(
                    canonical["friedel_branch_index"]
                ),
                "canonical_friedel_used": bool(canonical["friedel_used"]),
                "canonicalization_residual_deg": float(
                    canonical["canonicalization_residual_deg"]
                ),
                "beam_direction_crystal_cartesian": matrix[:, 2].tolist(),
                "sampling_seed": seed,
                "sampling_index": output_index,
                "sampling_candidate_index": candidate_index,
                "sampling_unit_cube": unit_cube.tolist(),
                "zone_axis_3index": None,
                "x_reference_uvw": None,
                "in_plane_rotation_deg": None,
            }
        )
        candidate_index += 1

    matrices = np.stack(
        [record["orientation_matrix_sample_to_crystal"] for record in records]
    )
    uniqueness = audit_symmetry_unique_orientations(
        matrices,
        symmetries,
        duplicate_tolerance_deg=float(sampling["duplicate_tolerance_deg"]),
        initial_neighbors=int(sampling["dedup_query_initial_neighbors"]),
    )
    for record, nearest, nearest_index in zip(
        records,
        uniqueness["nearest_equivalent_misorientation_deg"],
        uniqueness["nearest_distinct_index"],
        strict=True,
    ):
        record["nearest_clean_equivalent_misorientation_deg"] = nearest
        record["nearest_clean_equivalent_orientation_index"] = nearest_index

    summary = {
        "method": sampling["method"],
        "count": count,
        "seed": seed,
        "candidate_count": candidate_index,
        "rejected_exact_orientation_classes": rejected_exact_classes,
        "proper_crystal_symmetry_count": len(symmetries),
        "friedel_branch_count": 2,
        "duplicate_tolerance_deg": float(sampling["duplicate_tolerance_deg"]),
        "uniqueness": uniqueness,
        "distribution": orientation_distribution_summary(
            matrices,
            euler_sequence=str(sampling["distribution_euler_sequence"]),
        ),
    }
    return records, summary

