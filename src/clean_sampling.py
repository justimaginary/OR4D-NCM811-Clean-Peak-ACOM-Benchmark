from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from or4d_common import (
    canonicalize_clean_orientation,
    friedel_aware_misorientation_deg,
    low_discrepancy_so3_quaternion,
    make_orientation_matrix,
    proper_point_group_rotations,
    quaternion_wxyz_to_matrix,
    sobol_so3_quaternions,
)


def _minimum_clean_equivalent_distance(
    matrix: np.ndarray,
    records: list[dict[str, Any]],
    symmetries: list[np.ndarray],
) -> float | None:
    if not records:
        return None
    return min(
        friedel_aware_misorientation_deg(
            matrix,
            np.asarray(record["orientation_matrix_sample_to_crystal"], dtype=float),
            symmetries,
        )
        for record in records
    )


def _append_unique(
    records: list[dict[str, Any]],
    record: dict[str, Any],
    symmetries: list[np.ndarray],
    duplicate_tolerance_deg: float,
) -> None:
    orientation_id = str(record["orientation_id"])
    if any(existing["orientation_id"] == orientation_id for existing in records):
        raise ValueError(f"Duplicate orientation_id: {orientation_id}")

    orientation_class_id = record.get("orientation_class_id")
    if orientation_class_id is not None:
        if any(
            existing.get("orientation_class_id") == orientation_class_id
            for existing in records
        ):
            raise ValueError(
                f"{orientation_id} duplicates an earlier canonical "
                "Clean orientation class."
            )
        record["nearest_previous_clean_equivalent_misorientation_deg"] = None
        records.append(record)
        return

    matrix = np.asarray(record["orientation_matrix_sample_to_crystal"], dtype=float)
    minimum = _minimum_clean_equivalent_distance(matrix, records, symmetries)
    if minimum is not None and minimum <= duplicate_tolerance_deg:
        raise ValueError(
            f"{orientation_id} duplicates an earlier Clean-equivalent orientation "
            f"within {minimum:.9g} degrees."
        )
    record["nearest_previous_clean_equivalent_misorientation_deg"] = minimum
    records.append(record)


def _manual_records(
    config: dict[str, Any],
    lattice_matrix: np.ndarray,
) -> list[dict[str, Any]]:
    if not config.get("legacy_manual", {}).get("enabled", True):
        return []
    records: list[dict[str, Any]] = []
    for item in config["orientations"]:
        matrix = make_orientation_matrix(
            lattice_matrix=lattice_matrix,
            zone_axis_uvw=item["zone_axis_uvw"],
            x_reference_uvw=item["x_reference_uvw"],
            in_plane_rotation_deg=item["in_plane_rotation_deg"],
        )
        records.append(
            {
                "orientation_id": item["orientation_id"],
                "sampling_type": item.get("sampling_type", "manual"),
                "sample_role": "legacy_smoke",
                "acom_offgrid_policy": "report_only",
                "orientation_matrix_sample_to_crystal": matrix.tolist(),
                "zone_axis_3index": item["zone_axis_uvw"],
                "x_reference_uvw": item["x_reference_uvw"],
                "in_plane_rotation_deg": float(item["in_plane_rotation_deg"]),
                "beam_direction_crystal_cartesian": matrix[:, 2].tolist(),
            }
        )
    return records


def _legacy_so3_records(config: dict[str, Any]) -> list[dict[str, Any]]:
    sampling = config.get("legacy_so3_sampling", {})
    if not sampling.get("enabled", False):
        return []

    start = int(sampling["orientation_id_start"])
    offset = float(sampling.get("sequence_offset", 0.0))
    records: list[dict[str, Any]] = []
    for output_index, candidate_index in enumerate(sampling["candidate_indices"]):
        quaternion = low_discrepancy_so3_quaternion(
            int(candidate_index),
            offset=offset,
        )
        matrix = quaternion_wxyz_to_matrix(quaternion)
        records.append(
            {
                "orientation_id": f"ori_{start + output_index:03d}",
                "sampling_type": "so3_low_discrepancy",
                "sample_role": "legacy_smoke",
                "acom_offgrid_policy": "report_only",
                "orientation_matrix_sample_to_crystal": matrix.tolist(),
                "quaternion_wxyz": quaternion.tolist(),
                "zone_axis_3index": None,
                "x_reference_uvw": None,
                "in_plane_rotation_deg": None,
                "beam_direction_crystal_cartesian": matrix[:, 2].tolist(),
                "sequence_candidate_index": int(candidate_index),
            }
        )
    return records


def _headline_core_records(config: dict[str, Any]) -> list[dict[str, Any]]:
    sampling = config["clean_sampling"]["headline_core"]
    if not sampling.get("enabled", False):
        return []

    count = int(sampling["count"])
    quaternions = sobol_so3_quaternions(
        count,
        scramble=bool(sampling["scramble"]),
        seed=int(sampling["seed"]),
    )
    prefix = str(sampling["orientation_id_prefix"])
    records: list[dict[str, Any]] = []
    for index, quaternion in enumerate(quaternions):
        matrix = quaternion_wxyz_to_matrix(quaternion)
        records.append(
            {
                "orientation_id": f"{prefix}{index:04d}",
                "sampling_type": "sobol_so3",
                "sample_role": "headline_core",
                "acom_offgrid_policy": "report_only",
                "orientation_matrix_sample_to_crystal": matrix.tolist(),
                "quaternion_wxyz": quaternion.tolist(),
                "zone_axis_3index": None,
                "x_reference_uvw": None,
                "in_plane_rotation_deg": None,
                "beam_direction_crystal_cartesian": matrix[:, 2].tolist(),
                "sequence_index": index,
            }
        )
    return records


def _acom_probe_records(
    config: dict[str, Any],
    lattice_matrix: np.ndarray,
) -> list[dict[str, Any]]:
    sampling = config["clean_sampling"]["acom_grid_probes"]
    if not sampling.get("enabled", False):
        return []

    prefix = str(sampling["orientation_id_prefix"])
    offsets = [float(value) for value in sampling["in_plane_offsets_deg"]]
    records: list[dict[str, Any]] = []
    probe_index = 0
    for base in sampling["bases"]:
        for offset in offsets:
            matrix = make_orientation_matrix(
                lattice_matrix=lattice_matrix,
                zone_axis_uvw=base["zone_axis_uvw"],
                x_reference_uvw=base["x_reference_uvw"],
                in_plane_rotation_deg=offset,
            )
            records.append(
                {
                    "orientation_id": f"{prefix}{probe_index:03d}",
                    "sampling_type": "acom_in_plane_probe",
                    "sample_role": "acom_grid_probe",
                    "acom_offgrid_policy": "require_discrete_seed",
                    "orientation_matrix_sample_to_crystal": matrix.tolist(),
                    "zone_axis_3index": base["zone_axis_uvw"],
                    "x_reference_uvw": base["x_reference_uvw"],
                    "in_plane_rotation_deg": offset,
                    "beam_direction_crystal_cartesian": matrix[:, 2].tolist(),
                    "probe_axis_id": base["probe_axis_id"],
                    "probe_offset_deg": offset,
                    "probe_index": probe_index,
                }
            )
            probe_index += 1
    return records


def build_clean_orientation_records(
    config: dict[str, Any],
    structure: Any,
) -> list[dict[str, Any]]:
    """Build all v2 Clean orientation cohorts without consulting ACOM."""
    lattice_matrix = np.asarray(structure.lattice.matrix, dtype=float)
    symmetries = proper_point_group_rotations(structure)
    duplicate_tolerance = float(
        config["clean_sampling"]["headline_core"]["duplicate_tolerance_deg"]
    )

    candidates = [
        *_manual_records(config, lattice_matrix),
        *_legacy_so3_records(config),
        *_headline_core_records(config),
        *_acom_probe_records(config, lattice_matrix),
    ]
    if config["clean_sampling"]["headline_core"].get(
        "canonicalize_friedel", False
    ):
        for record in candidates:
            raw = np.asarray(
                record["orientation_matrix_sample_to_crystal"], dtype=float
            )
            canonical = canonicalize_clean_orientation(raw, symmetries)
            record["raw_orientation_matrix_sample_to_crystal"] = raw.tolist()
            record["orientation_matrix_sample_to_crystal"] = canonical[
                "canonical_matrix"
            ].tolist()
            record["canonical_quaternion_wxyz"] = canonical[
                "canonical_quaternion_wxyz"
            ].tolist()
            record["orientation_class_id"] = canonical[
                "orientation_class_key"
            ]
            record["canonical_crystal_symmetry_index"] = int(
                canonical["crystal_symmetry_index"]
            )
            record["canonical_friedel_branch_index"] = int(
                canonical["friedel_branch_index"]
            )
            record["canonical_friedel_used"] = bool(
                canonical["friedel_used"]
            )
            record["canonicalization_residual_deg"] = float(
                canonical["canonicalization_residual_deg"]
            )
            record["beam_direction_crystal_cartesian"] = canonical[
                "canonical_matrix"
            ][:, 2].tolist()
    records: list[dict[str, Any]] = []
    for record in candidates:
        _append_unique(
            records,
            record,
            symmetries,
            duplicate_tolerance_deg=duplicate_tolerance,
        )

    expected_counts = {
        str(role): int(count)
        for role, count in config["dataset"]["expected_sample_counts"].items()
        if int(count) > 0
    }
    actual_counts = Counter(record["sample_role"] for record in records)
    if dict(actual_counts) != expected_counts:
        raise RuntimeError(
            f"Unexpected sample-role counts: actual={dict(actual_counts)}, "
            f"expected={expected_counts}."
        )
    expected_total = int(config["dataset"]["expected_num_orientations"])
    if len(records) != expected_total:
        raise RuntimeError(
            f"Generated {len(records)} orientations, expected {expected_total}."
        )
    return records
