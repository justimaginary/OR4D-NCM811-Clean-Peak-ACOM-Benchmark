from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from or4d_common import (
    canonicalize_clean_orientation,
    make_orientation_matrix,
)


def _axis_angle_rotation(axis: Iterable[float], angle_deg: float) -> np.ndarray:
    axis_array = np.asarray(axis, dtype=float)
    axis_array /= np.linalg.norm(axis_array)
    x, y, z = axis_array
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    angle = np.deg2rad(float(angle_deg))
    return (
        np.eye(3)
        + np.sin(angle) * skew
        + (1.0 - np.cos(angle)) * (skew @ skew)
    )


def _controlled_orientation(
    lattice_matrix: np.ndarray,
    zone_axis_uvw: Iterable[int],
    x_reference_uvw: Iterable[int],
    in_plane_deg: float,
    tilt_deg: float,
    tilt_azimuth_deg: float,
) -> np.ndarray:
    base = make_orientation_matrix(
        lattice_matrix,
        zone_axis_uvw,
        x_reference_uvw,
        in_plane_deg,
    )
    azimuth = np.deg2rad(float(tilt_azimuth_deg))
    tilt_axis_sample = [np.cos(azimuth), np.sin(azimuth), 0.0]
    return base @ _axis_angle_rotation(tilt_axis_sample, tilt_deg)


def build_001_study_records(
    config: dict[str, Any],
    structure: Any,
    symmetry_rotations: list[np.ndarray],
) -> list[dict[str, Any]]:
    """Build the independent, non-headline [001] diagnostic cohort."""
    study = config["v5"]["study_001"]
    if not study.get("enabled", False):
        return []
    lattice = np.asarray(structure.lattice.matrix, dtype=float)
    raw_rows: list[dict[str, Any]] = []

    exact_count = int(study["exact_001_count"])
    for index, in_plane in enumerate(
        np.linspace(0.0, 60.0, exact_count, endpoint=False)
    ):
        raw_rows.append(
            {
                "orientation_id": f"v5_001_exact_{index:03d}",
                "study_group": "exact_001",
                "zone_axis_3index": [0, 0, 1],
                "tilt_deg": 0.0,
                "tilt_azimuth_deg": 0.0,
                "in_plane_rotation_deg": float(in_plane),
                "raw_matrix": _controlled_orientation(
                    lattice, [0, 0, 1], [1, 0, 0], in_plane, 0.0, 0.0
                ),
            }
        )

    local_azimuths = np.linspace(
        0.0, 360.0, int(study["local_tilt_azimuth_count"]), endpoint=False
    )
    local_in_planes = np.linspace(
        0.0, 60.0, int(study["local_in_plane_count"]), endpoint=False
    )
    for tilt in study["local_tilt_deg"]:
        for azimuth in local_azimuths:
            for in_plane in local_in_planes:
                raw_rows.append(
                    {
                        "orientation_id": f"v5_001_local_{len(raw_rows):04d}",
                        "study_group": "near_001",
                        "zone_axis_3index": [0, 0, 1],
                        "tilt_deg": float(tilt),
                        "tilt_azimuth_deg": float(azimuth),
                        "in_plane_rotation_deg": float(in_plane),
                        "raw_matrix": _controlled_orientation(
                            lattice,
                            [0, 0, 1],
                            [1, 0, 0],
                            in_plane,
                            tilt,
                            azimuth,
                        ),
                    }
                )

    transition_azimuths = np.linspace(
        0.0,
        360.0,
        int(study["transition_azimuth_count"]),
        endpoint=False,
    )
    transition_in_planes = np.linspace(
        0.0,
        60.0,
        int(study["transition_in_plane_count"]),
        endpoint=False,
    )
    for tilt in study["transition_tilt_deg"]:
        for azimuth in transition_azimuths:
            for in_plane in transition_in_planes:
                raw_rows.append(
                    {
                        "orientation_id": (
                            f"v5_001_transition_{len(raw_rows):04d}"
                        ),
                        "study_group": "transition_001",
                        "zone_axis_3index": [0, 0, 1],
                        "tilt_deg": float(tilt),
                        "tilt_azimuth_deg": float(azimuth),
                        "in_plane_rotation_deg": float(in_plane),
                        "raw_matrix": _controlled_orientation(
                            lattice,
                            [0, 0, 1],
                            [1, 0, 0],
                            in_plane,
                            tilt,
                            azimuth,
                        ),
                    }
                )

    control_specs = (
        ("control_100", [1, 0, 0], [0, 0, 1]),
        ("control_110", [1, 1, 0], [0, 0, 1]),
    )
    control_tilts = (0.25, 0.5, 1.0, 2.0)
    control_azimuths = np.linspace(0.0, 360.0, 8, endpoint=False)
    control_in_planes = np.linspace(0.0, 60.0, 8, endpoint=False)
    for group, zone_axis, x_reference in control_specs:
        for tilt in control_tilts:
            for azimuth in control_azimuths:
                for in_plane in control_in_planes:
                    raw_rows.append(
                        {
                            "orientation_id": f"v5_001_{group}_{len(raw_rows):04d}",
                            "study_group": group,
                            "zone_axis_3index": zone_axis,
                            "tilt_deg": tilt,
                            "tilt_azimuth_deg": azimuth,
                            "in_plane_rotation_deg": in_plane,
                            "raw_matrix": _controlled_orientation(
                                lattice,
                                zone_axis,
                                x_reference,
                                in_plane,
                                tilt,
                                azimuth,
                            ),
                        }
                    )
    records: list[dict[str, Any]] = []
    class_ids: set[str] = set()
    group_targets = {
        "exact_001": int(study["exact_001_count"]),
        "near_001": (
            len(study["local_tilt_deg"])
            * int(study["local_tilt_azimuth_count"])
            * int(study["local_in_plane_count"])
        ),
        "transition_001": (
            len(study["transition_tilt_deg"])
            * int(study["transition_azimuth_count"])
            * int(study["transition_in_plane_count"])
        ),
        "control_100": int(study["control_100_count"]),
        "control_110": int(study["control_110_count"]),
    }
    group_counts = {group: 0 for group in group_targets}
    for row in raw_rows:
        group = str(row["study_group"])
        if group_counts[group] >= group_targets[group]:
            continue
        canonical = canonicalize_clean_orientation(
            row.pop("raw_matrix"), symmetry_rotations
        )
        class_id = str(canonical["orientation_class_key"])
        if class_id in class_ids:
            if group.startswith("control_"):
                continue
            raise RuntimeError(
                f"Duplicate [001] study orientation class: {row['orientation_id']}"
            )
        class_ids.add(class_id)
        group_counts[group] += 1
        raw_matrix = canonical["raw_matrix"]
        canonical_matrix = canonical["canonical_matrix"]
        records.append(
            {
                **row,
                "sample_id": f"clean_{row['orientation_id']}",
                "sample_role": "study_001",
                "orientation_class_id": class_id,
                "raw_orientation_matrix_sample_to_crystal": raw_matrix.tolist(),
                "orientation_matrix_sample_to_crystal": canonical_matrix.tolist(),
                "canonical_quaternion_wxyz": canonical[
                    "canonical_quaternion_wxyz"
                ].tolist(),
                "canonical_crystal_symmetry_index": int(
                    canonical["crystal_symmetry_index"]
                ),
                "canonical_friedel_branch_index": int(
                    canonical["friedel_branch_index"]
                ),
                "canonicalization_residual_deg": float(
                    canonical["canonicalization_residual_deg"]
                ),
                "beam_direction_crystal_cartesian": canonical_matrix[:, 2].tolist(),
            }
        )

    expected_total = int(study["sample_count"])
    if group_counts != group_targets:
        raise RuntimeError(
            f"[001] study group counts {group_counts}, expected {group_targets}."
        )
    if len(records) != expected_total:
        raise RuntimeError(
            f"[001] study produced {len(records)} records, expected {expected_total}."
        )
    return records
