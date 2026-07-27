#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from or4d_common import (  # noqa: E402
    cif_path,
    load_config,
    low_discrepancy_so3_quaternion,
    make_orientation_matrix,
    nearest_rotation,
    quaternion_wxyz_to_matrix,
    symmetry_aware_misorientation_deg,
    write_jsonl,
)


def proper_point_group_rotations(structure: Structure) -> list[np.ndarray]:
    operations = SpacegroupAnalyzer(structure).get_point_group_operations(cartesian=True)
    rotations: list[np.ndarray] = []
    for operation in operations:
        matrix = nearest_rotation(np.asarray(operation.rotation_matrix, dtype=float))
        if np.linalg.det(matrix) > 0:
            rotations.append(matrix)
    return rotations


def main() -> None:
    config = load_config()
    structure = Structure.from_file(cif_path(config))
    lattice_matrix = np.asarray(structure.lattice.matrix, dtype=float)
    symmetries = proper_point_group_rotations(structure)

    records: list[dict] = []
    accepted_matrices: list[np.ndarray] = []

    # Hand-designed low-index and non-zero in-plane samples.
    for item in config["orientations"]:
        R = make_orientation_matrix(
            lattice_matrix=lattice_matrix,
            zone_axis_uvw=item["zone_axis_uvw"],
            x_reference_uvw=item["x_reference_uvw"],
            in_plane_rotation_deg=item["in_plane_rotation_deg"],
        )
        records.append(
            {
                "orientation_id": item["orientation_id"],
                "sampling_type": item.get("sampling_type", "manual"),
                "orientation_matrix_sample_to_crystal": R.tolist(),
                "zone_axis_3index": item["zone_axis_uvw"],
                "x_reference_uvw": item["x_reference_uvw"],
                "in_plane_rotation_deg": float(item["in_plane_rotation_deg"]),
                "beam_direction_crystal_cartesian": R[:, 2].tolist(),
            }
        )
        accepted_matrices.append(R)

    # Deterministic quasi-uniform SO(3) samples, filtered under crystal symmetry.
    sampling = config.get("so3_sampling", {})
    if sampling.get("enabled", False):
        count = int(sampling["count"])
        start = int(sampling["orientation_id_start"])
        offset = float(sampling.get("sequence_offset", 0.0))
        min_pairwise = float(sampling.get("min_pairwise_misorientation_deg", 0.0))
        max_candidates = int(sampling.get("max_candidates", 10000))

        generated = 0
        candidate_index = 0
        while generated < count and candidate_index < max_candidates:
            quaternion = low_discrepancy_so3_quaternion(candidate_index, offset=offset)
            R = quaternion_wxyz_to_matrix(quaternion)
            candidate_index += 1

            minimum = min(
                symmetry_aware_misorientation_deg(R, other, symmetries)
                for other in accepted_matrices
            )
            if minimum < min_pairwise:
                continue

            orientation_id = f"ori_{start + generated:03d}"
            records.append(
                {
                    "orientation_id": orientation_id,
                    "sampling_type": "so3_low_discrepancy",
                    "orientation_matrix_sample_to_crystal": R.tolist(),
                    "quaternion_wxyz": quaternion.tolist(),
                    "zone_axis_3index": None,
                    "x_reference_uvw": None,
                    "in_plane_rotation_deg": None,
                    "beam_direction_crystal_cartesian": R[:, 2].tolist(),
                    "nearest_previous_symmetry_misorientation_deg": float(minimum),
                    "sequence_candidate_index": candidate_index - 1,
                }
            )
            accepted_matrices.append(R)
            generated += 1

        if generated != count:
            raise RuntimeError(
                f"Generated only {generated}/{count} SO(3) samples. "
                "Reduce min_pairwise_misorientation_deg or increase max_candidates."
            )

    output = ROOT / "private" / "orientations.jsonl"
    write_jsonl(output, records)
    print(f"Wrote {len(records)} orientations to {output}")
    for record in records:
        matrix = np.asarray(record["orientation_matrix_sample_to_crystal"], dtype=float)
        det = np.linalg.det(matrix)
        print(
            record["orientation_id"],
            f"type={record['sampling_type']}",
            f"zone={record.get('zone_axis_3index')}",
            f"phi={record.get('in_plane_rotation_deg')}",
            f"det={det:.8f}",
        )


if __name__ == "__main__":
    main()
