from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _load_config_overlay(
    path: Path, *, inheritance_stack: tuple[Path, ...] = ()
) -> dict[str, Any]:
    resolved = path.resolve()
    if resolved in inheritance_stack:
        cycle = " -> ".join(str(value) for value in (*inheritance_stack, resolved))
        raise ValueError(f"Config inheritance cycle: {cycle}")
    with resolved.open("r", encoding="utf-8") as handle:
        overlay = yaml.safe_load(handle)
    if not isinstance(overlay, dict):
        raise ValueError(f"Config overlay must contain a mapping: {resolved}")
    parent = overlay.pop("extends", None)
    if parent is None:
        return overlay
    parent_path = Path(parent)
    if not parent_path.is_absolute():
        parent_path = resolved.parent / parent_path
    inherited = _load_config_overlay(
        parent_path,
        inheritance_stack=(*inheritance_stack, resolved),
    )
    return _deep_merge(inherited, overlay)


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    """Load the frozen base config and, optionally, a version overlay.

    ``OR4D_CONFIG`` is intentionally a path rather than a version name so every
    run manifest can record and hash the exact file that changed the defaults.
    """
    base_path = project_root() / "config" / "benchmark.yaml"
    with base_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    selected = path or os.environ.get("OR4D_CONFIG")
    if selected is None:
        return config
    overlay_path = Path(selected)
    if not overlay_path.is_absolute():
        overlay_path = project_root() / overlay_path
    if overlay_path.resolve() == base_path.resolve():
        return config
    overlay = _load_config_overlay(overlay_path)
    return _deep_merge(config, overlay)


def cif_path(config: dict[str, Any]) -> Path:
    return project_root() / config["dataset"]["cif_path"]


def normalize(v: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    if n < eps:
        raise ValueError("Cannot normalize a near-zero vector.")
    return v / n


def direction_cartesian(lattice_matrix: np.ndarray, uvw: Iterable[float]) -> np.ndarray:
    """Convert a direct-lattice direction [u v w] to Cartesian coordinates.

    pymatgen stores a, b, c as rows of lattice_matrix.
    """
    return np.asarray(uvw, dtype=float) @ np.asarray(lattice_matrix, dtype=float)


def make_orientation_matrix(
    lattice_matrix: np.ndarray,
    zone_axis_uvw: Iterable[float],
    x_reference_uvw: Iterable[float],
    in_plane_rotation_deg: float,
) -> np.ndarray:
    """Return R_sample_to_crystal with columns [x_crystal, y_crystal, z_crystal]."""
    z = normalize(direction_cartesian(lattice_matrix, zone_axis_uvw))
    x_ref = direction_cartesian(lattice_matrix, x_reference_uvw)
    x0 = x_ref - np.dot(x_ref, z) * z
    x0 = normalize(x0)
    y0 = normalize(np.cross(z, x0))

    phi = np.deg2rad(float(in_plane_rotation_deg))
    x = np.cos(phi) * x0 + np.sin(phi) * y0
    y = -np.sin(phi) * x0 + np.cos(phi) * y0
    R = np.column_stack([normalize(x), normalize(y), z])

    if not np.allclose(R.T @ R, np.eye(3), atol=1e-8):
        raise ValueError("Orientation matrix is not orthonormal.")
    if not np.isclose(np.linalg.det(R), 1.0, atol=1e-8):
        raise ValueError("Orientation matrix determinant is not +1.")
    return R



def quaternion_wxyz_to_matrix(quaternion_wxyz: Iterable[float]) -> np.ndarray:
    """Convert a unit quaternion [w, x, y, z] to a proper rotation matrix.

    The returned matrix uses the benchmark convention R_sample_to_crystal.
    """
    q = np.asarray(quaternion_wxyz, dtype=float)
    if q.shape != (4,):
        raise ValueError(f"Quaternion must have shape (4,), got {q.shape}.")
    q = q / np.linalg.norm(q)
    if q[0] < 0:
        q = -q
    w, x, y, z = q
    R = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )
    return nearest_rotation(R)


def matrix_to_quaternion_wxyz(matrix: np.ndarray) -> np.ndarray:
    """Convert a proper rotation matrix to a canonical-sign quaternion."""
    from scipy.spatial.transform import Rotation

    matrix = nearest_rotation(np.asarray(matrix, dtype=float))
    quaternion = Rotation.from_matrix(matrix).as_quat(scalar_first=True)
    nonzero = np.flatnonzero(np.abs(quaternion) > 1e-14)
    if len(nonzero) and quaternion[nonzero[0]] < 0.0:
        quaternion = -quaternion
    return quaternion / np.linalg.norm(quaternion)


def low_discrepancy_so3_quaternion(index: int, offset: float = 0.0) -> np.ndarray:
    """Return a deterministic quasi-uniform SO(3) quaternion [w, x, y, z].

    This is a low-discrepancy variant of Shoemake's uniform quaternion map.
    """
    if index < 0:
        raise ValueError("index must be non-negative")
    golden = (np.sqrt(5.0) - 1.0) / 2.0
    silver = np.sqrt(2.0) - 1.0
    u1 = np.mod((index + 0.5) * golden + offset, 1.0)
    u2 = np.mod((index + 0.5) * silver + 0.5 * offset, 1.0)
    u3 = np.mod((index + 0.5) * (np.sqrt(3.0) - 1.0) + 0.25 * offset, 1.0)
    q_xyzw = np.array(
        [
            np.sqrt(1.0 - u1) * np.sin(2.0 * np.pi * u2),
            np.sqrt(1.0 - u1) * np.cos(2.0 * np.pi * u2),
            np.sqrt(u1) * np.sin(2.0 * np.pi * u3),
            np.sqrt(u1) * np.cos(2.0 * np.pi * u3),
        ],
        dtype=float,
    )
    q_wxyz = q_xyzw[[3, 0, 1, 2]]
    if q_wxyz[0] < 0:
        q_wxyz = -q_wxyz
    return q_wxyz / np.linalg.norm(q_wxyz)


def shoemake_so3_quaternion(unit_cube_point: Iterable[float]) -> np.ndarray:
    """Map one point in [0, 1)^3 to a uniform SO(3) quaternion [w, x, y, z]."""
    u1, u2, u3 = np.asarray(unit_cube_point, dtype=float)
    if not np.all((0.0 <= np.array([u1, u2, u3])) & (np.array([u1, u2, u3]) < 1.0)):
        raise ValueError("Shoemake input coordinates must lie in [0, 1).")
    q_xyzw = np.array(
        [
            np.sqrt(1.0 - u1) * np.sin(2.0 * np.pi * u2),
            np.sqrt(1.0 - u1) * np.cos(2.0 * np.pi * u2),
            np.sqrt(u1) * np.sin(2.0 * np.pi * u3),
            np.sqrt(u1) * np.cos(2.0 * np.pi * u3),
        ],
        dtype=float,
    )
    q_wxyz = q_xyzw[[3, 0, 1, 2]]
    if q_wxyz[0] < 0:
        q_wxyz = -q_wxyz
    return q_wxyz / np.linalg.norm(q_wxyz)


def sobol_so3_quaternions(
    count: int,
    *,
    scramble: bool,
    seed: int,
) -> np.ndarray:
    """Generate a deterministic power-of-two Sobol sample on SO(3)."""
    if count <= 0 or count & (count - 1):
        raise ValueError("Sobol SO(3) count must be a positive power of two.")
    from scipy.stats import qmc

    exponent = count.bit_length() - 1
    points = qmc.Sobol(d=3, scramble=scramble, seed=seed).random_base2(exponent)
    return np.stack([shoemake_so3_quaternion(point) for point in points], axis=0)


def proper_point_group_rotations(structure: Any) -> list[np.ndarray]:
    """Return unique determinant +1 Cartesian point-group operations.

    Improper operations must be rejected before numerical orthogonalization.
    Otherwise ``nearest_rotation`` would turn mirrors or inversion into unrelated
    proper rotations.
    """
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    operations = SpacegroupAnalyzer(structure).get_point_group_operations(
        cartesian=True
    )
    rotations: list[np.ndarray] = []
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


def symmetry_aware_misorientation_deg(
    R_a: np.ndarray,
    R_b: np.ndarray,
    symmetry_rotations: Iterable[np.ndarray],
) -> float:
    """Minimum misorientation between two sample-to-crystal matrices."""
    R_a = nearest_rotation(np.asarray(R_a, dtype=float))
    R_b = nearest_rotation(np.asarray(R_b, dtype=float))
    return min(
        rotation_angle_deg(R_a @ (nearest_rotation(S) @ R_b).T)
        for S in symmetry_rotations
    )


# Right-acting sample-frame rotations used by the Clean-Peak evaluation.
# FRIEDEL_SAMPLE_ROTATION maps every diffraction vector q to -q while
# keeping the beam axis fixed. Ideal kinematical diffraction contains
# Friedel pairs, so R and R @ FRIEDEL_SAMPLE_ROTATION are observationally
# equivalent for the peak-set input used by this benchmark.
FRIEDEL_SAMPLE_ROTATION = np.diag([-1.0, -1.0, 1.0])

# py4DSTEM ACOM represents its optional projected mirror branch by
# negating columns 1 and 2 of the sample-to-crystal matrix, i.e. by
# right-multiplication with this 180 degree sample-x rotation.
ACOM_MIRROR_SAMPLE_ROTATION = np.diag([1.0, -1.0, -1.0])


def canonicalize_clean_orientation(
    matrix: np.ndarray,
    symmetry_rotations: Iterable[np.ndarray],
    *,
    decimals: int = 12,
) -> dict[str, Any]:
    """Choose one deterministic representative of a Clean orientation class.

    The class contains every ``S @ R @ F`` for proper crystal symmetries ``S``
    and the two detector-plane Friedel branches ``F``.  The representative is
    selected by a canonical-sign quaternion key, making construction
    independent of which equivalent raw matrix was sampled.
    """
    raw = nearest_rotation(np.asarray(matrix, dtype=float))
    symmetries = [
        nearest_rotation(np.asarray(symmetry, dtype=float))
        for symmetry in symmetry_rotations
    ]
    candidates: list[dict[str, Any]] = []
    friedel_branches = (np.eye(3), FRIEDEL_SAMPLE_ROTATION)
    for symmetry_index, proper_symmetry in enumerate(symmetries):
        for friedel_index, friedel in enumerate(friedel_branches):
            equivalent = nearest_rotation(proper_symmetry @ raw @ friedel)
            quaternion = matrix_to_quaternion_wxyz(equivalent)
            key = tuple(np.round(quaternion, decimals=decimals).tolist())
            candidates.append(
                {
                    "key": key,
                    "matrix": equivalent,
                    "quaternion_wxyz": quaternion,
                    "crystal_symmetry_index": symmetry_index,
                    "friedel_branch_index": friedel_index,
                    "friedel_used": bool(friedel_index),
                }
            )
    if not candidates:
        raise ValueError("At least one proper crystal symmetry is required.")
    selected = max(candidates, key=lambda item: item["key"])
    canonical = selected["matrix"]
    return {
        "raw_matrix": raw,
        "canonical_matrix": canonical,
        "canonical_quaternion_wxyz": selected["quaternion_wxyz"],
        "orientation_class_key": ",".join(
            f"{value:.{decimals}f}" for value in selected["key"]
        ),
        "crystal_symmetry_index": selected["crystal_symmetry_index"],
        "friedel_branch_index": selected["friedel_branch_index"],
        "friedel_used": selected["friedel_used"],
        "canonicalization_residual_deg": float(
            rotation_angle_deg(
                canonical
                @ (
                    nearest_rotation(
                        symmetries[selected["crystal_symmetry_index"]]
                    )
                    @ raw
                    @ friedel_branches[selected["friedel_branch_index"]]
                ).T
            )
        ),
    }


def friedel_aware_misorientation_deg(
    R_a: np.ndarray,
    R_b: np.ndarray,
    symmetry_rotations: Iterable[np.ndarray],
) -> float:
    """Minimum Clean-Peak misorientation including Friedel equivalence.

    Both matrices use the benchmark sample-to-crystal convention. The
    crystal point-group symmetry acts on the left, while the 180 degree
    detector-plane/Friedel ambiguity acts on the sample frame on the right.
    """
    R_a = nearest_rotation(np.asarray(R_a, dtype=float))
    R_b = nearest_rotation(np.asarray(R_b, dtype=float))
    sample_branches = (np.eye(3), FRIEDEL_SAMPLE_ROTATION)
    return min(
        rotation_angle_deg(
            R_a
            @ (nearest_rotation(S) @ R_b @ sample_branch).T
        )
        for S in symmetry_rotations
        for sample_branch in sample_branches
    )


def best_friedel_alignment(
    R_predicted: np.ndarray,
    R_ground_truth: np.ndarray,
    symmetry_rotations: Iterable[np.ndarray],
) -> dict[str, Any]:
    """Return the representative used by the Clean symmetry-aware metric.

    The evaluator compares ``R_predicted`` against every equivalent
    ``S @ R_ground_truth @ F``.  For visualization, the inverse operations
    must instead be applied to the prediction so it can be drawn in the same
    representative as the ground truth:

    ``R_aligned = S.T @ R_predicted @ F.T``.
    """
    predicted = nearest_rotation(np.asarray(R_predicted, dtype=float))
    ground_truth = nearest_rotation(np.asarray(R_ground_truth, dtype=float))
    candidates: list[tuple[float, np.ndarray, np.ndarray]] = []
    for symmetry in symmetry_rotations:
        proper_symmetry = nearest_rotation(np.asarray(symmetry, dtype=float))
        for friedel in (np.eye(3), FRIEDEL_SAMPLE_ROTATION):
            equivalent_ground_truth = proper_symmetry @ ground_truth @ friedel
            error = rotation_angle_deg(
                predicted @ equivalent_ground_truth.T
            )
            candidates.append((error, proper_symmetry, friedel))
    if not candidates:
        raise ValueError("At least one crystal symmetry rotation is required.")

    error, symmetry, friedel = min(candidates, key=lambda item: item[0])
    symmetry_aligned = nearest_rotation(symmetry.T @ predicted)
    aligned = nearest_rotation(symmetry_aligned @ friedel.T)
    return {
        "aligned_matrix": aligned,
        "symmetry_aligned_matrix": symmetry_aligned,
        "crystal_symmetry": symmetry,
        "friedel_matrix": friedel,
        "friedel_used": bool(
            np.allclose(friedel, FRIEDEL_SAMPLE_ROTATION, atol=1e-10)
        ),
        "equivalent_misorientation_deg": float(error),
        "symmetry_step_misorientation_deg": float(
            rotation_angle_deg(symmetry_aligned @ ground_truth.T)
        ),
        "raw_misorientation_deg": float(
            rotation_angle_deg(predicted @ ground_truth.T)
        ),
    }


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_peak_h5(path: Path, samples: list[dict[str, Any]], attrs: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    offsets = [0]
    qx_parts: list[np.ndarray] = []
    qy_parts: list[np.ndarray] = []
    intensity_parts: list[np.ndarray] = []
    sample_ids: list[str] = []
    peak_diagnostic_keys = sorted(
        {
            key
            for sample in samples
            for key in sample.get("peak_diagnostics", {})
        }
    )
    peak_diagnostic_parts: dict[str, list[np.ndarray]] = {
        key: [] for key in peak_diagnostic_keys
    }
    sample_metadata_keys = sorted(
        {
            key
            for sample in samples
            for key in sample.get("sample_metadata", {})
        }
    )

    for sample in samples:
        sample_ids.append(str(sample["sample_id"]))
        qx = np.asarray(sample["qx"], dtype=np.float32)
        qy = np.asarray(sample["qy"], dtype=np.float32)
        intensity = np.asarray(sample["intensity"], dtype=np.float32)
        if not (len(qx) == len(qy) == len(intensity)):
            raise ValueError(f"Peak array length mismatch for {sample['sample_id']}")
        qx_parts.append(qx)
        qy_parts.append(qy)
        intensity_parts.append(intensity)
        for key in peak_diagnostic_keys:
            values = sample.get("peak_diagnostics", {}).get(key)
            if values is None:
                array = np.full(len(qx), np.nan, dtype=np.float32)
            else:
                array = np.asarray(values, dtype=np.float32)
                if len(array) != len(qx):
                    raise ValueError(
                        f"Peak diagnostic {key} length mismatch for "
                        f"{sample['sample_id']}"
                    )
            peak_diagnostic_parts[key].append(array)
        offsets.append(offsets[-1] + len(qx))

    qx_all = np.concatenate(qx_parts) if qx_parts else np.empty(0, dtype=np.float32)
    qy_all = np.concatenate(qy_parts) if qy_parts else np.empty(0, dtype=np.float32)
    i_all = (
        np.concatenate(intensity_parts)
        if intensity_parts
        else np.empty(0, dtype=np.float32)
    )

    with h5py.File(path, "w") as h5:
        h5.create_dataset("sample_id", data=np.asarray(sample_ids, dtype=h5py.string_dtype("utf-8")))
        peaks = h5.create_group("peaks")
        peaks.create_dataset("qx", data=qx_all, compression="gzip")
        peaks.create_dataset("qy", data=qy_all, compression="gzip")
        peaks.create_dataset("intensity", data=i_all, compression="gzip")
        peaks.create_dataset("offsets", data=np.asarray(offsets, dtype=np.int64))
        if peak_diagnostic_keys:
            diagnostics = peaks.create_group("diagnostics")
            for key in peak_diagnostic_keys:
                values = np.concatenate(peak_diagnostic_parts[key])
                diagnostics.create_dataset(key, data=values, compression="gzip")
        if sample_metadata_keys:
            metadata = h5.create_group("sample_diagnostics")
            for key in sample_metadata_keys:
                values = [
                    sample.get("sample_metadata", {}).get(key, "")
                    for sample in samples
                ]
                if any(isinstance(value, str) for value in values):
                    metadata.create_dataset(
                        key,
                        data=np.asarray(
                            [str(value) for value in values],
                            dtype=h5py.string_dtype("utf-8"),
                        ),
                    )
                else:
                    metadata.create_dataset(
                        key, data=np.asarray(values, dtype=np.float64)
                    )
        for key, value in attrs.items():
            if isinstance(value, (dict, list, tuple)):
                h5.attrs[key] = json.dumps(value, ensure_ascii=False)
            else:
                h5.attrs[key] = value


def read_peak_h5(path: Path) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    with h5py.File(path, "r") as h5:
        ids = [x.decode() if isinstance(x, bytes) else str(x) for x in h5["sample_id"][:]]
        offsets = h5["peaks/offsets"][:]
        qx = h5["peaks/qx"][:]
        qy = h5["peaks/qy"][:]
        intensity = h5["peaks/intensity"][:]
        for i, sample_id in enumerate(ids):
            s, e = int(offsets[i]), int(offsets[i + 1])
            samples.append(
                {
                    "sample_id": sample_id,
                    "qx": qx[s:e],
                    "qy": qy[s:e],
                    "intensity": intensity[s:e],
                }
            )
    return samples


def normalize_intensities(intensity: np.ndarray) -> np.ndarray:
    intensity = np.asarray(intensity, dtype=float)
    max_value = float(np.max(intensity)) if intensity.size else 0.0
    if max_value > 0:
        intensity = intensity / max_value
    return intensity.astype(np.float32)


def nearest_rotation(matrix: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(np.asarray(matrix, dtype=float))
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def rotation_angle_deg(R: np.ndarray) -> float:
    value = (np.trace(R) - 1.0) / 2.0
    value = float(np.clip(value, -1.0, 1.0))
    return float(np.degrees(np.arccos(value)))
