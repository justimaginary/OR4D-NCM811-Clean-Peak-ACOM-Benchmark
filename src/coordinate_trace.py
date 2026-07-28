from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment


def reciprocal_cartesian_from_hkl(
    hkl: np.ndarray,
    reciprocal_lattice_matrix: np.ndarray,
) -> np.ndarray:
    """Map crystallographic Miller indices to Cartesian reciprocal coordinates.

    pymatgen stores reciprocal basis vectors as rows and the crystallographic
    reciprocal lattice is expressed in 1/angstrom without a 2*pi factor.
    """
    return (
        np.asarray(hkl, dtype=float)
        @ np.asarray(reciprocal_lattice_matrix, dtype=float)
    )


def crystal_to_sample_reciprocal(
    reciprocal_crystal_cartesian: np.ndarray,
    orientation_sample_to_crystal: np.ndarray,
) -> np.ndarray:
    """Express a crystal Cartesian reciprocal vector in sample coordinates."""
    return (
        np.asarray(orientation_sample_to_crystal, dtype=float).T
        @ np.asarray(reciprocal_crystal_cartesian, dtype=float)
    )


def reflection_coordinate_record(
    hkl: np.ndarray,
    reciprocal_lattice_matrix: np.ndarray,
    matrix_gt: np.ndarray,
    matrix_acom: np.ndarray,
    *,
    reported_qx: float,
    reported_qy: float,
    reported_intensity: float,
) -> dict[str, Any]:
    hkl_array = np.asarray(hkl, dtype=int)
    g_crystal = reciprocal_cartesian_from_hkl(
        hkl_array,
        reciprocal_lattice_matrix,
    )
    g_sample_gt = crystal_to_sample_reciprocal(g_crystal, matrix_gt)
    g_sample_acom = crystal_to_sample_reciprocal(g_crystal, matrix_acom)
    reported = np.array([reported_qx, reported_qy], dtype=float)
    residual = reported - g_sample_gt[:2]
    return {
        "hkl": hkl_array.tolist(),
        "g_crystal_cartesian_Ainv": g_crystal.tolist(),
        "standard_g_sample_Ainv": g_sample_gt.tolist(),
        "standard_qx_Ainv": float(g_sample_gt[0]),
        "standard_qy_Ainv": float(g_sample_gt[1]),
        "standard_qz_Ainv": float(g_sample_gt[2]),
        "reported_qx_Ainv": float(reported_qx),
        "reported_qy_Ainv": float(reported_qy),
        "reported_intensity_normalized": float(reported_intensity),
        "reported_minus_standard_qxy_Ainv": residual.tolist(),
        "reported_minus_standard_qxy_norm_Ainv": float(
            np.linalg.norm(residual)
        ),
        "acom_same_hkl_g_sample_Ainv": g_sample_acom.tolist(),
        "acom_same_hkl_minus_standard_qxy_Ainv": (
            g_sample_acom[:2] - g_sample_gt[:2]
        ).tolist(),
    }


def match_detector_peaks(
    observed: dict[str, np.ndarray],
    predicted: dict[str, np.ndarray],
    tolerance_Ainv: float,
) -> dict[str, Any]:
    """One-to-one q-space assignment, then reject pairs beyond the tolerance."""
    observed_q = np.column_stack([observed["qx"], observed["qy"]]).astype(float)
    predicted_q = np.column_stack([predicted["qx"], predicted["qy"]]).astype(float)
    if len(observed_q) == 0 or len(predicted_q) == 0:
        return {
            "matches": [],
            "unmatched_observed_indices": list(range(len(observed_q))),
            "unmatched_predicted_indices": list(range(len(predicted_q))),
        }

    distances = np.linalg.norm(
        observed_q[:, None, :] - predicted_q[None, :, :],
        axis=2,
    )
    observed_indices, predicted_indices = linear_sum_assignment(distances)
    accepted: list[dict[str, Any]] = []
    accepted_observed: set[int] = set()
    accepted_predicted: set[int] = set()
    for observed_index, predicted_index in zip(
        observed_indices,
        predicted_indices,
    ):
        distance = float(distances[observed_index, predicted_index])
        if distance > tolerance_Ainv:
            continue
        observed_hkl = np.asarray(observed["hkl"][observed_index], dtype=int)
        predicted_hkl = np.asarray(predicted["hkl"][predicted_index], dtype=int)
        accepted_observed.add(int(observed_index))
        accepted_predicted.add(int(predicted_index))
        accepted.append(
            {
                "observed_index": int(observed_index),
                "predicted_index": int(predicted_index),
                "observed_hkl": observed_hkl.tolist(),
                "predicted_hkl": predicted_hkl.tolist(),
                "raw_hkl_equal": bool(np.array_equal(observed_hkl, predicted_hkl)),
                "raw_hkl_friedel_equal": bool(
                    np.array_equal(observed_hkl, -predicted_hkl)
                ),
                "observed_qxy_Ainv": observed_q[observed_index].tolist(),
                "predicted_qxy_Ainv": predicted_q[predicted_index].tolist(),
                "predicted_minus_observed_qxy_Ainv": (
                    predicted_q[predicted_index] - observed_q[observed_index]
                ).tolist(),
                "q_distance_Ainv": distance,
                "observed_intensity_normalized": float(
                    observed["intensity"][observed_index]
                ),
                "predicted_intensity_normalized": float(
                    predicted["intensity"][predicted_index]
                ),
                "predicted_minus_observed_intensity": float(
                    predicted["intensity"][predicted_index]
                    - observed["intensity"][observed_index]
                ),
            }
        )
    return {
        "matches": accepted,
        "unmatched_observed_indices": sorted(
            set(range(len(observed_q))) - accepted_observed
        ),
        "unmatched_predicted_indices": sorted(
            set(range(len(predicted_q))) - accepted_predicted
        ),
    }
