#!/usr/bin/env python3
"""Write the self-contained V5 Clean benchmark visualization.

The page is generated from saved HDF5/JSON results. It never reruns indexing.
Large arrays are reduced to a small set of traceable, real diagnostic cases.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import yaml
from PIL import Image
from pymatgen.core import Structure

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clean_counting import add_gaussian_read_noise  # noqa: E402
from or4d_common import (  # noqa: E402
    FRIEDEL_SAMPLE_ROTATION,
    cif_path,
    load_config,
    proper_point_group_rotations,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Server/local V5 root containing datasets/, intermediates/, results/.",
    )
    parser.add_argument(
        "--acom-summary",
        type=Path,
        default=ROOT / "reports/v5/topk/V5_ACOM_TOP5_FULL_SUMMARY.json",
    )
    parser.add_argument(
        "--pyxem-summary",
        type=Path,
        default=ROOT / "reports/v5/topk/V5_PYXEM_TOP5_FULL_SUMMARY.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports/v5/ACOM_CLEAN_V5_VISUALIZATION.html",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def decode(values: np.ndarray) -> list[str]:
    return [
        value.decode() if isinstance(value, bytes) else str(value)
        for value in values
    ]


def rounded(value: object, digits: int = 6) -> object:
    if isinstance(value, (float, np.floating)):
        return round(float(value), digits)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, np.ndarray):
        return rounded(value.tolist(), digits)
    if isinstance(value, list):
        return [rounded(item, digits) for item in value]
    if isinstance(value, dict):
        return {str(key): rounded(item, digits) for key, item in value.items()}
    return value


def image_data_url(array: np.ndarray) -> str:
    values = np.asarray(array, dtype=np.float64)
    # Negative read-noise values are real stored/reconstructed values but cannot
    # be represented by the log display. Clipping is display-only and disclosed.
    display = np.maximum(values, 0.0)
    positive = display[display > 0]
    reference = float(np.percentile(positive, 50)) if positive.size else 1.0
    scaled = np.log1p(display / max(reference, 1e-12))
    nonzero = scaled[scaled > 0]
    ceiling = float(np.percentile(nonzero, 99.7)) if nonzero.size else 1.0
    normalized = np.clip(scaled / max(ceiling, 1e-12), 0.0, 1.0) ** 0.65
    rgb = np.empty((*normalized.shape, 3), dtype=np.uint8)
    rgb[..., 0] = (248 - 222 * normalized).astype(np.uint8)
    rgb[..., 1] = (250 - 205 * normalized).astype(np.uint8)
    rgb[..., 2] = (253 - 145 * normalized).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(
        buffer, format="PNG", optimize=True
    )
    return "data:image/png;base64," + base64.b64encode(
        buffer.getvalue()
    ).decode("ascii")


def rotation_angle_deg(relative: np.ndarray) -> float:
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.rad2deg(np.arccos(cosine)))


def best_alignment(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    symmetries: list[np.ndarray],
) -> dict:
    best: dict | None = None
    identity = np.eye(3)
    for symmetry_index, symmetry in enumerate(symmetries):
        for friedel_used, friedel in (
            (False, identity),
            (True, FRIEDEL_SAMPLE_ROTATION),
        ):
            equivalent = symmetry @ ground_truth @ friedel
            error = rotation_angle_deg(predicted @ equivalent.T)
            if best is None or error < best["error_deg"]:
                best = {
                    "error_deg": error,
                    "symmetry_index": symmetry_index,
                    "friedel_used": friedel_used,
                    "aligned_matrix": symmetry.T @ predicted @ friedel.T,
                }
    if best is None:
        raise RuntimeError("no alignment candidate")
    return best


def ragged_peaks(path: Path, sample_id: str) -> list[dict]:
    with h5py.File(path, "r") as handle:
        ids = decode(handle["sample_id"][:])
        sample_index = ids.index(sample_id)
        offsets = handle["peaks/offsets"]
        start, stop = (
            int(offsets[sample_index]),
            int(offsets[sample_index + 1]),
        )
        return [
            {"qx": float(qx), "qy": float(qy), "intensity": float(intensity)}
            for qx, qy, intensity in zip(
                handle["peaks/qx"][start:stop],
                handle["peaks/qy"][start:stop],
                handle["peaks/intensity"][start:stop],
            )
        ]


def reflection_trace(
    trace_path: Path,
    sample_index: int,
) -> tuple[list[dict], np.ndarray]:
    with h5py.File(trace_path, "r") as handle:
        offsets = handle["reflections/offsets"]
        start, stop = (
            int(offsets[sample_index]),
            int(offsets[sample_index + 1]),
        )
        rows = []
        for index in range(start, stop):
            rows.append(
                {
                    "hkl": handle["reflections/hkl"][index].astype(int).tolist(),
                    "g_crystal_Ainv": handle[
                        "reflections/g_crystal_Ainv"
                    ][index].astype(float).tolist(),
                    "g_sample_Ainv": handle[
                        "reflections/g_sample_Ainv"
                    ][index].astype(float).tolist(),
                    "q_Ainv": [
                        float(handle["reflections/qx_Ainv"][index]),
                        float(handle["reflections/qy_Ainv"][index]),
                        float(handle["reflections/qz_Ainv"][index]),
                    ],
                    "excitation_error_Ainv": float(
                        handle["reflections/excitation_error_center_Ainv"][
                            index
                        ]
                    ),
                    "structure_factor": [
                        float(handle["reflections/structure_factor_real"][index]),
                        float(handle["reflections/structure_factor_imag"][index]),
                    ],
                    "intensity_raw": float(
                        handle["reflections/intensity_raw"][index]
                    ),
                    "intensity_normalized": float(
                        handle["reflections/intensity_normalized"][index]
                    ),
                }
            )
        rows.sort(key=lambda row: row["intensity_normalized"], reverse=True)
        basis = np.asarray(
            handle["crystallography/reciprocal_basis_B_Ainv"][:], dtype=float
        )
    return rows, basis


def match_peaks(
    oracle: list[dict],
    detected: list[dict],
    q_per_px: float,
) -> dict:
    candidates = sorted(
        (
            (
                (o["qx"] - d["qx"]) ** 2 + (o["qy"] - d["qy"]) ** 2,
                oi,
                di,
            )
            for oi, o in enumerate(oracle)
            for di, d in enumerate(detected)
        ),
        key=lambda item: item[0],
    )
    used_o: set[int] = set()
    used_d: set[int] = set()
    matches = []
    for distance2, oracle_index, detected_index in candidates:
        distance_px = distance2**0.5 / q_per_px
        if distance_px > 1.0:
            break
        if oracle_index in used_o or detected_index in used_d:
            continue
        used_o.add(oracle_index)
        used_d.add(detected_index)
        matches.append(
            {
                "oracle_index": oracle_index,
                "detected_index": detected_index,
                "distance_px": distance_px,
            }
        )
    true_positive = len(matches)
    distances = [row["distance_px"] for row in matches]
    return {
        "oracle_count": len(oracle),
        "detected_count": len(detected),
        "true_positive": true_positive,
        "false_positive": len(detected) - true_positive,
        "false_negative": len(oracle) - true_positive,
        "precision": (
            true_positive / len(detected) if detected else 0.0
        ),
        "recall": true_positive / len(oracle) if oracle else 0.0,
        "position_rmse_px": (
            float(np.sqrt(np.mean(np.square(distances))))
            if distances
            else float("nan")
        ),
        "matches": matches,
    }


def candidate_rows(
    path: Path,
    sample_id: str,
    ground_truth: np.ndarray,
    symmetries: list[np.ndarray],
) -> list[dict]:
    with h5py.File(path, "r") as handle:
        ids = decode(handle["sample_id"][:])
        if sample_id not in ids:
            return []
        sample_index = ids.index(sample_id)
        result = []
        for rank in range(5):
            predicted = np.asarray(
                handle["orientation_matrix_sample_to_crystal"][
                    sample_index, rank
                ],
                dtype=float,
            )
            alignment = best_alignment(predicted, ground_truth, symmetries)
            result.append(
                {
                    "rank": rank + 1,
                    "correlation": float(
                        handle["correlation"][sample_index, rank]
                    ),
                    "strict_error_deg": float(
                        handle["strict_misorientation_deg"][
                            sample_index, rank
                        ]
                    ),
                    "friedel_error_deg": float(
                        handle["friedel_equivalent_misorientation_deg"][
                            sample_index, rank
                        ]
                    ),
                    "mirror_match": bool(
                        handle["mirror_match"][sample_index, rank]
                    ),
                    "matrix": predicted.tolist(),
                    **alignment,
                }
            )
    return result


def pyxem_candidate_rows(
    handle: h5py.File,
    *,
    sample_index: int,
    ground_truth: np.ndarray,
    symmetries: list[np.ndarray],
    track: str,
    dose_index: int | None,
    noise_index: int | None,
    repeat: int | None,
) -> list[dict]:
    if track == "expectation":
        matrices = handle["clean_e/orientation_matrix_sample_to_crystal"][
            sample_index
        ]
        correlations = handle["clean_e/correlation"][sample_index]
        mirrors = handle["clean_e/mirrored_template"][sample_index]
    elif noise_index is None:
        matrices = handle[
            "clean_c_noiseless/orientation_matrix_sample_to_crystal"
        ][dose_index, sample_index]
        correlations = handle["clean_c_noiseless/correlation"][
            dose_index, sample_index
        ]
        mirrors = handle["clean_c_noiseless/mirrored_template"][
            dose_index, sample_index
        ]
    else:
        matrices = handle[
            "clean_c_counted/orientation_matrix_sample_to_crystal"
        ][dose_index, noise_index, repeat, sample_index]
        correlations = handle["clean_c_counted/correlation"][
            dose_index, noise_index, repeat, sample_index
        ]
        mirrors = handle["clean_c_counted/mirrored_template"][
            dose_index, noise_index, repeat, sample_index
        ]
    result = []
    for rank, predicted in enumerate(np.asarray(matrices, dtype=float)):
        alignment = best_alignment(predicted, ground_truth, symmetries)
        result.append(
            {
                "rank": rank + 1,
                "correlation": float(correlations[rank]),
                "friedel_error_deg": alignment["error_deg"],
                "mirror_match": bool(mirrors[rank]),
                "matrix": predicted.tolist(),
                **alignment,
            }
        )
    return result


def select_sample(
    candidate_path: Path,
    position: str,
) -> str:
    with h5py.File(candidate_path, "r") as handle:
        ids = decode(handle["sample_id"][:])
        errors = np.asarray(
            handle["friedel_equivalent_misorientation_deg"][:, 0],
            dtype=float,
        )
    order = np.argsort(errors)
    lookup = {
        "best": int(order[0]),
        "median": int(order[len(order) // 2]),
        "p95": int(order[int(0.95 * (len(order) - 1))]),
        "worst": int(order[-1]),
    }
    return ids[lookup[position]]


def load_noise_manifest(path: Path) -> dict:
    with h5py.File(path, "r") as handle:
        return {
            "sample_ids": decode(handle["sample_id"][:]),
            "doses": np.asarray(handle["dose_electrons"][:], dtype=int),
            "levels": decode(handle["noise_level_id"][:]),
            "sigma": np.asarray(
                handle["read_noise_sigma_primary_e_rms_per_pixel"][:],
                dtype=float,
            ),
            "seeds": handle["read_noise_seed"][:],
        }


def read_case_image(
    *,
    case: dict,
    sample_index: int,
    expectation: h5py.File,
    counted: h5py.File,
    noiseless: h5py.File,
    noise: dict,
) -> np.ndarray:
    if case["track"] == "expectation":
        return np.asarray(
            expectation["expectation/intensity"][sample_index], dtype=np.float32
        )
    dose_index = int(
        np.where(noise["doses"] == int(case["dose"]))[0][0]
    )
    if case["noise"] == "noiseless":
        return np.asarray(
            noiseless["images/expected_counts"][sample_index, dose_index],
            dtype=np.float32,
        )
    image = np.asarray(
        counted["images/counts"][
            sample_index, dose_index, int(case["repeat"])
        ],
        dtype=np.float32,
    )
    level_index = noise["levels"].index(case["noise"])
    sigma = float(noise["sigma"][level_index])
    if sigma == 0.0:
        return image
    seed = int(
        noise["seeds"][
            sample_index, dose_index, level_index, int(case["repeat"])
        ]
    )
    return add_gaussian_read_noise(
        image, sigma, np.random.default_rng(seed)
    )


def detected_path(data_root: Path, case: dict) -> Path:
    if case["track"] == "expectation":
        return (
            data_root
            / "intermediates/detected_expectation_2048"
            / (
                "clean_expectation_"
                f"{case['detector']}_peaks_first_born.h5"
            )
        )
    if case["noise"] == "noiseless":
        return (
            data_root
            / "intermediates/detected_dose_noiseless_full_py4dstem"
            / (
                f"clean_dose{case['dose']}_noise_noiseless_"
                "py4dstem_peaks_first_born.h5"
            )
        )
    return (
        data_root
        / "intermediates/detected_counted_full_py4dstem"
        / (
            f"clean_dose{case['dose']}_noise_{case['noise']}_"
            f"repeat{case['repeat']}_py4dstem_peaks_first_born.h5"
        )
    )


def candidate_path(data_root: Path, case: dict) -> Path:
    if case["track"] == "expectation":
        return (
            data_root
            / "results/acom_top5_candidates/clean_e"
            / f"{case['detector']}_candidates.h5"
        )
    stem = (
        f"dose{case['dose']}_noise_{case['noise']}"
        + (
            ""
            if case["noise"] == "noiseless"
            else f"_repeat{case['repeat']}"
        )
        + "_py4dstem"
    )
    return (
        data_root
        / "results/acom_top5_candidates/clean_c"
        / f"{stem}_candidates.h5"
    )


def build_cases(
    data_root: Path,
    symmetries: list[np.ndarray],
) -> tuple[list[dict], np.ndarray, np.ndarray]:
    expectation_path = (
        data_root / "datasets/clean_v5_first_born_expectation_2048.h5"
    )
    counted_path = (
        data_root / "datasets/clean_v5_first_born_counted_2048.h5"
    )
    noiseless_path = (
        data_root / "datasets/clean_v5_first_born_dose_noiseless_2048.h5"
    )
    oracle_path = (
        data_root / "intermediates/clean_v5_first_born_oracle_2048.h5"
    )
    trace_path = (
        data_root / "intermediates/clean_v5_first_born_trace_2048.h5"
    )
    noise_path = (
        data_root / "manifests/clean_v5_instrument_noise_2048.h5"
    )
    pyxem_path = data_root / "results/pyxem_top5_merged.h5"
    clean_e_py = (
        data_root
        / "results/acom_top5_candidates/clean_e/py4dstem_candidates.h5"
    )
    clean_e_auto = (
        data_root
        / "results/acom_top5_candidates/clean_e/autodisk_candidates.h5"
    )
    cases = [
        {
            "id": "clean_e_median",
            "label": "Clean-E · typical ACOM / 典型",
            "description": "Clean-E + find_Bragg_disks Top-1 error median case.",
            "sample_id": select_sample(clean_e_py, "median"),
            "track": "expectation",
            "dose": None,
            "noise": "none",
            "repeat": None,
            "detector": "py4dstem",
        },
        {
            "id": "clean_e_autodisk_worst",
            "label": "Clean-E · AutoDisk difficult case / 困难",
            "description": "Clean-E + AutoDisk Top-1 orientation-error worst case.",
            "sample_id": select_sample(clean_e_auto, "worst"),
            "track": "expectation",
            "dose": None,
            "noise": "none",
            "repeat": None,
            "detector": "autodisk",
        },
        {
            "id": "clean_c_low_failure",
            "label": "Clean-C · 100 e⁻ · Poisson · no ACOM candidate",
            "description": "A real low-dose indexing failure caused by too few usable detected peaks.",
            "sample_id": None,
            "track": "counted",
            "dose": 100,
            "noise": "poisson_only",
            "repeat": 1,
            "detector": "py4dstem",
        },
        {
            "id": "clean_c_mid_noise",
            "label": "Clean-C · 10⁴ e⁻ · EMPAD-G2 16 frames",
            "description": "Intermediate dose with deterministic EMPAD-G2 read noise.",
            "sample_id": None,
            "track": "counted",
            "dose": 10_000,
            "noise": "empad_g2_16frames",
            "repeat": 0,
            "detector": "py4dstem",
        },
        {
            "id": "clean_c_high_noiseless",
            "label": "Clean-C · 10⁶ e⁻ · noiseless expected counts",
            "description": "High-dose deterministic expected-count image.",
            "sample_id": None,
            "track": "counted",
            "dose": 1_000_000,
            "noise": "noiseless",
            "repeat": None,
            "detector": "py4dstem",
        },
        {
            "id": "clean_c_read_noise",
            "label": "Clean-C · 100 e⁻ · EMPAD-G2 64 frames",
            "description": "Low dose with the strongest configured read-noise level.",
            "sample_id": None,
            "track": "counted",
            "dose": 100,
            "noise": "empad_g2_64frames",
            "repeat": 0,
            "detector": "py4dstem",
        },
    ]

    # Resolve representative samples from the saved condition candidates.
    for case in cases:
        if case["sample_id"] is not None:
            continue
        path = candidate_path(data_root, case)
        if case["id"] == "clean_c_low_failure":
            details_stem = path.name.removesuffix("_candidates.h5")
            details = load_json(
                data_root
                / "results/acom_top5/clean_c"
                / f"{details_stem}_details.json"
            )
            failures = list(details["indexing_failures"])
            if not failures:
                raise ValueError("expected a saved low-dose indexing failure")
            case["sample_id"] = str(failures[0]["sample_id"])
        elif case["id"] == "clean_c_read_noise":
            case["sample_id"] = select_sample(path, "worst")
        elif case["id"] == "clean_c_mid_noise":
            case["sample_id"] = select_sample(path, "median")
        else:
            case["sample_id"] = select_sample(path, "best")

    noise = load_noise_manifest(noise_path)
    with h5py.File(expectation_path, "r") as expectation, h5py.File(
        counted_path, "r"
    ) as counted, h5py.File(noiseless_path, "r") as noiseless, h5py.File(
        pyxem_path, "r"
    ) as pyxem:
        sample_ids = decode(expectation["sample_id"][:])
        qx = np.asarray(expectation["detector/qx_Ainv"][:], dtype=float)
        qy = np.asarray(expectation["detector/qy_Ainv"][:], dtype=float)
        q_per_px = float(np.median(np.diff(qx)))
        counted_levels = decode(pyxem["counted_noise_level_id"][:])
        for case in cases:
            sample_index = sample_ids.index(str(case["sample_id"]))
            ground_truth = np.asarray(
                expectation[
                    "orientation/canonical_matrix_sample_to_crystal"
                ][sample_index],
                dtype=float,
            )
            raw_orientation = np.asarray(
                expectation["orientation/raw_matrix_sample_to_crystal"][
                    sample_index
                ],
                dtype=float,
            )
            image = read_case_image(
                case=case,
                sample_index=sample_index,
                expectation=expectation,
                counted=counted,
                noiseless=noiseless,
                noise=noise,
            )
            oracle = ragged_peaks(oracle_path, str(case["sample_id"]))
            detected = ragged_peaks(
                detected_path(data_root, case), str(case["sample_id"])
            )
            trace, basis = reflection_trace(trace_path, sample_index)
            dose_index = (
                None
                if case["dose"] is None
                else int(
                    np.where(noise["doses"] == int(case["dose"]))[0][0]
                )
            )
            noise_index = (
                None
                if case["noise"] in {"none", "noiseless"}
                else counted_levels.index(str(case["noise"]))
            )
            case.update(
                {
                    "sample_index": sample_index,
                    "image_url": image_data_url(image),
                    "image_stats": {
                        "shape": list(image.shape),
                        "dtype": str(image.dtype),
                        "sum": float(np.sum(image)),
                        "minimum": float(np.min(image)),
                        "maximum": float(np.max(image)),
                        "nonzero_pixels": int(np.count_nonzero(image)),
                        "display_transform": (
                            "clip negative values to 0, log1p, 99.7th "
                            "percentile white-background scaling; display only"
                        ),
                    },
                    "q_axis": {
                        "qx_min": float(qx[0]),
                        "qx_max": float(qx[-1]),
                        "qy_min": float(qy[0]),
                        "qy_max": float(qy[-1]),
                        "q_pixel_size_Ainv": q_per_px,
                    },
                    "raw_orientation_matrix": raw_orientation,
                    "ground_truth_matrix": ground_truth,
                    "canonicalization": {
                        "crystal_symmetry_index": int(
                            expectation[
                                "orientation/canonical_crystal_symmetry_index"
                            ][sample_index]
                        ),
                        "friedel_branch_index": int(
                            expectation[
                                "orientation/canonical_friedel_branch_index"
                            ][sample_index]
                        ),
                        "orientation_class_id": decode(
                            expectation["orientation/orientation_class_id"][
                                sample_index : sample_index + 1
                            ]
                        )[0],
                    },
                    "oracle_peaks": oracle,
                    "detected_peaks": detected,
                    "peak_metrics": match_peaks(
                        oracle, detected, abs(q_per_px)
                    ),
                    "reflections": trace[:24],
                    "acom_candidates": candidate_rows(
                        candidate_path(data_root, case),
                        str(case["sample_id"]),
                        ground_truth,
                        symmetries,
                    ),
                    "pyxem_candidates": pyxem_candidate_rows(
                        pyxem,
                        sample_index=sample_index,
                        ground_truth=ground_truth,
                        symmetries=symmetries,
                        track=str(case["track"]),
                        dose_index=dose_index,
                        noise_index=noise_index,
                        repeat=case["repeat"],
                    ),
                }
            )
    return rounded(cases), basis, np.asarray(qx)


def aggregate_payload(summary: dict) -> list[dict]:
    return rounded(summary["aggregates"])


def build_payload(args: argparse.Namespace) -> dict:
    config = load_config()
    v5_config = yaml.safe_load(
        (ROOT / "config/benchmark_v5.yaml").read_text(encoding="utf-8")
    )
    structure = Structure.from_file(str(cif_path(config)))
    symmetries = proper_point_group_rotations(structure)
    cases, reciprocal_basis, _ = build_cases(
        args.data_root.resolve(), symmetries
    )
    acom = load_json(args.acom_summary.resolve())
    pyxem = load_json(args.pyxem_summary.resolve())
    generation = load_json(
        ROOT
        / "reports/v5/pipeline/first_born_generation_2048.json"
    )
    noise_report = load_json(
        ROOT / "reports/v5/pipeline/instrument_noise_manifest_2048.json"
    )
    study_001 = load_json(
        ROOT
        / "reports/v5/study_001/clean_v5_001_evaluation_expectation_512.json"
    )
    study_001_manifest = load_json(
        ROOT
        / "reports/v5/study_001/clean_v5_001_manifest_summary.json"
    )
    direct_basis = np.asarray(structure.lattice.matrix, dtype=float)
    payload = {
        "schema": "or4d-v5-clean-visualization-v1",
        "dataset": {
            "id": v5_config["dataset"]["id"],
            "samples": 2048,
            "image_shape": generation["image_shape"],
            "forward_model": generation["forward_model"],
            "kmax_Ainv": float(config["common"]["k_max_Ainv"]),
            "doses": v5_config["clean_image"]["counting"][
                "doses_electrons"
            ],
            "repeats": v5_config["clean_image"]["counting"]["repeats"],
            "counting_model": v5_config["clean_image"]["counting"]["model"],
            "canonicalize_friedel": bool(
                v5_config["clean_sampling"]["headline_core"][
                    "canonicalize_friedel"
                ]
            ),
            "direct_basis_A_Angstrom": direct_basis,
            "reciprocal_basis_B_Ainv": reciprocal_basis,
            "reciprocal_definition": (
                "rows(a*,b*,c*) with no 2π; A @ B.T = I"
            ),
        },
        "noise": rounded(noise_report),
        "acom": {
            "conditions": int(acom["num_conditions"]),
            "clean_e_conditions": int(acom["num_clean_e_conditions"]),
            "clean_c_conditions": int(acom["num_clean_c_conditions"]),
            "aggregates": aggregate_payload(acom),
        },
        "pyxem": {
            "conditions": len(pyxem["conditions"]),
            "aggregates": aggregate_payload(pyxem),
        },
        "cases": cases,
        "study_001": {
            "manifest": rounded(study_001_manifest),
            "evaluation": rounded(
                {
                    key: value
                    for key, value in study_001.items()
                    if key != "detectors"
                }
            ),
            "detectors": rounded(
                [
                    {
                        key: value
                        for key, value in row.items()
                        if key != "per_sample"
                    }
                    for row in study_001["detectors"]
                ]
            ),
        },
        "files": {
            "expectation": {
                "path": (
                    "datasets/clean_v5_first_born_expectation_2048.h5"
                ),
                "dataset": "expectation/intensity",
                "shape": [2048, 512, 512],
                "dtype": "float32",
            },
            "counted": {
                "path": "datasets/clean_v5_first_born_counted_2048.h5",
                "dataset": "images/counts",
                "shape": [2048, 9, 5, 512, 512],
                "dtype": "uint32",
            },
            "dose_noiseless": {
                "path": (
                    "datasets/clean_v5_first_born_dose_noiseless_2048.h5"
                ),
                "dataset": "images/expected_counts",
                "shape": [2048, 9, 512, 512],
                "dtype": "float32",
            },
            "oracle": {
                "path": (
                    "intermediates/clean_v5_first_born_oracle_2048.h5"
                ),
                "datasets": "peaks/{offsets,qx,qy,intensity}",
            },
            "trace": {
                "path": (
                    "intermediates/clean_v5_first_born_trace_2048.h5"
                ),
                "datasets": (
                    "reflections/{hkl,g_crystal_Ainv,g_sample_Ainv,"
                    "qx_Ainv,qy_Ainv,qz_Ainv,...}"
                ),
            },
            "acom_top5": {
                "local_path": "full_results/acom_top5_candidates/",
                "server_path": "results/acom_top5_candidates/",
            },
            "pyxem_top5": {
                "local_path": "full_results/pyxem_top5_merged.h5",
                "server_path": "results/pyxem_top5_merged.h5",
            },
        },
    }
    return rounded(payload)


HTML_TEMPLATE = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>OR4D Clean V5 · Diffraction Image → Top-5 Orientation</title>
<style>
:root{--ink:#172033;--muted:#64748b;--line:#dbe3ef;--panel:#f7f9fc;--blue:#2563eb;--cyan:#0891b2;--orange:#ea580c;--green:#15803d;--purple:#7c3aed;--red:#dc2626}
*{box-sizing:border-box}body{margin:0;background:#fff;color:var(--ink);font:15px/1.58 -apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans SC",sans-serif}main{max-width:1500px;margin:auto;padding:28px 34px 70px}h1{font-size:31px;line-height:1.2;margin:0 0 8px}h2{font-size:22px;margin:0 0 15px}h3{font-size:17px;margin:0 0 8px}.subtitle,.muted{color:var(--muted)}.topbar{display:flex;justify-content:space-between;gap:20px;align-items:flex-start;margin-bottom:24px}.nav{display:flex;flex-wrap:wrap;gap:8px}.nav a,.pill,.toggle button{border:1px solid var(--line);border-radius:9px;padding:8px 12px;text-decoration:none;color:var(--ink);background:#fff}.nav a.active,.toggle button.active{background:var(--ink);color:#fff;border-color:var(--ink)}.section{border-top:1px solid var(--line);padding-top:28px;margin-top:32px}.cards{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}.card,.panel{border:1px solid var(--line);border-radius:14px;background:#fff;padding:16px}.card strong{display:block;font-size:24px}.card span{color:var(--muted)}.pipeline{display:grid;grid-template-columns:repeat(7,1fr);gap:8px}.step{border:1px solid var(--line);border-radius:12px;padding:12px;background:var(--panel);min-height:110px}.step b{display:block;margin-bottom:5px}.step code{font-size:12px}.grid2{display:grid;grid-template-columns:1fr 1fr;gap:18px}.grid3{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}.formula{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;background:#f5f7fb;border:1px solid var(--line);border-radius:10px;padding:12px;white-space:pre-wrap}.controls{display:flex;flex-wrap:wrap;gap:12px;align-items:end;margin:12px 0 16px}.control label{display:block;font-weight:650;font-size:13px;margin-bottom:5px}.control select{min-width:210px;border:1px solid #cbd5e1;border-radius:9px;padding:9px 10px;background:#fff;color:var(--ink)}canvas.chart{width:100%;height:390px;border:1px solid var(--line);border-radius:12px;background:#fff}.chart-note{font-size:13px;color:var(--muted);margin-top:8px}.legend{display:flex;flex-wrap:wrap;gap:12px;margin:8px 0}.legend span:before{content:"";display:inline-block;width:14px;height:3px;background:var(--c);margin-right:5px;vertical-align:middle}.table-wrap{overflow:auto;border:1px solid var(--line);border-radius:12px}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{padding:9px 11px;border-bottom:1px solid #e8edf5;text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}th{background:#f7f9fc;font-size:13px}.status-ok{color:var(--green)}.status-no{color:var(--red)}details{border:1px solid var(--line);border-radius:12px;padding:12px 14px;background:#fff}details+details{margin-top:10px}summary{cursor:pointer;font-weight:700}.matrix{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;white-space:pre}.case-layout{display:grid;grid-template-columns:minmax(480px,1.2fr) minmax(420px,1fr);gap:18px}.image-wrap{position:relative;aspect-ratio:1;border:1px solid var(--line);border-radius:12px;overflow:hidden;background:#f8fafc}.image-wrap img,.image-wrap canvas{position:absolute;inset:0;width:100%;height:100%}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}.metric{background:var(--panel);border-radius:9px;padding:9px}.metric b{display:block}.candidate-tabs{display:flex;gap:8px;margin-bottom:10px}.candidate-tabs button{border:1px solid var(--line);border-radius:8px;background:#fff;padding:7px 10px}.candidate-tabs button.active{background:var(--ink);color:#fff}.warning{border-left:4px solid #f59e0b;background:#fffbeb;padding:12px 14px;border-radius:8px}.good{border-left:4px solid var(--green);background:#f0fdf4;padding:12px 14px;border-radius:8px}.file{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;overflow-wrap:anywhere}.small{font-size:12px}.axis-controls{display:flex;gap:8px;margin:8px 0}.axis-controls button{border:1px solid var(--line);background:#fff;padding:6px 9px;border-radius:8px}.axis-controls button.active{background:#172033;color:#fff}.small-multiples{display:grid;grid-template-columns:1fr 1fr;gap:14px}.mini-chart{border:1px solid var(--line);border-radius:12px;padding:11px}.mini-chart h3{font-size:14px}.mini-chart canvas{width:100%;height:240px}.trace-canvas{width:100%;height:320px;border:1px solid var(--line);border-radius:12px;background:#fff}@media(max-width:1050px){.cards{grid-template-columns:1fr 1fr}.pipeline{grid-template-columns:1fr 1fr}.grid2,.case-layout{grid-template-columns:1fr}}@media(max-width:900px){.small-multiples{grid-template-columns:1fr}}@media(max-width:650px){main{padding:20px 14px}.cards,.grid3{grid-template-columns:1fr}.topbar{display:block}.nav{margin-top:14px}.metrics{grid-template-columns:1fr 1fr}}
</style>
</head>
<body><main>
<header class="topbar">
 <div><h1>Clean V5：二维衍射图 → Top‑5 取向</h1><p class="subtitle">2,048 orientations · First‑Born Clean images · 9 electron doses · independent detector-noise ladder · ACOM and Pyxem</p></div>
 <nav class="nav"><a href="../v3/ACOM_COORDINATE_VISUALIZATION.html">V3 · 直接峰</a><a href="../v4/CLEAN_IMAGE_ACOM_VISUALIZATION.html">V4 · 图像接口</a><a class="active" href="ACOM_CLEAN_V5_VISUALIZATION.html">V5 · 剂量/噪声/Top‑5</a></nav>
</header>

<section class="cards">
 <div class="card"><strong>2,048</strong><span>headline orientations</span></div>
 <div class="card"><strong>9 × 6</strong><span>剂量 × 噪声阶梯</span></div>
 <div class="card"><strong>238 / 235</strong><span>ACOM / Pyxem completed conditions</span></div>
 <div class="card"><strong>Top‑1…Top‑5</strong><span>全部候选等级均保存并评测</span></div>
</section>

<section class="section"><h2>完整数据路径与处理链</h2>
<div class="pipeline">
 <div class="step"><b>① Orientation</b>SO(3) Sobol 取向；构建时做晶体对称性和 Friedel canonicalization。</div>
 <div class="step"><b>② Reflection trace</b><code>hkl → g_crystal → g_sample → q</code>，保存 B、结构因子、激发误差。</div>
 <div class="step"><b>③ Clean‑E</b>512×512 coherent First‑Born expectation image。</div>
 <div class="step"><b>④ Clean‑C</b>9 档剂量；无噪声 expected counts 或 Poisson 电子计数。</div>
 <div class="step"><b>⑤ Instrument noise</b>独立 EMPAD‑G2 read-noise 梯度，确定性 seed。</div>
 <div class="step"><b>⑥ Indexing</b>ACOM 读取检峰列表；Pyxem 直接读取二维图像模板匹配。</div>
 <div class="step"><b>⑦ Top‑K metric</b>前 K 个候选中至少一个 symmetry/Friedel error ≤2°。</div>
</div>
<div class="warning" style="margin-top:14px"><b>重要：</b>网页读取已保存结果，不会重新运行 benchmark。图像亮度使用显示专用的 clip/log 映射；ACOM/Pyxem 输入仍是原始 HDF5 数组。</div>
</section>

<section class="section"><h2>Top‑1…Top‑5 剂量曲线</h2>
 <p>每条曲线使用全部 2,048 个输入作为分母。ACOM 没有候选的样本计为错误；Pyxem 返回候选并不代表候选正确。</p>
 <div class="controls">
  <div class="control"><label>方法 / Method</label><select id="dose-method"><option value="acom">ACOM + find_Bragg_disks</option><option value="pyxem">Pyxem direct-image template matching</option></select></div>
  <div class="control"><label>噪声梯度 / Noise level</label><select id="dose-noise"></select></div>
  <div class="control"><label>纵轴指标 / Metric</label><select id="dose-metric"><option value="acc1">Top‑K Acc@1°</option><option value="acc2" selected>Top‑K Acc@2°</option><option value="acc5">Top‑K Acc@5°</option><option value="median">Median equivalent error</option><option value="p95">P95 equivalent error</option><option value="coverage">Prediction coverage</option></select></div>
 </div>
 <canvas class="chart" id="dose-chart" width="1280" height="390"></canvas>
 <div class="legend" id="dose-legend"></div>
 <p class="chart-note">横轴为电子数/图样（对数）；纵轴由上方指标选择。Top‑K 候选顺序来自算法相关分数，不使用 GT 重排。</p>
</section>

<section class="section"><h2>全噪声 × 全方法小多图</h2>
 <p>每一行固定一个噪声等级；左列是 ACOM + find_Bragg_disks，右列是 Pyxem 直接图像匹配。每张图同时保留 Top‑1…Top‑5。</p>
 <div class="controls"><div class="control"><label>纵轴指标 / Metric</label><select id="overview-metric"><option value="acc1">Top‑K Acc@1°</option><option value="acc2" selected>Top‑K Acc@2°</option><option value="acc5">Top‑K Acc@5°</option><option value="median">Median equivalent error</option><option value="p95">P95 equivalent error</option><option value="coverage">Prediction coverage</option></select></div></div>
 <div class="legend" id="overview-legend"></div>
 <div class="small-multiples" id="overview-grid"></div>
</section>

<section class="section"><h2>固定电子剂量下的噪声阶梯</h2>
 <div class="controls">
  <div class="control"><label>方法 / Method</label><select id="noise-method"><option value="acom">ACOM + find_Bragg_disks</option><option value="pyxem">Pyxem direct image</option></select></div>
  <div class="control"><label>电子剂量 / Dose</label><select id="noise-dose"></select></div>
  <div class="control"><label>纵轴指标 / Metric</label><select id="noise-metric"><option value="acc1">Top‑K Acc@1°</option><option value="acc2" selected>Top‑K Acc@2°</option><option value="acc5">Top‑K Acc@5°</option><option value="median">Median equivalent error</option><option value="p95">P95 equivalent error</option><option value="coverage">Prediction coverage</option></select></div>
 </div>
 <canvas class="chart" id="noise-chart" width="1280" height="390"></canvas>
 <div class="legend" id="noise-legend"></div>
 <p class="chart-note">噪声轴与电子剂量轴独立：noiseless 使用确定性 expected counts；其余使用同一 Poisson realization，再叠加不同 read noise。</p>
</section>

<section class="section"><h2>Clean‑E 输入/检峰器与 Top‑K</h2>
 <p>Clean‑E ACOM 全量比较 oracle、AutoDisk、DoG‑RGM、find_Bragg_disks；Pyxem 直接读取图像，不是检峰器。Clean‑C 全量 ACOM 当前只运行了 find_Bragg_disks。</p>
 <div class="controls"><div class="control"><label>纵轴指标 / Metric</label><select id="cleane-metric"><option value="acc1">Top‑K Acc@1°</option><option value="acc2" selected>Top‑K Acc@2°</option><option value="acc5">Top‑K Acc@5°</option><option value="median">Median equivalent error</option><option value="p95">P95 equivalent error</option><option value="coverage">Prediction coverage</option></select></div></div>
 <canvas class="chart" id="cleane-chart" width="1280" height="390"></canvas>
 <div class="legend" id="cleane-legend"></div>
 <div class="table-wrap" style="margin-top:14px"><table><thead><tr><th>Track / method</th><th>Clean‑E full</th><th>Clean‑C full</th><th>说明</th></tr></thead><tbody>
  <tr><td>ACOM + oracle</td><td class="status-ok">✓ 2,048</td><td class="status-no">—</td><td>只用于上限比较</td></tr>
  <tr><td>ACOM + AutoDisk</td><td class="status-ok">✓ 2,048</td><td class="status-no">未运行</td><td>不绘制不存在的 Clean‑C 曲线</td></tr>
  <tr><td>ACOM + DoG‑RGM</td><td class="status-ok">✓ 2,048</td><td class="status-no">未运行</td><td>不绘制不存在的 Clean‑C 曲线</td></tr>
  <tr><td>ACOM + find_Bragg_disks</td><td class="status-ok">✓ 2,048</td><td class="status-ok">✓ 234 conditions</td><td>Clean‑C ACOM 正式路径</td></tr>
  <tr><td>Pyxem direct image</td><td class="status-ok">✓ 2,048</td><td class="status-ok">✓ 234 conditions</td><td>直接图像模板匹配，无检峰步骤</td></tr>
 </tbody></table></div>
</section>

<section class="section"><h2>当前条件的精确 Top‑K 数值</h2>
 <div class="controls"><div class="control"><label>方法 / Method</label><select id="condition-method"><option value="acom">ACOM + find_Bragg_disks</option><option value="pyxem">Pyxem direct image</option></select></div><div class="control"><label>电子剂量 / Dose</label><select id="condition-dose"></select></div><div class="control"><label>噪声 / Noise</label><select id="condition-noise"></select></div></div>
 <p id="condition-caption"></p><div class="table-wrap"><table><thead><tr><th>K</th><th>Coverage</th><th>Acc@1°</th><th>Acc@2°</th><th>Acc@5°</th><th>Median error</th><th>P95 error</th></tr></thead><tbody id="condition-table"></tbody></table></div>
</section>

<section class="section"><h2>样本诊断：实际图像、检峰与候选矩阵</h2>
 <div class="controls"><div class="control" style="flex:1"><label>已保存案例 / Saved diagnostic case</label><select id="case-select" style="width:100%"></select></div></div>
 <p id="case-description" class="muted"></p>
 <div class="case-layout">
  <div class="panel"><h3 id="image-title"></h3><div class="image-wrap"><img id="case-image" alt="actual saved/reconstructed V5 diffraction image"><canvas id="peak-overlay" width="512" height="512"></canvas></div>
   <div class="controls small"><label><input id="show-oracle" type="checkbox" checked> Oracle peaks</label><label><input id="show-detected" type="checkbox" checked> Detected peaks</label><label><input id="show-links" type="checkbox" checked> Match links</label></div>
   <div class="metrics" id="peak-metrics"></div>
  </div>
  <div class="panel"><h3>图像与坐标中间量</h3><div id="image-stats"></div><details open><summary>反射坐标追踪 / Reflection trace</summary><div class="controls"><div class="control"><label>HKL（按强度）</label><select id="reflection-select"></select></div></div><div class="formula" id="reflection-detail"></div></details>
   <details><summary>原始与 canonical orientation</summary><div id="canonical-detail"></div></details>
  </div>
 </div>
 <div class="grid2" style="margin-top:18px">
  <div class="panel"><h3>ACOM Top‑5 candidates</h3><div id="acom-candidates"></div></div>
  <div class="panel"><h3>Pyxem Top‑5 candidates</h3><div id="pyxem-candidates"></div></div>
 </div>
 <div class="panel" style="margin-top:18px"><h3>逐反射坐标变换可视化</h3><canvas class="trace-canvas" id="coordinate-trace" width="1280" height="320"></canvas><p class="chart-note">紫色向量依次显示所选 HKL 在晶体倒空间、样品倒空间和探测器 q 平面的表示；数值与上方反射追踪框同步。</p></div>
 <div class="panel" style="margin-top:18px"><h3>取向轴投影：原始代表 / 对称性与 Friedel 对齐后</h3><div class="axis-controls"><button id="axis-raw">原始矩阵</button><button id="axis-aligned" class="active">对齐后</button></div><canvas class="chart" id="axis-chart" width="1280" height="390"></canvas><p class="chart-note">显示 R 的三列在 crystal XY 平面的投影。原始代表可能相差晶体对称操作或 detector-plane Friedel branch；对齐视图才与页面误差指标语义一致。</p></div>
</section>

<section class="section"><h2>晶格、倒易基矢和数值矩阵</h2>
 <div class="grid2"><div class="panel"><h3>直接晶格 A（行是 a,b,c；Å）</h3><div class="matrix" id="matrix-a"></div></div><div class="panel"><h3>倒易晶格 B（行是 a*,b*,c*；Å⁻¹；不含 2π）</h3><div class="matrix" id="matrix-b"></div></div></div>
 <div class="formula" style="margin-top:14px">a* = (b × c) / [a · (b × c)]
b* = (c × a) / [a · (b × c)]
c* = (a × b) / [a · (b × c)]

B = [a*; b*; c*],   A Bᵀ = I
g_crystal = [h k l] B
g_sample = R_sample→crystalᵀ g_crystal
q = [g_sample,x, g_sample,y],   ‖q‖ ≤ 1.5 Å⁻¹</div>
</section>

<section class="section"><h2>输入文件与中间变量</h2><div class="grid2" id="file-grid"></div></section>

<section class="section"><h2>[001] 独立研究集</h2><div id="study001"></div><p class="warning">[001] 的 512 个样本不混入 2,048 headline 指标。旧 Top‑1 ACOM 结果已按要求删除；本页只展示仍保留的构建与检峰中间结果，不把旧结果伪装成 Top‑5。</p></section>

<section class="section"><h2>运行参数与结果解释</h2>
 <details open><summary>Top‑K、对称性与 Friedel branch</summary><p>“Top‑K 含有正确答案”定义为：算法按相关分数给出的前 K 个候选中，至少一个候选在 proper crystal point-group rotations 与 detector-plane Friedel branch 两类等价变换后，取向误差 ≤2°。Friedel branch 处理二维衍射图无法区分的探测器平面反演；V5 同时在数据构建时保存 canonical Friedel branch，评价仍做等价搜索以避免代表选择影响指标。</p></details>
 <details><summary>电子计数与噪声</summary><div class="formula">Clean‑E: P(q), Σq P(q)=1
Noiseless expected counts: I_N(q)=N_e P(q)
Counted: n(q) ~ Poisson(N_e P(q))
Read noise: y(q)=n(q)+Normal(0, σ_level²)
σ_level = 0.008666… × √(summed frame count) primary-e⁻ RMS/pixel</div></details>
 <details><summary>为什么更多 frames 的 read noise 可能更大</summary><p>这里的变量是“多帧求和后的读出噪声”，每帧独立读出噪声按 √frames 累积；它不是把固定总曝光平均后降低噪声的模型。网页直接展示当前 benchmark 已冻结的定义与实测结果。</p></details>
</section>

<script>
const DATA=__DATA_JSON__;
const $=id=>document.getElementById(id);
const COLORS=["#dc2626","#ea580c","#ca8a04","#15803d","#2563eb"];
const NOISE_ORDER=["noiseless","poisson_only","empad_g2_1frame","empad_g2_4frames","empad_g2_16frames","empad_g2_64frames"];
const NOISE_LABEL={noiseless:"Noiseless expected counts",poisson_only:"Poisson only",empad_g2_1frame:"EMPAD-G2 · 1 frame",empad_g2_4frames:"EMPAD-G2 · 4 frames",empad_g2_16frames:"EMPAD-G2 · 16 frames",empad_g2_64frames:"EMPAD-G2 · 64 frames"};
const pct=v=>(100*v).toFixed(2)+"%";const num=(v,d=3)=>Number.isFinite(v)?Number(v).toFixed(d):"—";
const METRIC={acc1:{field:"accuracy_all_inputs_within_1deg",label:"Acc@1°",fixed:true},acc2:{field:"accuracy_all_inputs_within_2deg",label:"Acc@2°",fixed:true},acc5:{field:"accuracy_all_inputs_within_5deg",label:"Acc@5°",fixed:true},median:{field:"median_misorientation_deg_indexed",label:"Median equivalent error (°)",fixed:false},p95:{field:"p95_misorientation_deg_indexed",label:"P95 equivalent error (°)",fixed:false},coverage:{field:"prediction_coverage",label:"Prediction coverage",fixed:true}};
function matrixText(m){return m.map(r=>"["+r.map(v=>Number(v).toFixed(6).padStart(10)).join("  ")+"]").join("\n")}
function aggregate(method,group,fields){const rows=DATA[method].aggregates.filter(r=>r.group_by===group&&Object.entries(fields).every(([k,v])=>r[k]===v));if(rows.length!==1)throw new Error(`aggregate ${method}/${group} ${JSON.stringify(fields)} -> ${rows.length}`);return rows[0]}
function setupSelectors(){NOISE_ORDER.forEach(n=>{for(const id of ["dose-noise","condition-noise"]){const o=document.createElement("option");o.value=n;o.textContent=NOISE_LABEL[n];$(id).append(o)}});DATA.dataset.doses.forEach(d=>{for(const id of ["noise-dose","condition-dose"]){const o=document.createElement("option");o.value=d;o.textContent=d.toLocaleString()+" e⁻";$(id).append(o)}});$("noise-dose").value="10000";$("condition-dose").value="10000";for(const c of DATA.cases){const o=document.createElement("option");o.value=c.id;o.textContent=c.label+" · "+c.sample_id;$("case-select").append(o)}}
function chart(canvas,labels,series,{logX=false,yTitle="Acc@2°",fixed=true}={}){const ctx=canvas.getContext("2d"),W=canvas.width,H=canvas.height,p={l:72,r:24,t:28,b:64};ctx.clearRect(0,0,W,H);ctx.fillStyle="#fff";ctx.fillRect(0,0,W,H);const finite=series.flatMap(s=>s.values).filter(Number.isFinite),maxValue=fixed?1:Math.max(1,...finite)*1.08,minValue=0;ctx.font="12px system-ui";ctx.strokeStyle="#dbe3ef";ctx.fillStyle="#64748b";ctx.lineWidth=1;for(let i=0;i<=5;i++){const y=p.t+(H-p.t-p.b)*i/5,v=maxValue-(maxValue-minValue)*i/5;ctx.beginPath();ctx.moveTo(p.l,y);ctx.lineTo(W-p.r,y);ctx.stroke();ctx.textAlign="right";ctx.fillText(v<2?v.toFixed(1):v.toFixed(0),p.l-10,y+4)}const xs=labels.map((v,i)=>{if(logX){const lo=Math.log10(labels[0]),hi=Math.log10(labels[labels.length-1]);return p.l+(Math.log10(v)-lo)/(hi-lo)*(W-p.l-p.r)}return labels.length===1?(p.l+W-p.r)/2:p.l+i/(labels.length-1)*(W-p.l-p.r)});labels.forEach((v,i)=>{ctx.fillStyle="#64748b";ctx.textAlign="center";const text=logX?Number(v).toExponential(0):String(v);ctx.fillText(text,xs[i],H-p.b+22)});for(const s of series){ctx.strokeStyle=s.color;ctx.fillStyle=s.color;ctx.lineWidth=2.5;ctx.beginPath();s.values.forEach((v,i)=>{const y=p.t+(maxValue-v)/(maxValue-minValue)*(H-p.t-p.b);if(i===0)ctx.moveTo(xs[i],y);else ctx.lineTo(xs[i],y)});ctx.stroke();s.values.forEach((v,i)=>{const y=p.t+(maxValue-v)/(maxValue-minValue)*(H-p.t-p.b);ctx.beginPath();ctx.arc(xs[i],y,4,0,Math.PI*2);ctx.fill()})}ctx.save();ctx.translate(18,(p.t+H-p.b)/2);ctx.rotate(-Math.PI/2);ctx.textAlign="center";ctx.fillStyle="#172033";ctx.font="13px system-ui";ctx.fillText(yTitle,0,0);ctx.restore()}
function legend(id,series){$(id).innerHTML=series.map(s=>`<span style="--c:${s.color}">${s.name}</span>`).join("")}
function metricSeries(rows,metricKey){const m=METRIC[metricKey];return [1,2,3,4,5].map((k,i)=>({name:`Top-${k}`,color:COLORS[i],values:rows.map(r=>r.top_k[k-1][m.field])}))}
function drawDose(){const method=$("dose-method").value,noise=$("dose-noise").value,m=METRIC[$("dose-metric").value];const rows=DATA.dataset.doses.map(d=>aggregate(method,"dose_noise",{track:"Clean-C",dose_electrons:d,noise})),series=metricSeries(rows,$("dose-metric").value);chart($("dose-chart"),DATA.dataset.doses,series,{logX:true,yTitle:m.label,fixed:m.fixed});legend("dose-legend",series)}
function drawNoise(){const method=$("noise-method").value,dose=Number($("noise-dose").value),m=METRIC[$("noise-metric").value];const rows=NOISE_ORDER.map(noise=>aggregate(method,"dose_noise",{track:"Clean-C",dose_electrons:dose,noise})),series=metricSeries(rows,$("noise-metric").value);chart($("noise-chart"),NOISE_ORDER.map(n=>NOISE_LABEL[n].replace("EMPAD-G2 · ","").replace("Noiseless expected counts","Noiseless")),series,{yTitle:m.label,fixed:m.fixed});legend("noise-legend",series)}
function drawCleanE(){const inputs=[["oracle","ACOM oracle"],["autodisk","ACOM AutoDisk"],["dog_rgm","ACOM DoG-RGM"],["py4dstem","ACOM find_Bragg"],["pyxem","Pyxem image"]],m=METRIC[$("cleane-metric").value];const rows=inputs.map(([key])=>key==="pyxem"?aggregate("pyxem","track",{track:"Clean-E"}):aggregate("acom","clean_e_input",{track:"Clean-E",input:key})),series=metricSeries(rows,$("cleane-metric").value);chart($("cleane-chart"),inputs.map(x=>x[1]),series,{yTitle:m.label,fixed:m.fixed});legend("cleane-legend",series)}
function buildOverview(){const grid=$("overview-grid");grid.innerHTML="";for(const noise of NOISE_ORDER){for(const method of ["acom","pyxem"]){const box=document.createElement("div");box.className="mini-chart";box.innerHTML=`<h3>${NOISE_LABEL[noise]} · ${method==="acom"?"ACOM + find_Bragg":"Pyxem image"}</h3><canvas width="620" height="240"></canvas>`;grid.append(box)}}drawOverview()}
function drawOverview(){const key=$("overview-metric").value,m=METRIC[key],canvases=$("overview-grid").querySelectorAll("canvas");let index=0;for(const noise of NOISE_ORDER){for(const method of ["acom","pyxem"]){const rows=DATA.dataset.doses.map(d=>aggregate(method,"dose_noise",{track:"Clean-C",dose_electrons:d,noise})),series=metricSeries(rows,key);chart(canvases[index++],DATA.dataset.doses,series,{logX:true,yTitle:m.label,fixed:m.fixed})}}const dummy={top_k:[1,2,3,4,5].map(()=>({[m.field]:0}))};legend("overview-legend",metricSeries([dummy],key))}
function drawCondition(){const method=$("condition-method").value,noise=$("condition-noise").value,dose=Number($("condition-dose").value);const row=aggregate(method,"dose_noise",{track:"Clean-C",dose_electrons:dose,noise});$("condition-caption").textContent=`${method==="acom"?"ACOM + find_Bragg_disks":"Pyxem direct image"} · ${dose.toLocaleString()} e⁻ · ${NOISE_LABEL[noise]} · ${row.num_conditions} saved condition(s)`;$("condition-table").innerHTML=row.top_k.map(r=>`<tr><td>Top-${r.k}</td><td>${pct(r.prediction_coverage)}</td><td>${pct(r.accuracy_all_inputs_within_1deg)}</td><td><b>${pct(r.accuracy_all_inputs_within_2deg)}</b></td><td>${pct(r.accuracy_all_inputs_within_5deg)}</td><td>${num(r.median_misorientation_deg_indexed)}°</td><td>${num(r.p95_misorientation_deg_indexed)}°</td></tr>`).join("")}
function qToPixel(caseRow,qx,qy){const q=caseRow.q_axis;return [(qx-q.qx_min)/(q.qx_max-q.qx_min)*511,(qy-q.qy_min)/(q.qy_max-q.qy_min)*511]}
function drawOverlay(){const c=DATA.cases.find(x=>x.id===$("case-select").value),ctx=$("peak-overlay").getContext("2d");ctx.clearRect(0,0,512,512);if($("show-links").checked){ctx.strokeStyle="rgba(124,58,237,.7)";ctx.lineWidth=1;c.peak_metrics.matches.forEach(m=>{const a=c.oracle_peaks[m.oracle_index],b=c.detected_peaks[m.detected_index],p=qToPixel(c,a.qx,a.qy),q=qToPixel(c,b.qx,b.qy);ctx.beginPath();ctx.moveTo(p[0],p[1]);ctx.lineTo(q[0],q[1]);ctx.stroke()})}if($("show-oracle").checked){ctx.strokeStyle="#0891b2";ctx.lineWidth=1.6;c.oracle_peaks.forEach(x=>{const p=qToPixel(c,x.qx,x.qy);ctx.beginPath();ctx.arc(p[0],p[1],5,0,Math.PI*2);ctx.stroke()})}if($("show-detected").checked){ctx.strokeStyle="#ea580c";ctx.lineWidth=1.6;c.detected_peaks.forEach(x=>{const p=qToPixel(c,x.qx,x.qy);ctx.beginPath();ctx.moveTo(p[0]-4,p[1]-4);ctx.lineTo(p[0]+4,p[1]+4);ctx.moveTo(p[0]+4,p[1]-4);ctx.lineTo(p[0]-4,p[1]+4);ctx.stroke()})}const r=c.reflections[Number($("reflection-select").value)||0];if(r){const p=qToPixel(c,r.q_Ainv[0],r.q_Ainv[1]);ctx.strokeStyle="#7c3aed";ctx.lineWidth=3;ctx.beginPath();ctx.arc(p[0],p[1],10,0,Math.PI*2);ctx.stroke()}}
function candidateTable(rows,method){if(!rows.length)return `<div class="warning">ACOM 未生成候选：检峰列表中的可用峰不足。该样本仍保留在 2,048 输入分母中。</div>`;return `<div class="table-wrap"><table><thead><tr><th>Rank</th><th>Correlation</th><th>Equivalent error</th><th>Friedel used</th><th>Correct ≤2°</th></tr></thead><tbody>${rows.map(r=>`<tr><td>${r.rank}</td><td>${num(r.correlation,5)}</td><td>${num(r.friedel_error_deg,4)}°</td><td>${r.friedel_used?"yes":"no"}</td><td class="${r.friedel_error_deg<=2?"status-ok":"status-no"}">${r.friedel_error_deg<=2?"✓":"—"}</td></tr>`).join("")}</tbody></table></div><details><summary>Top‑1 raw/aligned matrices</summary><div class="grid2"><div><b>raw</b><pre class="matrix">${matrixText(rows[0].matrix)}</pre></div><div><b>aligned</b><pre class="matrix">${matrixText(rows[0].aligned_matrix)}</pre></div></div></details>`}
function drawCoordinateTrace(){const c=DATA.cases.find(x=>x.id===$("case-select").value),r=c.reflections[Number($("reflection-select").value)||0],canvas=$("coordinate-trace"),ctx=canvas.getContext("2d"),W=canvas.width,H=canvas.height;if(!r)return;ctx.clearRect(0,0,W,H);ctx.fillStyle="#fff";ctx.fillRect(0,0,W,H);const centers=[[210,H/2],[640,H/2],[1070,H/2]],vectors=[r.g_crystal_Ainv,r.g_sample_Ainv,r.q_Ainv],titles=["crystal reciprocal g","sample reciprocal Rᵀg","detector q / pixel"],scale=110/Math.max(.01,...vectors.flat().map(Math.abs));for(let i=0;i<3;i++){const [cx,cy]=centers[i],v=vectors[i];ctx.strokeStyle="#dbe3ef";ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(cx-130,cy);ctx.lineTo(cx+130,cy);ctx.moveTo(cx,cy-120);ctx.lineTo(cx,cy+120);ctx.stroke();ctx.strokeStyle="#7c3aed";ctx.fillStyle="#7c3aed";ctx.lineWidth=4;ctx.beginPath();ctx.moveTo(cx,cy);ctx.lineTo(cx+v[0]*scale,cy-v[1]*scale);ctx.stroke();ctx.beginPath();ctx.arc(cx+v[0]*scale,cy-v[1]*scale,6,0,Math.PI*2);ctx.fill();ctx.fillStyle="#172033";ctx.font="15px system-ui";ctx.textAlign="center";ctx.fillText(titles[i],cx,28);ctx.font="12px ui-monospace";ctx.fillText(`[${v.map(x=>num(x,4)).join(", ")}]`,cx,H-28)}ctx.strokeStyle="#94a3b8";ctx.lineWidth=2;for(let i=0;i<2;i++){ctx.beginPath();ctx.moveTo(centers[i][0]+150,centers[i][1]);ctx.lineTo(centers[i+1][0]-150,centers[i+1][1]);ctx.stroke();ctx.fillStyle="#64748b";ctx.font="13px system-ui";ctx.fillText(i===0?"Rᵀ":"project x,y → pixel",(centers[i][0]+centers[i+1][0])/2,H/2-10)}ctx.fillStyle="#7c3aed";ctx.font="14px system-ui";ctx.fillText(`HKL [${r.hkl.join(", ")}]`,W/2,H-4)}
function updateReflection(){const c=DATA.cases.find(x=>x.id===$("case-select").value),r=c.reflections[Number($("reflection-select").value)||0];if(!r){$("reflection-detail").textContent="No retained reflection";return}$("reflection-detail").textContent=`HKL = [${r.hkl.join(", ")}]\ng_crystal = [h k l] B = [${r.g_crystal_Ainv.map(x=>num(x,5)).join(", ")}] Å⁻¹\ng_sample = Rᵀ g_crystal = [${r.g_sample_Ainv.map(x=>num(x,5)).join(", ")}] Å⁻¹\nq = [qx,qy,qz] = [${r.q_Ainv.map(x=>num(x,5)).join(", ")}] Å⁻¹\nexcitation error = ${num(r.excitation_error_Ainv,6)} Å⁻¹\nF_g = ${num(r.structure_factor[0],5)} + ${num(r.structure_factor[1],5)}i\nI_raw = ${num(r.intensity_raw,6)}, I_normalized = ${num(r.intensity_normalized,6)}`;drawCoordinateTrace();drawOverlay()}
let axisAligned=true;
function drawAxes(){const c=DATA.cases.find(x=>x.id===$("case-select").value),ctx=$("axis-chart").getContext("2d"),W=1280,H=390,cx=W/2,cy=H/2,scale=155;ctx.clearRect(0,0,W,H);ctx.fillStyle="#fff";ctx.fillRect(0,0,W,H);ctx.strokeStyle="#dbe3ef";ctx.beginPath();ctx.moveTo(40,cy);ctx.lineTo(W-40,cy);ctx.moveTo(cx,30);ctx.lineTo(cx,H-35);ctx.stroke();const sets=[["GT",c.ground_truth_matrix,"#2563eb"],["ACOM",c.acom_candidates.length?(axisAligned?c.acom_candidates[0].aligned_matrix:c.acom_candidates[0].matrix):null,"#ea580c"],["Pyxem",axisAligned?c.pyxem_candidates[0].aligned_matrix:c.pyxem_candidates[0].matrix,"#15803d"]];for(const [name,m,color] of sets){if(!m)continue;for(let j=0;j<3;j++){const x=cx+m[0][j]*scale,y=cy-m[1][j]*scale;ctx.strokeStyle=color;ctx.fillStyle=color;ctx.lineWidth=name==="GT"?3:2;ctx.beginPath();ctx.moveTo(cx,cy);ctx.lineTo(x,y);ctx.stroke();ctx.beginPath();ctx.arc(x,y,4,0,Math.PI*2);ctx.fill();ctx.font="12px system-ui";ctx.fillText(`${name} ${["x","y","z"][j]}`,x+6,y-5)}}}
function updateCase(){const c=DATA.cases.find(x=>x.id===$("case-select").value);$("case-description").textContent=c.description;$("image-title").textContent=`${c.sample_id} · ${c.track==="expectation"?"Clean-E":`${Number(c.dose).toLocaleString()} e⁻ · ${NOISE_LABEL[c.noise]}`}`;$("case-image").src=c.image_url;const m=c.peak_metrics;$("peak-metrics").innerHTML=[["TP",m.true_positive],["FP",m.false_positive],["FN",m.false_negative],["P / R",`${pct(m.precision)} / ${pct(m.recall)}`]].map(x=>`<div class="metric"><b>${x[1]}</b><span>${x[0]}</span></div>`).join("");$("image-stats").innerHTML=`<div class="formula">HDF5 sample index = ${c.sample_index}\nshape/dtype = ${c.image_stats.shape.join("×")} / ${c.image_stats.dtype}\nsum = ${num(c.image_stats.sum,6)}\nmin / max = ${num(c.image_stats.minimum,6)} / ${num(c.image_stats.maximum,6)}\nnonzero pixels = ${c.image_stats.nonzero_pixels.toLocaleString()}\nq pixel = ${num(c.q_axis.q_pixel_size_Ainv,8)} Å⁻¹</div><p class="small muted">${c.image_stats.display_transform}</p>`;$("reflection-select").innerHTML=c.reflections.map((r,i)=>`<option value="${i}">[${r.hkl.join(", ")}] · I=${num(r.intensity_normalized,4)}</option>`).join("");$("canonical-detail").innerHTML=`<p>class <code>${c.canonicalization.orientation_class_id}</code>; crystal symmetry index ${c.canonicalization.crystal_symmetry_index}; Friedel branch index ${c.canonicalization.friedel_branch_index}</p><div class="grid2"><div><b>raw R</b><pre class="matrix">${matrixText(c.raw_orientation_matrix)}</pre></div><div><b>canonical GT R</b><pre class="matrix">${matrixText(c.ground_truth_matrix)}</pre></div></div>`;$("acom-candidates").innerHTML=candidateTable(c.acom_candidates,"acom");$("pyxem-candidates").innerHTML=candidateTable(c.pyxem_candidates,"pyxem");updateReflection();drawOverlay();drawAxes()}
function renderFiles(){const labels={expectation:"Clean‑E expectation",counted:"Clean‑C Poisson counts",dose_noiseless:"Noiseless expected counts",oracle:"Physical oracle peaks",trace:"Per-reflection trace",acom_top5:"ACOM Top‑5",pyxem_top5:"Pyxem Top‑5"};$("file-grid").innerHTML=Object.entries(DATA.files).map(([k,v])=>`<div class="panel"><h3>${labels[k]}</h3>${Object.entries(v).map(([a,b])=>`<div><b>${a}</b><div class="file">${Array.isArray(b)?JSON.stringify(b):b}</div></div>`).join("")}</div>`).join("")}
function renderStudy001(){const s=DATA.study_001;$("study001").innerHTML=`<div class="cards"><div class="card"><strong>${s.manifest.num_samples||512}</strong><span>independent samples</span></div><div class="card"><strong>${s.manifest.exact_001_count||32}</strong><span>exact [001]</span></div><div class="card"><strong>0.125°–6°</strong><span>local/transition tilts</span></div><div class="card"><strong>32 + 32</strong><span>[100] / [110] controls</span></div></div><div class="table-wrap" style="margin-top:14px"><table><thead><tr><th>Detector</th><th>Precision</th><th>Recall</th><th>Position RMSE</th><th>High-angle recall</th></tr></thead><tbody>${s.detectors.map(d=>`<tr><td>${d.detector}</td><td>${pct(d.precision)}</td><td>${pct(d.recall)}</td><td>${num(d.position_rmse_px)} px</td><td>${pct(d.high_angle_recall)}</td></tr>`).join("")}</tbody></table></div>`}
function init(){$("matrix-a").textContent=matrixText(DATA.dataset.direct_basis_A_Angstrom);$("matrix-b").textContent=matrixText(DATA.dataset.reciprocal_basis_B_Ainv);setupSelectors();renderFiles();renderStudy001();buildOverview();drawDose();drawNoise();drawCleanE();drawCondition();updateCase();["dose-method","dose-noise","dose-metric"].forEach(id=>$(id).addEventListener("change",drawDose));["noise-method","noise-dose","noise-metric"].forEach(id=>$(id).addEventListener("change",drawNoise));["condition-method","condition-dose","condition-noise"].forEach(id=>$(id).addEventListener("change",drawCondition));$("cleane-metric").addEventListener("change",drawCleanE);$("overview-metric").addEventListener("change",drawOverview);$("case-select").addEventListener("change",updateCase);$("reflection-select").addEventListener("change",updateReflection);["show-oracle","show-detected","show-links"].forEach(id=>$(id).addEventListener("change",drawOverlay));$("axis-raw").onclick=()=>{axisAligned=false;$("axis-raw").classList.add("active");$("axis-aligned").classList.remove("active");drawAxes()};$("axis-aligned").onclick=()=>{axisAligned=true;$("axis-aligned").classList.add("active");$("axis-raw").classList.remove("active");drawAxes()}}
init();
</script>
</main></body></html>"""


def main() -> None:
    args = parse_args()
    payload = build_payload(args)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    html = HTML_TEMPLATE.replace(
        "__DATA_JSON__",
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
    )
    output.write_text(html, encoding="utf-8")
    print(f"V5 visualization: {output}")
    print(f"Cases: {len(payload['cases'])}")
    print(f"Bytes: {output.stat().st_size}")


if __name__ == "__main__":
    main()
