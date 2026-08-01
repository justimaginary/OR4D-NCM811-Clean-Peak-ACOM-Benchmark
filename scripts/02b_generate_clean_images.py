#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import py4DSTEM
from pymatgen.core import Structure

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports" / "v4"
sys.path.insert(0, str(ROOT / "src"))

from kinematic_cbed import (  # noqa: E402
    ProjectedReflectionSet,
    ReflectionLibrary,
    render_acom_matched_cbed,
    render_kinematic_cbed,
    render_kinematic_cbed_batch_cuda,
)
from or4d_common import cif_path, load_config, read_jsonl, write_peak_h5  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate ideal kinematical Clean CBED images and analytic oracle peaks."
    )
    parser.add_argument(
        "--role",
        action="append",
        choices=("legacy_smoke", "headline_core", "acom_grid_probe", "study_001"),
        help="Generate only this sample role; repeat to select multiple roles.",
    )
    parser.add_argument(
        "--orientation-file",
        type=Path,
        default=ROOT / "private" / "orientations.jsonl",
        help="Orientation JSONL; V5 [001] diagnostics use their separate manifest.",
    )
    parser.add_argument("--sample-id", action="append", help="Generate only this sample ID.")
    parser.add_argument("--limit", type=int, help="Limit selected samples for smoke tests.")
    parser.add_argument("--output", type=Path, help="Image HDF5 output path.")
    parser.add_argument("--oracle-output", type=Path, help="Analytic peak HDF5 output path.")
    parser.add_argument("--raw-output", type=Path, help="Raw reflection HDF5 output path.")
    parser.add_argument("--report-output", type=Path)
    parser.add_argument(
        "--direct-beam-fraction",
        type=float,
        help="Override the canonical direct-beam probability fraction.",
    )
    parser.add_argument(
        "--forward-model",
        choices=("acom_matched", "coherent_first_born"),
        help="Override clean_image.forward_model.",
    )
    parser.add_argument(
        "--compute-backend",
        choices=("cpu", "cuda"),
        default="cpu",
        help="CUDA is available only for coherent_first_born.",
    )
    parser.add_argument("--batch-size", type=int)
    return parser.parse_args()


def selected_orientations(args: argparse.Namespace) -> tuple[list[dict], bool]:
    rows = read_jsonl(args.orientation_file.resolve())
    subset = bool(args.role or args.sample_id or args.limit is not None)
    if args.role:
        roles = set(args.role)
        rows = [row for row in rows if row["sample_role"] in roles]
    if args.sample_id:
        ids = set(args.sample_id)
        rows = [
            row
            for row in rows
            if f"clean_{row['orientation_id']}" in ids or row["orientation_id"] in ids
        ]
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("No orientations matched the requested subset")
    return rows, subset


def resolved_outputs(
    args: argparse.Namespace, subset: bool, forward_model: str
) -> tuple[Path, Path, Path]:
    suffix = "_smoke" if subset else ""
    model_suffix = "" if forward_model == "acom_matched" else "_first_born"
    image = (
        args.output
        or ROOT / "public" / f"clean_images{model_suffix}{suffix}.h5"
    )
    oracle = (
        args.oracle_output
        or ROOT
        / "private"
        / f"clean_physical_oracle_peaks{model_suffix}{suffix}.h5"
    )
    raw = (
        args.raw_output
        or ROOT
        / "private"
        / f"clean_physical_oracle_reflections{model_suffix}{suffix}.h5"
    )
    return image.resolve(), oracle.resolve(), raw.resolve()


def build_reflection_library(config: dict) -> tuple[ReflectionLibrary, object]:
    structure = Structure.from_file(cif_path(config))
    crystal = py4DSTEM.process.diffraction.Crystal.from_pymatgen_structure(
        structure=structure,
        conventional_standard_structure=False,
    )
    voltage = float(config["common"]["accelerating_voltage_V"])
    image_cfg = config["clean_image"]
    q_limit = float(image_cfg["q_max_Ainv"]) + 0.15
    crystal.setup_diffraction(accelerating_voltage=voltage)
    crystal.calculate_structure_factors(
        k_max=q_limit,
        tol_structure_factor=float(config["clean"]["tol_structure_factor"]),
    )
    return (
        ReflectionLibrary(
            g_crystal_Ainv=np.asarray(crystal.g_vec_all, dtype=float).T,
            hkl=np.rint(np.asarray(crystal.hkl, dtype=float).T).astype(np.int32),
            structure_factor=np.asarray(
                crystal.struct_factors, dtype=np.complex128
            ),
            wavelength_A=float(crystal.wavelength),
        ),
        crystal,
    )


def write_raw_reflections(
    path: Path,
    rows: list[dict],
    attrs: dict,
    reciprocal_basis_Ainv: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    offsets = [0]
    for row in rows:
        offsets.append(offsets[-1] + len(row["qx_Ainv"]))
    with h5py.File(temporary, "w") as h5:
        h5.create_dataset(
            "sample_id",
            data=np.asarray([row["sample_id"] for row in rows], dtype=h5py.string_dtype("utf-8")),
        )
        group = h5.create_group("reflections")
        for field, dtype in (
            ("qx_Ainv", np.float32),
            ("qy_Ainv", np.float32),
            ("qz_Ainv", np.float32),
            ("intensity_raw", np.float32),
            ("intensity_normalized", np.float32),
            ("structure_factor_real", np.float32),
            ("structure_factor_imag", np.float32),
            ("structure_factor_abs", np.float32),
            ("excitation_error_center_Ainv", np.float32),
        ):
            values = np.concatenate([np.asarray(row[field], dtype=dtype) for row in rows])
            group.create_dataset(field, data=values, compression="gzip")
        hkl = np.concatenate([np.asarray(row["hkl"], dtype=np.int32) for row in rows], axis=0)
        group.create_dataset("hkl", data=hkl, compression="gzip")
        for field in ("g_crystal_Ainv", "g_sample_Ainv"):
            values = np.concatenate(
                [np.asarray(row[field], dtype=np.float32) for row in rows],
                axis=0,
            )
            group.create_dataset(field, data=values, compression="gzip")
        group.create_dataset("offsets", data=np.asarray(offsets, dtype=np.int64))
        crystallography = h5.create_group("crystallography")
        crystallography.create_dataset(
            "reciprocal_basis_B_Ainv",
            data=np.asarray(reciprocal_basis_Ainv, dtype=np.float64),
        )
        crystallography.attrs["basis_convention"] = (
            "rows are a*, b*, c* in 1/angstrom without 2*pi"
        )
        crystallography.attrs["g_crystal_formula"] = "[h,k,l] @ B"
        crystallography.attrs["g_sample_formula"] = "g_crystal @ R_sample_to_crystal"
        diagnostics = h5.create_group("diagnostics")
        for field in (
            "candidate_reflection_count",
            "merged_disk_count",
            "rejected_edge_count",
            "rejected_low_intensity_count",
        ):
            diagnostics.create_dataset(
                field,
                data=np.asarray([row[field] for row in rows], dtype=np.int32),
            )
        for key, value in attrs.items():
            h5.attrs[key] = json.dumps(value) if isinstance(value, (dict, list, tuple)) else value
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    config = load_config()
    image_cfg = config["clean_image"]
    forward_model = str(args.forward_model or image_cfg["forward_model"])
    if args.compute_backend == "cuda" and forward_model != "coherent_first_born":
        raise ValueError(
            "--compute-backend cuda requires --forward-model coherent_first_born"
        )
    orientations, subset = selected_orientations(args)
    image_path, oracle_path, raw_path = resolved_outputs(
        args, subset, forward_model
    )
    for path in (image_path, oracle_path, raw_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    library, crystal = build_reflection_library(config)
    structure = Structure.from_file(cif_path(config))
    reciprocal_basis = np.asarray(
        structure.lattice.reciprocal_lattice_crystallographic.matrix,
        dtype=float,
    )
    structure_factor_by_hkl = {
        tuple(int(value) for value in hkl): value
        for hkl, value in zip(library.hkl, library.structure_factor)
    }
    ny, nx = (int(value) for value in image_cfg["gpts"])
    compression = str(image_cfg.get("compression", "gzip"))
    compression_opts = int(image_cfg.get("compression_level", 4))
    temporary_image = image_path.with_suffix(image_path.suffix + ".partial")
    oracle_samples: list[dict] = []
    raw_rows: list[dict] = []
    timings: list[float] = []
    render_timings: list[float] = []
    postprocess_timings: list[float] = []
    cuda_batch_timings: list[dict] = []
    generation_started = time.perf_counter()

    with h5py.File(temporary_image, "w") as h5:
        images = h5.create_dataset(
            "expectation/intensity",
            shape=(len(orientations), ny, nx),
            dtype=np.float32,
            chunks=(1, ny, nx),
            compression=compression,
            compression_opts=compression_opts,
        )
        h5.create_dataset(
            "sample_id",
            data=np.asarray(
                [f"clean_{row['orientation_id']}" for row in orientations],
                dtype=h5py.string_dtype("utf-8"),
            ),
        )
        orientation_group = h5.create_group("orientation")
        orientation_group.create_dataset(
            "canonical_matrix_sample_to_crystal",
            data=np.asarray(
                [
                    row["orientation_matrix_sample_to_crystal"]
                    for row in orientations
                ],
                dtype=np.float64,
            ),
        )
        orientation_group.create_dataset(
            "raw_matrix_sample_to_crystal",
            data=np.asarray(
                [
                    row.get(
                        "raw_orientation_matrix_sample_to_crystal",
                        row["orientation_matrix_sample_to_crystal"],
                    )
                    for row in orientations
                ],
                dtype=np.float64,
            ),
        )
        orientation_group.create_dataset(
            "orientation_class_id",
            data=np.asarray(
                [
                    row.get("orientation_class_id", row["orientation_id"])
                    for row in orientations
                ],
                dtype=h5py.string_dtype("utf-8"),
            ),
        )
        orientation_group.create_dataset(
            "canonical_crystal_symmetry_index",
            data=np.asarray(
                [
                    row.get("canonical_crystal_symmetry_index", -1)
                    for row in orientations
                ],
                dtype=np.int16,
            ),
        )
        orientation_group.create_dataset(
            "canonical_friedel_branch_index",
            data=np.asarray(
                [
                    row.get("canonical_friedel_branch_index", -1)
                    for row in orientations
                ],
                dtype=np.int8,
            ),
        )
        first = None
        rendered_batches: dict[int, object] = {}
        cuda_batch_size = int(
            args.batch_size
            or image_cfg.get("cuda_first_born", {}).get("batch_size", 16)
        )
        for index, orientation in enumerate(orientations):
            if args.compute_backend == "cuda" and index % cuda_batch_size == 0:
                batch_start = index
                batch_stop = min(
                    batch_start + cuda_batch_size, len(orientations)
                )
                batch_matrices = np.asarray(
                    [
                        row["orientation_matrix_sample_to_crystal"]
                        for row in orientations[batch_start:batch_stop]
                    ],
                    dtype=float,
                )
                cuda_batch_started = time.perf_counter()
                batch_rendered = render_kinematic_cbed_batch_cuda(
                    library,
                    batch_matrices,
                    image_cfg,
                    k_max_Ainv=float(config["common"]["k_max_Ainv"]),
                    direct_beam_fraction=args.direct_beam_fraction,
                )
                cuda_batch_elapsed = time.perf_counter() - cuda_batch_started
                cuda_batch_timings.append(
                    {
                        "start_index": batch_start,
                        "stop_index": batch_stop,
                        "num_patterns": batch_stop - batch_start,
                        "seconds": cuda_batch_elapsed,
                        "seconds_per_pattern": (
                            cuda_batch_elapsed / (batch_stop - batch_start)
                        ),
                    }
                )
                rendered_batches = {
                    batch_start + offset: value
                    for offset, value in enumerate(batch_rendered)
                }
            sample_id = f"clean_{orientation['orientation_id']}"
            matrix = np.asarray(
                orientation["orientation_matrix_sample_to_crystal"], dtype=float
            )
            if args.compute_backend == "cuda":
                rendered = rendered_batches[index]
                render_elapsed = cuda_batch_timings[-1][
                    "seconds_per_pattern"
                ]
            elif forward_model == "acom_matched":
                render_started = time.perf_counter()
                point_list = crystal.generate_diffraction_pattern(
                    orientation_matrix=matrix,
                    sigma_excitation_error=float(
                        config["clean"]["sigma_excitation_error_Ainv"]
                    ),
                    tol_excitation_error_mult=float(
                        config["clean"]["tol_excitation_error_mult"]
                    ),
                    tol_intensity=float(config["clean"]["tol_intensity"]),
                    k_max=float(config["common"]["k_max_Ainv"]),
                )
                data = point_list.data
                rendered = render_acom_matched_cbed(
                    ProjectedReflectionSet(
                        qx_Ainv=np.asarray(data["qx"], dtype=float),
                        qy_Ainv=np.asarray(data["qy"], dtype=float),
                        intensity=np.asarray(data["intensity"], dtype=float),
                        hkl=np.column_stack(
                            (data["h"], data["k"], data["l"])
                        ).astype(np.int32),
                        wavelength_A=library.wavelength_A,
                    ),
                    image_cfg,
                    k_max_Ainv=float(config["common"]["k_max_Ainv"]),
                    direct_beam_fraction=args.direct_beam_fraction,
                )
                render_elapsed = time.perf_counter() - render_started
            else:
                render_started = time.perf_counter()
                rendered = render_kinematic_cbed(
                    library,
                    matrix,
                    image_cfg,
                    k_max_Ainv=float(config["common"]["k_max_Ainv"]),
                    direct_beam_fraction=args.direct_beam_fraction,
                )
                render_elapsed = time.perf_counter() - render_started
            postprocess_started = time.perf_counter()
            images[index] = rendered.expectation
            if first is None:
                first = rendered
                detector = h5.create_group("detector")
                detector.create_dataset("qx_Ainv", data=rendered.qx_axis_Ainv)
                detector.create_dataset("qy_Ainv", data=rendered.qy_axis_Ainv)
                detector.create_dataset("vacuum_probe", data=rendered.vacuum_probe, compression="gzip")
                detector.create_dataset("valid_mask", data=np.ones((ny, nx), dtype=np.uint8))
            oracle_samples.append(
                {
                    "sample_id": sample_id,
                    "qx": rendered.oracle_qx_Ainv,
                    "qy": rendered.oracle_qy_Ainv,
                    "intensity": rendered.oracle_intensity_normalized,
                }
            )
            raw_rows.append(
                {
                    "sample_id": sample_id,
                    "qx_Ainv": rendered.oracle_qx_Ainv,
                    "qy_Ainv": rendered.oracle_qy_Ainv,
                    "intensity_raw": rendered.oracle_intensity_raw,
                    "intensity_normalized": rendered.oracle_intensity_normalized,
                    "hkl": rendered.oracle_hkl,
                    "g_crystal_Ainv": (
                        np.asarray(rendered.oracle_hkl, dtype=float)
                        @ reciprocal_basis
                    ),
                    "candidate_reflection_count": (
                        rendered.oracle_candidate_reflection_count
                    ),
                    "merged_disk_count": rendered.oracle_merged_disk_count,
                    "rejected_edge_count": rendered.oracle_rejected_edge_count,
                    "rejected_low_intensity_count": (
                        rendered.oracle_rejected_low_intensity_count
                    ),
                }
            )
            g_crystal = raw_rows[-1]["g_crystal_Ainv"]
            g_sample = g_crystal @ matrix
            structure_factors = np.asarray(
                [
                    structure_factor_by_hkl.get(
                        tuple(int(value) for value in hkl), np.nan + 1j * np.nan
                    )
                    for hkl in rendered.oracle_hkl
                ],
                dtype=np.complex128,
            )
            k0_Ainv = 1.0 / float(library.wavelength_A)
            qxy_sq = (
                np.asarray(rendered.oracle_qx_Ainv, dtype=float) ** 2
                + np.asarray(rendered.oracle_qy_Ainv, dtype=float) ** 2
            )
            excitation_error = (
                np.sqrt(np.maximum(k0_Ainv**2 - qxy_sq, 0.0))
                - k0_Ainv
                - g_sample[:, 2]
            )
            raw_rows[-1].update(
                {
                    "g_sample_Ainv": g_sample,
                    "qz_Ainv": g_sample[:, 2],
                    "structure_factor_real": structure_factors.real,
                    "structure_factor_imag": structure_factors.imag,
                    "structure_factor_abs": np.abs(structure_factors),
                    "excitation_error_center_Ainv": excitation_error,
                }
            )
            postprocess_elapsed = time.perf_counter() - postprocess_started
            elapsed = render_elapsed + postprocess_elapsed
            timings.append(elapsed)
            render_timings.append(render_elapsed)
            postprocess_timings.append(postprocess_elapsed)
            print(
                f"{index + 1}/{len(orientations)} {sample_id}: "
                f"peaks={len(rendered.oracle_qx_Ainv)}, "
                f"candidates={rendered.oracle_candidate_reflection_count}, "
                f"merged={rendered.oracle_merged_disk_count}, "
                f"max_probability={rendered.expectation.max():.4g}, "
                f"render_seconds={render_elapsed:.3f}, "
                f"postprocess_seconds={postprocess_elapsed:.3f}, "
                f"amortized_seconds={elapsed:.3f}"
            )
            h5.flush()
            rendered_batches.pop(index, None)

        assert first is not None
        h5.attrs["dataset_id"] = config["dataset"]["id"]
        h5.attrs["track"] = "clean_expectation"
        h5.attrs["forward_model"] = forward_model
        h5.attrs["compute_backend"] = args.compute_backend
        h5.attrs["input_type"] = "diffraction_image"
        h5.attrs["intensity_model"] = (
            "py4DSTEM ACOM-matched kinematical support/intensity with coherent finite disks"
            if forward_model == "acom_matched"
            else "coherent first-Born kinematical CBED with finite-thickness sinc amplitude"
        )
        if forward_model == "coherent_first_born":
            h5.attrs["sinc_convention"] = (
                "t_A * numpy.sinc(t_A * delta_kz_Ainv)"
            )
        h5.attrs["normalization"] = "sum(expectation/intensity)==1"
        h5.attrs["direct_beam_fraction"] = float(
            image_cfg["canonical_direct_beam_fraction"]
            if args.direct_beam_fraction is None
            else args.direct_beam_fraction
        )
        h5.attrs["coordinate_units"] = "1/angstrom"
        h5.attrs["beam_center_px_row_col"] = json.dumps(first.beam_center_px)
        h5.attrs["disk_radius_px"] = first.disk_radius_px
        h5.attrs["voltage_kV"] = float(config["common"]["accelerating_voltage_V"]) / 1000.0
        h5.attrs["semiangle_mrad"] = float(image_cfg["convergence_semiangle_mrad"])
        h5.attrs["thickness_nm"] = float(image_cfg["thickness_nm"])
        h5.attrs["q_max_Ainv"] = float(image_cfg["q_max_Ainv"])
        h5.attrs["k_max_Ainv"] = float(config["common"]["k_max_Ainv"])
        h5.attrs["generator"] = "scripts/02b_generate_clean_images.py"
        h5.attrs["source_orientation_file"] = str(
            args.orientation_file.resolve()
        )
        h5.attrs["scattered_reconstruction_formula"] = (
            "(expectation - direct_beam_fraction * normalized(vacuum_probe)) "
            "/ (1 - direct_beam_fraction)"
        )
        h5.attrs["generator_config"] = json.dumps(image_cfg, sort_keys=True)
    temporary_image.replace(image_path)

    common_attrs = {
        "dataset_id": config["dataset"]["id"],
        "track": "clean_physical_oracle",
        "forward_model": forward_model,
        "input_fields": ["qx", "qy", "intensity"],
        "coordinate_units": "1/angstrom",
        "k_max_Ainv": float(config["common"]["k_max_Ainv"]),
        "source_image_file": str(image_path.relative_to(ROOT)) if image_path.is_relative_to(ROOT) else str(image_path),
        "intensity_model": (
            "disk integration from the final coherent scattered expectation image"
        ),
        "oracle_definition": {
            "merge_distance_px": image_cfg["oracle_merge_distance_px"],
            "integration_radius_fraction": image_cfg[
                "oracle_integration_radius_fraction"
            ],
            "require_full_disk": image_cfg["oracle_require_full_disk"],
            "min_relative_intensity": image_cfg[
                "physical_oracle_min_relative_intensity"
            ],
        },
    }
    write_peak_h5(oracle_path, oracle_samples, common_attrs)
    write_raw_reflections(
        raw_path, raw_rows, common_attrs, reciprocal_basis
    )
    generation_wall_seconds = time.perf_counter() - generation_started

    report = {
        "num_samples": len(orientations),
        "forward_model": forward_model,
        "compute_backend": args.compute_backend,
        "k_max_Ainv": float(config["common"]["k_max_Ainv"]),
        "detector_q_max_Ainv": float(image_cfg["q_max_Ainv"]),
        "cuda_batch_size": (
            cuda_batch_size if args.compute_backend == "cuda" else None
        ),
        "roles": sorted({row["sample_role"] for row in orientations}),
        "image_path": str(image_path),
        "physical_oracle_path": str(oracle_path),
        "raw_reflections_path": str(raw_path),
        "image_shape": [ny, nx],
        "mean_amortized_seconds_per_pattern": float(np.mean(timings)),
        "total_amortized_pattern_seconds": float(np.sum(timings)),
        "mean_render_seconds_per_pattern": float(np.mean(render_timings)),
        "total_render_seconds": float(np.sum(render_timings)),
        "mean_postprocess_seconds_per_pattern": float(
            np.mean(postprocess_timings)
        ),
        "total_postprocess_seconds": float(np.sum(postprocess_timings)),
        "generation_wall_seconds": generation_wall_seconds,
        "cuda_batches": cuda_batch_timings,
        "oracle_diagnostics": {
            "candidate_reflections": int(
                sum(row["candidate_reflection_count"] for row in raw_rows)
            ),
            "merged_disks_before_threshold": int(
                sum(row["merged_disk_count"] for row in raw_rows)
            ),
            "retained_disks": int(
                sum(len(row["qx_Ainv"]) for row in raw_rows)
            ),
            "rejected_edge": int(
                sum(row["rejected_edge_count"] for row in raw_rows)
            ),
            "rejected_low_intensity": int(
                sum(row["rejected_low_intensity_count"] for row in raw_rows)
            ),
        },
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "py4DSTEM": py4DSTEM.__version__,
            "pymatgen": importlib.metadata.version("pymatgen"),
            "h5py": h5py.__version__,
        },
    }
    report_suffix = "_smoke" if subset else ""
    report_model_suffix = (
        "" if forward_model == "acom_matched" else "_first_born"
    )
    report_path = (
        args.report_output.resolve()
        if args.report_output is not None
        else REPORT_DIR
        / f"clean_image_generation{report_model_suffix}{report_suffix}.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Images: {image_path}")
    print(f"Physical oracle peaks: {oracle_path}")
    print(f"Raw physical reflections: {raw_path}")
    print(f"Generation report: {report_path}")


if __name__ == "__main__":
    main()
