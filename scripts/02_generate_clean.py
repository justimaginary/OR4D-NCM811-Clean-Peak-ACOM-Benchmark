#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import py4DSTEM
from pymatgen.core import Structure

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from or4d_common import (  # noqa: E402
    cif_path,
    load_config,
    normalize_intensities,
    read_jsonl,
    write_jsonl,
    write_peak_h5,
)


def main() -> None:
    config = load_config()
    orientations = read_jsonl(ROOT / "private" / "orientations.jsonl")
    structure = Structure.from_file(cif_path(config))

    crystal = py4DSTEM.process.diffraction.Crystal.from_pymatgen_structure(
        structure=structure,
        conventional_standard_structure=False,
    )

    voltage = float(config["common"]["accelerating_voltage_V"])
    k_max = float(config["common"]["k_max_Ainv"])
    central_exclusion = float(config["common"]["central_beam_exclusion_Ainv"])
    clean_cfg = config["clean"]

    crystal.setup_diffraction(accelerating_voltage=voltage)
    crystal.calculate_structure_factors(
        k_max=k_max,
        tol_structure_factor=float(clean_cfg["tol_structure_factor"]),
    )

    public_samples = []
    gt_records = []
    diagnostic_rows = []

    for orientation in orientations:
        orientation_id = orientation["orientation_id"]
        sample_id = f"clean_{orientation_id}"
        R = np.asarray(orientation["orientation_matrix_sample_to_crystal"], dtype=float)

        point_list = crystal.generate_diffraction_pattern(
            orientation_matrix=R,
            sigma_excitation_error=float(clean_cfg["sigma_excitation_error_Ainv"]),
            tol_excitation_error_mult=float(clean_cfg["tol_excitation_error_mult"]),
            tol_intensity=float(clean_cfg["tol_intensity"]),
            k_max=k_max,
        )
        data = point_list.data
        qx = np.asarray(data["qx"], dtype=float)
        qy = np.asarray(data["qy"], dtype=float)
        intensity = np.asarray(data["intensity"], dtype=float)

        radius = np.hypot(qx, qy)
        keep = (
            np.isfinite(qx)
            & np.isfinite(qy)
            & np.isfinite(intensity)
            & (radius >= central_exclusion)
            & (radius <= k_max + 1e-8)
            & (intensity > 0)
        )
        qx = qx[keep]
        qy = qy[keep]
        intensity = normalize_intensities(intensity[keep])

        order = np.argsort(np.hypot(qx, qy))
        qx, qy, intensity = qx[order], qy[order], intensity[order]
        public_samples.append(
            {"sample_id": sample_id, "qx": qx, "qy": qy, "intensity": intensity}
        )

        gt_record = {
            "sample_id": sample_id,
            "orientation_matrix_sample_to_crystal": R.tolist(),
            "zone_axis_3index": orientation.get("zone_axis_3index"),
            "x_reference_uvw": orientation.get("x_reference_uvw"),
            "in_plane_rotation_deg": orientation.get("in_plane_rotation_deg"),
            "beam_direction_crystal_cartesian": orientation.get(
                "beam_direction_crystal_cartesian"
            ),
            "sampling_type": orientation.get("sampling_type", "manual"),
            "track": "clean",
        }
        if orientation.get("quaternion_wxyz") is not None:
            gt_record["quaternion_wxyz"] = orientation["quaternion_wxyz"]
        gt_records.append(gt_record)

        fields = data.dtype.names or ()
        h = np.asarray(data["h"])[keep][order] if "h" in fields else np.zeros(len(qx), dtype=int)
        k = np.asarray(data["k"])[keep][order] if "k" in fields else np.zeros(len(qx), dtype=int)
        l = np.asarray(data["l"])[keep][order] if "l" in fields else np.zeros(len(qx), dtype=int)
        for idx in range(len(qx)):
            diagnostic_rows.append((sample_id, qx[idx], qy[idx], intensity[idx], h[idx], k[idx], l[idx]))

        print(f"{sample_id}: {len(qx)} peaks")

    write_peak_h5(
        ROOT / "public" / "clean_peaks.h5",
        public_samples,
        attrs={
            "dataset_id": config["dataset"]["id"],
            "track": "clean",
            "input_fields": ["qx", "qy", "intensity"],
            "coordinate_units": "1/angstrom",
            "py4DSTEM_version": py4DSTEM.__version__,
        },
    )
    write_jsonl(ROOT / "private" / "clean_ground_truth.jsonl", gt_records)

    preview = []
    for sample in public_samples:
        preview.append(
            {
                "sample_id": sample["sample_id"],
                "peaks": [
                    {"qx": float(x), "qy": float(y), "intensity": float(i)}
                    for x, y, i in zip(sample["qx"], sample["qy"], sample["intensity"])
                ],
            }
        )
    write_jsonl(ROOT / "diagnostics" / "clean_peaks_preview.jsonl", preview)

    dtype = np.dtype(
        [
            ("sample_id", h5py.string_dtype("utf-8")),
            ("qx", "f4"),
            ("qy", "f4"),
            ("intensity", "f4"),
            ("h", "i4"),
            ("k", "i4"),
            ("l", "i4"),
        ]
    )
    diag = np.empty(len(diagnostic_rows), dtype=dtype)
    for i, row in enumerate(diagnostic_rows):
        diag[i] = row
    with h5py.File(ROOT / "diagnostics" / "clean_reflections.h5", "w") as h5:
        h5.create_dataset("reflections", data=diag, compression="gzip")

    with (ROOT / "reports" / "clean_versions.json").open("w", encoding="utf-8") as f:
        json.dump({"py4DSTEM": py4DSTEM.__version__}, f, indent=2)

    print("Clean-Peak dataset finished.")


if __name__ == "__main__":
    main()
