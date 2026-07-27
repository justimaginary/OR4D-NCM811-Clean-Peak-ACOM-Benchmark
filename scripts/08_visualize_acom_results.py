#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import py4DSTEM
from pymatgen.core import Structure

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from or4d_common import (  # noqa: E402
    cif_path,
    load_config,
    normalize_intensities,
    read_peak_h5,
)


def load_details() -> dict:
    path = ROOT / "reports" / "acom_clean_details.json"
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def setup_crystal(config: dict):
    structure = Structure.from_file(cif_path(config))
    crystal = py4DSTEM.process.diffraction.Crystal.from_pymatgen_structure(
        structure=structure,
        conventional_standard_structure=False,
    )
    voltage = float(config["common"]["accelerating_voltage_V"])
    k_max = float(config["common"]["k_max_Ainv"])
    crystal.setup_diffraction(accelerating_voltage=voltage)
    crystal.calculate_structure_factors(
        k_max=k_max,
        tol_structure_factor=float(config["clean"]["tol_structure_factor"]),
    )
    return crystal


def simulate_filtered_peaks(crystal, R: np.ndarray, config: dict) -> dict:
    clean = config["clean"]
    common = config["common"]
    k_max = float(common["k_max_Ainv"])
    central_exclusion = float(common["central_beam_exclusion_Ainv"])
    point_list = crystal.generate_diffraction_pattern(
        orientation_matrix=R,
        sigma_excitation_error=float(clean["sigma_excitation_error_Ainv"]),
        tol_excitation_error_mult=float(clean["tol_excitation_error_mult"]),
        tol_intensity=float(clean["tol_intensity"]),
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
    return {
        "qx": qx[keep],
        "qy": qy[keep],
        "intensity": normalize_intensities(intensity[keep]),
    }


def plot_error_comparison(rows: list[dict]) -> None:
    sample_ids = [row["sample_id"].replace("clean_", "") for row in rows]
    strict = np.asarray([row["strict_misorientation_deg"] for row in rows])
    friedel = np.asarray(
        [row["friedel_equivalent_misorientation_deg"] for row in rows]
    )
    mirrors = np.asarray([row["mirror_match"] for row in rows], dtype=bool)

    x = np.arange(len(rows))
    width = 0.38
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.bar(x - width / 2, strict, width, label="Strict crystal orientation")
    ax.bar(x + width / 2, friedel, width, label="Friedel-equivalent Clean-Peak")
    for threshold, style in ((1.0, ":"), (2.0, "--"), (5.0, "-.")):
        ax.axhline(threshold, linestyle=style, linewidth=1, label=f"{threshold:g}°")
    mirror_y = np.maximum(strict, friedel) + 0.3
    ax.scatter(x[mirrors], mirror_y[mirrors], marker="*", s=80, label="ACOM mirror branch")
    ax.set_xticks(x)
    ax.set_xticklabels(sample_ids, rotation=55, ha="right")
    ax.set_ylabel("Misorientation (degrees)")
    ax.set_title("ACOM Clean-Peak orientation errors")
    ax.legend(ncol=3)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = ROOT / "reports" / "acom_error_comparison.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_offgrid_relation(rows: list[dict]) -> None:
    groups = {}
    for row in rows:
        groups.setdefault(row.get("sampling_type", "unknown"), []).append(row)

    fig, ax = plt.subplots(figsize=(8, 6))
    markers = ["o", "s", "^"]
    for marker, (name, group) in zip(markers, sorted(groups.items())):
        x = [
            row["nearest_search_node_friedel_equivalent_misorientation_deg"]
            for row in group
        ]
        y = [row["friedel_equivalent_misorientation_deg"] for row in group]
        ax.scatter(x, y, marker=marker, s=70, label=name)
        for row, x_value, y_value in zip(group, x, y):
            ax.annotate(
                row["sample_id"].replace("clean_", ""),
                (x_value, y_value),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )
    limit = max(
        3.2,
        max(
            row["nearest_search_node_friedel_equivalent_misorientation_deg"]
            for row in rows
        )
        + 0.3,
    )
    ax.plot([0, limit], [0, limit], linestyle="--", label="y = x")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Distance to nearest searched ACOM node (degrees)")
    ax.set_ylabel("Friedel-equivalent prediction error (degrees)")
    ax.set_title("Off-grid distance versus ACOM prediction error")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path = ROOT / "reports" / "acom_offgrid_vs_error.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_peak_overlays(rows: list[dict], samples: list[dict], config: dict) -> None:
    crystal = setup_crystal(config)
    samples_by_id = {sample["sample_id"]: sample for sample in samples}
    ncols = 4
    nrows = math.ceil(len(rows) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4.1 * nrows))
    axes = np.atleast_1d(axes).ravel()
    q_max = float(config["common"]["k_max_Ainv"])

    for ax, row in zip(axes, rows):
        observed = samples_by_id[row["sample_id"]]
        R_pred = np.asarray(
            row["predicted_orientation_matrix_sample_to_crystal"], dtype=float
        )
        predicted = simulate_filtered_peaks(crystal, R_pred, config)

        observed_sizes = 18.0 + 100.0 * np.asarray(observed["intensity"])
        predicted_sizes = 18.0 + 70.0 * np.asarray(predicted["intensity"])
        ax.scatter(
            observed["qx"],
            observed["qy"],
            s=observed_sizes,
            facecolors="none",
            edgecolors="C0",
            linewidths=1.0,
            label="Benchmark input peaks",
        )
        ax.scatter(
            predicted["qx"],
            predicted["qy"],
            s=predicted_sizes,
            marker="x",
            color="C1",
            linewidths=0.9,
            label="Peaks from ACOM prediction",
        )
        ax.scatter([0.0], [0.0], marker="+", s=50, color="C2")
        mirror_text = "mirror" if row["mirror_match"] else "normal"
        ax.set_title(
            f"{row['sample_id'].replace('clean_', '')} | {mirror_text}\n"
            f"strict={row['strict_misorientation_deg']:.2f}°, "
            f"Friedel={row['friedel_equivalent_misorientation_deg']:.2f}°",
            fontsize=10,
        )
        ax.set_xlim(-q_max, q_max)
        ax.set_ylim(-q_max, q_max)
        ax.set_aspect("equal")
        ax.grid(alpha=0.2)
        ax.set_xlabel(r"$q_x$ ($\AA^{-1}$)")
        ax.set_ylabel(r"$q_y$ ($\AA^{-1}$)")

    for ax in axes[len(rows):]:
        ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3)
    fig.suptitle("Clean-Peak input versus peaks simulated from ACOM predictions", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    path = ROOT / "reports" / "acom_peak_overlay.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved: {path}")


def main() -> None:
    config = load_config()
    details = load_details()
    rows = details["samples"]
    samples = read_peak_h5(ROOT / "public" / "clean_peaks.h5")
    plot_error_comparison(rows)
    plot_offgrid_relation(rows)
    plot_peak_overlays(rows, samples, config)


if __name__ == "__main__":
    main()
