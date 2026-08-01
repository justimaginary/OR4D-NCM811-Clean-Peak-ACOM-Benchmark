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
REPORT_DIR = ROOT / "reports" / "v3"
sys.path.insert(0, str(ROOT / "src"))

from or4d_common import (  # noqa: E402
    cif_path,
    load_config,
    normalize_intensities,
    read_peak_h5,
)


def load_details() -> dict:
    path = REPORT_DIR / "acom_clean_details.json"
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
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row.get("sample_role", "unspecified"), []).append(row)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for role, group in sorted(groups.items()):
        errors = np.sort(
            [row["friedel_equivalent_misorientation_deg"] for row in group]
        )
        cumulative = np.arange(1, len(errors) + 1) / len(errors)
        axes[0].step(errors, cumulative, where="post", label=f"{role} (n={len(group)})")
    for threshold, style in ((1.0, ":"), (2.0, "--"), (5.0, "-.")):
        axes[0].axvline(threshold, linestyle=style, linewidth=1)
    axes[0].set_xlabel("Friedel-equivalent misorientation (degrees)")
    axes[0].set_ylabel("Empirical cumulative fraction")
    axes[0].set_title("Error ECDF by sample role")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    headline = groups.get("headline_core", rows)
    friedel = np.sort(
        [row["friedel_equivalent_misorientation_deg"] for row in headline]
    )
    strict = np.sort([row["strict_misorientation_deg"] for row in headline])
    rank = np.arange(1, len(headline) + 1)
    axes[1].plot(rank, friedel, label="Friedel-equivalent")
    axes[1].plot(rank, strict, label="Strict", alpha=0.8)
    for threshold, style in ((1.0, ":"), (2.0, "--"), (5.0, "-.")):
        axes[1].axhline(threshold, linestyle=style, linewidth=1)
    axes[1].set_xlabel("Sorted headline sample rank")
    axes[1].set_ylabel("Misorientation (degrees)")
    axes[1].set_title("Headline strict versus Friedel error")
    axes[1].set_yscale("symlog", linthresh=5.0)
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    path = REPORT_DIR / "acom_error_comparison.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_offgrid_relation(rows: list[dict]) -> None:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row.get("sample_role", "unspecified"), []).append(row)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharex=True)
    markers = ["o", "s", "^", "D"]
    for ax in axes:
        for marker, (name, group) in zip(markers, sorted(groups.items())):
            x = [
                row["nearest_zone_axis_node_misorientation_deg"]
                for row in group
            ]
            y = [row["friedel_equivalent_misorientation_deg"] for row in group]
            ax.scatter(x, y, marker=marker, s=28, alpha=0.65, label=name)
    limit = max(
        3.2,
        max(
            row["nearest_zone_axis_node_misorientation_deg"]
            for row in rows
        )
        + 0.3,
    )
    for ax in axes:
        ax.plot([0, limit], [0, limit], linestyle="--", label="y = x")
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("Nearest zone-axis node distance (degrees)")
        ax.grid(alpha=0.25)
    axes[0].set_ylim(0, 5)
    axes[0].set_ylabel("Friedel-equivalent prediction error (degrees)")
    axes[0].set_title("Typical-error view")
    axes[0].legend()
    axes[1].set_title("Full range with catastrophic mismatches")
    fig.tight_layout()
    path = REPORT_DIR / "acom_offgrid_vs_error.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    print(f"Saved: {path}")


def select_overlay_rows(rows: list[dict], max_rows: int) -> list[dict]:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row.get("sample_role", "unspecified"), []).append(row)
    roles = sorted(groups)
    selected: list[dict] = []
    remaining = max_rows
    for role_index, role in enumerate(roles):
        group = sorted(
            groups[role],
            key=lambda row: row["friedel_equivalent_misorientation_deg"],
        )
        roles_left = len(roles) - role_index
        quota = max(1, remaining // roles_left)
        quota = min(quota, len(group))
        indices = np.linspace(0, len(group) - 1, quota, dtype=int)
        selected.extend(group[index] for index in indices)
        remaining -= quota
    return selected[:max_rows]


def plot_peak_overlays(rows: list[dict], samples: list[dict], config: dict) -> None:
    crystal = setup_crystal(config)
    samples_by_id = {sample["sample_id"]: sample for sample in samples}
    rows = select_overlay_rows(
        rows,
        max_rows=int(config["evaluation"]["max_peak_overlay_samples"]),
    )
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
    fig.suptitle(
        "Clean-Peak input versus peaks simulated from ACOM predictions",
        y=0.995,
    )
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.973),
        ncol=3,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path = REPORT_DIR / "acom_peak_overlay.png"
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
