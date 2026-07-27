#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import abtem
import numpy as np
from ase import Atoms
from pymatgen.core import Structure
from scipy.ndimage import gaussian_filter
from skimage.feature import peak_local_max

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


def species_symbol(species) -> str:
    element = getattr(species, "element", species)
    return str(getattr(element, "symbol", element))


def sample_explicit_disorder(
    cif: Path,
    supercell: list[int],
    seed: int,
) -> Atoms:
    structure = Structure.from_file(cif)
    structure.make_supercell(np.diag(supercell))
    rng = np.random.default_rng(seed)

    symbols: list[str] = []
    for site in structure:
        species_items = list(site.species.items())
        probs = np.asarray([float(occ) for _, occ in species_items], dtype=float)
        probs /= probs.sum()
        index = int(rng.choice(len(species_items), p=probs))
        symbols.append(species_symbol(species_items[index][0]))

    return Atoms(
        symbols=symbols,
        scaled_positions=np.asarray(structure.frac_coords, dtype=float),
        cell=np.asarray(structure.lattice.matrix, dtype=float),
        pbc=True,
    )


def extract_peaks(
    image: np.ndarray,
    qx_axis: np.ndarray,
    qy_axis: np.ndarray,
    *,
    k_max: float,
    central_exclusion: float,
    gaussian_sigma_px: float,
    threshold_rel: float,
    min_distance_px: int,
    integration_radius_px: int,
    max_num_peaks: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image = np.asarray(image, dtype=float)
    image = np.maximum(image, 0.0)
    if image.max() <= 0:
        raise RuntimeError("Dynamical diffraction image contains no positive intensity.")

    qx_grid, qy_grid = np.meshgrid(qx_axis, qy_axis)
    q_radius = np.hypot(qx_grid, qy_grid)
    work = np.log1p(image / max(np.percentile(image, 99.5), 1e-30))
    work[(q_radius < central_exclusion) | (q_radius > k_max)] = 0.0
    work = gaussian_filter(work, sigma=gaussian_sigma_px)

    coords = peak_local_max(
        work,
        min_distance=int(min_distance_px),
        threshold_rel=float(threshold_rel),
        num_peaks=int(max_num_peaks),
        exclude_border=False,
    )

    peaks: list[tuple[float, float, float]] = []
    ny, nx = image.shape
    radius = int(integration_radius_px)
    for row, col in coords:
        r0, r1 = max(0, row - radius), min(ny, row + radius + 1)
        c0, c1 = max(0, col - radius), min(nx, col + radius + 1)
        patch = image[r0:r1, c0:c1]
        weights = np.maximum(patch, 0.0)
        total = float(weights.sum())
        if total <= 0:
            continue

        rows = np.arange(r0, r1, dtype=float)[:, None]
        cols = np.arange(c0, c1, dtype=float)[None, :]
        row_c = float((weights * rows).sum() / total)
        col_c = float((weights * cols).sum() / total)

        qx = float(np.interp(col_c, np.arange(nx), qx_axis))
        qy = float(np.interp(row_c, np.arange(ny), qy_axis))
        q = math.hypot(qx, qy)
        if central_exclusion <= q <= k_max:
            peaks.append((qx, qy, total))

    if not peaks:
        raise RuntimeError("Peak extraction returned zero peaks.")

    arr = np.asarray(peaks, dtype=float)
    order = np.argsort(np.hypot(arr[:, 0], arr[:, 1]))
    arr = arr[order]
    return (
        arr[:, 0].astype(np.float32),
        arr[:, 1].astype(np.float32),
        normalize_intensities(arr[:, 2]),
    )


def get_q_axes(dp) -> tuple[np.ndarray, np.ndarray]:
    coords = tuple(np.asarray(x, dtype=float) for x in dp.coordinates)
    ny, nx = dp.shape[-2], dp.shape[-1]
    if len(coords[0]) == nx and len(coords[1]) == ny:
        return coords[0], coords[1]
    if len(coords[0]) == ny and len(coords[1]) == nx:
        return coords[1], coords[0]
    raise RuntimeError(
        f"Could not map abTEM coordinates to array axes: shape={dp.shape}, "
        f"coordinate lengths={[len(x) for x in coords]}"
    )


def main() -> None:
    config = load_config()
    dyn = config["dynamical"]
    common = config["common"]
    orientations = read_jsonl(ROOT / "private" / "orientations.jsonl")

    abtem.config.set({"device": "cpu"})

    public_samples = []
    gt_records = []
    metadata_records = []
    image_dir = ROOT / "diagnostics" / "dynamical_images"
    image_dir.mkdir(parents=True, exist_ok=True)

    target_thickness_A = float(dyn["thickness_nm"]) * 10.0
    energy = float(common["accelerating_voltage_V"])
    k_max = float(common["k_max_Ainv"])
    central_exclusion = float(common["central_beam_exclusion_Ainv"])

    for index, orientation in enumerate(orientations):
        orientation_id = orientation["orientation_id"]
        sample_id = f"dynamical_{orientation_id}_t{int(dyn['thickness_nm']):03d}"
        R = np.asarray(orientation["orientation_matrix_sample_to_crystal"], dtype=float)
        disorder_seed = int(dyn["disorder_seed_base"]) + index
        phonon_seed = int(dyn["frozen_phonon_seed_base"]) + index

        atoms = sample_explicit_disorder(
            cif=cif_path(config),
            supercell=[int(x) for x in dyn["disorder_supercell"]],
            seed=disorder_seed,
        )

        atoms, transform = abtem.orthogonalize_cell(
            atoms,
            max_repetitions=int(dyn["orthogonalize_max_repetitions"]),
            return_transform=True,
            plane=(tuple(R[:, 0]), tuple(R[:, 1])),
        )

        z_repeat = max(1, int(math.ceil(target_thickness_A / atoms.cell.lengths()[2])))
        xy_repeat = [int(x) for x in dyn["xy_repeat"]]
        atoms = atoms * (xy_repeat[0], xy_repeat[1], z_repeat)
        actual_thickness_A = float(atoms.cell.lengths()[2])

        frozen = abtem.FrozenPhonons(
            atoms,
            num_configs=int(dyn["frozen_phonon_configs"]),
            sigmas=float(dyn["frozen_phonon_sigma_A"]),
            seed=phonon_seed,
            ensemble_mean=True,
        )
        potential = abtem.Potential(
            frozen,
            sampling=float(dyn["real_space_sampling_A"]),
            slice_thickness=float(dyn["slice_thickness_A"]),
        )
        probe = abtem.Probe(
            energy=energy,
            semiangle_cutoff=float(dyn["semiangle_cutoff_mrad"]),
            defocus=float(dyn["defocus_A"]),
        )
        probe.grid.match(potential)

        scan = abtem.GridScan(
            start=(0.0, 0.0),
            end=(1.0, 1.0),
            gpts=tuple(int(x) for x in dyn["scan_gpts"]),
            fractional=True,
            potential=potential,
        )
        exit_waves = probe.multislice(potential, scan=scan)
        dp = exit_waves.diffraction_patterns(
            max_angle=float(dyn["max_angle_mrad"]),
            block_direct=False,
            fftshift=True,
            parity="odd",
        ).compute(progress_bar=True)

        image = np.asarray(dp.array, dtype=float)
        if image.ndim > 2:
            image = image.mean(axis=tuple(range(image.ndim - 2)))
        qx_axis, qy_axis = get_q_axes(dp)

        peak_cfg = dyn["peak_extraction"]
        qx, qy, intensity = extract_peaks(
            image,
            qx_axis,
            qy_axis,
            k_max=k_max,
            central_exclusion=central_exclusion,
            gaussian_sigma_px=float(peak_cfg["gaussian_sigma_px"]),
            threshold_rel=float(peak_cfg["threshold_rel"]),
            min_distance_px=int(peak_cfg["min_distance_px"]),
            integration_radius_px=int(peak_cfg["integration_radius_px"]),
            max_num_peaks=int(peak_cfg["max_num_peaks"]),
        )

        public_samples.append(
            {"sample_id": sample_id, "qx": qx, "qy": qy, "intensity": intensity}
        )
        gt_records.append(
            {
                "sample_id": sample_id,
                "orientation_matrix_sample_to_crystal": R.tolist(),
                "zone_axis_3index": orientation["zone_axis_3index"],
                "in_plane_rotation_deg": orientation["in_plane_rotation_deg"],
                "track": "dynamical",
                "thickness_nm": actual_thickness_A / 10.0,
            }
        )
        metadata_records.append(
            {
                "sample_id": sample_id,
                "disorder_seed": disorder_seed,
                "frozen_phonon_seed": phonon_seed,
                "actual_thickness_A": actual_thickness_A,
                "num_atoms": len(atoms),
                "cell_lengths_A": atoms.cell.lengths().tolist(),
                "orthogonalize_transform": [np.asarray(x).tolist() for x in transform],
                "num_peaks": int(len(qx)),
            }
        )
        np.savez_compressed(
            image_dir / f"{sample_id}.npz",
            intensity=image.astype(np.float32),
            qx_Ainv=qx_axis.astype(np.float32),
            qy_Ainv=qy_axis.astype(np.float32),
        )
        print(
            f"{sample_id}: atoms={len(atoms)}, thickness={actual_thickness_A / 10:.2f} nm, "
            f"peaks={len(qx)}"
        )

    write_peak_h5(
        ROOT / "public" / "dynamical_peaks.h5",
        public_samples,
        attrs={
            "dataset_id": config["dataset"]["id"],
            "track": "dynamical",
            "input_fields": ["qx", "qy", "intensity"],
            "coordinate_units": "1/angstrom",
            "abTEM_version": abtem.__version__,
        },
    )
    write_jsonl(ROOT / "private" / "dynamical_ground_truth.jsonl", gt_records)
    write_jsonl(ROOT / "diagnostics" / "dynamical_metadata.jsonl", metadata_records)

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
    write_jsonl(ROOT / "diagnostics" / "dynamical_peaks_preview.jsonl", preview)

    with (ROOT / "reports" / "dynamical_versions.json").open("w", encoding="utf-8") as f:
        json.dump({"abTEM": abtem.__version__}, f, indent=2)

    print("Dynamical-Peak dataset finished.")


if __name__ == "__main__":
    main()
