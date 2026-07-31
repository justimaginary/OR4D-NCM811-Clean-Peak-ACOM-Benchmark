from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np


def quarter_power_nonnegative(values):
    """Frozen Pyxem/ACOM intensity transform, valid for NumPy or CuPy arrays."""
    module = np
    if type(values).__module__.startswith("cupy"):
        import cupy

        module = cupy
    return module.maximum(values, 0) ** 0.25


def prepare_cartesian_patterns(
    images: np.ndarray,
    *,
    q_pixel_size_Ainv: float,
    central_beam_exclusion_Ainv: float,
) -> np.ndarray:
    """Mask the direct disk and align Pyxem's geometric center exactly.

    The benchmark has an even 512x512 detector with its direct beam at
    (255.5, 255.5).  The accelerated Pyxem API uses ``shape / 2`` as its
    polar origin. Padding one row and column before the image moves the beam
    to (256.5, 256.5), exactly equal to ``513 / 2``, without interpolation.
    """
    data = np.asarray(images, dtype=np.float32)
    if data.ndim != 3 or data.shape[-2:] != (512, 512):
        raise ValueError(f"expected [N,512,512] images, got {data.shape}")
    if q_pixel_size_Ainv <= 0 or central_beam_exclusion_Ainv < 0:
        raise ValueError("invalid reciprocal-space calibration")
    yy, xx = np.indices(data.shape[-2:], dtype=np.float32)
    radius = np.hypot(yy - 255.5, xx - 255.5) * q_pixel_size_Ainv
    prepared = data.copy()
    prepared[:, radius <= central_beam_exclusion_Ainv] = 0.0
    return np.pad(prepared, ((0, 0), (1, 0), (1, 0)))


def build_template_library(
    *,
    cif_path: Path,
    cache_path: Path,
    voltage_kV: float,
    q_pixel_size_Ainv: float,
    settings: dict[str, Any],
):
    """Build or load the frozen independent Pyxem/diffsims S2 library."""
    from diffsims.generators.diffraction_generator import DiffractionGenerator
    from diffsims.generators.library_generator import DiffractionLibraryGenerator
    from diffsims.libraries.diffraction_library import load_DiffractionLibrary
    from diffsims.libraries.structure_library import StructureLibrary
    from orix.crystal_map import Phase
    from orix.sampling import get_sample_reduced_fundamental

    metadata_path = cache_path.with_suffix(".json")
    expected = {
        "cache_schema": 2,
        "cif_path": str(cif_path.resolve()),
        "voltage_kV": float(voltage_kV),
        "q_pixel_size_Ainv": float(q_pixel_size_Ainv),
        "settings": settings,
    }
    if cache_path.is_file() and metadata_path.is_file():
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        if existing == expected:
            return load_DiffractionLibrary(str(cache_path), safety=True), existing
        raise ValueError(f"template cache metadata mismatch: {cache_path}")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    phase = Phase.from_cif(str(cif_path))
    # ``Phase.from_cif`` preserves oxidation-state strings such as ``Ni3+``.
    # diffsims' Lobato table is keyed by neutral element symbols; leaving the
    # charged labels in place silently produces an all-zero template library.
    # Rebuild only the diffpy atom list from pymatgen's element/occupancy data.
    from diffpy.structure import Atom, Lattice, Structure as DiffpyStructure
    from pymatgen.core import Structure as PymatgenStructure

    pmg = PymatgenStructure.from_file(str(cif_path))
    diffpy_structure = DiffpyStructure(
        lattice=Lattice(
            pmg.lattice.a,
            pmg.lattice.b,
            pmg.lattice.c,
            pmg.lattice.alpha,
            pmg.lattice.beta,
            pmg.lattice.gamma,
        )
    )
    for site in pmg:
        for species, occupancy in site.species.items():
            diffpy_structure.append(
                Atom(
                    atype=species.symbol,
                    xyz=np.asarray(site.frac_coords, dtype=float),
                    occupancy=float(occupancy),
                )
            )
    phase.structure = diffpy_structure
    rotations = get_sample_reduced_fundamental(
        resolution=float(settings["orientation_resolution_deg"]),
        point_group=phase.point_group,
    )
    euler_deg = rotations.to_euler(degrees=True)
    structure_library = StructureLibrary(
        [phase.name], [phase.structure], [euler_deg]
    )
    generator = DiffractionGenerator(
        accelerating_voltage=float(voltage_kV),
        scattering_params="lobato",
        shape_factor_model=str(settings["shape_factor_model"]),
        minimum_intensity=float(settings["minimum_relative_intensity"]),
        minima_number=int(settings["shape_factor_minima"]),
    )
    library = DiffractionLibraryGenerator(generator).get_diffraction_library(
        structure_library,
        calibration=float(q_pixel_size_Ainv),
        reciprocal_radius=float(settings["reciprocal_radius_Ainv"]),
        half_shape=(256.5, 256.5),
        with_direct_beam=False,
        max_excitation_error=float(settings["max_excitation_error_Ainv"]),
        shape_factor_width=float(settings["max_excitation_error_Ainv"]),
    )
    phase_key = next(iter(library))
    spot_counts = np.asarray(
        [
            simulation.intensities.size
            for simulation in library[phase_key]["simulations"]
        ],
        dtype=int,
    )
    if np.any(spot_counts == 0):
        raise RuntimeError(
            "Pyxem template construction produced empty orientations: "
            f"{int(np.sum(spot_counts == 0))}/{len(spot_counts)}"
        )
    library.pickle_library(str(cache_path))
    metadata_path.write_text(
        json.dumps(expected, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return library, expected


def match_prepared_batch(
    prepared: np.ndarray,
    *,
    library,
    q_pixel_size_Ainv: float,
    settings: dict[str, Any],
    target: str,
) -> dict[str, np.ndarray | float]:
    """Call Pyxem's official accelerated template matcher for one batch."""
    import hyperspy.api as hs
    from pyxem.utils.indexation_utils import (
        index_dataset_with_template_rotation,
    )

    if target not in {"cpu", "gpu"}:
        raise ValueError("target must be cpu or gpu")
    signal = hs.signals.Signal2D(prepared)
    start = time.perf_counter()
    result, phase_key = index_dataset_with_template_rotation(
        signal,
        library,
        n_best=int(settings["n_best"]),
        frac_keep=float(settings["frac_keep"]),
        delta_r=1.0,
        delta_theta=float(settings["in_plane_resolution_deg"]),
        max_r=float(settings["reciprocal_radius_Ainv"]) / q_pixel_size_Ainv,
        intensity_transform_function=quarter_power_nonnegative,
        normalize_images=True,
        normalize_templates=True,
        # Each caller batch is already bounded in memory.  ``None`` preserves
        # the single navigation chunk after Pyxem promotes a 1-D navigation
        # axis to [1, N]; explicitly splitting N triggers a Pyxem/Dask chunk
        # shape mismatch for batches larger than 32.
        chunks=(1, prepared.shape[0], None, None),
        parallel_workers=1 if target == "gpu" else 8,
        target=target,
        scheduler="threads",
        precision=np.float32,
    )
    elapsed = time.perf_counter() - start
    # A 1-D navigation signal is promoted by Pyxem to shape [1,N].
    return {
        "euler_deg": np.asarray(result["orientation"][0, :, 0], dtype=np.float64),
        "correlation": np.asarray(result["correlation"][0, :, 0], dtype=np.float64),
        "mirrored": np.asarray(
            result["mirrored_template"][0, :, 0], dtype=bool
        ),
        "template_index": np.asarray(
            result["template_index"][0, :, 0], dtype=np.int32
        ),
        "phase_index": np.asarray(result["phase_index"][0, :, 0], dtype=np.int8),
        "seconds": elapsed,
        "phase_key": phase_key,
    }


def euler_to_sample_to_crystal(euler_deg: np.ndarray) -> np.ndarray:
    """Use the same Bunge lab-to-crystal conversion as Pyxem CrystalMap."""
    from orix.quaternion import Rotation

    rotations = Rotation.from_euler(
        np.deg2rad(np.asarray(euler_deg, dtype=np.float64)),
        direction="lab2crystal",
    )
    return np.asarray(rotations.to_matrix(), dtype=np.float64)
