from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

import h5py
import numpy as np


@dataclass(frozen=True)
class ObservationCondition:
    index: int
    condition_id: str
    layer: str
    dose_index: int | None
    dose_e_per_A2: float | None
    repeat: int
    poisson_repeat: int | None
    poisson_shot_noise: bool
    summed_frame_count: int
    read_noise_sigma_e_per_pixel: float


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> bytes:
    return hashlib.sha256(value).digest()


def array_sha256(image: np.ndarray) -> bytes:
    array = np.ascontiguousarray(image)
    header = _canonical_json(
        {"dtype": array.dtype.str, "shape": list(array.shape)}
    )
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(memoryview(array).cast("B"))
    return digest.digest()


def recipe_sha256(payload: dict[str, Any]) -> bytes:
    return _sha256_bytes(_canonical_json(payload))


def stable_v6_seed(
    seed_base: int,
    namespace: str,
    sample_id: str,
    dose_e_per_A2: float,
    repeat: int,
    *,
    level_id: str = "",
) -> int:
    payload = (
        f"or4d-clean-v6|{int(seed_base)}|{namespace}|{sample_id}|"
        f"{float(dose_e_per_A2):.17g}|{int(repeat)}|{level_id}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")


def _axis_spacing(axis: np.ndarray, name: str) -> float:
    values = np.asarray(axis, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2:
        raise ValueError(f"{name} must be a one-dimensional coordinate axis")
    steps = np.diff(values)
    spacing = float(np.median(np.abs(steps)))
    if spacing <= 0.0 or not np.allclose(
        np.abs(steps), spacing, rtol=1e-5, atol=1e-10
    ):
        raise ValueError(f"{name} must be uniformly sampled")
    return spacing


def effective_probe_area_A2(
    vacuum_probe: np.ndarray,
    qx_axis_Ainv: np.ndarray,
    qy_axis_Ainv: np.ndarray,
    *,
    real_space_oversampling: int,
) -> dict[str, float]:
    """Compute the real-space intensity participation-ratio area.

    ``vacuum_probe`` is reciprocal-space probe intensity.  V6 has zero probe
    aberrations, so its non-negative square root is the reciprocal-space probe
    amplitude.  The inverse FFT gives the real-space wave; the effective area
    is ``(integral I)^2 / integral(I^2)`` in square angstroms.
    """
    probe = np.asarray(vacuum_probe, dtype=np.float64)
    if probe.ndim != 2 or np.any(probe < 0.0) or not np.all(np.isfinite(probe)):
        raise ValueError("vacuum_probe must be a finite non-negative 2D array")
    if not np.any(probe > 0.0):
        raise ValueError("vacuum_probe has zero intensity")
    oversampling = int(real_space_oversampling)
    if oversampling <= 0:
        raise ValueError("real_space_oversampling must be positive")
    dq_x = _axis_spacing(qx_axis_Ainv, "qx_axis_Ainv")
    dq_y = _axis_spacing(qy_axis_Ainv, "qy_axis_Ainv")
    ny, nx = probe.shape
    padded_shape = (ny * oversampling, nx * oversampling)
    amplitude_q = np.sqrt(probe)
    padded = np.zeros(padded_shape, dtype=np.complex128)
    row0 = (padded_shape[0] - ny) // 2
    col0 = (padded_shape[1] - nx) // 2
    padded[row0 : row0 + ny, col0 : col0 + nx] = amplitude_q
    wave = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(padded)))
    intensity = np.abs(wave) ** 2
    dr_x = 1.0 / (padded_shape[1] * dq_x)
    dr_y = 1.0 / (padded_shape[0] * dq_y)
    pixel_area = dr_x * dr_y
    integral = float(np.sum(intensity) * pixel_area)
    squared_integral = float(np.sum(intensity**2) * pixel_area)
    if integral <= 0.0 or squared_integral <= 0.0:
        raise RuntimeError("probe transform produced no positive intensity")
    area = integral**2 / squared_integral
    return {
        "effective_illumination_area_A2": float(area),
        "definition_numerator_integral_squared": float(integral**2),
        "definition_denominator_integral_I2": float(squared_integral),
        "real_space_pixel_size_x_A": float(dr_x),
        "real_space_pixel_size_y_A": float(dr_y),
        "reciprocal_pixel_size_x_Ainv": float(dq_x),
        "reciprocal_pixel_size_y_Ainv": float(dq_y),
        "fft_oversampling": oversampling,
    }


def expected_total_electrons(
    dose_e_per_A2: float, effective_illumination_area_A2: float
) -> float:
    dose = float(dose_e_per_A2)
    area = float(effective_illumination_area_A2)
    if dose <= 0.0 or area <= 0.0:
        raise ValueError("dose and effective illumination area must be positive")
    return dose * area


def normalized_expectation(expectation: np.ndarray) -> np.ndarray:
    probability = np.asarray(expectation, dtype=np.float64)
    if probability.ndim != 2:
        raise ValueError("expectation must be two-dimensional")
    if np.any(probability < 0.0) or not np.all(np.isfinite(probability)):
        raise ValueError("expectation must contain finite non-negative values")
    total = float(probability.sum())
    if total <= 0.0:
        raise ValueError("expectation has zero total intensity")
    return probability / total


def deterministic_count_image(
    expectation: np.ndarray, expected_total: float
) -> np.ndarray:
    if float(expected_total) <= 0.0:
        raise ValueError("expected_total must be positive")
    return (normalized_expectation(expectation) * float(expected_total)).astype(
        np.float32
    )


def poisson_count_image(
    expectation: np.ndarray,
    expected_total: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if float(expected_total) <= 0.0:
        raise ValueError("expected_total must be positive")
    counts = rng.poisson(normalized_expectation(expectation) * float(expected_total))
    if np.any(counts > np.iinfo(np.uint32).max):
        raise OverflowError("Poisson count exceeds uint32 range")
    return counts.astype(np.uint32)


def read_noise_sigma_e_per_pixel(level: dict[str, Any], config: dict[str, Any]) -> float:
    frames = int(level["summed_frame_count"])
    if frames < 0:
        raise ValueError("summed_frame_count must be non-negative")
    reference = float(
        config["clean_image"]["instrument_noise"][
            "reference_read_noise_primary_e_rms_per_pixel"
        ]
    )
    return reference * np.sqrt(frames)


def build_observation_conditions(config: dict[str, Any]) -> list[ObservationCondition]:
    image = config["clean_image"]
    doses = [float(value) for value in image["counting"]["doses_e_per_A2"]]
    levels = image["instrument_noise"]["levels"]
    stochastic_repeats = int(image["counting"]["stochastic_repeats"])
    conditions = [
        ObservationCondition(
            index=0,
            condition_id="clean_e_expectation",
            layer="expectation",
            dose_index=None,
            dose_e_per_A2=None,
            repeat=0,
            poisson_repeat=None,
            poisson_shot_noise=False,
            summed_frame_count=0,
            read_noise_sigma_e_per_pixel=0.0,
        )
    ]
    for dose_index, dose in enumerate(doses):
        for level in levels:
            level_id = str(level["id"])
            repeats = int(level["stochastic_repeats"])
            if bool(level["poisson_shot_noise"]) and repeats != stochastic_repeats:
                raise ValueError(
                    f"{level_id} repeats={repeats} does not match counting "
                    f"stochastic_repeats={stochastic_repeats}"
                )
            sigma = read_noise_sigma_e_per_pixel(level, config)
            for repeat in range(repeats):
                conditions.append(
                    ObservationCondition(
                        index=len(conditions),
                        condition_id=(
                            f"dose_{dose:.17g}e_per_A2__{level_id}__repeat_{repeat}"
                        ),
                        layer=level_id,
                        dose_index=dose_index,
                        dose_e_per_A2=dose,
                        repeat=repeat,
                        poisson_repeat=(
                            repeat if bool(level["poisson_shot_noise"]) else None
                        ),
                        poisson_shot_noise=bool(level["poisson_shot_noise"]),
                        summed_frame_count=int(level["summed_frame_count"]),
                        read_noise_sigma_e_per_pixel=float(sigma),
                    )
                )
    return conditions


def logical_observation_count(config: dict[str, Any], sample_count: int) -> int:
    return int(sample_count) * len(build_observation_conditions(config))


def _hdf5_compression(config: dict[str, Any], *, sparse: bool) -> dict[str, Any]:
    storage = config["v6"]["observation_store"]
    codec = str(storage["sparse_codec" if sparse else "dense_codec"])
    level = int(storage["sparse_codec_level" if sparse else "dense_codec_level"])
    shuffle = str(storage["shuffle"])
    if codec == "gzip":
        return {"compression": "gzip", "compression_opts": level, "shuffle": True}
    if codec != "zstd":
        raise ValueError(f"Unsupported observation codec: {codec}")
    try:
        import hdf5plugin
    except ImportError as error:
        raise RuntimeError(
            "V6 zstd storage requires hdf5plugin in the active environment"
        ) from error
    if shuffle == "bitshuffle":
        return dict(hdf5plugin.Bitshuffle(nelems=0, cname="zstd", clevel=level))
    if shuffle == "none":
        return dict(hdf5plugin.Zstd(clevel=level))
    raise ValueError(f"Unsupported observation shuffle: {shuffle}")


def _decode_strings(values: np.ndarray) -> list[str]:
    return [value.decode() if isinstance(value, bytes) else str(value) for value in values]


def _read_poisson_from_group(group: h5py.Group, flat_index: int) -> np.ndarray:
    shape = tuple(int(value) for value in group.attrs["image_shape"])
    encoding = str(group.attrs["encoding"])
    if encoding == "dense_uint32":
        repeats = int(group.attrs["repeats"])
        return np.asarray(
            group["counts"][flat_index // repeats, flat_index % repeats],
            dtype=np.uint32,
        )
    if encoding != "sparse_csr_uint32":
        raise ValueError(f"Unsupported Poisson encoding: {encoding}")
    offsets = group["offsets"]
    start, stop = int(offsets[flat_index]), int(offsets[flat_index + 1])
    flat = np.zeros(shape[0] * shape[1], dtype=np.uint32)
    flat[np.asarray(group["indices"][start:stop], dtype=np.int64)] = np.asarray(
        group["values"][start:stop], dtype=np.uint32
    )
    return flat.reshape(shape)


def _iter_counts(
    expectation: h5py.Dataset,
    sample_ids: list[str],
    sample_start: int,
    sample_stop: int,
    dose: float,
    expected_total: float,
    repeats: int,
    seed_base: int,
) -> Iterator[tuple[int, int, int, np.ndarray]]:
    for local_index, source_index in enumerate(range(sample_start, sample_stop)):
        probability = np.asarray(expectation[source_index], dtype=np.float64)
        for repeat in range(repeats):
            seed = stable_v6_seed(
                seed_base,
                "poisson",
                sample_ids[source_index],
                dose,
                repeat,
            )
            counts = poisson_count_image(
                probability, expected_total, np.random.default_rng(seed)
            )
            yield local_index, repeat, seed, counts


def write_observation_shard(
    expectation_file: Path | str,
    output_file: Path | str,
    config: dict[str, Any],
    *,
    sample_start: int,
    sample_stop: int,
) -> dict[str, Any]:
    """Persist all independent V6 Poisson realizations for one sample shard."""
    source_path = Path(expectation_file).resolve()
    output_path = Path(output_file).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".partial")
    conditions = build_observation_conditions(config)
    counting = config["clean_image"]["counting"]
    doses = [float(value) for value in counting["doses_e_per_A2"]]
    repeats = int(counting["stochastic_repeats"])
    seed_base = int(counting["seed_base"])
    read_seed_base = int(config["clean_image"]["instrument_noise"]["seed_base"])
    store_config = config["v6"]["observation_store"]
    pilot_samples = int(store_config["sample_block_size"])
    sparse_threshold = float(store_config["sparse_switch_fraction"])

    with h5py.File(source_path, "r") as source:
        expectation = source["expectation/intensity"]
        total_samples, ny, nx = expectation.shape
        if not 0 <= sample_start < sample_stop <= total_samples:
            raise ValueError(
                f"Invalid sample range [{sample_start}, {sample_stop}) for {total_samples} samples"
            )
        sample_ids = _decode_strings(source["sample_id"][:])
        qx = np.asarray(source["detector/qx_Ainv"][:], dtype=np.float64)
        qy = np.asarray(source["detector/qy_Ainv"][:], dtype=np.float64)
        probe = np.asarray(source["detector/vacuum_probe"][:], dtype=np.float64)
        area_record = effective_probe_area_A2(
            probe,
            qx,
            qy,
            real_space_oversampling=int(
                config["v6"]["effective_probe_area"]["real_space_oversampling"]
            ),
        )
        area = area_record["effective_illumination_area_A2"]
        sample_count = sample_stop - sample_start
        expectation_hashes = np.empty((sample_count, 32), dtype=np.uint8)
        logical_seeds = np.zeros((sample_count, len(conditions)), dtype=np.uint64)
        logical_hashes = np.empty(
            (sample_count, len(conditions), 32), dtype=np.uint8
        )
        logical_actual_electrons = np.full(
            (sample_count, len(conditions)), np.nan, dtype=np.float64
        )
        poisson_hashes = np.empty(
            (sample_count, len(doses), repeats, 32), dtype=np.uint8
        )
        poisson_seeds = np.empty(
            (sample_count, len(doses), repeats), dtype=np.uint64
        )
        poisson_totals = np.empty(
            (sample_count, len(doses), repeats), dtype=np.uint64
        )
        expected_totals = np.asarray(
            [expected_total_electrons(dose, area) for dose in doses],
            dtype=np.float64,
        )

        with h5py.File(temporary, "w") as target:
            target.create_dataset(
                "sample_id",
                data=np.asarray(
                    sample_ids[sample_start:sample_stop],
                    dtype=h5py.string_dtype("utf-8"),
                ),
            )
            target.create_dataset("dose_e_per_A2", data=np.asarray(doses))
            target.create_dataset("expected_total_electrons", data=expected_totals)
            effective_area_group = target.create_group("effective_probe_area")
            effective_area_group.create_dataset("value_A2", data=np.asarray(area))
            for key, value in area_record.items():
                effective_area_group.attrs[key] = value
            target.attrs["schema_version"] = config["v6"]["schema_version"]
            target.attrs["sample_start"] = sample_start
            target.attrs["sample_stop"] = sample_stop
            target.attrs["source_expectation_file"] = str(source_path)
            target.attrs["rng_algorithm"] = str(counting["rng_algorithm"])
            target.attrs["logical_hash_definition"] = (
                "SHA-256 of persisted pixels for expectation/Poisson; SHA-256 of "
                "the lossless reconstruction recipe for deterministic/read-noise layers"
            )
            target.attrs["conditions_json"] = json.dumps(
                [asdict(condition) for condition in conditions], sort_keys=True
            )

            for local_index, source_index in enumerate(range(sample_start, sample_stop)):
                probability = np.asarray(expectation[source_index], dtype=np.float32)
                normalization_error = abs(float(probability.sum(dtype=np.float64)) - 1.0)
                if normalization_error > float(
                    config["v6"]["effective_probe_area"]["normalization_tolerance"]
                ):
                    raise ValueError(
                        f"Expectation sample {source_index} is not normalized: "
                        f"sum={probability.sum(dtype=np.float64):.12g}"
                    )
                digest = array_sha256(probability)
                expectation_hashes[local_index] = np.frombuffer(digest, dtype=np.uint8)
                logical_hashes[local_index, 0] = expectation_hashes[local_index]
                logical_actual_electrons[local_index, 0] = 1.0

            poisson_root = target.create_group("poisson")
            for dose_index, (dose, expected_total) in enumerate(
                zip(doses, expected_totals, strict=True)
            ):
                group = poisson_root.create_group(f"dose_{dose_index:02d}")
                group.attrs["dose_e_per_A2"] = dose
                group.attrs["expected_total_electrons"] = expected_total
                group.attrs["repeats"] = repeats
                group.attrs["image_shape"] = (ny, nx)
                iterator = _iter_counts(
                    expectation,
                    sample_ids,
                    sample_start,
                    sample_stop,
                    dose,
                    expected_total,
                    repeats,
                    seed_base,
                )
                pilot_count = min(sample_count, pilot_samples) * repeats
                pilot = [next(iterator) for _ in range(pilot_count)]
                nonzero_fraction = float(
                    np.mean([np.count_nonzero(row[3]) / row[3].size for row in pilot])
                )
                sparse = nonzero_fraction <= sparse_threshold
                group.attrs["pilot_nonzero_fraction"] = nonzero_fraction
                group.attrs["encoding"] = (
                    "sparse_csr_uint32" if sparse else "dense_uint32"
                )
                compression = _hdf5_compression(config, sparse=sparse)
                entries = itertools.chain(pilot, iterator)
                if sparse:
                    offsets = group.create_dataset(
                        "offsets", shape=(sample_count * repeats + 1,), dtype=np.uint64
                    )
                    indices = group.create_dataset(
                        "indices",
                        shape=(0,),
                        maxshape=(None,),
                        dtype=np.uint32,
                        chunks=(int(store_config["sparse_value_chunk_length"]),),
                        **compression,
                    )
                    values = group.create_dataset(
                        "values",
                        shape=(0,),
                        maxshape=(None,),
                        dtype=np.uint32,
                        chunks=(int(store_config["sparse_value_chunk_length"]),),
                        **compression,
                    )
                    cursor = 0
                    offsets[0] = 0
                else:
                    counts_dataset = group.create_dataset(
                        "counts",
                        shape=(sample_count, repeats, ny, nx),
                        dtype=np.uint32,
                        chunks=(
                            min(
                                sample_count,
                                int(store_config["dense_chunk_patterns"]),
                            ),
                            1,
                            ny,
                            nx,
                        ),
                        **compression,
                    )
                flat_index = 0
                append_count = max(1, pilot_samples * repeats)
                while True:
                    batch = list(itertools.islice(entries, append_count))
                    if not batch:
                        break
                    sparse_indices: list[np.ndarray] = []
                    sparse_values: list[np.ndarray] = []
                    batch_offsets = [cursor]
                    for local_index, repeat, seed, counts in batch:
                        poisson_seeds[local_index, dose_index, repeat] = seed
                        poisson_totals[local_index, dose_index, repeat] = int(
                            counts.sum()
                        )
                        digest = array_sha256(counts)
                        poisson_hashes[local_index, dose_index, repeat] = (
                            np.frombuffer(digest, dtype=np.uint8)
                        )
                        if sparse:
                            flat = counts.ravel()
                            nz = np.flatnonzero(flat).astype(np.uint32)
                            nz_values = flat[nz].astype(np.uint32, copy=False)
                            sparse_indices.append(nz)
                            sparse_values.append(nz_values)
                            cursor += len(nz)
                            batch_offsets.append(cursor)
                        else:
                            counts_dataset[local_index, repeat] = counts
                    if sparse:
                        start_cursor = int(indices.shape[0])
                        indices.resize((cursor,))
                        values.resize((cursor,))
                        if cursor > start_cursor:
                            indices[start_cursor:cursor] = np.concatenate(
                                sparse_indices
                            )
                            values[start_cursor:cursor] = np.concatenate(
                                sparse_values
                            )
                        offsets[
                            flat_index + 1 : flat_index + len(batch) + 1
                        ] = np.asarray(batch_offsets[1:], dtype=np.uint64)
                    flat_index += len(batch)
                if flat_index != sample_count * repeats:
                    raise RuntimeError(
                        f"Dose {dose:g} wrote {flat_index} images, expected "
                        f"{sample_count * repeats}"
                    )

            target.create_dataset("poisson/seed", data=poisson_seeds)
            target.create_dataset("poisson/actual_total_electrons", data=poisson_totals)
            target.create_dataset("poisson/pixel_sha256", data=poisson_hashes)

            for local_index, sample_id in enumerate(
                sample_ids[sample_start:sample_stop]
            ):
                expectation_hex = bytes(expectation_hashes[local_index]).hex()
                for condition in conditions[1:]:
                    assert condition.dose_index is not None
                    dose_index = condition.dose_index
                    expected_total = expected_totals[dose_index]
                    if not condition.poisson_shot_noise:
                        logical_actual_electrons[local_index, condition.index] = expected_total
                        digest = recipe_sha256(
                            {
                                "schema": "v6-deterministic-count-v1",
                                "expectation_sha256": expectation_hex,
                                "expected_total_electrons": expected_total,
                            }
                        )
                    else:
                        assert condition.poisson_repeat is not None
                        repeat = condition.poisson_repeat
                        poisson_seed = int(
                            poisson_seeds[local_index, dose_index, repeat]
                        )
                        poisson_digest = bytes(
                            poisson_hashes[local_index, dose_index, repeat]
                        )
                        logical_actual_electrons[local_index, condition.index] = float(
                            poisson_totals[local_index, dose_index, repeat]
                        )
                        if condition.summed_frame_count == 0:
                            logical_seeds[local_index, condition.index] = poisson_seed
                            digest = poisson_digest
                        else:
                            read_seed = stable_v6_seed(
                                read_seed_base,
                                "read-noise",
                                sample_id,
                                float(condition.dose_e_per_A2),
                                condition.repeat,
                                level_id=condition.layer,
                            )
                            logical_seeds[local_index, condition.index] = read_seed
                            digest = recipe_sha256(
                                {
                                    "schema": "v6-empad-read-noise-v1",
                                    "poisson_pixel_sha256": poisson_digest.hex(),
                                    "read_noise_seed": read_seed,
                                    "read_noise_sigma_e_per_pixel": (
                                        condition.read_noise_sigma_e_per_pixel
                                    ),
                                    "rng_algorithm": "numpy.PCG64",
                                }
                            )
                    logical_hashes[local_index, condition.index] = np.frombuffer(
                        digest, dtype=np.uint8
                    )

            logical = target.create_group("logical")
            logical.create_dataset("seed", data=logical_seeds)
            logical.create_dataset("validation_sha256", data=logical_hashes)
            logical.create_dataset(
                "actual_total_electrons", data=logical_actual_electrons
            )
            logical.create_dataset(
                "expectation_pixel_sha256", data=expectation_hashes
            )

    temporary.replace(output_path)
    with h5py.File(output_path, "r") as result:
        encodings = {
            f"{float(result['dose_e_per_A2'][index]):.17g}": str(
                result[f"poisson/dose_{index:02d}"].attrs["encoding"]
            )
            for index in range(len(doses))
        }
    return {
        "path": str(output_path),
        "sample_start": sample_start,
        "sample_stop": sample_stop,
        "sample_count": sample_count,
        "condition_count_per_sample": len(conditions),
        "logical_observation_count": sample_count * len(conditions),
        "effective_illumination_area_A2": area,
        "expected_total_electrons": expected_totals.tolist(),
        "encodings": encodings,
    }


class V6ObservationShardLoader:
    def __init__(
        self,
        expectation_file: Path | str,
        shard_file: Path | str,
        config: dict[str, Any],
    ) -> None:
        self.expectation_h5 = h5py.File(Path(expectation_file), "r")
        self.shard_h5 = h5py.File(Path(shard_file), "r")
        self.config = config
        self.conditions = build_observation_conditions(config)
        self.sample_start = int(self.shard_h5.attrs["sample_start"])
        self.sample_stop = int(self.shard_h5.attrs["sample_stop"])
        self.sample_ids = _decode_strings(self.shard_h5["sample_id"][:])

    def close(self) -> None:
        self.shard_h5.close()
        self.expectation_h5.close()

    def __enter__(self) -> "V6ObservationShardLoader":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _poisson(self, local_index: int, dose_index: int, repeat: int) -> np.ndarray:
        group = self.shard_h5[f"poisson/dose_{dose_index:02d}"]
        repeats = int(group.attrs["repeats"])
        return _read_poisson_from_group(group, local_index * repeats + repeat)

    def image(self, local_index: int, condition_index: int) -> np.ndarray:
        if not 0 <= local_index < len(self.sample_ids):
            raise IndexError(local_index)
        condition = self.conditions[condition_index]
        source_index = self.sample_start + local_index
        expectation = np.asarray(
            self.expectation_h5["expectation/intensity"][source_index],
            dtype=np.float32,
        )
        if condition.layer == "expectation":
            return expectation
        assert condition.dose_index is not None
        expected_total = float(
            self.shard_h5["expected_total_electrons"][condition.dose_index]
        )
        if not condition.poisson_shot_noise:
            return deterministic_count_image(expectation, expected_total)
        assert condition.poisson_repeat is not None
        image = self._poisson(
            local_index, condition.dose_index, condition.poisson_repeat
        )
        if condition.summed_frame_count == 0:
            return image
        seed = int(self.shard_h5["logical/seed"][local_index, condition.index])
        noise = np.random.default_rng(seed).normal(
            0.0, condition.read_noise_sigma_e_per_pixel, size=image.shape
        )
        return (image.astype(np.float32) + noise).astype(np.float32)

    def metadata(self, local_index: int, condition_index: int) -> dict[str, Any]:
        condition = self.conditions[condition_index]
        return {
            "sample_id": self.sample_ids[local_index],
            "global_sample_index": self.sample_start + local_index,
            "condition": asdict(condition),
            "seed": int(self.shard_h5["logical/seed"][local_index, condition_index]),
            "validation_sha256": bytes(
                self.shard_h5["logical/validation_sha256"][
                    local_index, condition_index
                ]
            ).hex(),
            "effective_illumination_area_A2": float(
                self.shard_h5["effective_probe_area/value_A2"][()]
            ),
            "expected_total_electrons": (
                None
                if condition.dose_index is None
                else float(
                    self.shard_h5["expected_total_electrons"][condition.dose_index]
                )
            ),
            "actual_total_electrons": float(
                self.shard_h5["logical/actual_total_electrons"][
                    local_index, condition_index
                ]
            ),
        }
