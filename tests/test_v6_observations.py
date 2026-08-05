from __future__ import annotations

import copy
from pathlib import Path

import h5py
import numpy as np

from or4d_common import load_config
from v6_observations import (
    V6ObservationShardLoader,
    array_sha256,
    build_observation_conditions,
    effective_probe_area_A2,
    logical_observation_count,
    stable_v6_seed,
    write_observation_shard,
)


ROOT = Path(__file__).resolve().parents[1]


def v6_test_config() -> dict:
    config = copy.deepcopy(load_config(ROOT / "config" / "benchmark_v6.yaml"))
    config["clean_image"]["counting"]["doses_e_per_A2"] = [0.3, 3.0]
    config["v6"]["observation_store"]["dense_codec"] = "gzip"
    config["v6"]["observation_store"]["sparse_codec"] = "gzip"
    config["v6"]["observation_store"]["dense_codec_level"] = 1
    config["v6"]["observation_store"]["sparse_codec_level"] = 1
    config["v6"]["observation_store"]["sample_block_size"] = 1
    config["v6"]["effective_probe_area"]["real_space_oversampling"] = 2
    return config


def write_tiny_expectation(path: Path) -> None:
    yy, xx = np.mgrid[:16, :16]
    probe = (np.hypot(yy - 7.5, xx - 7.5) <= 2.2).astype(np.float32)
    first = probe + 0.05 * (
        np.hypot(yy - 4.0, xx - 11.0) <= 1.5
    ).astype(np.float32)
    second = probe + 0.04 * (
        np.hypot(yy - 11.0, xx - 4.0) <= 1.5
    ).astype(np.float32)
    images = np.stack([first / first.sum(), second / second.sum()]).astype(
        np.float32
    )
    qx = (np.arange(16) - 7.5) * 0.01
    qy = (7.5 - np.arange(16)) * 0.01
    with h5py.File(path, "w") as h5:
        h5.create_dataset("expectation/intensity", data=images)
        h5.create_dataset(
            "sample_id",
            data=np.asarray(
                ["clean_v6_core_00000", "clean_v6_core_00001"],
                dtype=h5py.string_dtype("utf-8"),
            ),
        )
        h5.create_dataset("detector/qx_Ainv", data=qx)
        h5.create_dataset("detector/qy_Ainv", data=qy)
        h5.create_dataset("detector/vacuum_probe", data=probe)


def test_v6_condition_grid_matches_specification() -> None:
    config = load_config(ROOT / "config" / "benchmark_v6.yaml")
    conditions = build_observation_conditions(config)
    assert len(conditions) == 91
    assert logical_observation_count(config, 16384) == 1_490_944
    assert conditions[0].condition_id == "clean_e_expectation"
    assert sum(row.layer == "deterministic_count" for row in conditions) == 9
    assert sum(row.layer == "poisson_shot" for row in conditions) == 27
    assert sum(row.layer == "empad_g2_1frame" for row in conditions) == 27
    assert sum(
        row.layer == "empad_g2_16frame_fixed_total_dose" for row in conditions
    ) == 27


def test_effective_probe_area_is_scale_invariant() -> None:
    yy, xx = np.mgrid[:32, :32]
    probe = (np.hypot(yy - 15.5, xx - 15.5) <= 4.0).astype(float)
    qx = (np.arange(32) - 15.5) * 0.01
    qy = (15.5 - np.arange(32)) * 0.01
    first = effective_probe_area_A2(
        probe, qx, qy, real_space_oversampling=2
    )
    second = effective_probe_area_A2(
        probe * 7.0, qx, qy, real_space_oversampling=2
    )
    assert first["effective_illumination_area_A2"] > 0.0
    assert np.isclose(
        first["effective_illumination_area_A2"],
        second["effective_illumination_area_A2"],
        rtol=1e-12,
    )


def test_seed_is_stable_and_condition_specific() -> None:
    first = stable_v6_seed(7, "poisson", "sample", 0.3, 1)
    assert first == stable_v6_seed(7, "poisson", "sample", 0.3, 1)
    assert first != stable_v6_seed(7, "poisson", "sample", 0.3, 2)
    assert first != stable_v6_seed(7, "read-noise", "sample", 0.3, 1)


def test_compressed_shard_roundtrip(tmp_path: Path) -> None:
    config = v6_test_config()
    expectation = tmp_path / "expectation.h5"
    shard = tmp_path / "observations.h5"
    write_tiny_expectation(expectation)
    report = write_observation_shard(
        expectation,
        shard,
        config,
        sample_start=0,
        sample_stop=2,
    )
    assert report["logical_observation_count"] == 2 * (1 + 2 * 10)
    assert set(report["encodings"].values()) <= {
        "dense_uint32",
        "sparse_csr_uint32",
    }

    conditions = build_observation_conditions(config)
    expectation_index = 0
    deterministic_index = next(
        row.index for row in conditions if row.layer == "deterministic_count"
    )
    poisson_index = next(
        row.index for row in conditions if row.layer == "poisson_shot"
    )
    empad_1_index = next(
        row.index for row in conditions if row.layer == "empad_g2_1frame"
    )
    empad_16_index = next(
        row.index
        for row in conditions
        if row.layer == "empad_g2_16frame_fixed_total_dose"
    )
    with V6ObservationShardLoader(expectation, shard, config) as loader:
        clean_e = loader.image(0, expectation_index)
        deterministic = loader.image(0, deterministic_index)
        poisson = loader.image(0, poisson_index)
        empad_1_a = loader.image(0, empad_1_index)
        empad_1_b = loader.image(0, empad_1_index)
        empad_16 = loader.image(0, empad_16_index)
        assert np.isclose(clean_e.sum(), 1.0)
        assert np.isclose(
            deterministic.sum(),
            loader.metadata(0, deterministic_index)["expected_total_electrons"],
        )
        assert poisson.dtype == np.uint32
        assert np.array_equal(empad_1_a, empad_1_b)
        assert not np.array_equal(empad_1_a, empad_16)
        assert (
            loader.metadata(0, poisson_index)["validation_sha256"]
            == array_sha256(poisson).hex()
        )
