from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import h5py
import numpy as np

from or4d_common import load_config, read_peak_h5, write_peak_h5


ROOT = Path(__file__).resolve().parents[1]


def load_merge_script():
    path = ROOT / "scripts" / "41_merge_v6_expectation_shards.py"
    spec = importlib.util.spec_from_file_location("v6_expectation_merge", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_image_shard(path: Path, sample_id: str, value: float) -> None:
    with h5py.File(path, "w") as h5:
        h5.attrs["semiangle_mrad"] = 1.2
        h5.create_dataset(
            "sample_id",
            data=np.asarray([sample_id], dtype=h5py.string_dtype("utf-8")),
        )
        detector = h5.create_group("detector")
        detector.create_dataset("qx_Ainv", data=np.arange(4))
        detector.create_dataset("qy_Ainv", data=np.arange(4))
        detector.create_dataset("vacuum_probe", data=np.eye(4))
        detector.create_dataset("valid_mask", data=np.ones((4, 4)))
        h5.create_dataset(
            "expectation/intensity", data=np.full((1, 4, 4), value)
        )
        orientation = h5.create_group("orientation")
        orientation.create_dataset(
            "canonical_matrix_sample_to_crystal", data=np.eye(3)[None]
        )
        orientation.create_dataset(
            "orientation_class_id",
            data=np.asarray([sample_id], dtype=h5py.string_dtype("utf-8")),
        )


def write_reflection_shard(path: Path, sample_id: str, offset: float) -> None:
    with h5py.File(path, "w") as h5:
        h5.attrs["k_max_Ainv"] = 1.5
        h5.create_dataset(
            "sample_id",
            data=np.asarray([sample_id], dtype=h5py.string_dtype("utf-8")),
        )
        reflections = h5.create_group("reflections")
        reflections.create_dataset("offsets", data=np.asarray([0, 2]))
        reflections.create_dataset("qx_Ainv", data=np.asarray([offset, offset + 1]))
        reflections.create_dataset(
            "hkl", data=np.asarray([[1, 0, 0], [0, 1, 0]], dtype=np.int32)
        )
        crystal = h5.create_group("crystallography")
        crystal.create_dataset("reciprocal_basis_B_Ainv", data=np.eye(3))
        diagnostics = h5.create_group("diagnostics")
        diagnostics.create_dataset("candidate_reflection_count", data=np.asarray([2]))


def test_merge_v6_expectation_products(tmp_path: Path) -> None:
    module = load_merge_script()
    image_shards = [tmp_path / "image0.h5", tmp_path / "image1.h5"]
    reflection_shards = [tmp_path / "raw0.h5", tmp_path / "raw1.h5"]
    oracle_shards = [tmp_path / "oracle0.h5", tmp_path / "oracle1.h5"]
    sample_ids = ["clean_v6_core_00000", "clean_v6_core_00001"]
    for index, sample_id in enumerate(sample_ids):
        write_image_shard(image_shards[index], sample_id, float(index + 1))
        write_reflection_shard(reflection_shards[index], sample_id, float(index))
        write_peak_h5(
            oracle_shards[index],
            [
                {
                    "sample_id": sample_id,
                    "qx": np.asarray([0.1 + index]),
                    "qy": np.asarray([0.2 + index]),
                    "intensity": np.asarray([1.0]),
                }
            ],
            {
                "k_max_Ainv": 1.5,
                "source_image_file": str(image_shards[index]),
            },
        )

    config = copy.deepcopy(load_config(ROOT / "config" / "benchmark_v6.yaml"))
    config["clean_image"]["gpts"] = [4, 4]
    config["clean_image"]["compression"] = "gzip"
    output_image = tmp_path / "image.h5"
    output_oracle = tmp_path / "oracle.h5"
    output_reflections = tmp_path / "raw.h5"
    assert module.merge_images(
        image_shards, output_image, sample_ids, config
    ) == sample_ids
    assert module.merge_oracles(
        oracle_shards, output_oracle, output_image
    ) == 2
    assert module.merge_reflections(reflection_shards, output_reflections) == 2

    with h5py.File(output_image, "r") as h5:
        assert h5["expectation/intensity"].shape == (2, 4, 4)
        assert np.all(h5["expectation/intensity"][0] == 1.0)
        assert np.all(h5["expectation/intensity"][1] == 2.0)
    assert [row["sample_id"] for row in read_peak_h5(output_oracle)] == sample_ids
    with h5py.File(output_oracle, "r") as h5:
        assert h5.attrs["source_image_file"] == str(output_image)
    with h5py.File(output_reflections, "r") as h5:
        assert h5["reflections/offsets"][:].tolist() == [0, 2, 4]
        assert h5["reflections/qx_Ainv"][:].tolist() == [0.0, 1.0, 1.0, 2.0]
