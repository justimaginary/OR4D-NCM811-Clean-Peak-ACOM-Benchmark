from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_acom_shard_script():
    path = ROOT / "scripts" / "39_run_v6_acom_peak_shard.py"
    spec = importlib.util.spec_from_file_location("v6_acom_peak_shard", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_expanded_ground_truth_uses_source_orientation(tmp_path: Path) -> None:
    module = load_acom_shard_script()
    orientation_file = tmp_path / "orientations.jsonl"
    records = [
        {
            "orientation_id": "v6_core_00000",
            "orientation_matrix_sample_to_crystal": np.eye(3).tolist(),
        },
        {
            "orientation_id": "v6_core_00001",
            "orientation_matrix_sample_to_crystal": np.diag([-1, -1, 1]).tolist(),
        },
    ]
    orientation_file.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    peak_file = tmp_path / "peaks.h5"
    string_dtype = h5py.string_dtype("utf-8")
    with h5py.File(peak_file, "w") as h5:
        h5.create_dataset(
            "sample_id",
            data=np.asarray(
                [
                    "clean_v6_core_00000::condition_004",
                    "clean_v6_core_00001::condition_090",
                ],
                dtype=string_dtype,
            ),
        )
        h5.create_dataset(
            "sample/source_sample_id",
            data=np.asarray(
                ["clean_v6_core_00000", "clean_v6_core_00001"],
                dtype=string_dtype,
            ),
        )

    expanded = module.expanded_ground_truth(peak_file, orientation_file)

    assert [record["sample_id"] for record in expanded] == [
        "clean_v6_core_00000::condition_004",
        "clean_v6_core_00001::condition_090",
    ]
    assert [record["source_sample_id"] for record in expanded] == [
        "clean_v6_core_00000",
        "clean_v6_core_00001",
    ]
    assert expanded[0]["orientation_matrix_sample_to_crystal"] == records[0][
        "orientation_matrix_sample_to_crystal"
    ]
    assert expanded[1]["orientation_matrix_sample_to_crystal"] == records[1][
        "orientation_matrix_sample_to_crystal"
    ]
