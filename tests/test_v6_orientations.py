from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from or4d_common import FRIEDEL_SAMPLE_ROTATION
from v6_orientations import (
    audit_symmetry_unique_orientations,
    build_v6_orientation_records,
)


class _Structure:
    pass


def _config(count: int = 32) -> dict:
    return {
        "clean_sampling": {
            "headline_core": {
                "enabled": True,
                "count": count,
                "orientation_id_prefix": "v6_test_",
                "method": "haar_uniform_so3",
                "seed": 1234,
                "canonicalize_crystal_symmetry": True,
                "canonicalize_friedel": True,
                "canonical_quaternion_decimals": 12,
                "duplicate_tolerance_deg": 1.0e-6,
                "dedup_query_initial_neighbors": 4,
                "distribution_euler_sequence": "ZYZ",
            }
        }
    }


def test_v6_orientation_generation_is_deterministic_and_unique(monkeypatch):
    monkeypatch.setattr(
        "v6_orientations.proper_point_group_rotations",
        lambda structure: [np.eye(3)],
    )
    first, first_summary = build_v6_orientation_records(_config(), _Structure())
    second, second_summary = build_v6_orientation_records(_config(), _Structure())
    assert first == second
    assert len(first) == 32
    assert len({row["orientation_class_id"] for row in first}) == 32
    assert (
        first_summary["uniqueness"]["minimum_equivalent_misorientation_deg"]
        > 1.0e-6
    )
    assert first_summary["distribution"] == second_summary["distribution"]


def test_symmetry_audit_rejects_friedel_duplicate():
    identity = np.eye(3)
    duplicate = identity @ FRIEDEL_SAMPLE_ROTATION
    with pytest.raises(ValueError, match="duplicated"):
        audit_symmetry_unique_orientations(
            np.stack([identity, duplicate]),
            [np.eye(3)],
            duplicate_tolerance_deg=1.0e-6,
            initial_neighbors=2,
        )


def test_v6_requires_symmetry_and_friedel_canonicalization(monkeypatch):
    monkeypatch.setattr(
        "v6_orientations.proper_point_group_rotations",
        lambda structure: [np.eye(3)],
    )
    config = _config()
    broken = deepcopy(config)
    broken["clean_sampling"]["headline_core"]["canonicalize_friedel"] = False
    with pytest.raises(ValueError, match="Friedel"):
        build_v6_orientation_records(broken, _Structure())
