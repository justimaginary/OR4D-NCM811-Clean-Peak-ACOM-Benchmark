#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from or4d_common import read_peak_h5, write_jsonl  # noqa: E402


def main() -> None:
    records = []
    for track in ("clean", "dynamical"):
        path = ROOT / "public" / f"{track}_peaks.h5"
        if path.exists():
            for sample in read_peak_h5(path):
                records.append(
                    {
                        "sample_id": sample["sample_id"],
                        "orientation_matrix_sample_to_crystal": np.eye(3).tolist(),
                    }
                )
    output = ROOT / "submissions" / "submission_example.jsonl"
    write_jsonl(output, records)
    print(f"Wrote submission-format example to {output}")


if __name__ == "__main__":
    main()
