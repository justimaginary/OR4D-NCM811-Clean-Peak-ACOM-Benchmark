#!/usr/bin/env python3
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
from pymatgen.core import Structure

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clean_sampling import build_clean_orientation_records  # noqa: E402
from or4d_common import cif_path, load_config, write_jsonl  # noqa: E402


def main() -> None:
    config = load_config()
    structure = Structure.from_file(cif_path(config))
    records = build_clean_orientation_records(config, structure)

    output = ROOT / "private" / "orientations.jsonl"
    write_jsonl(output, records)
    print(f"Wrote {len(records)} orientations to {output}")
    role_counts = Counter(record["sample_role"] for record in records)
    determinants = np.asarray(
        [
            np.linalg.det(
                np.asarray(
                    record["orientation_matrix_sample_to_crystal"],
                    dtype=float,
                )
            )
            for record in records
        ]
    )
    print(f"Sample roles: {dict(role_counts)}")
    print(
        "Orientation determinants: "
        f"min={determinants.min():.8f}, max={determinants.max():.8f}"
    )


if __name__ == "__main__":
    main()
