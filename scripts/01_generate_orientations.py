#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from pymatgen.core import Structure

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clean_sampling import build_clean_orientation_records  # noqa: E402
from or4d_common import cif_path, load_config, write_jsonl  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate deterministic Clean orientation records."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "private" / "orientations.jsonl",
    )
    parser.add_argument("--summary-output", type=Path)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    args = parse_args()
    config = load_config()
    structure = Structure.from_file(cif_path(config))
    records = build_clean_orientation_records(config, structure)

    output = args.output.resolve()
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
    if args.summary_output is not None:
        summary_output = args.summary_output.resolve()
        summary_output.parent.mkdir(parents=True, exist_ok=True)
        summary_output.write_text(
            json.dumps(
                {
                    "dataset_id": config["dataset"]["id"],
                    "output": str(output),
                    "sha256": sha256_file(output),
                    "num_orientations": len(records),
                    "sample_roles": dict(role_counts),
                    "friedel_canonical": bool(
                        config["clean_sampling"]["headline_core"].get(
                            "canonicalize_friedel", False
                        )
                    ),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Summary: {summary_output}")


if __name__ == "__main__":
    main()
