#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
from pymatgen.core import Structure

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from or4d_common import cif_path, load_config, write_jsonl  # noqa: E402
from v6_orientations import build_v6_orientation_records  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config" / "benchmark_v6.yaml",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    records, summary = build_v6_orientation_records(
        config,
        Structure.from_file(cif_path(config)),
    )
    expected = int(config["v6"]["headline_orientation_count"])
    if len(records) != expected:
        raise RuntimeError(f"Generated {len(records)} records, expected {expected}")

    output = args.output.resolve()
    write_jsonl(output, records)
    matrices = np.asarray(
        [record["orientation_matrix_sample_to_crystal"] for record in records]
    )
    determinants = np.linalg.det(matrices)
    summary.update(
        {
            "dataset_id": config["dataset"]["id"],
            "config": str(args.config.resolve()),
            "config_environment_override": os.environ.get("OR4D_CONFIG"),
            "output": str(output),
            "output_sha256": sha256_file(output),
            "determinant_min": float(determinants.min()),
            "determinant_max": float(determinants.max()),
        }
    )
    summary_output = args.summary_output.resolve()
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "orientations": len(records),
                "output": str(output),
                "sha256": summary["output_sha256"],
                "minimum_equivalent_misorientation_deg": summary[
                    "uniqueness"
                ]["minimum_equivalent_misorientation_deg"],
                "summary": str(summary_output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

