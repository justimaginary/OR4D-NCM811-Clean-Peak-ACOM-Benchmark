#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from pymatgen.core import Structure

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clean_001_study import build_001_study_records  # noqa: E402
from or4d_common import (  # noqa: E402
    cif_path,
    load_config,
    proper_point_group_rotations,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the independent V5 [001] diagnostic manifest."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "private" / "clean_v5_001_orientations.jsonl",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=ROOT / "reports" / "clean_v5_001_manifest_summary.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config()
    if "v5" not in config:
        raise ValueError("The [001] study requires the V5 config overlay.")
    structure = Structure.from_file(cif_path(config))
    records = build_001_study_records(
        config, structure, proper_point_group_rotations(structure)
    )
    output = args.output.resolve()
    write_jsonl(output, records)
    output_display = (
        str(output.relative_to(ROOT))
        if output.is_relative_to(ROOT)
        else str(output)
    )
    summary = {
        "dataset_id": config["dataset"]["id"],
        "sample_count": len(records),
        "groups": dict(Counter(row["study_group"] for row in records)),
        "output": output_display,
        "headline_included": False,
    }
    report = args.report_output.resolve()
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
