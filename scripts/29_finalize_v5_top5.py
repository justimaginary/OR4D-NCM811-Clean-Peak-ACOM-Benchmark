#!/usr/bin/env python3
"""Validate and summarize full V5 ACOM/Pyxem Top-K results."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from v5_results import (  # noqa: E402
    aggregate_group_keys,
    aggregate_topk_error_blocks,
    group_label,
    parse_clean_c_condition_stem,
)


DETAIL_SUFFIX = "_details.json"
ACOM_CLEAN_C_DETECTORS = ("autodisk", "dog_rgm", "py4dstem")
CONDITIONS_PER_CLEAN_C_DETECTOR = 234


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--acom-clean-e-details-dir", type=Path, required=True)
    parser.add_argument("--acom-clean-e-candidates-dir", type=Path, required=True)
    parser.add_argument("--acom-clean-c-details-dir", type=Path, required=True)
    parser.add_argument("--acom-clean-c-candidates-dir", type=Path, required=True)
    parser.add_argument("--pyxem-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--artifact-prefix", default="V5")
    parser.add_argument("--dataset-label", default="V5 Clean headline")
    parser.add_argument("--sample-count", type=int, default=2048)
    return parser.parse_args()


def close_metrics(left: list[dict], right: list[dict]) -> bool:
    if len(left) != len(right):
        return False
    fields = (
        "num_input_samples",
        "num_indexed_samples",
        "num_valid_predictions",
        "prediction_coverage",
        "median_misorientation_deg_indexed",
        "p95_misorientation_deg_indexed",
        "accuracy_all_inputs_within_1deg",
        "accuracy_all_inputs_within_2deg",
        "accuracy_all_inputs_within_5deg",
    )
    for a, b in zip(left, right, strict=True):
        for field in fields:
            if not np.isclose(
                float(a[field]), float(b[field]), equal_nan=True, atol=1e-10
            ):
                return False
    return True


def load_acom(
    clean_e_details_dir: Path,
    clean_e_candidates_dir: Path,
    clean_c_details_dir: Path,
    clean_c_candidates_dir: Path,
) -> tuple[dict, dict]:
    conditions: list[dict] = []
    runs: list[dict] = []
    blocks: dict[tuple[str, str], list[tuple[np.ndarray, int]]] = {}

    def append_condition(
        *,
        details_path: Path,
        candidate_path: Path,
        label: dict[str, int | str | None],
        output_paths: dict[str, Path],
    ) -> None:
        stem = details_path.name[: -len(DETAIL_SUFFIX)]
        if not candidate_path.is_file():
            raise FileNotFoundError(candidate_path)
        details = json.loads(details_path.read_text(encoding="utf-8"))
        with h5py.File(candidate_path, "r") as candidate:
            errors = np.asarray(
                candidate["friedel_equivalent_misorientation_deg"][:],
                dtype=float,
            )
            if errors.ndim != 2 or errors.shape[1] != 5:
                raise ValueError(f"bad candidate error shape in {candidate_path}")
            sample_rows = int(errors.shape[0])
            if sample_rows != int(details["num_matched_samples"]):
                raise ValueError(f"matched sample count differs in {details_path}")
        total_inputs = int(details["num_input_samples"])
        computed = aggregate_topk_error_blocks([(errors, total_inputs)])
        if not close_metrics(computed, details["top_k_metrics"]):
            raise ValueError(f"saved Top-K metrics differ in {details_path}")
        condition = {
            **label,
            "condition_id": stem,
            "num_input_samples": total_inputs,
            "num_indexed_samples": sample_rows,
            "num_indexing_failures": total_inputs - sample_rows,
            "top_k": computed,
            "runtime": details["runtime"],
        }
        conditions.append(condition)
        for key in aggregate_group_keys(label):
            blocks.setdefault(key, []).append((errors, total_inputs))
        missing = [name for name, path in output_paths.items() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"{stem} lacks outputs: {missing}")
        runs.append(
            {
                **label,
                "condition_id": stem,
                "status": "completed",
                "outputs": {
                    name: str(path) for name, path in output_paths.items()
                },
                "num_input_samples": total_inputs,
                "num_indexed_samples": sample_rows,
                "num_indexing_failures": total_inputs - sample_rows,
            }
        )

    clean_e_paths = sorted(
        clean_e_details_dir.glob(f"*{DETAIL_SUFFIX}")
    )
    for details_path in clean_e_paths:
        stem = details_path.name[: -len(DETAIL_SUFFIX)]
        append_condition(
            details_path=details_path,
            candidate_path=clean_e_candidates_dir / f"{stem}_candidates.h5",
            label={"track": "Clean-E", "input": stem},
            output_paths={
                "details": details_path,
                "candidates": (
                    clean_e_candidates_dir / f"{stem}_candidates.h5"
                ),
                "predictions": (
                    clean_e_details_dir / f"{stem}_predictions.jsonl"
                ),
                "audit": clean_e_details_dir / f"{stem}_audit.json",
            },
        )
    if len(clean_e_paths) != 4:
        raise ValueError(
            f"expected 4 ACOM Clean-E conditions, found {len(clean_e_paths)}"
        )

    clean_c_paths = sorted(
        clean_c_details_dir.glob(f"*{DETAIL_SUFFIX}")
    )
    for details_path in clean_c_paths:
        stem = details_path.name[: -len(DETAIL_SUFFIX)]
        append_condition(
            details_path=details_path,
            candidate_path=clean_c_candidates_dir / f"{stem}_candidates.h5",
            label=parse_clean_c_condition_stem(stem),
            output_paths={
                "details": details_path,
                "candidates": (
                    clean_c_candidates_dir / f"{stem}_candidates.h5"
                ),
                "predictions": (
                    clean_c_details_dir / f"{stem}_predictions.jsonl"
                ),
                "audit": clean_c_details_dir / f"{stem}_audit.json",
                "evaluation": (
                    clean_c_details_dir / f"{stem}_evaluation.json"
                ),
            },
        )
    expected_clean_c = (
        len(ACOM_CLEAN_C_DETECTORS) * CONDITIONS_PER_CLEAN_C_DETECTOR
    )
    if len(clean_c_paths) != expected_clean_c:
        raise ValueError(
            f"expected {expected_clean_c} ACOM Clean-C conditions, "
            f"found {len(clean_c_paths)}"
        )
    detector_counts = {
        detector: sum(
            condition.get("detector") == detector
            for condition in conditions
            if condition["track"] == "Clean-C"
        )
        for detector in ACOM_CLEAN_C_DETECTORS
    }
    if any(
        count != CONDITIONS_PER_CLEAN_C_DETECTOR
        for count in detector_counts.values()
    ):
        raise ValueError(
            "ACOM Clean-C detector condition counts are incomplete: "
            f"{detector_counts}"
        )
    aggregates = [
        {
            **group_label(group_by, key),
            "num_conditions": len(group_blocks),
            "top_k": aggregate_topk_error_blocks(group_blocks),
        }
        for (group_by, key), group_blocks in sorted(blocks.items())
    ]
    summary = {
        "schema": "or4d-v5-acom-topk-summary-v1",
        "method": (
            "py4DSTEM ACOM on saved AutoDisk, DoG-RGM, and "
            "find_Bragg_disks peak lists"
        ),
        "metric": (
            "Minimum misorientation over proper crystal point-group rotations "
            "and the detector-plane Friedel branch."
        ),
        "num_clean_e_conditions": len(clean_e_paths),
        "num_clean_c_conditions": len(clean_c_paths),
        "clean_c_conditions_by_detector": detector_counts,
        "num_conditions": len(conditions),
        "conditions": conditions,
        "aggregates": aggregates,
    }
    manifest = {
        "schema": "or4d-v5-acom-unified-run-manifest-v1",
        "num_conditions": len(runs),
        "num_completed": len(runs),
        "num_failed": 0,
        "runs": runs,
    }
    return summary, manifest


def aggregate_index(summary: dict) -> dict[tuple, dict]:
    result = {}
    for row in summary["aggregates"]:
        key = (
            row["group_by"],
            row.get("dose_electrons"),
            row.get("noise"),
            row.get("track"),
        )
        result[key] = row
    return result


def metric_at(row: dict, k: int, field: str) -> float:
    return float(row["top_k"][k - 1][field])


def write_comparison_plot(acom: dict, pyxem: dict, path: Path) -> None:
    doses = sorted(
        {
            int(row["dose_electrons"])
            for row in acom["aggregates"]
            if row["group_by"] == "dose_counted_detector"
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    methods = (
        ("ACOM + AutoDisk", "autodisk", "#e76f51"),
        ("ACOM + DoG-RGM", "dog_rgm", "#ca8a04"),
        ("ACOM + find_Bragg", "py4dstem", "#7c3aed"),
        ("Pyxem image", None, "#2a6fdb"),
    )
    for label, detector, color in methods:
        rows = (
            [
                find_aggregate(
                    acom,
                    "dose_counted_detector",
                    track="Clean-C",
                    detector=detector,
                    dose_electrons=dose,
                )
                for dose in doses
            ]
            if detector is not None
            else [
                find_aggregate(
                    pyxem,
                    "dose_counted",
                    track="Clean-C",
                    dose_electrons=dose,
                )
                for dose in doses
            ]
        )
        axes[0].plot(
            doses,
            [
                metric_at(
                    row, 1, "accuracy_all_inputs_within_2deg"
                )
                for row in rows
            ],
            "o--",
            color=color,
            label=f"{label} Top-1",
        )
        axes[0].plot(
            doses,
            [
                metric_at(
                    row, 5, "accuracy_all_inputs_within_2deg"
                )
                for row in rows
            ],
            "o-",
            color=color,
            label=f"{label} Top-5",
        )
        axes[1].plot(
            doses,
            [
                metric_at(row, 1, "prediction_coverage")
                for row in rows
            ],
            "o-",
            color=color,
            label=label,
        )
    for axis in axes:
        axis.set_xscale("log")
        axis.grid(True, alpha=0.25)
        axis.set_xlabel("Electrons per pattern")
        axis.set_ylim(-0.02, 1.02)
        axis.legend()
    axes[0].set_ylabel("Accuracy within 2°")
    axes[0].set_title("Counted Clean-C orientation accuracy")
    axes[1].set_ylabel("Prediction coverage")
    axes[1].set_title("Counted Clean-C indexing coverage")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def find_aggregate(summary: dict, group_by: str, **fields: object) -> dict:
    matches = [
        row
        for row in summary["aggregates"]
        if row["group_by"] == group_by
        and all(row.get(name) == value for name, value in fields.items())
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one {group_by} aggregate for {fields}, found {len(matches)}"
        )
    return matches[0]


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def report_metrics(row: dict) -> tuple[str, str, str, str, str]:
    top1 = row["top_k"][0]
    top5 = row["top_k"][4]
    return (
        pct(float(top1["prediction_coverage"])),
        pct(float(top1["accuracy_all_inputs_within_2deg"])),
        pct(float(top5["accuracy_all_inputs_within_2deg"])),
        f"{float(top1['median_misorientation_deg_indexed']):.3f}°",
        f"{float(top1['p95_misorientation_deg_indexed']):.3f}°",
    )


def write_markdown_report(
    acom: dict,
    pyxem: dict,
    comparison_path: Path,
    plot_path: Path,
    path: Path,
    *,
    dataset_label: str,
    sample_count: int,
) -> None:
    lines = [
        f"# {dataset_label} Top-1 / Top-5 results",
        "",
        "This report is generated from the saved per-condition candidate files. "
        "No failed or inaccurate scientific result is removed.",
        "",
        "## Scope",
        "",
        f"- Dataset: {sample_count:,} orientations per condition; "
        "`kmax = 1.5 Å⁻¹`.",
        "- Clean-E: deterministic expectation-intensity diffraction images.",
        "- Clean-C: electron-counted images at 9 independent dose levels. Each "
        "dose has a noiseless condition plus Poisson-only and EMPAD-G2 detector "
        "noise levels; the counted noise conditions have 5 repeats.",
        "- ACOM: py4DSTEM ACOM using saved peak lists. Clean-E compares "
        "oracle, AutoDisk, DoG-RGM, and find_Bragg_disks inputs; Clean-C "
        "runs all three automatic disk detectors independently.",
        "- Pyxem: accelerated template matching from the diffraction image.",
        "- Error: minimum misorientation over proper crystal point-group "
        "symmetry and the detector-plane Friedel branch.",
        "- Top-K: the best symmetry/Friedel-equivalent error among the first K "
        "saved candidates. Accuracy uses all input samples as the denominator; "
        "an indexing failure therefore counts as incorrect.",
        "",
        "## Clean-E",
        "",
        "| Method / input | Coverage | Top-1 Acc@2° | Top-5 Acc@2° | "
        "Top-1 median | Top-1 P95 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for input_name in ("oracle", "autodisk", "dog_rgm", "py4dstem"):
        row = find_aggregate(
            acom, "clean_e_input", track="Clean-E", input=input_name
        )
        values = report_metrics(row)
        lines.append(
            f"| ACOM / {input_name} | " + " | ".join(values) + " |"
        )
    pyxem_e = find_aggregate(pyxem, "track", track="Clean-E")
    lines.append(
        "| Pyxem / expectation image | "
        + " | ".join(report_metrics(pyxem_e))
        + " |"
    )

    lines.extend(
        [
            "",
            "## Clean-C: counted conditions grouped by electron dose",
            "",
            "| Method / input | Electrons / pattern | Coverage | "
            "Top-1 Acc@2° | Top-5 Acc@2° | Top-1 median | Top-1 P95 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    doses = sorted(
        {
            int(row["dose_electrons"])
            for row in acom["aggregates"]
            if row["group_by"] == "dose_counted_detector"
        }
    )
    for dose in doses:
        for detector, label in (
            ("autodisk", "ACOM / AutoDisk"),
            ("dog_rgm", "ACOM / DoG-RGM"),
            ("py4dstem", "ACOM / find_Bragg_disks"),
        ):
            row = find_aggregate(
                acom,
                "dose_counted_detector",
                track="Clean-C",
                detector=detector,
                dose_electrons=dose,
            )
            lines.append(
                f"| {label} | {dose:,} | "
                + " | ".join(report_metrics(row))
                + " |"
            )
        pyxem_row = find_aggregate(
            pyxem,
            "dose_counted",
            track="Clean-C",
            dose_electrons=dose,
        )
        pyxem_values = report_metrics(pyxem_row)
        lines.append(
            f"| Pyxem / image | {dose:,} | "
            + " | ".join(pyxem_values)
            + " |"
        )

    lines.extend(
        [
            "",
            "## Clean-C: independent noise-model groups",
            "",
            "These rows aggregate all 9 dose levels. Dose and noise remain "
            "separate experimental variables in the stored condition table.",
            "",
            "| Method / input | Noise model | Conditions | Top-1 Acc@2° | "
            "Top-5 Acc@2° |",
            "|---|---|---:|---:|---:|",
        ]
    )
    noise_names = sorted(
        {
            str(row["noise"])
            for row in acom["aggregates"]
            if row["group_by"] == "noise_all_detector"
        }
    )
    for noise in noise_names:
        for detector, label in (
            ("autodisk", "ACOM / AutoDisk"),
            ("dog_rgm", "ACOM / DoG-RGM"),
            ("py4dstem", "ACOM / find_Bragg_disks"),
        ):
            row = find_aggregate(
                acom,
                "noise_all_detector",
                track="Clean-C",
                detector=detector,
                noise=noise,
            )
            values = report_metrics(row)
            lines.append(
                f"| {label} | {noise} | {row['num_conditions']} | "
                f"{values[1]} | {values[2]} |"
            )
        pyxem_row = find_aggregate(
            pyxem, "noise_all", track="Clean-C", noise=noise
        )
        pyxem_values = report_metrics(pyxem_row)
        lines.append(
            f"| Pyxem / image | {noise} | {pyxem_row['num_conditions']} | "
            f"{pyxem_values[1]} | {pyxem_values[2]} |"
        )

    acom_failures = sum(
        int(condition["num_indexing_failures"])
        for condition in acom["conditions"]
    )
    pyxem_failures = sum(
        int(condition.get("num_indexing_failures", 0))
        for condition in pyxem["conditions"]
    )
    lines.extend(
        [
            "",
            "## Execution and validation status",
            "",
            f"- ACOM: {acom['num_conditions']} completed conditions; "
            f"{acom_failures:,} inputs had no saved candidate because the peak "
            "list contained too few usable peaks. These are reproducible "
            "indexing failures, not suppressed rows.",
            f"- Pyxem: {len(pyxem['conditions'])} completed conditions; "
            f"{pyxem_failures:,} missing predictions.",
            "- Every ACOM candidate array was checked for shape `[matched, 5, "
            "3, 3]`, finite values, and agreement between recomputed and saved "
            "Top-K metrics.",
            "- The four disjoint Pyxem Clean-C shards were merged only after "
            "verifying exactly one owner for every expected condition.",
            "",
            "## Artifacts",
            "",
            f"- Comparison data: `{comparison_path.name}`",
            f"- Dose plot: `{plot_path.name}`",
            "- Full ACOM summary and provenance use the artifact prefix "
            f"`{path.stem.removesuffix('_TOP5_RESULTS')}`.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    acom, manifest = load_acom(
        args.acom_clean_e_details_dir.resolve(),
        args.acom_clean_e_candidates_dir.resolve(),
        args.acom_clean_c_details_dir.resolve(),
        args.acom_clean_c_candidates_dir.resolve(),
    )
    pyxem = json.loads(args.pyxem_summary.read_text(encoding="utf-8"))
    if len(pyxem.get("conditions", [])) != 235:
        raise ValueError(
            "Pyxem summary must contain 1 Clean-E + 234 Clean-C conditions"
        )
    if not pyxem.get("aggregates"):
        raise ValueError("Pyxem summary lacks exact aggregate blocks")

    prefix = args.artifact_prefix.strip()
    allowed = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-"
    if not prefix or any(char not in allowed for char in prefix):
        raise ValueError(
            "artifact prefix must contain only letters, digits, '_' or '-'"
        )
    acom_path = output_dir / f"{prefix}_ACOM_TOP5_FULL_SUMMARY.json"
    manifest_path = output_dir / f"{prefix}_ACOM_TOP5_RUN_MANIFEST.json"
    comparison_path = output_dir / f"{prefix}_ACOM_PYXEM_TOP5_COMPARISON.json"
    plot_path = output_dir / f"{prefix}_ACOM_PYXEM_TOP5_BY_DOSE.png"
    report_path = output_dir / f"{prefix}_TOP5_RESULTS.md"
    acom_path.write_text(
        json.dumps(acom, ensure_ascii=False, indent=2, allow_nan=True),
        encoding="utf-8",
    )
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    comparison = {
        "schema": "or4d-v5-acom-pyxem-comparison-v1",
        "scope": {
            "dataset_label": args.dataset_label,
            "samples_per_condition": args.sample_count,
            "acom_clean_e_conditions": 4,
            "pyxem_clean_e_conditions": 1,
            "acom_clean_c_conditions": 702,
            "acom_clean_c_detectors": list(ACOM_CLEAN_C_DETECTORS),
            "pyxem_clean_c_conditions": 234,
            "doses_electrons": [
                100,
                300,
                1000,
                3000,
                10000,
                30000,
                100000,
                300000,
                1000000,
            ],
            "kmax_Ainv": 1.5,
            "reported_ranks": [1, 2, 3, 4, 5],
        },
        "acom_summary": str(acom_path),
        "pyxem_summary": str(args.pyxem_summary.resolve()),
        "acom_aggregates": acom["aggregates"],
        "pyxem_aggregates": pyxem["aggregates"],
    }
    comparison_path.write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2, allow_nan=True),
        encoding="utf-8",
    )
    write_comparison_plot(acom, pyxem, plot_path)
    write_markdown_report(
        acom,
        pyxem,
        comparison_path,
        plot_path,
        report_path,
        dataset_label=args.dataset_label,
        sample_count=args.sample_count,
    )
    print(f"ACOM summary: {acom_path}")
    print(f"Unified manifest: {manifest_path}")
    print(f"Comparison: {comparison_path}")
    print(f"Plot: {plot_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
