#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = Path(
    os.environ.get("OR4D_REPORT_V3_DIR", ROOT / "reports" / "v3")
).resolve()
sys.path.insert(0, str(ROOT / "src"))

from or4d_common import load_config  # noqa: E402


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def summarize(values: list[float]) -> dict:
    array = np.asarray(values, dtype=float)
    return {
        "num_samples": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "max": float(array.max()),
        "acc1": float(np.mean(array <= 1.0)),
        "acc2": float(np.mean(array <= 2.0)),
        "acc5": float(np.mean(array <= 5.0)),
    }


def metric_row(label: str, metrics: dict) -> str:
    return (
        f"| {label} | {metrics['num_samples']} | "
        f"{metrics['mean_misorientation_deg']:.3f} | "
        f"{metrics['median_misorientation_deg']:.3f} | "
        f"{metrics['p90_misorientation_deg']:.3f} | "
        f"{metrics['p95_misorientation_deg']:.3f} | "
        f"{metrics['max_misorientation_deg']:.3f} | "
        f"{100 * metrics['accuracy_within_1deg']:.1f}% | "
        f"{100 * metrics['accuracy_within_2deg']:.1f}% | "
        f"{100 * metrics['accuracy_within_5deg']:.1f}% |"
    )


def compact_metric_row(label: str, metrics: dict) -> str:
    return (
        f"| {label} | {metrics['num_samples']} | {metrics['mean']:.3f} | "
        f"{metrics['median']:.3f} | {metrics['p90']:.3f} | "
        f"{metrics['p95']:.3f} | {metrics['max']:.3f} | "
        f"{100 * metrics['acc1']:.1f}% | {100 * metrics['acc2']:.1f}% | "
        f"{100 * metrics['acc5']:.1f}% |"
    )


def step_tag(step: float) -> str:
    return f"angle_{f'{step:g}'.replace('.', 'p')}deg"


def load_sweep_rows(config: dict) -> list[dict]:
    canonical = float(config["acom"]["angle_step_zone_axis_deg"])
    rows: list[dict] = []
    for step_value in config["acom"]["sweep_angle_steps_deg"]:
        step = float(step_value)
        suffix = "" if step == canonical else f"_{step_tag(step)}"
        evaluation_path = REPORT_DIR / f"acom_clean_evaluation{suffix}.json"
        details_path = REPORT_DIR / f"acom_clean_details{suffix}.json"
        audit_path = REPORT_DIR / f"acom_plan_audit{suffix}.json"
        if not (
            evaluation_path.exists()
            and details_path.exists()
            and audit_path.exists()
        ):
            continue
        sweep_evaluation = load_json(evaluation_path)
        sweep_details = load_json(details_path)
        sweep_audit = load_json(audit_path)
        rows.append(
            {
                "step": step,
                "metrics": sweep_evaluation["metrics"],
                "runtime": sweep_details["runtime"]["matching"],
                "plan": sweep_audit["orientation_plan"],
                "canonical": step == canonical,
                "sha256": sweep_details["sha256"],
                "versions": sweep_details["versions"],
            }
        )
    return rows


def distance_bin_rows(
    rows: list[dict],
    edges: list[float],
) -> list[tuple[str, dict]]:
    output: list[tuple[str, dict]] = []
    for index, lower in enumerate(edges):
        upper = edges[index + 1] if index + 1 < len(edges) else np.inf
        selected = [
            row["friedel_equivalent_misorientation_deg"]
            for row in rows
            if lower
            <= row["nearest_zone_axis_node_misorientation_deg"]
            < upper
        ]
        if selected:
            label = (
                f"[{lower:g}, {upper:g})°"
                if np.isfinite(upper)
                else f"[{lower:g}, +∞)°"
            )
            output.append((label, summarize(selected)))
    return output


def main() -> None:
    config = load_config()
    details = load_json(REPORT_DIR / "acom_clean_details.json")
    evaluation = load_json(REPORT_DIR / "acom_clean_evaluation.json")
    audit = load_json(REPORT_DIR / "acom_plan_audit.json")

    headline_role = str(config["evaluation"]["headline_sample_role"])
    headline_rows = [
        row for row in details["samples"] if row["sample_role"] == headline_role
    ]
    if not headline_rows:
        raise ValueError(f"No ACOM rows found for headline role {headline_role}")

    zone_distance = np.asarray(
        [row["nearest_zone_axis_node_misorientation_deg"] for row in headline_rows],
        dtype=float,
    )
    seed_distance = np.asarray(
        [
            row["nearest_discrete_search_seed_misorientation_deg"]
            for row in headline_rows
        ],
        dtype=float,
    )
    errors = np.asarray(
        [row["friedel_equivalent_misorientation_deg"] for row in headline_rows],
        dtype=float,
    )
    zone_error_correlation = float(np.corrcoef(zone_distance, errors)[0, 1])
    seed_error_correlation = float(np.corrcoef(seed_distance, errors)[0, 1])
    above_5deg = int(np.sum(errors > 5.0))
    above_10deg = int(np.sum(errors > 10.0))

    role_metrics = evaluation["metrics_by_sample_role"]
    plan = audit["orientation_plan"]
    runtime = details["runtime"]
    matching = runtime["matching"]
    sweep_rows = load_sweep_rows(config)
    canonical_angle = float(config["acom"]["angle_step_zone_axis_deg"])
    lines = [
        "# NCM811 Clean-Peak ACOM Report",
        "",
        f"- Dataset: `{details['dataset_id']}`",
        f"- Headline cohort: `{headline_role}` "
        f"({evaluation['metrics']['num_samples']} samples)",
        "- Primary metric: Friedel-equivalent misorientation under proper "
        "crystal point-group rotations",
        f"- Canonical baseline: py4DSTEM ACOM, {canonical_angle:g}° zone-axis "
        f"step and {canonical_angle:g}° in-plane step",
        "",
        "## Headline result",
        "",
        "| Metric | n | Mean | Median | P90 | P95 | Max | Acc@1° | Acc@2° | Acc@5° |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        metric_row("Friedel-equivalent", evaluation["metrics"]),
        metric_row("Strict", evaluation["metrics_strict"]),
        "",
        "Strict/Friedel disagreement rate: "
        f"{100 * evaluation['strict_friedel_disagreement_rate']:.2f}%.",
        f"Catastrophic mismatches: {above_5deg}/{len(errors)} above 5°; "
        f"{above_10deg}/{len(errors)} above 10°.",
    ]
    if sweep_rows:
        reference_hashes = sweep_rows[0]["sha256"]
        reference_versions = sweep_rows[0]["versions"]
        if not all(
            row["sha256"] == reference_hashes
            and row["versions"] == reference_versions
            for row in sweep_rows
        ):
            raise ValueError(
                "ACOM sweep runs do not share identical inputs and versions"
            )
        lines.extend(
            [
                "",
                "## ACOM angular-resolution sweep",
                "",
                "Controlled comparison: all sweep rows have identical config, "
                "CIF, orientation-manifest, public-peak and ground-truth SHA256 "
                "values and identical software versions. Only the zone-axis and "
                "in-plane angular steps change.",
                "",
                "| Step | n | Mean | Median | P95 | Max | Acc@1° | Acc@2° | "
                "Acc@5° | Seeds incl. mirror | Match time | Throughput |",
                "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
                "---: | ---: | ---: | ---: |",
            ]
        )
        for row in sorted(sweep_rows, key=lambda value: value["step"], reverse=True):
            metrics = row["metrics"]
            runtime_row = row["runtime"]
            plan_row = row["plan"]
            marker = " (canonical)" if row["canonical"] else ""
            lines.append(
                f"| {row['step']:g}°{marker} | {metrics['num_samples']} | "
                f"{metrics['mean_misorientation_deg']:.3f} | "
                f"{metrics['median_misorientation_deg']:.3f} | "
                f"{metrics['p95_misorientation_deg']:.3f} | "
                f"{metrics['max_misorientation_deg']:.3f} | "
                f"{100 * metrics['accuracy_within_1deg']:.1f}% | "
                f"{100 * metrics['accuracy_within_2deg']:.1f}% | "
                f"{100 * metrics['accuracy_within_5deg']:.1f}% | "
                f"{plan_row['num_discrete_seeds_including_mirror']} | "
                f"{runtime_row['total_seconds']:.1f} s | "
                f"{runtime_row['throughput_samples_per_second']:.1f}/s |"
            )
    lines.extend(
        [
        "",
        "## Result by sample role",
        "",
        "| Role | n | Mean | Median | P90 | P95 | Max | Acc@1° | Acc@2° | Acc@5° |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for role, metrics in sorted(role_metrics.items()):
        lines.append(metric_row(role, metrics))

    lines.extend(
        [
            "",
            "## Headline result by nearest zone-axis node distance",
            "",
            "| Distance bin | n | Mean | Median | P90 | P95 | Max | Acc@1° | Acc@2° | Acc@5° |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    edges = [
        float(value)
        for value in config["evaluation"]["plan_distance_bin_edges_deg"]
    ]
    for label, metrics in distance_bin_rows(headline_rows, edges):
        lines.append(compact_metric_row(label, metrics))

    lines.extend(
        [
            "",
            "## ACOM diagnostics",
            "",
            f"- Zone-axis node distance/error Pearson correlation: "
            f"{zone_error_correlation:.3f}",
            f"- Discrete seed distance/error Pearson correlation: "
            f"{seed_error_correlation:.3f}",
            "- Discrete seed distance is diagnostic only. ACOM performs a "
            "parabolic sub-grid fit of the in-plane correlation peak, so it is "
            "not an error lower bound.",
            f"- Plan nodes: {plan['num_zone_axes']} zone axes × "
            f"{plan['num_in_plane_steps']} in-plane steps; "
            f"{plan['num_discrete_seeds_including_mirror']} seeds including mirror.",
            f"- Plan build: {runtime['plan_build_seconds']:.3f} s.",
            f"- Matching: {matching['total_seconds']:.3f} s total, "
            f"{matching['p50_seconds']:.4f} s p50, "
            f"{matching['p90_seconds']:.4f} s p90, "
            f"{matching['p99_seconds']:.4f} s p99, "
            f"{matching['throughput_samples_per_second']:.1f} samples/s.",
            "",
            "## Reproducibility",
            "",
            f"- Source git revision: `{details['source_git_revision']}`",
            f"- Versions: Python {details['versions']['python']}, "
            f"NumPy {details['versions']['numpy']}, "
            f"py4DSTEM {details['versions']['py4DSTEM']}, "
            f"pymatgen {details['versions']['pymatgen']}, "
            f"h5py {details['versions']['h5py']}",
        ]
    )
    for name, digest in sorted(details["sha256"].items()):
        lines.append(f"- SHA256 `{name}`: `{digest}`")

    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            details["matched_model_limitation"],
            "",
            "![ACOM error distributions](acom_error_comparison.png)",
            "",
            "![Zone-axis distance versus error](acom_offgrid_vs_error.png)",
            "",
            "![Representative peak overlays](acom_peak_overlay.png)",
            "",
            "逐坐标、逐反射、HKL 和 ACOM 峰差异见 "
            "[ACOM coordinate analysis](ACOM_COORDINATE_ANALYSIS.md)。",
            "",
            "可离线打开的交互图见 "
            "[ACOM coordinate visualization](ACOM_COORDINATE_VISUALIZATION.html)。",
            "",
        ]
    )

    output = REPORT_DIR / "ACOM_CLEAN_REPORT.md"
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
