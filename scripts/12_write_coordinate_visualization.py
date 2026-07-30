#!/usr/bin/env python3
from __future__ import annotations

import gzip
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from pymatgen.core import Structure

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from or4d_common import (  # noqa: E402
    best_friedel_alignment,
    proper_point_group_rotations,
)

CONFIG_PATH = ROOT / "config" / "benchmark.yaml"
TRACE_PATH = ROOT / "diagnostics" / "clean_coordinate_trace.jsonl.gz"
DETAILS_PATH = ROOT / "reports" / "acom_clean_details.json"
PLAN_AUDIT_PATH = ROOT / "reports" / "acom_plan_audit.json"
OUTPUT_PATH = ROOT / "reports" / "ACOM_COORDINATE_VISUALIZATION.html"
EVALUATION_PATHS = {
    4.0: ROOT / "reports" / "acom_clean_evaluation_angle_4deg.json",
    3.0: ROOT / "reports" / "acom_clean_evaluation_angle_3deg.json",
    2.0: ROOT / "reports" / "acom_clean_evaluation.json",
}


def representative_ids() -> list[str]:
    details = json.loads(DETAILS_PATH.read_text(encoding="utf-8"))
    rows = sorted(
        (
            row
            for row in details["samples"]
            if row["sample_role"] == "headline_core"
        ),
        key=lambda row: row["friedel_equivalent_misorientation_deg"],
    )
    positions = (0, len(rows) // 2, int(0.95 * (len(rows) - 1)), len(rows) - 1)
    selected = [rows[position]["sample_id"] for position in positions]
    friedel_candidates = [
        row
        for row in rows
        if (
            row["strict_misorientation_deg"]
            - row["friedel_equivalent_misorientation_deg"]
        )
        > 10.0
    ]
    if not friedel_candidates:
        raise ValueError("No clear Friedel-branch example was found.")
    friedel_example = min(
        friedel_candidates,
        key=lambda row: row["friedel_equivalent_misorientation_deg"],
    )["sample_id"]
    if friedel_example not in selected:
        selected.append(friedel_example)
    return selected


def rounded(value: object) -> object:
    if isinstance(value, float):
        return round(value, 5)
    if isinstance(value, list):
        return [rounded(item) for item in value]
    return value


def parse_config() -> dict:
    """Read the mapping/scalar subset used by this repository's YAML config."""
    root: dict = {}
    stack: list[tuple[int, dict]] = [(-1, root)]
    for raw_line in CONFIG_PATH.read_text(encoding="utf-8").splitlines():
        content = raw_line.split("#", 1)[0].rstrip()
        if not content.strip() or content.lstrip().startswith("- "):
            continue
        indent = len(content) - len(content.lstrip())
        key, separator, raw_value = content.strip().partition(":")
        if not separator:
            continue
        while stack[-1][0] >= indent:
            stack.pop()
        parent = stack[-1][1]
        value_text = raw_value.strip()
        if not value_text:
            child: dict = {}
            parent[key] = child
            stack.append((indent, child))
            continue
        try:
            parent[key] = json.loads(value_text)
        except json.JSONDecodeError:
            parent[key] = value_text
    return root


def runtime_parameters() -> dict[str, list[list[str]]]:
    config = parse_config()
    dataset = config["dataset"]
    common = config["common"]
    clean = config["clean"]
    acom = config["acom"]
    evaluation = config["evaluation"]
    sampling = config["clean_sampling"]["headline_core"]
    details = json.loads(DETAILS_PATH.read_text(encoding="utf-8"))
    role_counts = Counter(row["sample_role"] for row in details["samples"])
    audit = json.loads(PLAN_AUDIT_PATH.read_text(encoding="utf-8"))
    plan = audit["orientation_plan"]

    benchmark = [
        ["数据集 / Dataset", str(dataset["id"]), "本次结果对应的数据版本"],
        [
            "样本组成 / Sample composition",
            (
                f"{sum(role_counts.values())} = "
                f"{role_counts['legacy_smoke']} legacy + "
                f"{role_counts['headline_core']} headline + "
                f"{role_counts['acom_grid_probe']} probe"
            ),
            "headline 指标只统计 headline_core",
        ],
        [
            "headline 取向采样 / Orientation sampling",
            f"{sampling['method']}; seed={sampling['seed']}; scramble={sampling['scramble']}",
            "独立于 ACOM 搜索网格的 SO(3) 采样",
        ],
        [
            "加速电压 / Accelerating voltage",
            f"{float(common['accelerating_voltage_V']) / 1000:g} kV",
            "电子波长与衍射几何",
        ],
        ["最大倒空间半径 / Kmax", f"{common['k_max_Ainv']} Å⁻¹", "保留的最大探测器倒空间半径"],
        [
            "中心束排除半径 / Central-beam exclusion",
            f"{common['central_beam_exclusion_Ainv']} Å⁻¹",
            "去掉透射中心束附近峰",
        ],
        [
            "激发误差 σ / Excitation-error sigma",
            f"{clean['sigma_excitation_error_Ainv']} Å⁻¹",
            "衍射峰沿 Ewald 球偏离的权重尺度",
        ],
        [
            "激发误差截断 / Excitation-error cutoff",
            (
                f"{clean['tol_excitation_error_mult']}σ = "
                f"{float(clean['tol_excitation_error_mult']) * float(clean['sigma_excitation_error_Ainv']):g} Å⁻¹"
            ),
            "超出该范围的反射不进入图样",
        ],
        ["结构因子阈值 / Structure-factor tolerance", str(clean["tol_structure_factor"]), "生成候选反射时过滤弱结构因子"],
        ["峰强度阈值 / Peak-intensity tolerance", str(clean["tol_intensity"]), "生成图样时过滤弱峰"],
        ["强度归一化 / Intensity normalization", str(common["normalize_peak_intensity"]), "每个样本以最强峰归一化为 1"],
    ]
    acom_rows = [
        ["py4DSTEM 版本 / Version", str(audit["py4DSTEM_version"]), "ACOM 实现版本"],
        ["晶带轴范围 / Zone-axis range", str(acom["zone_axis_range"]), "取向搜索覆盖范围"],
        [
            "标准运行角步长 / Canonical angular step",
            f"zone={acom['angle_step_zone_axis_deg']}°; in-plane={acom['angle_step_in_plane_deg']}°",
            "当前报告的 canonical ACOM 网格",
        ],
        [
            "对照扫描角步长 / Sweep angular steps",
            ", ".join(f"{value}°" for value in acom["sweep_angle_steps_deg"]),
            "同一方法分别运行，不混用检测规则",
        ],
        [
            "实际搜索网格 / Actual search grid",
            (
                f"{plan['num_zone_axes']} zone axes × "
                f"{plan['num_in_plane_steps']} in-plane; "
                f"{plan['num_discrete_seeds_including_mirror']} seeds"
            ),
            "由 canonical 2° 参数生成",
        ],
        ["相关核半径 / Correlation-kernel size", f"{acom['corr_kernel_size_Ainv']} Å⁻¹", "峰位置相关的空间尺度"],
        ["ACOM 激发误差 σ / Excitation-error sigma", f"{acom['sigma_excitation_error_Ainv']} Å⁻¹", "模拟模板中的激发误差尺度"],
        ["径向权重指数 / Radial power", str(acom["power_radial"]), "相关评分中的径向权重"],
        [
            "强度权重指数 / Intensity powers",
            f"sim={acom['power_intensity_simulated']}; exp={acom['power_intensity_experiment']}",
            "模拟峰与观测峰的强度加权",
        ],
        ["ACOM 峰距离容差 / Peak-distance tolerance", f"{acom['tol_distance_Ainv']} Å⁻¹", "orientation_plan 内部峰相关容差"],
        ["最少峰数 / Minimum peaks", str(acom["min_number_peaks"]), "少于该数量不做可靠取向匹配"],
        ["反演对称 / Inversion symmetry", str(acom["inversion_symmetry"]), "搜索时包含 Friedel / mirror 等价"],
        ["返回候选数 / Returned matches", str(acom["num_matches_return"]), "每个样本保留的 ACOM 取向候选"],
        ["CUDA / GPU acceleration", str(acom["cuda"]), "本次运行是否使用 GPU"],
        [
            "评估峰匹配容差 / Evaluation match tolerance",
            f"{evaluation['coordinate_match_tolerance_Ainv']} Å⁻¹",
            "GT 与 ACOM 峰在探测器二维 q 空间的一对一匹配阈值",
        ],
    ]
    return {"benchmark": benchmark, "acom": acom_rows}


def overview_results() -> dict:
    runs = []
    for angle_step, path in EVALUATION_PATHS.items():
        evaluation = json.loads(path.read_text(encoding="utf-8"))
        metrics = evaluation["metrics"]
        runs.append(
            {
                "angle_step_deg": angle_step,
                "num_samples": metrics["num_samples"],
                "mean_deg": metrics["mean_misorientation_deg"],
                "median_deg": metrics["median_misorientation_deg"],
                "p90_deg": metrics["p90_misorientation_deg"],
                "p95_deg": metrics["p95_misorientation_deg"],
                "max_deg": metrics["max_misorientation_deg"],
                "within_1deg": metrics["accuracy_within_1deg"],
                "within_2deg": metrics["accuracy_within_2deg"],
                "within_5deg": metrics["accuracy_within_5deg"],
                "strict_friedel_disagreement_rate": evaluation[
                    "strict_friedel_disagreement_rate"
                ],
            }
        )
    details = json.loads(DETAILS_PATH.read_text(encoding="utf-8"))
    return {
        "runs": runs,
        "canonical_angle_step_deg": details["acom_angle_step_zone_axis_deg"],
        "runtime": details["runtime"],
        "model_limitation": details["matched_model_limitation"],
    }


def symmetry_description(matrix: np.ndarray) -> dict:
    matrix = np.asarray(matrix, dtype=float)
    cosine = float(np.clip((np.trace(matrix) - 1.0) / 2.0, -1.0, 1.0))
    angle = float(np.degrees(np.arccos(cosine)))
    if angle < 1e-8:
        axis = np.array([0.0, 0.0, 1.0])
        signed_angle = 0.0
    elif abs(np.sin(np.deg2rad(angle))) > 1e-7:
        axis = np.array(
            [
                matrix[2, 1] - matrix[1, 2],
                matrix[0, 2] - matrix[2, 0],
                matrix[1, 0] - matrix[0, 1],
            ]
        )
        axis /= np.linalg.norm(axis)
        signed_angle = angle
    else:
        eigenvalues, eigenvectors = np.linalg.eig(matrix)
        axis = np.real(eigenvectors[:, np.argmin(np.abs(eigenvalues - 1.0))])
        axis /= np.linalg.norm(axis)
        signed_angle = angle
    dominant = int(np.argmax(np.abs(axis)))
    if axis[dominant] < 0:
        axis = -axis
        signed_angle = -signed_angle
    if np.allclose(axis, [0.0, 0.0, 1.0], atol=1e-5):
        text = f"绕 crystal Z 轴 {signed_angle:.3f}°"
    else:
        text = (
            f"绕 crystal axis [{axis[0]:.3f}, {axis[1]:.3f}, "
            f"{axis[2]:.3f}] 旋转 {signed_angle:.3f}°"
        )
    return {
        "axis": axis.tolist(),
        "angle_deg": signed_angle,
        "text": text,
    }


def compact_trace(
    row: dict,
    label: str,
    symmetries: list[np.ndarray],
) -> dict:
    standard_matrix = np.asarray(
        row["standard_orientation_matrix_sample_to_crystal"],
        dtype=float,
    )
    acom_matrix = np.asarray(
        row["acom_orientation_matrix_sample_to_crystal"],
        dtype=float,
    )
    reciprocal_matrix = np.asarray(
        row["reciprocal_lattice_matrix_B_Ainv"],
        dtype=float,
    )
    alignment = best_friedel_alignment(
        acom_matrix,
        standard_matrix,
        symmetries,
    )
    reported_error = float(
        row["acom_result"]["friedel_equivalent_misorientation_deg"]
    )
    if not np.isclose(
        alignment["equivalent_misorientation_deg"],
        reported_error,
        atol=1e-6,
    ):
        raise ValueError(
            f"Visualization alignment does not match evaluation for "
            f"{row['sample_id']}: computed "
            f"{alignment['equivalent_misorientation_deg']}, "
            f"reported {reported_error}"
        )
    symmetry = np.asarray(alignment["crystal_symmetry"], dtype=float)
    aligned_matrix = np.asarray(alignment["aligned_matrix"], dtype=float)
    hkl_transform = reciprocal_matrix @ symmetry.T @ np.linalg.inv(
        reciprocal_matrix
    )
    friedel_sign = -1 if alignment["friedel_used"] else 1
    observed_ranked = sorted(
        enumerate(row["standard_observed_reflections"]),
        key=lambda item: item[1]["reported_intensity_normalized"],
        reverse=True,
    )
    predicted_ranked = sorted(
        enumerate(row["acom_simulated_reflections"]),
        key=lambda item: item[1]["reported_by_py4DSTEM_intensity_normalized"],
        reverse=True,
    )[:12]
    observed_index = {
        original_index: compact_index
        for compact_index, (original_index, _) in enumerate(observed_ranked)
    }
    predicted_index = {
        original_index: compact_index
        for compact_index, (original_index, _) in enumerate(predicted_ranked)
    }
    observed = [
        {
            "hkl": reflection["hkl"],
            "q": [
                reflection["reported_qx_Ainv"],
                reflection["reported_qy_Ainv"],
            ],
            "intensity": reflection["reported_intensity_normalized"],
            "g_crystal": reflection["g_crystal_cartesian_Ainv"],
            "q_standard": reflection["standard_g_sample_Ainv"],
            "q_acom_same_hkl_raw": reflection[
                "acom_same_hkl_g_sample_Ainv"
            ],
        }
        for _, reflection in observed_ranked
    ]
    for reflection in observed:
        hkl = np.asarray(reflection["hkl"], dtype=float)
        related_float = friedel_sign * (hkl @ hkl_transform)
        related_hkl = np.rint(related_float).astype(int)
        if not np.allclose(related_float, related_hkl, atol=1e-5):
            raise ValueError(
                f"Symmetry-related HKL is not integral for "
                f"{row['sample_id']}: {related_float.tolist()}"
            )
        g_crystal = np.asarray(reflection["g_crystal"], dtype=float)
        related_g = related_hkl @ reciprocal_matrix
        reflection["acom_related_hkl"] = related_hkl.tolist()
        reflection["acom_related_g_crystal"] = related_g.tolist()
        reflection["q_acom_related_raw"] = (related_g @ acom_matrix).tolist()
        reflection["q_acom_aligned_same_hkl"] = (
            g_crystal @ aligned_matrix
        ).tolist()
    predicted = [
        {
            "hkl": reflection["hkl"],
            "q": [
                reflection["reported_by_py4DSTEM_qx_Ainv"],
                reflection["reported_by_py4DSTEM_qy_Ainv"],
            ],
            "intensity": reflection[
                "reported_by_py4DSTEM_intensity_normalized"
            ],
        }
        for _, reflection in predicted_ranked
    ]
    matches = [
        {
            "observed": observed_index[match["observed_index"]],
            "predicted": predicted_index[match["predicted_index"]],
        }
        for match in row["detector_q_assignment"]["matches"]
        if match["observed_index"] in observed_index
        and match["predicted_index"] in predicted_index
    ]
    return rounded(
        {
            "sample_id": row["sample_id"],
            "label": label,
            "orientation_error_deg": reported_error,
            "strict_misorientation_deg": row["acom_result"][
                "strict_misorientation_deg"
            ],
            "raw_misorientation_deg": alignment["raw_misorientation_deg"],
            "observed_match_fraction": row["comparison_summary"][
                "observed_match_fraction"
            ],
            "q_rmse_Ainv": row["comparison_summary"]["q_distance_rmse_Ainv"],
            "standard_matrix": standard_matrix.tolist(),
            "acom_matrix": acom_matrix.tolist(),
            "acom_aligned_matrix": aligned_matrix.tolist(),
            "best_crystal_symmetry": symmetry.tolist(),
            "best_crystal_symmetry_description": symmetry_description(
                symmetry
            ),
            "friedel_used": alignment["friedel_used"],
            "friedel_matrix": np.asarray(
                alignment["friedel_matrix"]
            ).tolist(),
            "reciprocal_matrix": reciprocal_matrix.tolist(),
            "observed": observed,
            "predicted": predicted,
            "matches": matches,
        }
    )


def load_samples() -> list[dict]:
    config = parse_config()
    structure = Structure.from_file(ROOT / config["dataset"]["cif_path"])
    symmetries = proper_point_group_rotations(structure)
    selected_ids = representative_ids()
    base_labels = ("Best", "Median", "P95", "Worst")
    labels = {
        sample_id: (
            base_labels[index]
            if index < len(base_labels)
            else "Friedel branch"
        )
        for index, sample_id in enumerate(selected_ids)
    }
    selected: dict[str, dict] = {}
    with gzip.open(TRACE_PATH, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            sample_id = row["sample_id"]
            if sample_id in labels:
                selected[sample_id] = compact_trace(
                    row,
                    labels[sample_id],
                    symmetries,
                )
    missing = sorted(set(selected_ids) - set(selected))
    if missing:
        raise ValueError(f"Coordinate trace is missing representative IDs: {missing}")
    return [selected[sample_id] for sample_id in selected_ids]


HTML_TEMPLATE = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Clean v3 coordinate trace visualization</title>
<style>
:root {
  color-scheme: light;
  --background: #ffffff;
  --foreground: #172033;
  --muted: #f5f7fa;
  --muted-foreground: #627087;
  --border: #c9d0dc;
  --series-1: #2774d8;
  --series-2: #d45c37;
  --series-3: #7b52ab;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: #ffffff;
  color: var(--foreground);
  font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
main { max-width: 1320px; margin: 0 auto; padding: 24px; }
h1 { margin: 0 0 6px; font-size: 24px; font-weight: 600; }
.page-header { display: flex; justify-content: space-between; align-items: flex-start; gap: 18px; margin-bottom: 16px; }
.page-switch {
  display: inline-block; flex: none; padding: 9px 12px; color: var(--foreground);
  background: var(--muted); border: 1px solid var(--border); border-radius: 8px;
  text-decoration: none; font-weight: 600; white-space: nowrap;
}
.page-switch:hover { border-color: var(--series-3); background: #f7f4fc; }
.overview { margin: 18px 0 24px; padding-bottom: 20px; border-bottom: 1px solid var(--border); }
.overview-head { display: flex; justify-content: space-between; gap: 12px; align-items: baseline; flex-wrap: wrap; }
.overview h2 { margin: 0; font-size: 19px; }
.overview-subtitle { color: var(--muted-foreground); font-size: 13px; }
.overview-metrics { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; margin: 12px 0; }
.overview-metric { padding: 10px 0; border-top: 2px solid var(--series-1); }
.overview-metric span { display: block; color: var(--muted-foreground); font-size: 12px; }
.overview-metric strong { display: block; margin-top: 3px; font-size: 21px; }
.overview-text { margin: 8px 0 14px; line-height: 1.6; }
.overview-charts { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 22px; }
.overview-charts h3 { margin: 0 0 5px; font-size: 15px; }
.overview-chart { display: block; width: 100%; height: auto; }
.overview-table { margin-top: 14px; }
.overview-caveat { margin: 10px 0 0; color: var(--muted-foreground); font-size: 12px; }
.controls, .tabs, .legend, .reflection-control {
  display: flex; align-items: center; gap: 10px; flex-wrap: wrap;
}
.controls { justify-content: space-between; margin-bottom: 14px; }
button, select {
  font: inherit; color: var(--foreground); background: var(--muted);
  border: 1px solid var(--border); border-radius: 7px; padding: 7px 10px;
}
button { cursor: pointer; }
button[aria-pressed="true"] { background: var(--foreground); color: var(--background); }
button:focus-visible, select:focus-visible { outline: 3px solid var(--series-1); outline-offset: 2px; }
.metrics { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; margin-bottom: 18px; }
.metric { border: 1px solid var(--border); border-radius: 9px; padding: 11px 13px; display: flex; justify-content: space-between; gap: 10px; }
.metric span { color: var(--muted-foreground); }
.notation { margin: 4px 0 22px; }
.notation h2 { margin: 0 0 8px; font-size: 17px; }
.notation-note { margin: 9px 0 0; color: var(--muted-foreground); font-size: 13px; }
.notation code { color: var(--foreground); }
.process-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 22px; margin: 2px 0 20px; }
.process-grid article { padding-top: 12px; border-top: 3px solid var(--series-1); }
.process-grid article:last-child { border-top-color: var(--series-2); }
.process-grid h2 { margin: 0 0 7px; font-size: 17px; }
.process-grid p { margin: 0; color: var(--muted-foreground); font-size: 13px; line-height: 1.55; }
.parameters { margin: 0 0 22px; }
.parameters > summary {
  cursor: pointer; padding: 10px 0; border-top: 1px solid var(--border);
  border-bottom: 1px solid var(--border); font-weight: 600; font-size: 17px;
}
.parameters[open] > summary { margin-bottom: 10px; }
.parameter-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 22px; }
.parameter-grid h3 { margin: 0 0 7px; font-size: 15px; }
.parameter-grid table { font-size: 12px; }
.parameter-grid th, .parameter-grid td { white-space: normal; vertical-align: top; }
.parameter-grid td:nth-child(1) { width: 27%; font-weight: 600; }
.parameter-grid td:nth-child(2) { width: 31%; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
.view-control { display: flex; align-items: center; justify-content: space-between; gap: 12px; flex-wrap: wrap; margin: 0 0 12px; }
.view-control .tabs { margin-left: auto; }
.plots { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 20px; }
.plot-head { display: flex; justify-content: space-between; gap: 12px; flex-wrap: wrap; min-height: 30px; font-weight: 600; }
.legend { color: var(--muted-foreground); font-weight: 400; }
.dot { width: 10px; height: 10px; border: 2px solid var(--series-1); border-radius: 50%; }
.cross { color: var(--series-2); font-size: 20px; line-height: 1; }
.line { width: 20px; border-top: 2px solid var(--series-1); }
.line.acom { border-top-color: var(--series-2); border-top-style: dashed; }
svg { display: block; width: 100%; height: auto; color: var(--foreground); }
.plot-note { min-height: 20px; margin-top: -4px; color: var(--muted-foreground); font-size: 13px; text-align: center; }
.reflection-control { margin-top: 18px; }
.transform { margin-top: 14px; padding-top: 14px; border-top: 1px solid var(--border); }
.transform h2, .all-reflections h2 { margin: 0 0 10px; font-size: 17px; }
.equation-grid {
  display: grid;
  grid-template-columns: minmax(130px, .7fr) auto minmax(300px, 1.4fr) auto minmax(220px, 1fr);
  gap: 10px;
  align-items: stretch;
}
.equation-grid article, .matrix-wrap section { min-width: 0; }
.equation-grid h3, .matrix-wrap h3 { margin: 0 0 6px; font-size: 13px; color: var(--muted-foreground); font-weight: 600; }
.equation-note { margin: 6px 0 0; color: var(--muted-foreground); font-size: 12px; }
.operator { align-self: center; color: var(--muted-foreground); font-size: 22px; }
.matrix-wrap { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 18px; margin-top: 14px; }
.matrix-equation { display: grid; grid-template-columns: minmax(260px, 1fr) auto minmax(220px, .8fr); gap: 10px; align-items: center; }
.representation-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; margin: 12px 0 18px; }
.representation-grid article { min-width: 0; }
.representation-grid h3 { margin: 0 0 6px; font-size: 13px; color: var(--muted-foreground); }
.representation-note { margin: 8px 0 14px; padding: 10px 12px; border-left: 4px solid var(--series-3); background: #f7f4fc; font-size: 13px; }
.friedel-explanation { margin: 14px 0 20px; border: 1px solid var(--border); border-radius: 9px; padding: 12px 14px; }
.friedel-explanation summary { cursor: pointer; font-weight: 600; }
.friedel-explanation p { margin: 10px 0 0; line-height: 1.6; }
.friedel-formula { margin-top: 10px; padding: 10px 12px; background: var(--muted); border-radius: 7px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
pre {
  margin: 0; padding: 10px; border: 1px solid var(--border); border-radius: 7px;
  background: var(--muted); overflow-x: auto; font-size: 13px; line-height: 1.45;
}
.all-reflections { margin-top: 22px; }
.table-wrap { overflow-x: auto; border-top: 1px solid var(--border); }
table { width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; font-size: 13px; }
th, td { padding: 8px 10px; border-bottom: 1px solid var(--border); text-align: left; white-space: nowrap; }
th { color: var(--muted-foreground); font-weight: 600; background: var(--muted); }
tbody tr.is-selected { background: #eef4ff; }
@media (max-width: 720px) {
  main { padding: 16px; }
  .page-header { display: block; }
  .page-switch { margin-top: 12px; }
  .plots, .matrix-wrap, .metrics, .overview-metrics, .overview-charts, .equation-grid, .matrix-equation, .process-grid, .parameter-grid, .representation-grid { grid-template-columns: 1fr; }
  .operator { display: none; }
}
@media print {
  @page { size: landscape; margin: 10mm; }
  body { color: #000000; background: #ffffff; }
  main { max-width: none; padding: 0; }
  button, select, pre, th { color: #000000; background: #ffffff; }
  .page-switch { display: none; }
  details.parameters > :not(summary) { display: block !important; }
  .process-grid, .parameters, .notation, .plots, .transform, .all-reflections, .matrix-wrap { break-inside: avoid; }
  .all-reflections { break-before: page; }
}
</style>
</head>
<body>
<main>
  <header class="page-header">
    <h1>Clean v3 坐标与倒空间中间变量</h1>
    <a class="page-switch" href="CLEAN_IMAGE_ACOM_VISUALIZATION.html">进入二维衍射图版本 →</a>
  </header>
  <section class="overview">
    <div class="overview-head">
      <h2>Overview｜总体结果</h2>
      <span class="overview-subtitle">Headline core · n=1024 · 主指标：Friedel-equivalent misorientation</span>
    </div>
    <div class="overview-metrics">
      <div class="overview-metric"><span>标准角步长 / Canonical step</span><strong id="overview-step"></strong></div>
      <div class="overview-metric"><span>中位取向误差 / Median error</span><strong id="overview-median"></strong></div>
      <div class="overview-metric"><span>P95 取向误差 / P95 error</span><strong id="overview-p95"></strong></div>
      <div class="overview-metric"><span>误差 ≤2° / Accuracy within 2°</span><strong id="overview-within2"></strong></div>
    </div>
    <p id="overview-text" class="overview-text"></p>
    <div class="overview-charts">
      <section>
        <h3>角步长与取向误差 / Angular step vs. error</h3>
        <svg id="error-chart" class="overview-chart" viewBox="0 0 600 270" role="img" aria-label="4度、3度、2度角步长的中位和P95取向误差对比"></svg>
      </section>
      <section>
        <h3>阈值准确率 / Accuracy within thresholds</h3>
        <svg id="accuracy-chart" class="overview-chart" viewBox="0 0 600 270" role="img" aria-label="4度、3度、2度角步长在1度、2度、5度阈值内的准确率"></svg>
      </section>
    </div>
    <div class="table-wrap overview-table">
      <table>
        <thead><tr><th>角步长 / Step</th><th>Mean</th><th>Median</th><th>P90</th><th>P95</th><th>≤1°</th><th>≤2°</th><th>≤5°</th><th>Max</th></tr></thead>
        <tbody id="overview-rows"></tbody>
      </table>
    </div>
    <p id="overview-caveat" class="overview-caveat"></p>
  </section>
  <div class="controls">
    <strong>代表样本 / Representative samples</strong>
    <div id="tabs" class="tabs"></div>
  </div>
  <div class="metrics">
    <div class="metric"><span>Friedel 等价误差 / Primary</span><strong id="error"></strong></div>
    <div class="metric"><span>仅晶体对称误差 / Strict</span><strong id="strict-error"></strong></div>
    <div class="metric"><span>原始代表矩阵差 / Raw</span><strong id="raw-error"></strong></div>
    <div class="metric"><span>最佳晶体对称操作</span><strong id="best-symmetry"></strong></div>
    <div class="metric"><span>Friedel branch</span><strong id="friedel-branch"></strong></div>
    <div class="metric"><span>探测器观测峰匹配率</span><strong id="match"></strong></div>
    <div class="metric"><span>探测器峰匹配 q-RMSE（二维）</span><strong id="rmse"></strong></div>
  </div>
  <details class="friedel-explanation" open>
    <summary>Friedel branch 做了什么，为什么需要它？ / What and why</summary>
    <p><b>它处理理想运动学衍射中的 Friedel 二义性。</b>本 Clean benchmark 的峰集同时包含 <code>q</code> 与 <code>−q</code> 的 Friedel 对，因此把样品坐标的 x、y 方向同时反转、保持电子束 z 方向不变，得到的二维峰位置集合不变。仅凭这种理想峰集，算法不能区分这两个取向代表。</p>
    <div class="friedel-formula">F = diag(−1, −1, 1)<br>R<sub>equiv</sub> = S R<sub>GT</sub> F<br>R<sub>ACOM aligned</sub> = S<sub>best</sub><sup>T</sup> R<sub>ACOM raw</sub> F<sub>best</sub><sup>T</sup></div>
    <p>评测会同时检查 <code>F = I</code> 和 <code>F = diag(−1,−1,1)</code>，只采用误差更小的分支。“Strict”只搜索晶体点群操作 <code>S</code>；“Friedel-equivalent”再加入 <code>F</code>。该操作<b>不会修改 ACOM 原始输出、不会移动检测峰，也不会制造更好的结果</b>，只是把观测上不可区分的代表放到同一个坐标表示中比较。它也不同于晶体三重旋转：晶体对称在左侧作用，Friedel branch 在样品坐标右侧作用。</p>
    <p id="friedel-example-summary"></p>
  </details>
  <section class="process-grid" aria-label="Benchmark 构建与 ACOM 预测流程">
    <article>
      <h2>过程 A｜构建原始 Benchmark（GT）</h2>
      <p>晶体结构 → 倒易基矩阵 B 与结构因子 → 采样标准取向 R<sup>GT</sup> → 按激发误差、强度阈值和 Kmax 生成衍射峰 → 对外只提供观测峰 <code>{qₓ, qᵧ, I}</code>。HKL 和 R<sup>GT</sup> 属于诊断/标准答案，不提供给 ACOM。</p>
    </article>
    <article>
      <h2>过程 B｜运行 ACOM 预测</h2>
      <p>读取公开观测峰 <code>{qₓ, qᵧ, I}</code> → ACOM 搜索取向 → 输出 R<sup>ACOM</sup> → 用该取向模拟预测峰 → 在探测器二维 q 空间与 GT 观测峰使用同一种一对一匹配方法计算指标。</p>
    </article>
  </section>
  <details class="parameters">
    <summary>运行参数 / Runtime parameters（点击展开 / Click to expand）</summary>
    <div class="parameter-grid">
      <article>
        <h3>过程 A｜Benchmark 生成参数 / Generation parameters</h3>
        <div class="table-wrap">
          <table><thead><tr><th>参数 / Parameter</th><th>本次取值 / Value</th><th>作用 / Purpose</th></tr></thead><tbody id="benchmark-parameters"></tbody></table>
        </div>
      </article>
      <article>
        <h3>过程 B｜ACOM 预测与评估参数 / Prediction and evaluation</h3>
        <div class="table-wrap">
          <table><thead><tr><th>参数 / Parameter</th><th>本次取值 / Value</th><th>作用 / Purpose</th></tr></thead><tbody id="acom-parameters"></tbody></table>
        </div>
      </article>
    </div>
  </details>
  <section class="notation">
    <h2>变量定义与坐标约定</h2>
    <div class="table-wrap">
      <table>
        <thead><tr><th>符号</th><th>完整含义</th><th>坐标系 / 维度</th><th>单位</th><th>本页用途</th></tr></thead>
        <tbody>
          <tr><td><code>h = [h,k,l]</code></td><td>一个衍射反射的 Miller 指数；不是整个晶体取向</td><td>晶格基底，整数三元组</td><td>无量纲</td><td>指定一条晶面反射</td></tr>
          <tr><td><code>B</code></td><td>晶体倒易基矩阵；三行依次是 a*、b*、c*</td><td>晶体笛卡尔坐标，3×3</td><td>Å⁻¹</td><td><code>g_c = h B</code>；不含 2π</td></tr>
          <tr><td><code>g_c</code></td><td>g<sub>crystal</sub>，该 HKL 在晶体笛卡尔坐标中的倒易向量</td><td>晶体笛卡尔坐标，三维</td><td>Å⁻¹</td><td>由 HKL 和 B 唯一确定</td></tr>
          <tr><td><code>R<sup>GT</sup></code></td><td>标准答案的 sample→crystal 取向矩阵</td><td>旋转矩阵，3×3</td><td>无量纲</td><td>其列是样品 x、y、z 轴在晶体坐标中的方向</td></tr>
          <tr><td><code>R<sup>ACOM</sup></code></td><td>ACOM 估计的 sample→crystal 取向矩阵</td><td>旋转矩阵，3×3</td><td>无量纲</td><td>与 R<sup>GT</sup> 比较取向误差</td></tr>
          <tr><td><code>g_s</code></td><td>同一倒易向量在样品坐标中的表示</td><td>样品笛卡尔坐标，[q<sub>x</sub>,q<sub>y</sub>,q<sub>z</sub>]</td><td>Å⁻¹</td><td><code>g_s = g_c R</code>（本页行向量写法）</td></tr>
          <tr><td><code>q = [q_x,q_y]</code></td><td>探测器平面中的二维衍射峰坐标</td><td>样品 x-y / 探测器平面，二维</td><td>Å⁻¹</td><td>取 g<sub>s</sub> 的前两项；q<sub>z</sub> 沿电子束且不在公开峰表中</td></tr>
          <tr><td><code>I</code></td><td>该衍射峰的归一化强度</td><td>标量</td><td>无量纲</td><td>决定点大小及强度排序</td></tr>
        </tbody>
      </table>
    </div>
    <p class="notation-note">本页数值数组统一按行显示，因此使用 <code>g_c = h B</code>、<code>g_s = g_c R</code>；与代码中的列向量写法 <code>g_s = Rᵀ g_c</code> 完全等价。GT = 生成数据时的标准答案，ACOM = 算法估计结果。</p>
  </section>
  <div class="view-control">
    <strong>ACOM 取向代表</strong>
    <div id="orientation-tabs" class="tabs" aria-label="选择对称对齐后或原始 ACOM 取向矩阵"></div>
  </div>
  <div id="representation-note" class="representation-note"></div>
  <section class="representation-grid" aria-label="GT、原始 ACOM 与对齐后 ACOM 矩阵">
    <article><h3>R<sup>GT</sup> / Ground Truth</h3><pre id="summary-gt-matrix"></pre></article>
    <article><h3>R<sup>ACOM raw</sup> / 原始输出</h3><pre id="summary-raw-matrix"></pre></article>
    <article><h3>R<sup>ACOM aligned</sup> / 对称对齐后</h3><pre id="summary-aligned-matrix"></pre></article>
  </section>
  <div class="view-control">
    <strong>图层显示</strong>
    <div id="view-tabs" class="tabs" aria-label="选择 GT、ACOM 或叠加显示"></div>
  </div>
  <div class="plots">
    <section>
      <div class="plot-head">
        <span>探测器倒空间 q = [qₓ, qᵧ]</span>
        <span id="detector-legend" class="legend"></span>
      </div>
      <svg id="detector" viewBox="0 0 560 400" role="img" aria-label="标准与 ACOM 衍射峰"></svg>
      <div id="detector-note" class="plot-note"></div>
    </section>
    <section>
      <div class="plot-head">
        <span>R 的三列：样品坐标轴在晶体 XY 平面的投影</span>
        <span id="axes-legend" class="legend"></span>
      </div>
      <svg id="axes" viewBox="0 0 560 400" role="img" aria-label="标准与 ACOM 样品坐标轴"></svg>
      <div id="axes-note" class="plot-note"></div>
    </section>
  </div>
  <div class="reflection-control">
    <label for="reflection"><strong>选择一个 GT 反射做坐标诊断</strong></label>
    <select id="reflection"></select>
  </div>
  <section class="transform">
    <h2>所选 GT 反射的坐标链：h → g<sub>c</sub> → g<sub>s</sub> → q</h2>
    <div class="equation-grid">
      <article><h3>① h = [h,k,l]（Miller 指数）</h3><pre id="hkl-vector"></pre></article>
      <div class="operator">×</div>
      <article><h3>② B（倒易基矩阵；行是 a*、b*、c*；Å⁻¹）</h3><pre id="reciprocal-matrix"></pre></article>
      <div class="operator">=</div>
      <article><h3>③ g<sub>c</sub> = g<sub>crystal</sub>（晶体坐标；Å⁻¹）</h3><pre id="g-vector"></pre></article>
    </div>
  </section>
  <div class="matrix-wrap">
    <section>
      <h3>④ GT：g<sub>s</sub><sup>GT</sup> = g<sub>c</sub> R<sup>GT</sup>（样品坐标）</h3>
      <div class="matrix-equation"><pre id="standard-matrix"></pre><div class="operator">→</div><pre id="standard-q"></pre></div>
      <p class="equation-note">右侧三项依次为 [qₓ, qᵧ, q<sub>z</sub>]；探测器只使用前两项。</p>
    </section>
    <section>
      <h3 id="acom-transform-title"></h3>
      <p id="acom-reflection-map" class="equation-note"></p>
      <div class="matrix-equation"><pre id="acom-matrix"></pre><div class="operator">→</div><pre id="acom-q"></pre></div>
      <p id="acom-transform-note" class="equation-note"></p>
    </section>
  </div>
  <section class="all-reflections">
    <h2 id="all-reflections-title"></h2>
    <div class="table-wrap">
      <table>
        <thead><tr><th>#</th><th>GT HKL</th><th id="acom-hkl-heading">ACOM HKL</th><th>I<sup>GT</sup></th><th>g<sub>c</sub><sup>GT</sup> (Å⁻¹)</th><th id="acom-g-heading">g<sub>c</sub><sup>ACOM</sup></th><th>g<sub>s</sub><sup>GT</sup> = [qₓ,qᵧ,qz]</th><th id="acom-q-heading">g<sub>s</sub><sup>ACOM</sup></th><th>‖Δg<sub>s</sub>‖₂</th></tr></thead>
        <tbody id="reflection-rows"></tbody>
      </table>
    </div>
  </section>
</main>
<script>
const reportData = __DATA__;
const samples = reportData.samples;
const runtimeParameters = reportData.runtime_parameters;
const overviewResults = reportData.overview_results;
const kMaxAinv = Number(reportData.k_max_Ainv);
const tabs = document.getElementById("tabs");
const viewTabs = document.getElementById("view-tabs");
const orientationTabs = document.getElementById("orientation-tabs");
const detector = document.getElementById("detector");
const axes = document.getElementById("axes");
const reflection = document.getElementById("reflection");
let sampleIndex = 1;
let reflectionIndex = 0;
let displayMode = "overlay";
let orientationMode = "aligned";
const esc = value => String(value).replace(/[&<>"']/g, character => {
  if (character === "&") return "&amp;";
  if (character === "<") return "&lt;";
  if (character === ">") return "&gt;";
  if (character === "'") return "&#39;";
  return "&quot;";
});
const vector = values => `[${values.map(value => Number(value).toFixed(4)).join(", ")}]`;
const matrix = values => values.map(vector).join("\\n");
const qPixelsPerAinv = 165 / kMaxAinv;
const qX = value => 280 + value * qPixelsPerAinv;
const qY = value => 200 - value * qPixelsPerAinv;
const axisScale = value => value * 125;
const selectedAcomMatrix = sample =>
  orientationMode === "aligned" ? sample.acom_aligned_matrix : sample.acom_matrix;
const selectedAcomReflection = peak => orientationMode === "aligned"
  ? {
      hkl: peak.hkl,
      gCrystal: peak.g_crystal,
      q: peak.q_acom_aligned_same_hkl,
    }
  : {
      hkl: peak.acom_related_hkl,
      gCrystal: peak.acom_related_g_crystal,
      q: peak.q_acom_related_raw,
    };

function renderOverview() {
  const runs = overviewResults.runs;
  const canonical = runs.find(run => run.angle_step_deg === overviewResults.canonical_angle_step_deg);
  const coarse = runs.find(run => run.angle_step_deg === 4);
  const medianReduction = 100 * (coarse.median_deg - canonical.median_deg) / coarse.median_deg;
  const p95Reduction = 100 * (coarse.p95_deg - canonical.p95_deg) / coarse.p95_deg;
  const within2Gain = 100 * (canonical.within_2deg - coarse.within_2deg);
  const above5 = 100 * (1 - canonical.within_5deg);
  document.getElementById("overview-step").textContent = `${canonical.angle_step_deg.toFixed(0)}°`;
  document.getElementById("overview-median").textContent = `${canonical.median_deg.toFixed(3)}°`;
  document.getElementById("overview-p95").textContent = `${canonical.p95_deg.toFixed(3)}°`;
  document.getElementById("overview-within2").textContent = `${(canonical.within_2deg * 100).toFixed(1)}%`;
  document.getElementById("overview-text").innerHTML =
    `<strong>总体结论 / Overall:</strong> 2° 是本次 4°、3°、2° 三组同方法测试中最优的网格。` +
    `相对 4°，中位误差下降 ${medianReduction.toFixed(1)}%，P95 下降 ${p95Reduction.toFixed(1)}%，` +
    `≤2° 准确率提高 ${within2Gain.toFixed(1)} 个百分点。` +
    ` <span class="overview-subtitle">The 2° grid performs best across the tested angular steps.</span>`;
  document.getElementById("overview-caveat").textContent =
    `尾部仍有失败样本：2° 运行中 ${above5.toFixed(1)}% 的 headline 样本误差大于 5°，最大误差 ${canonical.max_deg.toFixed(2)}°；` +
    `strict/Friedel 判定不一致率 ${(canonical.strict_friedel_disagreement_rate * 100).toFixed(1)}%。` +
    `模型边界 / Limitation: clean inputs and ACOM templates use the same CIF and py4DSTEM kinematical model; this measures self-consistency, not real-data generalization.`;
  document.getElementById("overview-rows").innerHTML = runs.map(run =>
    `<tr><td>${run.angle_step_deg.toFixed(0)}°</td>` +
    `<td>${run.mean_deg.toFixed(3)}°</td><td>${run.median_deg.toFixed(3)}°</td>` +
    `<td>${run.p90_deg.toFixed(3)}°</td><td>${run.p95_deg.toFixed(3)}°</td>` +
    `<td>${(run.within_1deg * 100).toFixed(1)}%</td>` +
    `<td>${(run.within_2deg * 100).toFixed(1)}%</td>` +
    `<td>${(run.within_5deg * 100).toFixed(1)}%</td>` +
    `<td>${run.max_deg.toFixed(2)}°</td></tr>`
  ).join("");

  const chartWidth = 600;
  const chartHeight = 270;
  const left = 48;
  const right = 18;
  const top = 24;
  const bottom = 42;
  const plotWidth = chartWidth - left - right;
  const plotHeight = chartHeight - top - bottom;
  const centers = runs.map((_, index) => left + plotWidth * (index + 0.5) / runs.length);

  const errorMax = 6;
  const errorY = value => top + plotHeight * (1 - value / errorMax);
  let errorMarkup = "";
  [0, 2, 4, 6].forEach(tick => {
    const y = errorY(tick);
    errorMarkup += `<line x1="${left}" y1="${y}" x2="${chartWidth-right}" y2="${y}" stroke="var(--border)"/>` +
      `<text x="${left-8}" y="${y}" dy="0.35em" text-anchor="end" fill="var(--muted-foreground)" font-size="12">${tick}°</text>`;
  });
  runs.forEach((run, index) => {
    const values = [
      {value: run.median_deg, offset: -22, color: "var(--series-1)"},
      {value: run.p95_deg, offset: 22, color: "var(--series-2)"},
    ];
    values.forEach(item => {
      const y = errorY(item.value);
      errorMarkup += `<rect x="${centers[index]+item.offset-16}" y="${y}" width="32" height="${top+plotHeight-y}" fill="${item.color}"/>` +
        `<text x="${centers[index]+item.offset}" y="${y-5}" text-anchor="middle" fill="var(--foreground)" font-size="11">${item.value.toFixed(2)}°</text>`;
    });
    errorMarkup += `<text x="${centers[index]}" y="${chartHeight-14}" text-anchor="middle" fill="var(--foreground)">${run.angle_step_deg.toFixed(0)}° step</text>`;
  });
  errorMarkup += `<rect x="405" y="8" width="10" height="10" fill="var(--series-1)"/><text x="421" y="17" fill="var(--foreground)" font-size="12">Median</text>` +
    `<rect x="488" y="8" width="10" height="10" fill="var(--series-2)"/><text x="504" y="17" fill="var(--foreground)" font-size="12">P95</text>`;
  document.getElementById("error-chart").innerHTML = errorMarkup;

  const accuracyY = value => top + plotHeight * (1 - value);
  let accuracyMarkup = "";
  [0, 0.25, 0.5, 0.75, 1].forEach(tick => {
    const y = accuracyY(tick);
    accuracyMarkup += `<line x1="${left}" y1="${y}" x2="${chartWidth-right}" y2="${y}" stroke="var(--border)"/>` +
      `<text x="${left-8}" y="${y}" dy="0.35em" text-anchor="end" fill="var(--muted-foreground)" font-size="12">${(tick*100).toFixed(0)}%</text>`;
  });
  const accuracySeries = [
    {key: "within_1deg", label: "≤1°", offset: -32, color: "var(--series-1)"},
    {key: "within_2deg", label: "≤2°", offset: 0, color: "var(--series-2)"},
    {key: "within_5deg", label: "≤5°", offset: 32, color: "var(--series-3)"},
  ];
  runs.forEach((run, index) => {
    accuracySeries.forEach(series => {
      const value = run[series.key];
      const y = accuracyY(value);
      accuracyMarkup += `<rect x="${centers[index]+series.offset-11}" y="${y}" width="22" height="${top+plotHeight-y}" fill="${series.color}"/>`;
    });
    accuracyMarkup += `<text x="${centers[index]}" y="${chartHeight-14}" text-anchor="middle" fill="var(--foreground)">${run.angle_step_deg.toFixed(0)}° step</text>`;
  });
  accuracySeries.forEach((series, index) => {
    const x = 348 + index * 78;
    accuracyMarkup += `<rect x="${x}" y="8" width="10" height="10" fill="${series.color}"/><text x="${x+16}" y="17" fill="var(--foreground)" font-size="12">${series.label}</text>`;
  });
  document.getElementById("accuracy-chart").innerHTML = accuracyMarkup;
}

function renderParameters() {
  [
    ["benchmark-parameters", runtimeParameters.benchmark],
    ["acom-parameters", runtimeParameters.acom],
  ].forEach(([elementId, rows]) => {
    document.getElementById(elementId).innerHTML = rows.map(row =>
      `<tr><td>${esc(row[0])}</td><td>${esc(row[1])}</td><td>${esc(row[2])}</td></tr>`
    ).join("");
  });
}

function renderTabs() {
  tabs.innerHTML = "";
  samples.forEach((sample, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.setAttribute("aria-pressed", index === sampleIndex ? "true" : "false");
    button.textContent = `${sample.label} · ${sample.sample_id.replace("clean_", "")}`;
    button.addEventListener("click", () => {
      sampleIndex = index;
      reflectionIndex = 0;
      render();
    });
    tabs.appendChild(button);
  });
}

function renderViewTabs() {
  const modes = [
    ["gt", "仅 Benchmark / GT"],
    ["acom", "仅 ACOM 预测"],
    ["overlay", "GT + ACOM 叠加"],
  ];
  viewTabs.innerHTML = "";
  modes.forEach(([mode, label]) => {
    const button = document.createElement("button");
    button.type = "button";
    button.setAttribute("aria-pressed", mode === displayMode ? "true" : "false");
    button.textContent = label;
    button.addEventListener("click", () => {
      displayMode = mode;
      render();
    });
    viewTabs.appendChild(button);
  });
  const showGT = displayMode !== "acom";
  const showACOM = displayMode !== "gt";
  document.getElementById("detector-legend").innerHTML =
    `${showGT ? '<i class="dot"></i>GT 观测峰' : ''}` +
    `${showACOM ? '<i class="cross">×</i>ACOM 预测峰' : ''}`;
  document.getElementById("axes-legend").innerHTML =
    `${showGT ? '<i class="line"></i>R<sup>GT</sup>' : ''}` +
    `${showACOM ? `<i class="line acom"></i>R<sup>ACOM ${orientationMode}</sup>` : ''}`;
}

function renderOrientationTabs(sample) {
  const modes = [
    ["aligned", "对称对齐后（默认）"],
    ["raw", "原始 ACOM 输出"],
  ];
  orientationTabs.innerHTML = "";
  modes.forEach(([mode, label]) => {
    const button = document.createElement("button");
    button.type = "button";
    button.setAttribute("aria-pressed", mode === orientationMode ? "true" : "false");
    button.textContent = label;
    button.addEventListener("click", () => {
      orientationMode = mode;
      render();
    });
    orientationTabs.appendChild(button);
  });
  document.getElementById("representation-note").innerHTML =
    orientationMode === "aligned"
      ? `<strong>当前绘图使用对称/Friedel 对齐后代表。</strong> R<sup>ACOM aligned</sup> = S<sub>best</sub><sup>T</sup> R<sup>ACOM raw</sup> F<sub>best</sub><sup>T</sup>；它与上方 Friedel 等价误差使用同一个最佳分支。当前 F<sub>best</sub> = ${sample.friedel_used ? "diag(-1,-1,1)（已使用）" : "I（未使用）"}。`
      : `<strong>当前绘图使用原始 ACOM 输出。</strong> 坐标轴可能因晶体对称等价操作而与 GT 相差很大；这时应同时查看原始代表矩阵差、对称相关 HKL 和对称等价误差。`;
}

function renderDetector(sample) {
  const plottedObserved = sample.observed.slice(0, 12).map((peak, index) => ({peak, index}));
  if (reflectionIndex >= 12) {
    plottedObserved.push({peak: sample.observed[reflectionIndex], index: reflectionIndex});
  }
  let markup = `
    <line x1="82" y1="200" x2="500" y2="200" stroke="var(--border)"/>
    <line x1="280" y1="22" x2="280" y2="378" stroke="var(--border)"/>
    <circle cx="280" cy="200" r="165" fill="none" stroke="var(--border)"/>
    <text x="495" y="222" text-anchor="end" fill="var(--muted-foreground)">qₓ (Å⁻¹)</text>
    <text x="290" y="36" fill="var(--muted-foreground)">qᵧ (Å⁻¹)</text>`;
  if (displayMode === "overlay") {
    sample.matches.filter(match => match.observed < 12).forEach(match => {
      const observed = sample.observed[match.observed];
      const predicted = sample.predicted[match.predicted];
      markup += `<line x1="${qX(observed.q[0])}" y1="${qY(observed.q[1])}" x2="${qX(predicted.q[0])}" y2="${qY(predicted.q[1])}" stroke="var(--border)"/>`;
    });
  }
  if (displayMode !== "acom") {
    plottedObserved.forEach(({peak, index}) => {
      const radius = 4 + 7 * Math.sqrt(peak.intensity);
      const selected = index === reflectionIndex;
      markup += `<circle cx="${qX(peak.q[0])}" cy="${qY(peak.q[1])}" r="${selected ? radius + 3 : radius}" fill="none" stroke="${selected ? "var(--series-3)" : "var(--series-1)"}" stroke-width="${selected ? 3 : 1.8}"><title>GT 观测反射 HKL ${peak.hkl.join(",")} · I=${peak.intensity}</title></circle>`;
    });
  }
  if (displayMode !== "gt") {
    sample.predicted.forEach(peak => {
      const x = qX(peak.q[0]);
      const y = qY(peak.q[1]);
      const size = 4 + 6 * Math.sqrt(peak.intensity);
      markup += `<path d="M ${x-size} ${y-size} L ${x+size} ${y+size} M ${x-size} ${y+size} L ${x+size} ${y-size}" stroke="var(--series-2)" stroke-width="2"><title>ACOM 模拟预测反射 HKL ${peak.hkl.join(",")} · I=${peak.intensity}</title></path>`;
    });
  }
  detector.innerHTML = markup;
  const modeText = displayMode === "gt" ? "GT 观测峰" : displayMode === "acom" ? "ACOM 预测峰" : "GT 与 ACOM 叠加";
  document.getElementById("detector-note").textContent =
    `当前显示：${modeText}（强度前 12）；诊断选择为 GT HKL [${sample.observed[reflectionIndex].hkl.join(", ")}]`;
}

function renderAxes(sample) {
  const centerX = 280;
  const centerY = 200;
  const names = ["x", "y", "z beam"];
  let markup = `
    <circle cx="${centerX}" cy="${centerY}" r="145" fill="none" stroke="var(--border)"/>
    <line x1="80" y1="${centerY}" x2="480" y2="${centerY}" stroke="var(--border)"/>
    <line x1="${centerX}" y1="22" x2="${centerX}" y2="378" stroke="var(--border)"/>
    <text x="475" y="${centerY + 22}" text-anchor="end" fill="var(--muted-foreground)">crystal X</text>
    <text x="${centerX + 10}" y="34" fill="var(--muted-foreground)">crystal Y</text>`;
  [
    {matrix: sample.standard_matrix, key: "gt"},
    {matrix: selectedAcomMatrix(sample), key: "acom"},
  ].filter(item => displayMode === "overlay" || displayMode === item.key).forEach(item => {
    const matrix = item.matrix;
    const matrixIndex = item.key === "gt" ? 0 : 1;
    names.forEach((name, column) => {
      const x = centerX + axisScale(matrix[0][column]);
      const y = centerY - axisScale(matrix[1][column]);
      const dx = x - centerX;
      const dy = y - centerY;
      const length = Math.hypot(dx, dy) || 1;
      const radialX = dx / length;
      const radialY = dy / length;
      const perpendicularX = -radialY;
      const perpendicularY = radialX;
      const side = matrixIndex === 0 ? -1 : 1;
      const labelX = Math.max(34, Math.min(526, x + radialX * 12 + perpendicularX * side * 14));
      const labelY = Math.max(22, Math.min(378, y + radialY * 12 + perpendicularY * side * 14));
      const color = matrixIndex === 0 ? "var(--series-1)" : "var(--series-2)";
      const dash = matrixIndex === 0 ? "" : `stroke-dasharray="7 5"`;
      markup += `<line x1="${centerX}" y1="${centerY}" x2="${x}" y2="${y}" stroke="${color}" stroke-width="2.5" ${dash}/>` +
        `<circle cx="${x}" cy="${y}" r="3.5" fill="${color}"/>` +
        `<line x1="${x}" y1="${y}" x2="${labelX}" y2="${labelY}" stroke="${color}" stroke-width="1"/>` +
        `<text x="${labelX}" y="${labelY}" dy="0.35em" text-anchor="middle" fill="var(--foreground)" font-size="12">${matrixIndex === 0 ? "Std" : "ACOM"} ${name}</text>`;
    });
  });
  if (displayMode !== "acom") {
    const selected = sample.observed[reflectionIndex];
    const norm = Math.hypot(selected.g_crystal[0], selected.g_crystal[1]) || 1;
    const gx = centerX + 105 * selected.g_crystal[0] / norm;
    const gy = centerY - 105 * selected.g_crystal[1] / norm;
    markup += `<line x1="${centerX}" y1="${centerY}" x2="${gx}" y2="${gy}" stroke="var(--series-3)" stroke-width="4"/><circle cx="${gx}" cy="${gy}" r="4" fill="var(--series-3)"/>`;
  }
  axes.innerHTML = markup;
  document.getElementById("axes-note").innerHTML =
    displayMode === "gt"
      ? "仅显示 R<sup>GT</sup>；紫色为所选 GT 反射的 g<sub>c</sub> 方向"
      : displayMode === "acom"
        ? `仅显示 R<sup>ACOM ${orientationMode}</sup>；虚线表示算法估计的样品坐标轴`
        : `叠加 R<sup>GT</sup>（实线）与 R<sup>ACOM ${orientationMode}</sup>（虚线）；紫色为所选 GT 反射的 g<sub>c</sub> 方向`;
}

function renderTransform(sample) {
  const selected = sample.observed[reflectionIndex];
  const acomReflection = selectedAcomReflection(selected);
  const acomMatrix = selectedAcomMatrix(sample);
  document.getElementById("hkl-vector").textContent = `[${selected.hkl.join(", ")}]`;
  document.getElementById("reciprocal-matrix").textContent = matrix(sample.reciprocal_matrix);
  document.getElementById("g-vector").textContent = vector(selected.g_crystal);
  document.getElementById("standard-matrix").textContent = matrix(sample.standard_matrix);
  document.getElementById("acom-matrix").textContent = matrix(acomMatrix);
  document.getElementById("standard-q").textContent = vector(selected.q_standard);
  document.getElementById("acom-q").textContent = vector(acomReflection.q);
  document.getElementById("acom-transform-title").innerHTML =
    orientationMode === "aligned"
      ? "④ 对齐后：g<sub>s</sub><sup>ACOM aligned</sup> = g<sub>c</sub><sup>GT</sup> R<sup>ACOM aligned</sup>"
      : "④ 原始代表：g<sub>s</sub><sup>ACOM raw</sup> = g<sub>c</sub><sup>related</sup> R<sup>ACOM raw</sup>";
  document.getElementById("acom-reflection-map").innerHTML =
    orientationMode === "aligned"
      ? `对齐后使用同一个 GT HKL [${selected.hkl.join(", ")}]。`
      : `晶体对称重标记：GT HKL [${selected.hkl.join(", ")}] ↔ raw ACOM HKL [${acomReflection.hkl.join(", ")}]；不是强行比较同一个 HKL。`;
  document.getElementById("acom-transform-note").textContent =
    orientationMode === "aligned"
      ? "该矩阵和坐标与对称等价误差使用同一个最佳晶体对称/Friedel 分支。"
      : "原始矩阵保留 ACOM 实际输出；对称相关 HKL 在原始代表下投到同一探测器峰。";
}

function renderReflectionTable(sample) {
  document.getElementById("all-reflections-title").innerHTML =
    `当前 GT 图样中的 ${sample.observed.length} 个有效反射（Kmax = ${kMaxAinv} Å⁻¹）· ${orientationMode === "aligned" ? "对称对齐后代表" : "原始 ACOM 代表"}`;
  document.getElementById("acom-hkl-heading").textContent =
    orientationMode === "aligned" ? "Aligned ACOM HKL（同 GT）" : "Raw ACOM 对称相关 HKL";
  document.getElementById("acom-g-heading").innerHTML =
    orientationMode === "aligned"
      ? "g<sub>c</sub><sup>ACOM aligned</sup>"
      : "g<sub>c</sub><sup>ACOM related</sup>";
  document.getElementById("acom-q-heading").innerHTML =
    orientationMode === "aligned"
      ? "g<sub>s</sub><sup>ACOM aligned</sup>"
      : "g<sub>s</sub><sup>ACOM raw</sup>";
  document.getElementById("reflection-rows").innerHTML = sample.observed.map((peak, index) => {
    const acomReflection = selectedAcomReflection(peak);
    const delta = Math.hypot(
      peak.q_standard[0] - acomReflection.q[0],
      peak.q_standard[1] - acomReflection.q[1],
      peak.q_standard[2] - acomReflection.q[2]
    );
    return `<tr class="${index === reflectionIndex ? "is-selected" : ""}">` +
      `<td>${index + 1}</td>` +
      `<td>[${esc(peak.hkl.join(", "))}]</td>` +
      `<td>[${esc(acomReflection.hkl.join(", "))}]</td>` +
      `<td>${peak.intensity.toFixed(4)}</td>` +
      `<td>${esc(vector(peak.g_crystal))}</td>` +
      `<td>${esc(vector(acomReflection.gCrystal))}</td>` +
      `<td>${esc(vector(peak.q_standard))}</td>` +
      `<td>${esc(vector(acomReflection.q))}</td>` +
      `<td>${delta.toFixed(5)}</td></tr>`;
  }).join("");
}

function render() {
  const sample = samples[sampleIndex];
  const friedelExample = samples.find(item => item.label === "Friedel branch");
  renderTabs();
  renderOrientationTabs(sample);
  renderViewTabs();
  document.getElementById("error").textContent = `${sample.orientation_error_deg.toFixed(3)}°`;
  document.getElementById("strict-error").textContent = `${sample.strict_misorientation_deg.toFixed(3)}°`;
  document.getElementById("raw-error").textContent = `${sample.raw_misorientation_deg.toFixed(3)}°`;
  document.getElementById("best-symmetry").textContent =
    sample.best_crystal_symmetry_description.text;
  document.getElementById("friedel-branch").textContent =
    sample.friedel_used ? "使用 / used" : "未使用 / identity";
  document.getElementById("match").textContent = `${(sample.observed_match_fraction * 100).toFixed(1)}%`;
  document.getElementById("rmse").textContent = `${sample.q_rmse_Ainv.toFixed(4)} Å⁻¹`;
  document.getElementById("summary-gt-matrix").textContent = matrix(sample.standard_matrix);
  document.getElementById("summary-raw-matrix").textContent = matrix(sample.acom_matrix);
  document.getElementById("summary-aligned-matrix").textContent = matrix(sample.acom_aligned_matrix);
  document.getElementById("friedel-example-summary").innerHTML =
    `<b>本页实例：</b><code>${friedelExample.sample_id}</code> 的 Strict 误差为 ` +
    `${friedelExample.strict_misorientation_deg.toFixed(3)}°，加入 Friedel 等价后为 ` +
    `${friedelExample.orientation_error_deg.toFixed(3)}°，最佳分支明确使用 ` +
    `<code>F = diag(−1,−1,1)</code>。点击上方“Friedel branch”样本即可查看原始矩阵、` +
    `对齐后矩阵及 Friedel/晶体对称相关 HKL。`;
  reflection.innerHTML = sample.observed.map((peak, index) =>
    `<option value="${index}" ${index === reflectionIndex ? "selected" : ""}>GT HKL [${esc(peak.hkl.join(", "))}] · I=${peak.intensity.toFixed(3)}</option>`
  ).join("");
  renderDetector(sample);
  renderAxes(sample);
  renderTransform(sample);
  renderReflectionTable(sample);
}

reflection.addEventListener("change", event => {
  reflectionIndex = Number(event.target.value);
  render();
});
renderParameters();
renderOverview();
render();
</script>
</body>
</html>
"""


def main() -> None:
    samples = load_samples()
    config = parse_config()
    payload = json.dumps(
        {
            "samples": samples,
            "runtime_parameters": runtime_parameters(),
            "overview_results": overview_results(),
            "k_max_Ainv": config["common"]["k_max_Ainv"],
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    output = HTML_TEMPLATE.replace("__DATA__", payload)
    OUTPUT_PATH.write_text(output, encoding="utf-8")
    print(f"Standalone coordinate visualization: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
