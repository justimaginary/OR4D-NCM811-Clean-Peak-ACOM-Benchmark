#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import io
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, TextIO

import h5py
import numpy as np
import py4DSTEM
from pymatgen.core import Structure

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports" / "v3"
sys.path.insert(0, str(ROOT / "src"))

from coordinate_trace import (  # noqa: E402
    crystal_to_sample_reciprocal,
    match_detector_peaks,
    reciprocal_cartesian_from_hkl,
    reflection_coordinate_record,
)
from or4d_common import (  # noqa: E402
    cif_path,
    load_config,
    normalize_intensities,
    read_jsonl,
    read_peak_h5,
)


def unique_by_id(records: list[dict], source: str) -> dict[str, dict]:
    output: dict[str, dict] = {}
    for record in records:
        sample_id = str(record["sample_id"])
        if sample_id in output:
            raise ValueError(f"Duplicate sample_id in {source}: {sample_id}")
        output[sample_id] = record
    return output


def setup_crystal(config: dict, structure: Structure):
    crystal = py4DSTEM.process.diffraction.Crystal.from_pymatgen_structure(
        structure=structure,
        conventional_standard_structure=False,
    )
    crystal.setup_diffraction(
        accelerating_voltage=float(config["common"]["accelerating_voltage_V"])
    )
    crystal.calculate_structure_factors(
        k_max=float(config["common"]["k_max_Ainv"]),
        tol_structure_factor=float(config["clean"]["tol_structure_factor"]),
    )
    return crystal


def filtered_simulated_peaks(crystal, matrix: np.ndarray, config: dict) -> dict:
    clean = config["clean"]
    common = config["common"]
    k_max = float(common["k_max_Ainv"])
    central_exclusion = float(common["central_beam_exclusion_Ainv"])
    point_list = crystal.generate_diffraction_pattern(
        orientation_matrix=matrix,
        sigma_excitation_error=float(clean["sigma_excitation_error_Ainv"]),
        tol_excitation_error_mult=float(clean["tol_excitation_error_mult"]),
        tol_intensity=float(clean["tol_intensity"]),
        k_max=k_max,
    )
    data = point_list.data
    qx = np.asarray(data["qx"], dtype=float)
    qy = np.asarray(data["qy"], dtype=float)
    intensity = np.asarray(data["intensity"], dtype=float)
    radius = np.hypot(qx, qy)
    keep = (
        np.isfinite(qx)
        & np.isfinite(qy)
        & np.isfinite(intensity)
        & (radius >= central_exclusion)
        & (radius <= k_max + 1e-8)
        & (intensity > 0)
    )
    qx = qx[keep]
    qy = qy[keep]
    intensity = normalize_intensities(intensity[keep]).astype(float)
    hkl = np.column_stack(
        [
            np.asarray(data[field], dtype=int)[keep]
            for field in ("h", "k", "l")
        ]
    )
    radius = np.hypot(qx, qy)
    order = np.lexsort((qy, qx, radius))
    return {
        "qx": qx[order],
        "qy": qy[order],
        "intensity": intensity[order],
        "hkl": hkl[order],
    }


def load_diagnostic_reflections() -> dict[str, dict[str, np.ndarray]]:
    path = ROOT / "diagnostics" / "clean_reflections.h5"
    groups: defaultdict[str, list[Any]] = defaultdict(list)
    with h5py.File(path, "r") as h5:
        for row in h5["reflections"]:
            raw_id = row["sample_id"]
            sample_id = raw_id.decode() if isinstance(raw_id, bytes) else str(raw_id)
            groups[sample_id].append(row)
    output: dict[str, dict[str, np.ndarray]] = {}
    for sample_id, rows in groups.items():
        output[sample_id] = {
            "qx": np.asarray([row["qx"] for row in rows], dtype=float),
            "qy": np.asarray([row["qy"] for row in rows], dtype=float),
            "intensity": np.asarray(
                [row["intensity"] for row in rows],
                dtype=float,
            ),
            "hkl": np.asarray(
                [[row["h"], row["k"], row["l"]] for row in rows],
                dtype=int,
            ),
        }
    return output


def summarize_peak_match(
    match: dict[str, Any],
    num_observed: int,
    num_predicted: int,
) -> dict[str, Any]:
    rows = match["matches"]
    distances = np.asarray([row["q_distance_Ainv"] for row in rows], dtype=float)
    intensity_delta = np.asarray(
        [row["predicted_minus_observed_intensity"] for row in rows],
        dtype=float,
    )
    num_matches = len(rows)
    return {
        "num_observed_peaks": int(num_observed),
        "num_acom_simulated_peaks": int(num_predicted),
        "num_q_matched_peaks": int(num_matches),
        "observed_match_fraction": (
            float(num_matches / num_observed) if num_observed else 0.0
        ),
        "predicted_match_fraction": (
            float(num_matches / num_predicted) if num_predicted else 0.0
        ),
        "q_distance_rmse_Ainv": (
            float(np.sqrt(np.mean(distances**2))) if distances.size else None
        ),
        "q_distance_median_Ainv": (
            float(np.median(distances)) if distances.size else None
        ),
        "q_distance_max_Ainv": (
            float(np.max(distances)) if distances.size else None
        ),
        "intensity_mae_on_q_matches": (
            float(np.mean(np.abs(intensity_delta)))
            if intensity_delta.size
            else None
        ),
        "raw_hkl_equal_fraction_on_q_matches": (
            float(np.mean([row["raw_hkl_equal"] for row in rows]))
            if rows
            else None
        ),
        "raw_hkl_equal_or_friedel_fraction_on_q_matches": (
            float(
                np.mean(
                    [
                        row["raw_hkl_equal"] or row["raw_hkl_friedel_equal"]
                        for row in rows
                    ]
                )
            )
            if rows
            else None
        ),
    }


def predicted_reflection_records(
    peaks: dict[str, np.ndarray],
    reciprocal_matrix: np.ndarray,
    matrix_acom: np.ndarray,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, hkl in enumerate(peaks["hkl"]):
        g_crystal = reciprocal_cartesian_from_hkl(hkl, reciprocal_matrix)
        g_sample = crystal_to_sample_reciprocal(g_crystal, matrix_acom)
        records.append(
            {
                "predicted_index": index,
                "hkl": np.asarray(hkl, dtype=int).tolist(),
                "g_crystal_cartesian_Ainv": g_crystal.tolist(),
                "g_sample_Ainv": g_sample.tolist(),
                "reported_by_py4DSTEM_qx_Ainv": float(peaks["qx"][index]),
                "reported_by_py4DSTEM_qy_Ainv": float(peaks["qy"][index]),
                "reported_by_py4DSTEM_intensity_normalized": float(
                    peaks["intensity"][index]
                ),
                "reported_minus_derived_qxy_Ainv": (
                    np.asarray([peaks["qx"][index], peaks["qy"][index]])
                    - g_sample[:2]
                ).tolist(),
            }
        )
    return records


def build_sample_trace(
    sample: dict,
    gt: dict,
    detail: dict,
    diagnostic: dict[str, np.ndarray],
    predicted_peaks: dict[str, np.ndarray],
    structure: Structure,
    tolerance_Ainv: float,
    config: dict,
) -> dict[str, Any]:
    matrix_gt = np.asarray(
        gt["orientation_matrix_sample_to_crystal"],
        dtype=float,
    )
    matrix_acom = np.asarray(
        detail["predicted_orientation_matrix_sample_to_crystal"],
        dtype=float,
    )
    direct_matrix = np.asarray(structure.lattice.matrix, dtype=float)
    reciprocal_matrix = np.asarray(
        structure.lattice.reciprocal_lattice_crystallographic.matrix,
        dtype=float,
    )
    for field in ("qx", "qy", "intensity"):
        np.testing.assert_allclose(
            np.asarray(sample[field], dtype=float),
            diagnostic[field],
            atol=5e-7,
            rtol=0.0,
            err_msg=f"{sample['sample_id']} diagnostic/public {field} mismatch",
        )

    observed_reflections: list[dict[str, Any]] = []
    for index, hkl in enumerate(diagnostic["hkl"]):
        record = reflection_coordinate_record(
            hkl,
            reciprocal_matrix,
            matrix_gt,
            matrix_acom,
            reported_qx=float(diagnostic["qx"][index]),
            reported_qy=float(diagnostic["qy"][index]),
            reported_intensity=float(diagnostic["intensity"][index]),
        )
        record["observed_index"] = index
        observed_reflections.append(record)

    matching = match_detector_peaks(
        diagnostic,
        predicted_peaks,
        tolerance_Ainv=tolerance_Ainv,
    )
    match_summary = summarize_peak_match(
        matching,
        len(diagnostic["qx"]),
        len(predicted_peaks["qx"]),
    )
    coordinate_residuals = np.asarray(
        [
            row["reported_minus_standard_qxy_norm_Ainv"]
            for row in observed_reflections
        ],
        dtype=float,
    )
    match_summary["standard_coordinate_identity_max_residual_Ainv"] = float(
        coordinate_residuals.max(initial=0.0)
    )
    match_summary["standard_coordinate_identity_rmse_Ainv"] = float(
        np.sqrt(np.mean(coordinate_residuals**2))
        if coordinate_residuals.size
        else 0.0
    )

    return {
        "schema_version": 1,
        "sample_id": sample["sample_id"],
        "sample_role": gt.get("sample_role"),
        "sampling_type": gt.get("sampling_type"),
        "coordinate_convention": {
            "orientation": "R_sample_to_crystal",
            "matrix_columns": (
                "sample x, y, z axes expressed in crystal Cartesian coordinates"
            ),
            "direct_lattice_matrix_A": "row vectors a, b, c in angstrom",
            "reciprocal_lattice_matrix_B": (
                "row vectors a*, b*, c* in 1/angstrom, without 2*pi"
            ),
            "hkl_to_crystal": "g_crystal = [h,k,l] @ B",
            "crystal_to_sample": "g_sample = R_sample_to_crystal.T @ g_crystal",
            "detector_projection": (
                "qx = g_sample[0], qy = g_sample[1]; qz is beam-axis component"
            ),
            "k_max_Ainv": float(config["common"]["k_max_Ainv"]),
            "central_beam_exclusion_Ainv": float(
                config["common"]["central_beam_exclusion_Ainv"]
            ),
        },
        "direct_lattice_matrix_A": direct_matrix.tolist(),
        "reciprocal_lattice_matrix_B_Ainv": reciprocal_matrix.tolist(),
        "standard_orientation_matrix_sample_to_crystal": matrix_gt.tolist(),
        "standard_sample_axes_in_crystal_cartesian": {
            "x": matrix_gt[:, 0].tolist(),
            "y": matrix_gt[:, 1].tolist(),
            "z_beam": matrix_gt[:, 2].tolist(),
        },
        "acom_orientation_matrix_sample_to_crystal": matrix_acom.tolist(),
        "acom_sample_axes_in_crystal_cartesian": {
            "x": matrix_acom[:, 0].tolist(),
            "y": matrix_acom[:, 1].tolist(),
            "z_beam": matrix_acom[:, 2].tolist(),
        },
        "acom_result": {
            "correlation_score": float(detail["correlation_score"]),
            "zone_axis_plan_index": int(detail["zone_axis_plan_index"]),
            "in_plane_plan_index": int(detail["in_plane_plan_index"]),
            "mirror_match": bool(detail["mirror_match"]),
            "euler_angles_deg": detail["euler_angles_deg"],
            "strict_misorientation_deg": float(
                detail["strict_misorientation_deg"]
            ),
            "friedel_equivalent_misorientation_deg": float(
                detail["friedel_equivalent_misorientation_deg"]
            ),
        },
        "comparison_summary": match_summary,
        "standard_observed_reflections": observed_reflections,
        "acom_simulated_reflections": predicted_reflection_records(
            predicted_peaks,
            reciprocal_matrix,
            matrix_acom,
        ),
        "detector_q_assignment": {
            "tolerance_Ainv": float(tolerance_Ainv),
            **matching,
            "unmatched_observed_hkl": [
                diagnostic["hkl"][index].tolist()
                for index in matching["unmatched_observed_indices"]
            ],
            "unmatched_acom_hkl": [
                predicted_peaks["hkl"][index].tolist()
                for index in matching["unmatched_predicted_indices"]
            ],
        },
    }


def representative_ids(rows: list[dict]) -> list[str]:
    headline = [row for row in rows if row["sample_role"] == "headline_core"]

    def error(row: dict) -> float:
        if "friedel_equivalent_misorientation_deg" in row:
            return float(row["friedel_equivalent_misorientation_deg"])
        return float(
            row["acom_result"]["friedel_equivalent_misorientation_deg"]
        )

    ordered = sorted(
        headline,
        key=error,
    )
    if not ordered:
        return [row["sample_id"] for row in rows[:4]]
    positions = [0, len(ordered) // 2, int(0.95 * (len(ordered) - 1)), -1]
    return list(dict.fromkeys(ordered[position]["sample_id"] for position in positions))


def open_text_output(path: Path) -> TextIO:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".gz":
        raw = path.open("wb")
        compressed = gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=raw,
            mtime=0,
        )
        return io.TextIOWrapper(compressed, encoding="utf-8")
    return path.open("w", encoding="utf-8")


def optional_number(value: Any, fmt: str = ".4f") -> str:
    if value is None:
        return "n/a"
    return format(float(value), fmt)


def aggregate_rows(traces: list[dict], role: str | None = None) -> dict[str, float]:
    selected = [
        trace
        for trace in traces
        if role is None or trace["sample_role"] == role
    ]
    summaries = [trace["comparison_summary"] for trace in selected]
    if not summaries:
        return {}
    return {
        "num_samples": len(selected),
        "median_observed_match_fraction": float(
            np.median([row["observed_match_fraction"] for row in summaries])
        ),
        "mean_observed_match_fraction": float(
            np.mean([row["observed_match_fraction"] for row in summaries])
        ),
        "median_q_rmse_Ainv": float(
            np.median(
                [
                    row["q_distance_rmse_Ainv"]
                    for row in summaries
                    if row["q_distance_rmse_Ainv"] is not None
                ]
            )
        ),
        "median_intensity_mae": float(
            np.median(
                [
                    row["intensity_mae_on_q_matches"]
                    for row in summaries
                    if row["intensity_mae_on_q_matches"] is not None
                ]
            )
        ),
        "median_raw_hkl_equal_fraction": float(
            np.median(
                [
                    row["raw_hkl_equal_fraction_on_q_matches"]
                    for row in summaries
                    if row["raw_hkl_equal_fraction_on_q_matches"] is not None
                ]
            )
        ),
        "median_raw_hkl_equal_or_friedel_fraction": float(
            np.median(
                [
                    row["raw_hkl_equal_or_friedel_fraction_on_q_matches"]
                    for row in summaries
                    if row[
                        "raw_hkl_equal_or_friedel_fraction_on_q_matches"
                    ]
                    is not None
                ]
            )
        ),
        "max_standard_coordinate_residual_Ainv": float(
            max(
                row["standard_coordinate_identity_max_residual_Ainv"]
                for row in summaries
            )
        ),
    }


def write_summary(
    path: Path,
    traces: list[dict],
    config: dict,
    trace_path: Path,
) -> None:
    headline = [
        trace for trace in traces if trace["sample_role"] == "headline_core"
    ]
    errors = np.asarray(
        [
            trace["acom_result"]["friedel_equivalent_misorientation_deg"]
            for trace in headline
        ],
        dtype=float,
    )
    all_metrics = aggregate_rows(traces)
    headline_metrics = aggregate_rows(traces, "headline_core")
    reps = set(representative_ids(traces))
    representative = [
        trace
        for trace in sorted(
            traces,
            key=lambda row: row["acom_result"][
                "friedel_equivalent_misorientation_deg"
            ],
        )
        if trace["sample_id"] in reps
    ]
    worst = sorted(
        headline,
        key=lambda row: row["acom_result"][
            "friedel_equivalent_misorientation_deg"
        ],
        reverse=True,
    )[:20]
    example = representative[len(representative) // 2]
    example_reflections = example["standard_observed_reflections"][:3]
    example_input = {
        "sample_id": example["sample_id"],
        "peaks": [
            {
                "qx": row["reported_qx_Ainv"],
                "qy": row["reported_qy_Ainv"],
                "intensity": row["reported_intensity_normalized"],
            }
            for row in example_reflections
        ],
    }
    example_output = {
        "standard_orientation_matrix_sample_to_crystal": example[
            "standard_orientation_matrix_sample_to_crystal"
        ],
        "acom_orientation_matrix_sample_to_crystal": example[
            "acom_orientation_matrix_sample_to_crystal"
        ],
        "friedel_equivalent_misorientation_deg": example["acom_result"][
            "friedel_equivalent_misorientation_deg"
        ],
    }
    example_hkl_flow = {
        key: example_reflections[0][key]
        for key in (
            "hkl",
            "g_crystal_cartesian_Ainv",
            "standard_g_sample_Ainv",
            "reported_qx_Ainv",
            "reported_qy_Ainv",
            "acom_same_hkl_g_sample_Ainv",
        )
    }

    lines = [
        "# Clean v3 坐标链路与 ACOM 对比",
        "",
        f"- 数据集：`{config['dataset']['id']}`",
        f"- 已追踪样本：{len(traces)}",
        f"- `k_max`：{config['common']['k_max_Ainv']:.3f} Å⁻¹",
        "- 公共输入：每个样本一个变长峰表 `{qx, qy, intensity}`，HKL 不公开给算法。",
        "- 标准输出与 ACOM 输出：`3×3 orientation_matrix_sample_to_crystal`。",
        f"- 完整逐样本逐反射数据：`{trace_path.name}`",
        "",
        "## 坐标是怎么来的",
        "",
        "直接晶格矩阵 `A` 的行是实空间 `a,b,c`（Å）；晶体学倒易矩阵 "
        "`B` 的行是 `a*,b*,c*`（Å⁻¹，不含 `2π`）。对反射 `(h,k,l)`：",
        "",
        "```text",
        "g_crystal = [h, k, l] @ B",
        "g_sample  = R_sample_to_crystal.T @ g_crystal",
        "qx = g_sample[0], qy = g_sample[1], qz = g_sample[2]",
        "```",
        "",
        "`R` 的三列分别是样品 `x,y,z` 轴在晶体笛卡尔坐标中的方向；因此 "
        "`qx` 也等于 `g_crystal · x_crystal`。`qz` 沿电子束方向，公开峰表只保留探测器平面的 `qx,qy`。",
        "",
        "## 标准结果与 ACOM 结果",
        "",
        f"- 标准 HKL→q 恒等式的全局最大残差："
        f"{all_metrics['max_standard_coordinate_residual_Ainv']:.3e} Å⁻¹。",
        f"- 全部样本 ACOM 峰的中位观测匹配率："
        f"{100 * all_metrics['median_observed_match_fraction']:.2f}%；"
        f"平均 {100 * all_metrics['mean_observed_match_fraction']:.2f}%。",
        f"- Headline ACOM 峰的中位观测匹配率："
        f"{100 * headline_metrics['median_observed_match_fraction']:.2f}%；"
        f"中位 q-RMSE {headline_metrics['median_q_rmse_Ainv']:.4f} Å⁻¹。",
        f"- Headline 已匹配峰的中位强度 MAE："
        f"{headline_metrics['median_intensity_mae']:.4f}；逐样本原始 HKL "
        f"相同率的中位数为 "
        f"{100 * headline_metrics['median_raw_hkl_equal_fraction']:.1f}%，"
        f"计入 `(h,k,l)↔(-h,-k,-l)` 后为 "
        f"{100 * headline_metrics['median_raw_hkl_equal_or_friedel_fraction']:.1f}%。",
    ]
    if errors.size:
        lines.extend(
            [
                f"- Headline 取向误差：median {np.median(errors):.3f}°，"
                f"p95 {np.percentile(errors, 95):.3f}°，max {errors.max():.3f}°。",
                f"- 大于 5°：{int(np.sum(errors > 5.0))}/{len(errors)}；"
                f"大于 10°：{int(np.sum(errors > 10.0))}/{len(errors)}。",
            ]
        )
    lines.extend(
        [
            "",
            "峰匹配是在探测器 `q` 平面做一对一最小距离分配，并以 "
            f"{config['evaluation']['coordinate_match_tolerance_Ainv']:.3f} Å⁻¹ 截断。"
            "原始 HKL 标签可能因晶体对称操作或 Friedel 等价而改变，所以 HKL "
            "相同率只是诊断；主判断依据仍是 q 匹配与 symmetry/Friedel-aware 取向误差。",
            "",
            "## 数据和结果长什么样",
            "",
            "公共输入不含 HKL，下面只截取一个样本的前三个峰：",
            "",
            "```json",
            *json.dumps(example_input, ensure_ascii=False, indent=2).splitlines(),
            "```",
            "",
            "标准答案与 ACOM 都是 sample→crystal 的 3×3 旋转矩阵：",
            "",
            "```json",
            *json.dumps(example_output, ensure_ascii=False, indent=2).splitlines(),
            "```",
            "",
            "诊断文件才把 HKL 和完整坐标链路连起来；同一条反射的实际记录例如：",
            "",
            "```json",
            *json.dumps(example_hkl_flow, ensure_ascii=False, indent=2).splitlines(),
            "```",
            "",
            "差异主要有三类：取向矩阵/样品坐标轴不同，使同一 HKL 的 "
            "`qx,qy,qz` 改变；激发误差筛选使两套峰表出现缺峰或多峰；晶体对称和 "
            "Friedel 等价会让几乎相同的 detector `q` 使用不同原始 HKL 标签。",
            "",
            "## 代表样本",
            "",
            "| sample | role | 取向误差 | observed/predicted/matched | observed match | q-RMSE | raw HKL 相同 |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for trace in representative:
        summary = trace["comparison_summary"]
        lines.append(
            f"| `{trace['sample_id']}` | {trace['sample_role']} | "
            f"{trace['acom_result']['friedel_equivalent_misorientation_deg']:.3f}° | "
            f"{summary['num_observed_peaks']}/"
            f"{summary['num_acom_simulated_peaks']}/"
            f"{summary['num_q_matched_peaks']} | "
            f"{100 * summary['observed_match_fraction']:.1f}% | "
            f"{optional_number(summary['q_distance_rmse_Ainv'])} | "
            f"{optional_number(summary['raw_hkl_equal_fraction_on_q_matches'], '.3f')} |"
        )
    lines.extend(
        [
            "",
            "## Headline 差异最大的 20 个样本",
            "",
            "| sample | 取向误差 | corr | peaks observed/predicted/matched | observed match | q-RMSE | mirror |",
            "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for trace in worst:
        summary = trace["comparison_summary"]
        acom = trace["acom_result"]
        lines.append(
            f"| `{trace['sample_id']}` | "
            f"{acom['friedel_equivalent_misorientation_deg']:.3f}° | "
            f"{acom['correlation_score']:.4f} | "
            f"{summary['num_observed_peaks']}/"
            f"{summary['num_acom_simulated_peaks']}/"
            f"{summary['num_q_matched_peaks']} | "
            f"{100 * summary['observed_match_fraction']:.1f}% | "
            f"{optional_number(summary['q_distance_rmse_Ainv'])} | "
            f"{acom['mirror_match']} |"
        )
    lines.extend(
        [
            "",
            "## 如何打印所有中间变量",
            "",
            "```bash",
            "# 打印一个样本（含矩阵、每个 HKL、g_crystal、qx/qy/qz、ACOM 峰和差异）",
            "conda run -n or4d-clean python scripts/10_trace_clean_coordinates.py \\",
            "  --sample-id clean_core_0000 --stdout",
            "",
            "# 打印全部样本；输出量很大",
            "conda run -n or4d-clean python scripts/10_trace_clean_coordinates.py \\",
            "  --all --stdout",
            "```",
            "",
            "压缩 JSONL 中每行对应一个样本，可直接解压后按 "
            "`sample_id`、`standard_observed_reflections[].hkl` 或 "
            "`detector_q_assignment.matches[]` 检索。",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Trace real/reciprocal coordinate transformations and compare "
            "standard Clean peaks with peaks generated from ACOM orientations."
        )
    )
    parser.add_argument(
        "--sample-id",
        action="append",
        default=[],
        help="Trace one sample ID; repeat for multiple samples.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Trace all samples instead of representative samples.",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Print every selected intermediate variable as indented JSON.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Trace path. Defaults to the canonical full path with --all and "
            "to a separate selected-sample path otherwise."
        ),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        help=(
            "Markdown path. Defaults to the canonical report with --all and "
            "to a separate selected-sample report otherwise."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.all and args.sample_id:
        raise ValueError("Use either --all or --sample-id, not both")
    config = load_config()
    structure = Structure.from_file(cif_path(config))
    crystal = setup_crystal(config, structure)
    samples = read_peak_h5(ROOT / "public" / "clean_peaks.h5")
    samples_by_id = unique_by_id(samples, "public/clean_peaks.h5")
    ground_truth = unique_by_id(
        read_jsonl(ROOT / "private" / "clean_ground_truth.jsonl"),
        "private/clean_ground_truth.jsonl",
    )
    details_payload = json.loads(
        (REPORT_DIR / "acom_clean_details.json").read_text(
            encoding="utf-8"
        )
    )
    details = unique_by_id(
        details_payload["samples"],
        "reports/v3/acom_clean_details.json",
    )
    diagnostics = load_diagnostic_reflections()
    all_ids = [sample["sample_id"] for sample in samples]
    if not (set(all_ids) == set(ground_truth) == set(details) == set(diagnostics)):
        raise ValueError(
            "Public, ground-truth, ACOM-detail, and reflection IDs do not match"
        )

    if args.sample_id:
        unknown = sorted(set(args.sample_id) - set(all_ids))
        if unknown:
            raise ValueError(f"Unknown sample IDs: {unknown}")
        selected_ids = list(dict.fromkeys(args.sample_id))
    elif args.all:
        selected_ids = all_ids
    else:
        selected_ids = representative_ids(list(details.values()))

    output_path = args.output or (
        ROOT
        / "diagnostics"
        / (
            "clean_coordinate_trace.jsonl.gz"
            if args.all
            else "clean_coordinate_trace_selected.jsonl.gz"
        )
    )
    summary_output_path = args.summary_output or (
        REPORT_DIR
        / (
            "ACOM_COORDINATE_ANALYSIS.md"
            if args.all
            else "ACOM_COORDINATE_ANALYSIS_SELECTED.md"
        )
    )
    tolerance = float(config["evaluation"]["coordinate_match_tolerance_Ainv"])
    traces: list[dict] = []
    with open_text_output(output_path) as output:
        for index, sample_id in enumerate(selected_ids):
            predicted_peaks = filtered_simulated_peaks(
                crystal,
                np.asarray(
                    details[sample_id][
                        "predicted_orientation_matrix_sample_to_crystal"
                    ],
                    dtype=float,
                ),
                config,
            )
            trace = build_sample_trace(
                samples_by_id[sample_id],
                ground_truth[sample_id],
                details[sample_id],
                diagnostics[sample_id],
                predicted_peaks,
                structure,
                tolerance,
                config,
            )
            traces.append(trace)
            output.write(json.dumps(trace, ensure_ascii=False) + "\n")
            if args.stdout:
                print(json.dumps(trace, ensure_ascii=False, indent=2))
            if (index + 1) % 50 == 0 or index + 1 == len(selected_ids):
                print(
                    f"Traced {index + 1}/{len(selected_ids)} samples",
                    file=sys.stderr,
                )

    write_summary(summary_output_path, traces, config, output_path)
    print(f"Full coordinate trace: {output_path}", file=sys.stderr)
    print(f"Coordinate analysis: {summary_output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
