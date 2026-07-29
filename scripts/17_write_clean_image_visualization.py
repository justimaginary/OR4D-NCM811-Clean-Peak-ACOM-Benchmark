#!/usr/bin/env python3
"""Write a self-contained, offline visualization of the full Clean image benchmark."""
from __future__ import annotations

import base64
import gzip
import io
import json
from pathlib import Path

import h5py
import numpy as np
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports" / "CLEAN_IMAGE_ACOM_VISUALIZATION.html"
IMAGE_FILE = ROOT / "public" / "clean_images.h5"
COUNTED_FILE = ROOT / "public" / "clean_counted_images.h5"
ORACLE_FILE = ROOT / "private" / "clean_physical_oracle_reflections.h5"
GT_FILE = ROOT / "private" / "clean_ground_truth.jsonl"
TRACE_FILE = ROOT / "diagnostics" / "clean_coordinate_trace.jsonl.gz"
CONFIG_FILE = ROOT / "config" / "benchmark.yaml"
DETAILS_ORACLE = ROOT / "reports" / "acom_clean_details_physical_oracle.json"
DETAILS_V3 = ROOT / "reports" / "acom_clean_details.json"
EVALUATION_V3 = ROOT / "reports" / "acom_clean_evaluation.json"
PEAK_REPORT_E = ROOT / "reports" / "clean_image_pipeline_evaluation.json"
PEAK_REPORT_C = ROOT / "reports" / "clean_counted_pipeline_evaluation.json"
ACOM_REPORT_E = ROOT / "reports" / "clean_acom_comparison.json"
ACOM_REPORT_C = ROOT / "reports" / "clean_counted_acom_comparison.json"
DOSES = (10_000, 100_000, 1_000_000)
DETECTORS = ("autodisk", "py4dstem")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def rounded(value, digits: int = 6):
    if isinstance(value, (float, np.floating)):
        return round(float(value), digits)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, np.ndarray):
        return rounded(value.tolist(), digits)
    if isinstance(value, list):
        return [rounded(item, digits) for item in value]
    if isinstance(value, dict):
        return {key: rounded(item, digits) for key, item in value.items()}
    return value


def row_map(path: Path, wanted: set[str]) -> dict[str, dict]:
    rows = load_json(path)["samples"]
    return {row["sample_id"]: row for row in rows if row["sample_id"] in wanted}


def representative_samples() -> list[tuple[str, str]]:
    rows = [
        row
        for row in load_json(DETAILS_ORACLE)["samples"]
        if row["sample_role"] == "headline_core"
    ]
    rows.sort(key=lambda row: row["friedel_equivalent_misorientation_deg"])
    positions = (0, len(rows) // 2, int(0.95 * (len(rows) - 1)), len(rows) - 1)
    chosen = [
        ("Best / 最佳", rows[positions[0]]["sample_id"]),
        ("Median / 中位", rows[positions[1]]["sample_id"]),
        ("P95 / 第95百分位", rows[positions[2]]["sample_id"]),
        ("Worst / 最差", rows[positions[3]]["sample_id"]),
    ]
    failure = "clean_core_0744"
    if failure not in {sample_id for _, sample_id in chosen}:
        chosen.append(("AutoDisk failure / 新增失败", failure))
    return [(sample_id, label) for label, sample_id in chosen]


def read_ragged(path: Path, sample_indices: dict[str, int]) -> dict[str, list[dict]]:
    result: dict[str, list[dict]] = {}
    with h5py.File(path, "r") as handle:
        offsets = handle["peaks/offsets"][()]
        qx = handle["peaks/qx"][()]
        qy = handle["peaks/qy"][()]
        intensity = handle["peaks/intensity"][()]
        for sample_id, index in sample_indices.items():
            start, stop = int(offsets[index]), int(offsets[index + 1])
            result[sample_id] = [
                {"qx": float(x), "qy": float(y), "intensity": float(i)}
                for x, y, i in zip(qx[start:stop], qy[start:stop], intensity[start:stop])
            ]
    return result


def read_oracle(sample_indices: dict[str, int]) -> dict[str, list[dict]]:
    result: dict[str, list[dict]] = {}
    with h5py.File(ORACLE_FILE, "r") as handle:
        offsets = handle["reflections/offsets"][()]
        for sample_id, index in sample_indices.items():
            start, stop = int(offsets[index]), int(offsets[index + 1])
            result[sample_id] = [
                {
                    "hkl": [int(v) for v in hkl],
                    "qx": float(qx),
                    "qy": float(qy),
                    "intensity": float(intensity),
                }
                for hkl, qx, qy, intensity in zip(
                    handle["reflections/hkl"][start:stop],
                    handle["reflections/qx_Ainv"][start:stop],
                    handle["reflections/qy_Ainv"][start:stop],
                    handle["reflections/intensity_normalized"][start:stop],
                )
            ]
    return result


def image_data_url(array: np.ndarray) -> str:
    values = np.asarray(array, dtype=np.float64)
    positive_values = values[values > 0]
    reference = (
        float(np.percentile(positive_values, 50))
        if positive_values.size
        else 1.0
    )
    scaled = np.log1p(values / max(reference, 1e-12))
    positive_scaled = scaled[scaled > 0]
    ceiling = (
        float(np.percentile(positive_scaled, 99.5))
        if positive_scaled.size
        else 1.0
    )
    pixels = np.clip(scaled / max(ceiling, 1e-12), 0, 1) ** 0.65
    # A restrained blue-white map keeps weak disks visible without obscuring overlays.
    rgb = np.empty((*pixels.shape, 3), dtype=np.uint8)
    rgb[..., 0] = (8 + 235 * pixels).astype(np.uint8)
    rgb[..., 1] = (18 + 225 * pixels).astype(np.uint8)
    rgb[..., 2] = (35 + 205 * pixels).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(buffer, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def match_metrics(oracle: list[dict], detected: list[dict], q_per_px: float) -> dict:
    candidates = sorted(
        (
            ((o["qx"] - d["qx"]) ** 2 + (o["qy"] - d["qy"]) ** 2, oi, di)
            for oi, o in enumerate(oracle)
            for di, d in enumerate(detected)
        ),
        key=lambda item: item[0],
    )
    used_o: set[int] = set()
    used_d: set[int] = set()
    distances = []
    matches = []
    for distance2, oi, di in candidates:
        distance_px = distance2**0.5 / q_per_px
        if distance_px > 1.0:
            break
        if oi not in used_o and di not in used_d:
            used_o.add(oi)
            used_d.add(di)
            distances.append(distance_px)
            matches.append(
                {
                    "oracle_index": oi,
                    "detected_index": di,
                    "distance_px": distance_px,
                }
            )
    tp = len(distances)
    return {
        "oracle": len(oracle),
        "detected": len(detected),
        "tp": tp,
        "precision": tp / len(detected) if detected else 0.0,
        "recall": tp / len(oracle) if oracle else 0.0,
        "rmse_px": float(np.sqrt(np.mean(np.square(distances)))) if distances else None,
        "p95_px": float(np.percentile(distances, 95)) if distances else None,
        "matches": matches,
        "false_negative_indices": sorted(set(range(len(oracle))) - used_o),
        "false_positive_indices": sorted(set(range(len(detected))) - used_d),
    }


def reciprocal_matrix() -> list[list[float]]:
    with gzip.open(TRACE_FILE, "rt", encoding="utf-8") as handle:
        row = json.loads(next(handle))
    return rounded(row["reciprocal_lattice_matrix_B_Ainv"])


def read_v3_traces(wanted: set[str]) -> dict[str, dict]:
    """Load the detailed direct-peak coordinate chain retained by v3."""
    result: dict[str, dict] = {}
    with gzip.open(TRACE_FILE, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            sample_id = row["sample_id"]
            if sample_id not in wanted:
                continue
            result[sample_id] = {
                "direct_lattice_matrix_A": row["direct_lattice_matrix_A"],
                "reciprocal_lattice_matrix_B": row[
                    "reciprocal_lattice_matrix_B_Ainv"
                ],
                "gt_matrix": row[
                    "standard_orientation_matrix_sample_to_crystal"
                ],
                "acom_matrix": row[
                    "acom_orientation_matrix_sample_to_crystal"
                ],
                "gt_axes": row["standard_sample_axes_in_crystal_cartesian"],
                "acom_axes": row["acom_sample_axes_in_crystal_cartesian"],
                "orientation_error_deg": row["acom_result"][
                    "friedel_equivalent_misorientation_deg"
                ],
                "observed_match_fraction": row["comparison_summary"][
                    "observed_match_fraction"
                ],
                "q_rmse_Ainv": row["comparison_summary"][
                    "q_distance_rmse_Ainv"
                ],
                "observed": [
                    {
                        "hkl": reflection["hkl"],
                        "intensity": reflection[
                            "reported_intensity_normalized"
                        ],
                        "q": [
                            reflection["reported_qx_Ainv"],
                            reflection["reported_qy_Ainv"],
                        ],
                        "g_crystal": reflection[
                            "g_crystal_cartesian_Ainv"
                        ],
                        "g_sample_gt": reflection[
                            "standard_g_sample_Ainv"
                        ],
                        "g_sample_v3_acom_same_hkl": reflection[
                            "acom_same_hkl_g_sample_Ainv"
                        ],
                    }
                    for reflection in row["standard_observed_reflections"]
                ],
                "predicted": [
                    {
                        "hkl": reflection["hkl"],
                        "intensity": reflection[
                            "reported_by_py4DSTEM_intensity_normalized"
                        ],
                        "q": [
                            reflection["reported_by_py4DSTEM_qx_Ainv"],
                            reflection["reported_by_py4DSTEM_qy_Ainv"],
                        ],
                        "g_crystal": reflection[
                            "g_crystal_cartesian_Ainv"
                        ],
                        "g_sample": reflection["g_sample_Ainv"],
                    }
                    for reflection in row["acom_simulated_reflections"]
                ],
                "matches": [
                    {
                        "observed_index": int(match["observed_index"]),
                        "predicted_index": int(match["predicted_index"]),
                    }
                    for match in row["detector_q_assignment"]["matches"]
                ],
            }
            if len(result) == len(wanted):
                break
    missing = wanted - set(result)
    if missing:
        raise ValueError(f"v3 coordinate trace is missing: {sorted(missing)}")
    return result


def parameter_tables() -> dict[str, list[list[str]]]:
    config = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
    common, clean, image, acom = (
        config["common"],
        config["clean"],
        config["clean_image"],
        config["acom"],
    )
    v3 = [
        ["公开输入 / Public input", "input_type", "PointList: qx, qy, intensity"],
        ["输入来源 / Input source", "forward_model", "py4DSTEM kinematical diffraction pattern"],
        ["最大倒空间半径 / Kmax", "k_max_Ainv", f"{common['k_max_Ainv']} Å⁻¹"],
        ["中心束排除 / Central exclusion", "central_beam_exclusion_Ainv", f"{common['central_beam_exclusion_Ainv']} Å⁻¹"],
        ["加速电压 / Voltage", "accelerating_voltage_V", f"{common['accelerating_voltage_V']/1000:g} kV"],
        ["激发误差宽度 / Excitation σ", "sigma_excitation_error_Ainv", f"{clean['sigma_excitation_error_Ainv']} Å⁻¹"],
        ["激发误差截断 / Excitation cutoff", "tol_excitation_error_mult", f"{clean['tol_excitation_error_mult']} σ"],
        ["结构因子阈值 / Structure-factor tol.", "tol_structure_factor", str(clean["tol_structure_factor"])],
        ["峰强度阈值 / Intensity tol.", "tol_intensity", str(clean["tol_intensity"])],
        ["强度归一化 / Normalization", "normalize_peak_intensity", str(common["normalize_peak_intensity"])],
    ]
    image_rows = [
        ["公开输入 / Public input", "input_type", "512 × 512 diffraction image"],
        ["图像模型 / Image model", "forward_model", str(image["forward_model"])],
        ["图像倒空间范围 / Image q range", "q_max_Ainv", f"±{image['q_max_Ainv']} Å⁻¹"],
        ["ACOM 截止 / ACOM Kmax", "k_max_Ainv", f"{common['k_max_Ainv']} Å⁻¹"],
        ["加速电压 / Voltage", "voltage_kV", f"{common['accelerating_voltage_V']/1000:g} kV"],
        ["会聚半角 / Semiangle", "convergence_semiangle_mrad", f"{image['convergence_semiangle_mrad']} mrad"],
        ["样品厚度 / Thickness", "thickness_nm", f"{image['thickness_nm']} nm"],
        ["过采样 / Oversampling", "oversampling", str(image["oversampling"])],
        ["孔径软边 / Aperture soft edge", "aperture_soft_edge_fraction", str(image["aperture_soft_edge_fraction"])],
        ["中心束比例 / Direct beam", "canonical_direct_beam_fraction", str(image["canonical_direct_beam_fraction"])],
        ["计数模型 / Counting", "counting.model", str(image["counting"]["model"])],
        ["剂量 / Doses", "doses_electrons", ", ".join(str(value) for value in image["counting"]["doses_electrons"])],
        ["随机重复 / Repeats", "counting.repeats", str(image["counting"]["repeats"])],
        ["噪声/背景/畸变 / Noise/background/distortion", "detector", "disabled / disabled / disabled"],
    ]
    acom_rows = [
        ["搜索范围 / Zone-axis range", "zone_axis_range", str(acom["zone_axis_range"])],
        ["晶带轴角步长 / Zone-axis step", "angle_step_zone_axis_deg", f"{acom['angle_step_zone_axis_deg']}°"],
        ["面内角步长 / In-plane step", "angle_step_in_plane_deg", f"{acom['angle_step_in_plane_deg']}°"],
        ["对照角步长 / Sweep", "sweep_angle_steps_deg", ", ".join(f"{value}°" for value in acom["sweep_angle_steps_deg"])],
        ["相关核半径 / Correlation kernel", "corr_kernel_size_Ainv", f"{acom['corr_kernel_size_Ainv']} Å⁻¹"],
        ["ACOM 激发误差 σ", "sigma_excitation_error_Ainv", f"{acom['sigma_excitation_error_Ainv']} Å⁻¹"],
        ["径向权重 / Radial power", "power_radial", str(acom["power_radial"])],
        ["模拟强度权重 / Sim. intensity power", "power_intensity_simulated", str(acom["power_intensity_simulated"])],
        ["输入强度权重 / Input intensity power", "power_intensity_experiment", str(acom["power_intensity_experiment"])],
        ["峰距离容差 / Distance tol.", "tol_distance_Ainv", f"{acom['tol_distance_Ainv']} Å⁻¹"],
        ["最少峰数 / Minimum peaks", "min_number_peaks", str(acom["min_number_peaks"])],
        ["反演/Friedel 对称 / Inversion symmetry", "inversion_symmetry", str(acom["inversion_symmetry"])],
        ["返回候选数 / Matches returned", "num_matches_return", str(acom["num_matches_return"])],
        ["GPU", "cuda", str(acom["cuda"])],
    ]
    return {"v3": v3, "image": image_rows, "acom": acom_rows}


def details_path(track: str, dose: int | None, repeat: int, detector: str) -> Path:
    if track == "expectation":
        return ROOT / "reports" / f"acom_clean_details_expectation_{detector}.json"
    return (
        ROOT
        / "reports"
        / f"acom_clean_details_counted_dose{dose}_repeat{repeat}_{detector}.json"
    )


def peaks_path(track: str, dose: int | None, repeat: int, detector: str) -> Path:
    if track == "expectation":
        return ROOT / "diagnostics" / f"clean_expectation_{detector}_peaks.h5"
    return (
        ROOT
        / "diagnostics"
        / f"clean_counted_dose{dose}_repeat{repeat}_{detector}_peaks.h5"
    )


def comparison_index(report: dict) -> dict[str, dict]:
    return {candidate["label"]: candidate for candidate in report["candidates"]}


def build_data() -> dict:
    representatives = representative_samples()
    wanted = {sample_id for sample_id, _ in representatives}
    v3_traces = read_v3_traces(wanted)
    gt_rows = {
        row["sample_id"]: row
        for row in (
            json.loads(line)
            for line in GT_FILE.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        if row["sample_id"] in wanted
    }
    oracle_rows = row_map(DETAILS_ORACLE, wanted)
    with h5py.File(IMAGE_FILE, "r") as handle:
        ids = [
            value.decode() if isinstance(value, bytes) else str(value)
            for value in handle["sample_id"][()]
        ]
        sample_indices = {sample_id: ids.index(sample_id) for sample_id in wanted}
        expectation_images = {
            sample_id: image_data_url(handle["expectation/intensity"][index])
            for sample_id, index in sample_indices.items()
        }
        image_attrs = {key: value for key, value in handle.attrs.items()}
        qx = handle["detector/qx_Ainv"][()]
        qy = handle["detector/qy_Ainv"][()]
        q_per_px = float(abs(qx[1] - qx[0]))
        q_bounds = [float(qx[0]), float(qx[-1]), float(qy[-1]), float(qy[0])]
    oracle_peaks = read_oracle(sample_indices)
    acom_e = load_json(ACOM_REPORT_E)
    acom_c = load_json(ACOM_REPORT_C)
    comparisons = {**comparison_index(acom_e), **comparison_index(acom_c)}

    variants: dict[str, dict] = {}
    variant_specs = [
        ("expectation", None, 0, detector) for detector in DETECTORS
    ] + [
        ("counted", dose, repeat, detector)
        for dose in DOSES
        for repeat in range(5)
        for detector in DETECTORS
    ]
    for track, dose, repeat, detector in variant_specs:
        label = (
            f"{detector}_expectation"
            if track == "expectation"
            else f"counted_dose{dose}_repeat{repeat}_{detector}"
        )
        peaks = read_ragged(peaks_path(track, dose, repeat, detector), sample_indices)
        detail_rows = row_map(details_path(track, dose, repeat, detector), wanted)
        comparison = comparisons[label]
        deltas = {
            row["sample_id"]: row
            for row in comparison["per_sample"]
            if row["sample_id"] in wanted
        }
        variants[label] = {
            "track": track,
            "dose": dose,
            "repeat": repeat,
            "detector": detector,
            "peaks": peaks,
            "details": {
                sample_id: {
                    "matrix": row["predicted_orientation_matrix_sample_to_crystal"],
                    "gt_error_deg": row["friedel_equivalent_misorientation_deg"],
                    "correlation": row["correlation_score"],
                    "num_peaks": row["num_peaks"],
                    "delta_oracle_deg": deltas[sample_id][
                        "delta_from_physical_oracle_acom_deg"
                    ],
                }
                for sample_id, row in detail_rows.items()
            },
            "sample_peak_metrics": {
                sample_id: match_metrics(oracle_peaks[sample_id], peaks[sample_id], q_per_px)
                for sample_id in wanted
            },
        }

    counted_images: dict[str, dict[str, str]] = {sample_id: {} for sample_id in wanted}
    seeds: dict[str, dict[str, int]] = {sample_id: {} for sample_id in wanted}
    with h5py.File(COUNTED_FILE, "r") as handle:
        for sample_id, index in sample_indices.items():
            for dose_index, dose in enumerate(DOSES):
                for repeat in range(5):
                    key = f"{dose}:{repeat}"
                    counted_images[sample_id][key] = image_data_url(
                        handle["images/counts"][index, dose_index, repeat]
                    )
                    seeds[sample_id][key] = int(
                        handle["rng_seed"][index, dose_index, repeat]
                    )

    parameter_config = json.loads(str(image_attrs["generator_config"]))
    parameters = [
        ["输入尺寸 / Image size", "gpts", "512 × 512 px"],
        ["倒空间范围 / Reciprocal-space extent", "q_max_Ainv", f"±{parameter_config['q_max_Ainv']} Å⁻¹"],
        ["ACOM 截止 / ACOM Kmax", "k_max_Ainv", "1.5 Å⁻¹"],
        ["加速电压 / Accelerating voltage", "voltage_kV", f"{image_attrs['voltage_kV']} kV"],
        ["会聚半角 / Convergence semiangle", "semiangle_mrad", f"{image_attrs['semiangle_mrad']} mrad"],
        ["样品厚度 / Thickness", "thickness_nm", f"{image_attrs['thickness_nm']} nm"],
        ["中心束比例 / Direct-beam fraction", "direct_beam_fraction", str(image_attrs["direct_beam_fraction"])],
        ["盘半径 / Disk radius", "disk_radius_px", f"{image_attrs['disk_radius_px']:.3f} px"],
        ["期望图归一化 / Expectation normalization", "normalization", str(image_attrs["normalization"])],
        ["计数模型 / Counting model", "counting_model", "multinomial_fixed_total"],
        ["剂量 / Electron doses", "doses_electrons", "10⁴, 10⁵, 10⁶ e⁻/pattern"],
        ["随机重复 / Repeats", "repeats", "5 per dose"],
        ["ACOM 角步长 / Angular step", "angle_step_deg", "2° zone + 2° in-plane"],
        ["检峰匹配容差 / Peak match tolerance", "match_tolerance_px", "1.0 px"],
    ]

    samples = {}
    for sample_id, label in representatives:
        oracle = oracle_rows[sample_id]
        samples[sample_id] = {
            "label": label,
            "index": sample_indices[sample_id],
            "gt_matrix": gt_rows[sample_id]["orientation_matrix_sample_to_crystal"],
            "oracle_matrix": oracle["predicted_orientation_matrix_sample_to_crystal"],
            "oracle_gt_error_deg": oracle["friedel_equivalent_misorientation_deg"],
            "oracle_correlation": oracle["correlation_score"],
            "oracle_peaks": oracle_peaks[sample_id],
            "expectation_image": expectation_images[sample_id],
            "counted_images": counted_images[sample_id],
            "seeds": seeds[sample_id],
            "v3": v3_traces[sample_id],
        }

    peak_e = load_json(PEAK_REPORT_E)
    peak_c = load_json(PEAK_REPORT_C)
    v3_sweep = []
    for angle, filename in (
        (4, "acom_clean_evaluation_angle_4deg.json"),
        (3, "acom_clean_evaluation_angle_3deg.json"),
        (2, "acom_clean_evaluation.json"),
    ):
        metrics = load_json(ROOT / "reports" / filename)["metrics"]
        v3_sweep.append(
            {
                "angle_step_deg": angle,
                "median_deg": metrics["median_misorientation_deg"],
                "p95_deg": metrics["p95_misorientation_deg"],
                "acc_at_2deg": metrics["accuracy_within_2deg"],
                "acc_at_5deg": metrics["accuracy_within_5deg"],
            }
        )
    return rounded(
        {
            "generated_from": {
                "patterns": 1081,
                "headline_core": 1024,
                "counted_images": 1081 * 3 * 5,
                "acom_paths": 33,
                "note": "All aggregate values come from completed full-run JSON reports.",
            },
            "q_bounds": q_bounds,
            "q_per_px": q_per_px,
            "disk_radius_px": image_attrs["disk_radius_px"],
            "reciprocal_matrix_B": reciprocal_matrix(),
            "samples": samples,
            "sample_order": [sample_id for sample_id, _ in representatives],
            "variants": variants,
            "parameters": parameters,
            "parameter_tables": parameter_tables(),
            "v3_sweep": v3_sweep,
            "v3_metrics": load_json(EVALUATION_V3)["metrics"],
            "peak_expectation": peak_e["detectors"],
            "peak_dose_summary": peak_c["dose_summary"],
            "acom_oracle": acom_e["physical_oracle_versus_ground_truth"],
            "acom_expectation": [
                {
                    "label": row["label"],
                    "versus_ground_truth": row["versus_ground_truth"],
                    "versus_oracle": row["versus_physical_oracle_acom"],
                    "new_failures": row["new_gt_5deg_failures"],
                }
                for row in acom_e["candidates"]
            ],
            "acom_dose_summary": acom_c["dose_summary"],
        }
    )


HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Clean Image → Peak Detection → ACOM</title>
<style>
:root{--ink:#172033;--muted:#697386;--line:#dce2ea;--paper:#fff;--wash:#f5f7fa;--blue:#246bce;--orange:#e56b3f;--green:#16866b;--purple:#7251b5}
*{box-sizing:border-box} body{margin:0;background:#fff;color:var(--ink);font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC",sans-serif}
main{max-width:1440px;margin:auto;padding:34px 42px 70px} h1{font-size:30px;margin:0 0 6px} h2{font-size:21px;margin:36px 0 14px} h3{font-size:16px;margin:0 0 9px}
.lead{color:var(--muted);margin:0}.stamp{display:inline-block;margin-top:14px;padding:5px 10px;border-radius:99px;background:#e9f6f1;color:#126850;font-weight:650}
.flow{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:24px 0}.flow div{padding:14px;border:1px solid var(--line);border-radius:10px;background:var(--wash);font-weight:650}.flow b{color:var(--blue)}
.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.card,.panel{border:1px solid var(--line);border-radius:12px;background:var(--paper);padding:16px}.value{font-size:26px;font-weight:760}.unit{font-size:13px;color:var(--muted)}
.summary-grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}.chart{width:100%;height:280px}.chart text{font-size:11px;fill:var(--muted)}.chart .grid{stroke:#e5e9ef}.chart .axis{stroke:#9ca7b7}.chart .auto{stroke:var(--orange);fill:none;stroke-width:2.4}.chart .py{stroke:var(--blue);fill:none;stroke-width:2.4}.chart circle{stroke-width:2;fill:#fff}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{text-align:left;padding:8px 9px;border-bottom:1px solid var(--line);vertical-align:top}th{color:#4f596b;background:#fafbfc;position:sticky;top:0}
.verdict{border-left:4px solid var(--green);background:#f1faf7;padding:13px 15px;margin-top:14px}.warn{border-left-color:#dc8b1d;background:#fff8e9}
.toolbar{display:flex;flex-wrap:wrap;gap:10px;align-items:end;padding:14px;background:var(--wash);border-radius:12px}.control label{display:block;font-size:11px;color:var(--muted);margin-bottom:3px}.control select,.seg button{height:36px;border:1px solid #cbd3df;background:#fff;border-radius:7px;padding:0 10px;color:var(--ink)}.seg{display:flex;gap:5px}.seg button.active{background:var(--ink);color:#fff;border-color:var(--ink)}
.diag{display:grid;grid-template-columns:minmax(480px,1.1fr) minmax(410px,.9fr);gap:16px;margin-top:14px}.image-panel{position:relative;background:#081223;border-radius:10px;overflow:hidden;aspect-ratio:1}.image-panel img,.image-panel svg{position:absolute;inset:0;width:100%;height:100%}.image-panel img{image-rendering:auto}.image-panel svg{overflow:visible}.legend{display:flex;gap:18px;flex-wrap:wrap;font-size:13px;margin:9px 0}.dot{display:inline-block;width:11px;height:11px;border-radius:50%;border:2px solid var(--blue);margin-right:5px}.cross{color:var(--orange);font-size:18px;vertical-align:-1px}.matrix-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:8px}.matrix{background:var(--wash);border-radius:8px;padding:10px}.matrix>b,.matrix>.mini{display:block}.matrix pre{font:11px/1.55 ui-monospace,SFMono-Regular,Menlo,monospace;margin:6px 0;white-space:pre-wrap}.metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin:11px 0}.metric{border:1px solid var(--line);border-radius:8px;padding:9px}.metric b{display:block;font-size:19px}.metric span{font-size:11px;color:var(--muted)}
.path-compare{display:grid;grid-template-columns:1fr 1fr;gap:14px}.path{border:1px solid var(--line);border-radius:12px;padding:16px}.path.v3{border-top:4px solid var(--purple)}.path.image{border-top:4px solid var(--green)}.path code{display:block;margin:10px 0;padding:10px;background:var(--wash);border-radius:7px;white-space:normal}.equation-grid{display:grid;grid-template-columns:.7fr 1.2fr 1fr;gap:10px}.transform-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.definitions{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.definition{border:1px solid var(--line);border-radius:12px;padding:16px}.definition.v3{border-top:4px solid var(--purple)}.definition.e{border-top:4px solid var(--blue)}.definition.c{border-top:4px solid var(--orange)}.definition h3{font-size:18px}.definition .tag{display:inline-block;background:var(--wash);border-radius:99px;padding:3px 8px;font-size:12px;margin-bottom:8px}.formation{display:grid;grid-template-columns:repeat(6,1fr);gap:8px}.formation>div{position:relative;border:1px solid var(--line);border-radius:9px;padding:11px;background:var(--wash);min-height:112px}.formation>div:not(:last-child):after{content:"→";position:absolute;right:-10px;top:43%;z-index:2;color:var(--blue);font-weight:800}.formation b{display:block;color:var(--blue);margin-bottom:5px}.detector-compare{display:grid;grid-template-columns:1fr 1fr;gap:14px}.compare-image{position:relative;aspect-ratio:1;background:#081223;border-radius:10px;overflow:hidden}.compare-image img,.compare-image svg{position:absolute;inset:0;width:100%;height:100%}.compare-metric{margin-top:8px;font-size:13px}.v3-diagnostic{display:grid;grid-template-columns:minmax(500px,1.15fr) minmax(360px,.85fr);gap:14px;margin-top:14px}.v3-plot{width:100%;background:#fafbfc;border:1px solid var(--line);border-radius:10px}.v3-plot text{font-size:10px;fill:#5e6879}
.input-examples{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}.input-example{border:1px solid var(--line);border-radius:12px;padding:12px}.input-example img{display:block;width:100%;aspect-ratio:1;background:#081223;border-radius:9px}.input-example h3{margin:9px 0 3px}.input-example p{margin:0;color:var(--muted);font-size:12px}
.case-description{margin-top:10px;padding:11px 13px;border-left:4px solid var(--purple);background:#f7f4fc;border-radius:0 8px 8px 0}.control-help{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-top:10px}.control-help div{padding:10px;border:1px solid var(--line);border-radius:8px;background:#fff}.control-help b{display:block;margin-bottom:3px}.control-help span{font-size:12px;color:var(--muted)}.input-schema{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:12px}.input-schema section{border:1px solid var(--line);border-radius:9px;padding:13px}.input-schema h3{margin-bottom:6px}.input-schema code{font-size:12px}.schema-note{padding:10px 12px;background:var(--wash);border-radius:8px;margin-top:10px}
.peak-table-wrap{max-height:360px;overflow:auto}.formula{font:14px/1.7 ui-monospace,SFMono-Regular,Menlo,monospace;background:#f6f8fb;border-radius:8px;padding:12px}.mini{font-size:12px;color:var(--muted)}details{border:1px solid var(--line);border-radius:10px;padding:11px 13px;margin-top:10px}summary{cursor:pointer;font-weight:670}.hidden{display:none!important}
@media(max-width:900px){main{padding:24px 16px}.cards,.flow,.definitions,.input-examples,.control-help{grid-template-columns:1fr 1fr}.formation{grid-template-columns:1fr 1fr}.formation>div:after{display:none}.summary-grid,.diag,.path-compare,.equation-grid,.transform-grid,.detector-compare,.v3-diagnostic,.input-schema{grid-template-columns:1fr}.matrix-grid{grid-template-columns:1fr}.image-panel{min-width:0}}
</style>
</head>
<body><main>
<header>
  <h1>Clean 图像 Benchmark：衍射图 → 检峰 → ACOM</h1>
  <p class="lead">Full Clean image benchmark · expectation + counted electrons · AutoDisk vs py4DSTEM find_Bragg_disks</p>
  <span class="stamp">全量实际运行结果 / completed full run</span>
</header>
<div class="flow"><div><b>1</b> 运动学 CBED<br><span class="mini">physical expectation</span></div><div><b>2</b> 电子计数<br><span class="mini">10⁴ / 10⁵ / 10⁶ e⁻</span></div><div><b>3</b> 衍射盘检测<br><span class="mini">AutoDisk / find_Bragg_disks</span></div><div><b>4</b> ACOM + GT 评测<br><span class="mini">same 2° orientation plan</span></div></div>

<h2>输入类型与评测层级 / Input tracks and evaluation levels</h2>
<div class="definitions">
 <section class="definition v3"><span class="tag">历史基线 / frozen baseline</span><h3>v3：直接峰输入</h3><p><b>输入不是衍射图。</b>每个样本直接向 ACOM 提供连续浮点峰列表 <code>(qx, qy, intensity)</code>。该路径假设峰中心和强度已经获得，用于测量 ACOM 取向恢复能力。</p><p class="mini">评测目标：给定理想峰列表时的 ACOM 基线性能。</p></section>
 <section class="definition e"><span class="tag">E = Expectation / 期望强度</span><h3>Clean-E：无随机噪声衍射图</h3><p>由同一 CIF 和取向生成的 <b>512×512 浮点期望强度图 P(q)</b>。图像包含中心盘和有限尺寸衍射盘，不包含电子计数涨落、背景、读出噪声、椭圆畸变或饱和。</p><p class="mini">评测目标：理想二维图像形成与自动检峰引入的性能变化。</p></section>
 <section class="definition c"><span class="tag">C = Counted / 电子计数</span><h3>Clean-C：有限剂量计数图</h3><p>从同一张 Clean-E 概率图逐电子抽样得到的 <b>uint32 整数计数图</b>。每张图严格包含 10⁴、10⁵ 或 10⁶ 个电子，每个剂量使用 5 个独立随机重复。</p><p class="mini">评测目标：有限电子剂量对检峰和 ACOM 的影响。</p></section>
</div>
<div class="verdict warn"><b>关系：</b>Clean-C 不是另一种散射模型，而是 Clean-E 的有限电子观测；v3 则根本没有图像输入。三者必须分开比较，才能区分 ACOM 误差、图像检峰误差和剂量噪声。</div>

<h2>二维衍射图生成 / 2D diffraction image formation</h2>
<section class="panel">
 <div class="formation">
  <div><b>① CIF + R</b>NCM811 晶体结构与样品→晶体取向矩阵。</div>
  <div><b>② 运动学反射</b>py4DSTEM 生成与 ACOM 模板一致的 HKL、(qx,qy) 和积分强度 I<sub>g</sub>。</div>
  <div><b>③ 有限衍射盘</b>在每个亚像素中心放置软边圆孔振幅；盘半径由 α/λ 决定。</div>
  <div><b>④ 相干成像</b>4× 过采样，累加 √I<sub>g</sub>A(q−g)e<sup>−iχ</sup>，再取模平方并做像素积分。</div>
  <div><b>⑤ Clean-E</b>中心盘与散射盘分别归一化后按 0.90/0.10 合成概率图 P(q)。</div>
  <div><b>⑥ Clean-C</b>counts ∼ Multinomial(N<sub>e</sub>, P)，然后交给两个检峰器。</div>
 </div>
 <div class="formula" style="margin-top:12px">disk radius = α / λ = 4.063 px<br>Ψ<sub>scat</sub>(q) = Σ<sub>g</sub> √I<sub>g</sub> A(q−g) exp[−iχ(q−g)]<br>P<sub>E</sub>(q) = 0.90 P<sub>direct</sub>(q) + 0.10 P<sub>scattered</sub>(q)<br>Clean-C: n(q) ∼ Multinomial(N<sub>e</sub>, P<sub>E</sub>), &nbsp; Σ<sub>q</sub>n(q)=N<sub>e</sub></div>
 <p class="mini"><b>当前 canonical 模型的边界：</b>反射支持和积分强度来自与 ACOM 匹配的 py4DSTEM 运动学模型；图像形成使用有限会聚盘，而不是高斯点。它不是 multislice，也不声称模拟 dynamical scattering。有限厚度 First-Born+sinc 模型只保留为诊断模式，没有混入本页 canonical 结果。</p>
</section>

<h2>二维衍射图输入接口 / Diffraction-image input interface</h2>
<section class="panel">
 <p>检峰器接收的不是 PNG、网页截图或峰坐标表，而是下面两个 <b>HDF5 文件中的二维数值数组</b>。对样本索引 <code>i</code>，图像像素 <code>[row, col]</code> 的倒空间坐标为 <code>(detector/qx_Ainv[col], detector/qy_Ainv[row])</code>，单位 Å⁻¹；<code>sample_id[i]</code> 给出该图对应的样本编号。</p>
 <div class="input-schema">
  <section><h3>Clean-E：<code>public/clean_images.h5</code></h3><p>实际检峰输入为 <code>expectation/intensity[i, :, :]</code>，整体形状 <code>[1081, 512, 512]</code>，dtype 为 <code>float32</code>。每张图是归一化概率/期望强度，满足 <code>image.sum() = 1</code>。</p></section>
  <section><h3>Clean-C：<code>public/clean_counted_images.h5</code></h3><p>实际检峰输入为 <code>images/counts[i, d, r, :, :]</code>，整体形状 <code>[1081, 3, 5, 512, 512]</code>，dtype 为 <code>uint32</code>。其中 <code>d</code> 选择 10⁴/10⁵/10⁶ e⁻，<code>r</code> 选择 5 次独立抽样；每张图的像素和严格等于所选剂量。</p></section>
 </div>
 <div class="schema-note"><b>两个文件共用的标定数据：</b><code>sample_id [1081]</code>、<code>detector/qx_Ainv [512]</code>、<code>detector/qy_Ainv [512]</code>、<code>detector/vacuum_probe [512,512]</code> 和 <code>detector/valid_mask [512,512]</code>。Clean-C 另外保存 <code>dose_electrons [3]</code> 与 <code>rng_seed [1081,3,5]</code>。AutoDisk 和 <code>find_Bragg_disks</code> 读取同一幅原始数组、同一套 q 轴和同一个真空探针模板；网页的亮度映射不参与检测。</div>
 <details><summary>HDF5 层级与索引示例 / Schema and indexing example</summary><pre class="formula">public/clean_images.h5
├── expectation/intensity   float32 [sample, row, col]
├── sample_id
└── detector/{qx_Ainv, qy_Ainv, vacuum_probe, valid_mask}

public/clean_counted_images.h5
├── expectation/intensity   float32 [sample, row, col]
├── images/counts           uint32  [sample, dose, repeat, row, col]
├── dose_electrons          int64   [3]
├── rng_seed                uint64  [sample, dose, repeat]
├── sample_id
└── detector/{qx_Ainv, qy_Ainv, vacuum_probe, valid_mask}

# 检峰脚本的实际入口
python scripts/03_extract_clean_disks.py \
  --image-file public/clean_images.h5 --track expectation

python scripts/03_extract_clean_disks.py \
  --image-file public/clean_counted_images.h5 --track counted</pre></details>
</section>

<h2>二维输入图像示例 / Example 2D inputs</h2>
<section class="panel">
 <p>以下三幅图来自同一个实际运行样本 <code id="example-sample-id"></code>。Clean-C 图像均由左侧同一张 Clean-E 期望图采样得到，因此盘的位置不变，低剂量时可见性和计数涨落发生变化。</p>
 <div class="input-examples">
  <div class="input-example"><img id="example-clean-e" alt="Clean-E expectation diffraction image"><h3>Clean-E</h3><p>float32 期望强度图；无随机电子计数。</p></div>
  <div class="input-example"><img id="example-clean-c-low" alt="Clean-C low dose diffraction image"><h3>Clean-C · 10⁴ e⁻</h3><p>低剂量整数计数图；弱盘可能仅获得少量或零电子。</p></div>
  <div class="input-example"><img id="example-clean-c-high" alt="Clean-C high dose diffraction image"><h3>Clean-C · 10⁶ e⁻</h3><p>高剂量整数计数图；更接近期望强度分布。</p></div>
 </div>
 <p class="mini"><b>显示说明：</b>网页对每幅图分别使用以非零像素中位数为参考的自适应对数映射、99.5% 百分位截断和 0.65 显示伽马，以同时显示中心束与弱盘。因此三幅图的显示亮度不能用于比较绝对计数；该变换只用于网页预览，HDF5 中参与检峰和 ACOM 的原始 float32/uint32 像素值未被修改。</p>
</section>

<h2>总体结果 / Overview</h2>
<div class="cards" id="headline-cards"></div>
<div class="summary-grid" style="margin-top:14px">
 <section class="panel"><h3>剂量对检峰的影响 / Peak recovery vs dose</h3><svg id="peak-chart" class="chart" viewBox="0 0 620 280"></svg></section>
 <section class="panel"><h3>剂量对端到端取向的影响 / ACOM Acc@2° vs dose</h3><svg id="acom-chart" class="chart" viewBox="0 0 620 280"></svg></section>
</div>
<section class="panel" style="margin-top:14px"><h3>全量指标 / Full-run metrics</h3><div id="overview-table"></div><div class="verdict"><b>结论：</b>电子量增大有效。10⁴ e⁻ 时两种检峰器均明显丢峰；10⁵ e⁻ 已接近 oracle ACOM；10⁶ e⁻ 时 find_Bragg_disks 达到 100% Precision/Recall，且相对 oracle 的 ACOM P95 差异仅 0.0067°。当前正式推荐路径是 <b>10⁶ e⁻ + find_Bragg_disks</b>。</div></section>

<h2>v3 直接峰输入 vs 新图像输入 / Pipeline comparison</h2>
<div class="path-compare">
 <section class="path v3"><h3>冻结的 v3 基线 / Direct-peak baseline</h3><code>orientation → py4DSTEM 运动学峰 (qx, qy, I) → ACOM → orientation</code><p>公开输入直接是浮点峰列表。它验证取向采样、坐标变换、ACOM 搜索和评测，但没有检验图像形成、电子统计或检峰器。</p><div id="v3-summary"></div></section>
 <section class="path image"><h3>当前完整 Clean / Image-first benchmark</h3><code>orientation → physical CBED image → AutoDisk / find_Bragg_disks → (qx, qy, I) → same ACOM → orientation</code><p>ACOM 入口仍然是同一种 PointList；新增损失只来自物理图像形成、计数观测和自动检峰。两条路径共享同一个 2° orientation plan 和同一套对称性评测。</p><div id="image-summary"></div></section>
</div>
<section class="panel" style="margin-top:14px"><h3>v3 角步长基线 / Frozen v3 angular sweep</h3><div id="v3-sweep-table"></div><p class="mini">该表是之前 v3 真实运行的 4°、3°、2° 结果；新图像链路使用同一个 canonical 2° 检测与评价方法，不把不同角步长的方法混在一起。</p></section>

<h2>单张衍射图检查 / Per-pattern inspection</h2>
<p class="lead" style="margin-bottom:10px">所有控件都可以修改，修改后下方主诊断立即显示所选组合的实际运行结果。“案例预设”只是一次性填入一组有代表性的参数，不会限制后续选择。</p>
<div class="toolbar">
 <div class="control"><label>案例预设（可选）/ Example preset</label><select id="sample"></select></div>
 <div class="control"><label>图像层 / Image track</label><div class="seg"><button id="track-e">Clean-E 期望图</button><button id="track-c" class="active">Clean-C 计数图</button></div></div>
 <div class="control counted"><label>电子剂量 / Dose</label><select id="dose"><option value="10000" selected>10⁴ e⁻</option><option value="100000">10⁵ e⁻</option><option value="1000000">10⁶ e⁻</option></select></div>
 <div class="control counted"><label>随机重复 / Repeat</label><select id="repeat"><option>0</option><option>1</option><option>2</option><option>3</option><option>4</option></select></div>
 <div class="control"><label>检峰器 / Detector</label><select id="detector"><option value="py4dstem">py4DSTEM find_Bragg_disks</option><option value="autodisk">AutoDisk</option></select></div>
 <div class="control"><label>图层 / Overlay</label><div class="seg"><button id="v3-toggle" class="active">v3 direct peaks</button><button id="oracle-toggle" class="active">Physical oracle</button><button id="detected-toggle" class="active">Detected</button></div></div>
</div>
<div id="case-description" class="case-description"></div>
<details><summary>查看参数与图例说明 / Terms and controls</summary>
 <div class="control-help">
  <div><b>期望图 / Expectation</b><span>Clean-E 的确定性浮点图。每个像素表示模型预测的平均强度或电子落点概率；没有进行随机电子抽样。</span></div>
  <div><b>计数图 / Counted</b><span>Clean-C 的整数图。从期望图按所选 Dose 抽样；Repeat 是相同取向和剂量下的独立随机实现。</span></div>
  <div><b>Physical oracle</b><span>根据已知物理模型和最终无噪声图得到的参考盘中心及积分强度，只用于私有评测，不提供给检峰器。</span></div>
  <div><b>Detected</b><span>当前所选 AutoDisk 或 <code>find_Bragg_disks</code> 从二维图像自动输出的盘中心；它是算法结果，不是 Ground Truth。</span></div>
  <div><b>v3 direct peaks</b><span>旧 v3 直接交给 ACOM 的浮点峰列表，不是从当前二维图像检测得到。</span></div>
  <div><b>Dose</b><span>每张 Clean-C 图的固定总电子数。剂量越低，弱盘获得零电子或少量电子的概率越高。</span></div>
  <div><b>Repeat</b><span>同一个期望图、同一个剂量下的独立多项分布采样编号；用于估计随机波动。</span></div>
  <div><b>Overlay</b><span>只控制网页上显示哪些标记，不改变输入图、检峰结果或 ACOM 结果。</span></div>
 </div>
</details>
<div class="diag">
 <section>
  <div class="image-panel"><img id="pattern" alt="diffraction pattern"><svg id="overlay" viewBox="0 0 512 512"></svg></div>
  <div class="legend"><span style="color:var(--purple)">□ v3 直接输入峰</span><span><i class="dot"></i>物理 oracle + HKL</span><span><b class="cross">×</b>当前检峰</span><span id="seed-label" class="mini"></span></div>
 </section>
 <section class="panel">
  <h3 id="selection-title"></h3>
  <div class="metrics" id="sample-metrics"></div>
  <div class="matrix-grid">
   <div class="matrix"><b>Ground Truth</b><span class="mini">R<sub>sample→crystal</sub></span><pre id="gt-matrix"></pre></div>
   <div class="matrix"><b>v3 Direct-Peak ACOM</b><span class="mini">R<sub>sample→crystal</sub></span><pre id="v3-matrix"></pre></div>
   <div class="matrix"><b>Oracle-ACOM</b><span class="mini">R<sub>sample→crystal</sub></span><pre id="oracle-matrix"></pre></div>
   <div class="matrix"><b>当前 ACOM / Selected</b><span class="mini">R<sub>sample→crystal</sub></span><pre id="selected-matrix"></pre></div>
  </div>
  <p class="mini">四者均为同一语义的样品坐标→晶体坐标旋转矩阵。矩阵元素外观可能因晶体/Friedel 对称等价表示而明显不同；上方角度使用对称性约化后的取向误差。图像检峰误差与 ACOM 相对 GT 的取向误差分开报告。</p>
 </section>
</div>

<h2>v3 直接峰样本诊断 / v3 direct-input sample diagnostics</h2>
<div class="v3-diagnostic">
 <section><svg id="v3-direct-plot" class="v3-plot" viewBox="0 0 620 520" role="img" aria-label="v3 direct peaks and ACOM template peaks"></svg><div class="legend"><span style="color:var(--blue)">○ v3 直接输入峰 / observed</span><span style="color:var(--orange)">× v3 ACOM 模板峰 / simulated</span><span style="color:#9aa4b2">— 一对一匹配 / assignment</span></div></section>
 <section class="panel"><h3>v3 输入与模板匹配</h3><p>蓝色圆圈是 v3 直接交给 ACOM 的 <code>(qx,qy,I)</code>；橙色叉是预测取向下 orientation plan 生成的模拟模板峰。灰线表示 v3 报告保存的一对一峰匹配。</p><div class="metrics" id="v3-plot-metrics"></div><p class="mini">v3 路径不包含检峰器，蓝色圆圈本身就是 ACOM 输入。该图与二维衍射图诊断并列显示，用于区分直接峰基线和图像输入链路。</p></section>
</div>

<h2>衍射盘检测结果 / Visual disk-detection accuracy</h2>
<section class="panel">
 <p>下面两幅图使用当前选择的同一张 Clean-E 或 Clean-C 图像，同时显示两个检峰器。<b>绿色圆圈和连线</b>是 1 px 容差内的一对一真阳性；<b>黄色圆圈</b>是漏检 Oracle 盘；<b>红色叉</b>是误检盘。线段长度就是位置误差。</p>
 <div class="verdict warn"><b>默认错误案例：</b><code>clean_core_0970 · Clean-C · 10⁴ e⁻ · repeat 0</code>。AutoDisk 为 TP 14 / FP 6 / FN 7；<code>find_Bragg_disks</code> 为 TP 16 / FP 5 / FN 5。可在上方切换到 Clean-E 或更高剂量观察错误如何减少。</div>
 <div class="detector-compare">
  <div><h3>AutoDisk</h3><div class="compare-image"><img id="compare-image-autodisk" alt="AutoDisk comparison image"><svg id="compare-overlay-autodisk" viewBox="0 0 512 512"></svg></div><div id="compare-metric-autodisk" class="compare-metric"></div></div>
  <div><h3>py4DSTEM find_Bragg_disks</h3><div class="compare-image"><img id="compare-image-py4dstem" alt="find Bragg disks comparison image"><svg id="compare-overlay-py4dstem" viewBox="0 0 512 512"></svg></div><div id="compare-metric-py4dstem" class="compare-metric"></div></div>
 </div>
 <div class="legend"><span style="color:#42d392">○ TP + error line</span><span style="color:#ffd166">○ FN / 漏检</span><span style="color:#ff5d5d">× FP / 误检</span></div>
</section>

<section class="panel" style="margin-top:14px"><h3>物理 Oracle 逐反射表 / All oracle reflections</h3><p class="mini">HKL 是倒易晶格反射索引，不是“样品只有一个晶面”。每个可见衍射盘对应一个满足当前探测范围和激发条件的 (h,k,l)。</p><div class="peak-table-wrap"><table><thead><tr><th>#</th><th>HKL</th><th>q<sub>x</sub> (Å⁻¹)</th><th>q<sub>y</sub> (Å⁻¹)</th><th>强度 / Intensity</th></tr></thead><tbody id="peak-rows"></tbody></table></div></section>

<h2>v3 逐反射坐标追踪 / Direct-peak coordinate trace</h2>
<section class="panel">
 <p>这一节保留旧 v3 的完整诊断语义。选择任意一个 v3 输入反射，逐步查看 <b>HKL → g<sub>crystal</sub> → g<sub>sample</sub> → detector q</b>；这里的 v3 峰是 ACOM 的直接输入，不是从衍射图检测得到。</p>
 <div class="control" style="max-width:500px"><label>选择 v3 输入反射 / Select direct-input reflection</label><select id="v3-reflection" style="width:100%"></select></div>
 <div class="equation-grid" style="margin-top:12px">
  <div class="matrix"><b>① h = [h,k,l]</b><span class="mini">Miller reflection index</span><pre id="trace-hkl"></pre></div>
  <div class="matrix"><b>② B = [a*; b*; c*]</b><span class="mini">reciprocal basis · Å⁻¹ · no 2π</span><pre id="trace-b"></pre></div>
  <div class="matrix"><b>③ g<sub>crystal</sub> = hB</b><span class="mini">crystal Cartesian · Å⁻¹</span><pre id="trace-gcrystal"></pre></div>
 </div>
 <div class="transform-grid" style="margin-top:10px">
  <div class="matrix"><b>④ GT: g<sub>sample</sub> = g<sub>crystal</sub> R<sup>GT</sup></b><span class="mini">[qx, qy, qz] · detector uses qx,qy</span><pre id="trace-gsample-gt"></pre></div>
  <div class="matrix"><b>④ v3 ACOM: same HKL under R<sup>v3</sup></b><span class="mini">diagnostic transform, not a separate ACOM input</span><pre id="trace-gsample-v3"></pre></div>
 </div>
 <div class="formula" id="trace-equation" style="margin-top:10px"></div>
</section>
<section class="panel" style="margin-top:14px"><h3>全部 v3 输入反射 / All v3 direct-input reflections</h3><div class="peak-table-wrap"><table><thead><tr><th>#</th><th>HKL</th><th>I</th><th>g<sub>crystal</sub> (Å⁻¹)</th><th>g<sub>sample</sub><sup>GT</sup> = [qx,qy,qz]</th><th>g<sub>sample</sub><sup>v3 ACOM|same h</sup></th><th>‖Δg‖₂</th></tr></thead><tbody id="v3-reflection-rows"></tbody></table></div></section>
<section class="panel" style="margin-top:14px"><h3>v3 ACOM 模拟模板反射 / v3 ACOM simulated reflections</h3><p class="mini">这些是预测取向下 orientation plan 生成的模板峰，用于与上表的 v3 观测峰做稀疏相关；它们不是 Ground Truth HKL 的逐项拷贝。</p><div class="peak-table-wrap"><table><thead><tr><th>#</th><th>Predicted HKL</th><th>q<sub>x</sub></th><th>q<sub>y</sub></th><th>I</th><th>g<sub>crystal</sub></th><th>g<sub>sample</sub></th></tr></thead><tbody id="v3-predicted-rows"></tbody></table></div></section>

<h2>坐标与变量 / Coordinates and variables</h2>
<section class="panel">
 <div class="formula">B = [a*; b*; c*] (Å⁻¹, without 2π)<br>g<sub>crystal</sub> = [h k l] B<br>g<sub>sample</sub> = R<sub>sample→crystal</sub><sup>T</sup> g<sub>crystal</sub><sup>T</sup><br>[q<sub>x</sub>, q<sub>y</sub>] = detector-plane components of g<sub>sample</sub></div>
 <div class="matrix" style="margin-top:10px;max-width:520px"><b>B 倒易基矩阵 / Reciprocal basis</b><pre id="b-matrix"></pre></div>
 <p class="mini">A 的行是实空间基矢 a、b、c（Å）；B 的行是倒易基矢 a*、b*、c*（Å⁻¹）。本 benchmark 的 B 不含 2π。图像像素坐标仅用于检测；转换回 q 后才进入 ACOM。</p>
</section>

<h2>运行参数 / Runtime parameters</h2>
<details open><summary>v3 直接峰生成参数 · Direct-peak generation parameters</summary><div id="v3-parameter-table" style="margin-top:10px"></div></details>
<details><summary>新图像生成与计数参数 · Image formation and counting parameters</summary><div id="image-parameter-table" style="margin-top:10px"></div></details>
<details><summary>两条路径共用的 ACOM 参数 · Shared ACOM parameters</summary><div id="acom-parameter-table" style="margin-top:10px"></div></details>
<details><summary>当前页面显示参数 · Current display/runtime parameters</summary><div id="parameter-table" style="margin-top:10px"></div></details>
<details><summary>结果来源与限制 / Provenance and limitation</summary><p>网页由 <code>scripts/17_write_clean_image_visualization.py</code> 从本地 HDF5 和 JSON 结果重新生成。汇总图使用全部 1081 个图样；headline 取向指标统计 1024 个 headline_core。Clean 图像与 ACOM 模板共享同一 CIF 和匹配的运动学模型，因此这里验证的是端到端自洽性，不代表真实实验数据或跨模拟器泛化能力。</p></details>
</main>
<script>
const DATA = __DATA__;
const $ = id => document.getElementById(id);
const CASES=[
 {id:"error_low_auto",label:"错误案例 · Clean-C 10⁴ e⁻ · AutoDisk",sample:"clean_core_0970",track:"counted",dose:10000,repeat:0,detector:"autodisk",description:"低剂量检峰错误：AutoDisk 为 TP 14、FP 6、FN 7。该场景用于查看黄色漏检盘、红色误检盘及位置误差线。"},
 {id:"error_low_py",label:"错误案例 · Clean-C 10⁴ e⁻ · find_Bragg_disks",sample:"clean_core_0970",track:"counted",dose:10000,repeat:0,detector:"py4dstem",description:"同一张低剂量图上的另一种检测结果：find_Bragg_disks 为 TP 16、FP 5、FN 5，可与 AutoDisk 直接比较。"},
 {id:"error_clean_e_auto",label:"错误案例 · Clean-E · AutoDisk 新增 >5° ACOM 失败",sample:"clean_core_0744",track:"expectation",dose:1000000,repeat:0,detector:"autodisk",description:"无噪声 Clean-E 中 AutoDisk 的峰 Precision/Recall 仍为 100%，但亚像素位置与积分强度变化使该样本成为相对 Oracle 新增的 >5° ACOM 失败。"},
 {id:"correct_clean_e_py",label:"正确案例 · Clean-E · find_Bragg_disks",sample:"clean_core_0970",track:"expectation",dose:1000000,repeat:0,detector:"py4dstem",description:"理想期望图的正确检峰案例：全部物理 Oracle 盘被恢复，位置误差约为百分之一像素，ACOM 与 Oracle 基本一致。"},
 {id:"median_clean_e",label:"代表案例 · Clean-E · 中位 ACOM 误差",sample:"clean_core_0530",track:"expectation",dose:1000000,repeat:0,detector:"py4dstem",description:"按 Physical-Oracle ACOM 对 Ground Truth 的 headline 误差排序选取的中位样本，用于查看典型而非极端结果。"},
 {id:"worst_clean_e",label:"极端案例 · Clean-E · 最差 ACOM 误差",sample:"clean_core_0037",track:"expectation",dose:1000000,repeat:0,detector:"py4dstem",description:"headline 中 Ground Truth 取向误差最大的代表样本。用于区分检峰正确与 ACOM 对称性或模板歧义造成的灾难性错误。"}
];
let activeCase=CASES[0].id;
let state={sample:CASES[0].sample,track:CASES[0].track,dose:CASES[0].dose,repeat:CASES[0].repeat,detector:CASES[0].detector,v3:true,oracle:true,detected:true,v3Reflection:0};
const fmt=(x,n=3)=>x==null?"—":Number(x).toFixed(n);
const pct=x=>fmt(100*x,2)+"%";
const matrix=m=>m.map(r=>"["+r.map(x=>(x>=0?" ":"")+Number(x).toFixed(5)).join(", ")+"]").join("\n");
const vector=(v,n=5)=>"["+v.map(x=>(x>=0?" ":"")+Number(x).toFixed(n)).join(", ")+"]";
const norm=v=>Math.sqrt(v.reduce((s,x)=>s+x*x,0));
function variantKeyFor(detector){return state.track==="expectation"?`${detector}_expectation`:`counted_dose${state.dose}_repeat${state.repeat}_${detector}`}
function variantKey(){return variantKeyFor(state.detector)}
function svgChart(id,rows,field,yLabel,percent=false){
 const svg=$(id),W=620,H=280,L=52,R=18,T=22,B=42, xs=[10000,100000,1000000];
 const vals=rows.map(r=>r[field]); let ymin=Math.min(...vals),ymax=Math.max(...vals); if(percent){ymin=Math.max(0,ymin-.05);ymax=Math.min(1,ymax+.03)} else {ymin*=.92;ymax*=1.08}
 const x=i=>L+i*(W-L-R)/2, y=v=>T+(ymax-v)/(ymax-ymin||1)*(H-T-B);
 let h=""; for(let i=0;i<5;i++){let v=ymin+(ymax-ymin)*i/4,yy=y(v);h+=`<line class="grid" x1="${L}" y1="${yy}" x2="${W-R}" y2="${yy}"/><text x="${L-7}" y="${yy+4}" text-anchor="end">${percent?(v*100).toFixed(0)+"%":v.toFixed(2)}</text>`}
 h+=`<line class="axis" x1="${L}" y1="${H-B}" x2="${W-R}" y2="${H-B}"/><text x="12" y="15">${yLabel}</text>`;
 xs.forEach((v,i)=>h+=`<text x="${x(i)}" y="${H-17}" text-anchor="middle">10${["⁴","⁵","⁶"][i]} e⁻</text>`);
 ["autodisk","py4dstem"].forEach((det,di)=>{let rs=xs.map(d=>rows.find(r=>r.detector===det&&r.dose_electrons===d));let cls=det==="autodisk"?"auto":"py";h+=`<polyline class="${cls}" points="${rs.map((r,i)=>x(i)+","+y(r[field])).join(" ")}"/>`;rs.forEach((r,i)=>h+=`<circle class="${cls}" cx="${x(i)}" cy="${y(r[field])}" r="4"/>`)});
 h+=`<line class="auto" x1="390" y1="13" x2="415" y2="13"/><text x="421" y="17">AutoDisk</text><line class="py" x1="490" y1="13" x2="515" y2="13"/><text x="521" y="17">find_Bragg</text>`;svg.innerHTML=h;
}
function initOverview(){
 const pyE=DATA.peak_expectation.find(r=>r.detector==="py4dstem"), py1m=DATA.peak_dose_summary.find(r=>r.detector==="py4dstem"&&r.dose_electrons===1000000), a1m=DATA.acom_dose_summary.find(r=>r.detector==="py4dstem"&&r.dose_electrons===1000000);
 $("headline-cards").innerHTML=[
  ["图样 / Patterns",DATA.generated_from.patterns,"1081 = 17 legacy + 1024 headline + 40 probe"],
  ["计数图 / Count images",DATA.generated_from.counted_images,"3 doses × 5 repeats"],
  ["Clean-E 检峰 / Peak P/R",pct(pyE.precision)+" / "+pct(pyE.recall),"find_Bragg_disks"],
  ["推荐端到端 / Acc@2°",pct(a1m.ground_truth_acc_at_2deg_mean),"10⁶ e⁻ + find_Bragg_disks"]
 ].map(x=>`<div class="card"><div class="unit">${x[0]}</div><div class="value">${x[1]}</div><div class="unit">${x[2]}</div></div>`).join("");
 svgChart("peak-chart",DATA.peak_dose_summary,"recall_mean","Recall",true); svgChart("acom-chart",DATA.acom_dose_summary,"ground_truth_acc_at_2deg_mean","Acc@2°",true);
 let rows=[...DATA.peak_dose_summary].map(p=>{let a=DATA.acom_dose_summary.find(x=>x.detector===p.detector&&x.dose_electrons===p.dose_electrons);return `<tr><td>Clean-C</td><td>${p.dose_electrons.toLocaleString()}</td><td>${p.detector==="py4dstem"?"find_Bragg_disks":"AutoDisk"}</td><td>${pct(p.precision_mean)}</td><td>${pct(p.recall_mean)}</td><td>${fmt(p.position_rmse_px_mean,3)}</td><td>${pct(a.ground_truth_acc_at_2deg_mean)}</td><td>${fmt(a.delta_p95_deg_mean,3)}°</td></tr>`});
 DATA.peak_expectation.forEach(p=>{let a=DATA.acom_expectation.find(x=>x.label===`${p.detector}_expectation`);rows.unshift(`<tr><td>Clean-E</td><td>∞ / expectation</td><td>${p.detector==="py4dstem"?"find_Bragg_disks":"AutoDisk"}</td><td>${pct(p.precision)}</td><td>${pct(p.recall)}</td><td>${fmt(p.position_rmse_px,3)}</td><td>${pct(a.versus_ground_truth.acc_at_2deg)}</td><td>${fmt(a.versus_oracle.p95_deg,3)}°</td></tr>`)});
 $("overview-table").innerHTML=`<table><thead><tr><th>Track</th><th>Dose</th><th>Detector</th><th>Precision</th><th>Recall</th><th>位置 RMSE (px)</th><th>GT Acc@2°</th><th>vs Oracle P95</th></tr></thead><tbody>${rows.join("")}</tbody></table>`;
 const v3=DATA.v3_metrics,oracle=DATA.acom_oracle,py=DATA.acom_expectation.find(x=>x.label==="py4dstem_expectation");
 $("v3-summary").innerHTML=`<b>2° headline：</b>Median ${fmt(v3.median_misorientation_deg,3)}° · P95 ${fmt(v3.p95_misorientation_deg,3)}° · Acc@2° ${pct(v3.accuracy_within_2deg)} · Acc@5° ${pct(v3.accuracy_within_5deg)}`;
 $("image-summary").innerHTML=`<b>Physical oracle：</b>Acc@2° ${pct(oracle.acc_at_2deg)}；<b>Clean-E + find_Bragg：</b>Acc@2° ${pct(py.versus_ground_truth.acc_at_2deg)} · vs oracle P95 ${fmt(py.versus_oracle.p95_deg,4)}°`;
 $("v3-sweep-table").innerHTML=`<table><thead><tr><th>Angle step</th><th>Median</th><th>P95</th><th>Acc@2°</th><th>Acc@5°</th></tr></thead><tbody>${DATA.v3_sweep.map(r=>`<tr><td>${r.angle_step_deg}°</td><td>${fmt(r.median_deg,3)}°</td><td>${fmt(r.p95_deg,3)}°</td><td>${pct(r.acc_at_2deg)}</td><td>${pct(r.acc_at_5deg)}</td></tr>`).join("")}</tbody></table>`;
}
function setButton(id,on){$(id).classList.toggle("active",on)}
function renderV3Plot(s){
 const trace=s.v3,W=620,H=520,cx=W/2,cy=H/2,scale=215/1.5,qx=q=>cx+q*scale,qy=q=>cy-q*scale;
 let svg=`<circle cx="${cx}" cy="${cy}" r="215" fill="none" stroke="#cfd6e1"/><line x1="${cx}" y1="30" x2="${cx}" y2="${H-30}" stroke="#cfd6e1"/><line x1="70" y1="${cy}" x2="${W-70}" y2="${cy}" stroke="#cfd6e1"/><text x="${W-82}" y="${cy-8}">qₓ</text><text x="${cx+8}" y="45">qᵧ</text>`;
 trace.matches.forEach(m=>{let o=trace.observed[m.observed_index],p=trace.predicted[m.predicted_index];if(o&&p)svg+=`<line x1="${qx(o.q[0])}" y1="${qy(o.q[1])}" x2="${qx(p.q[0])}" y2="${qy(p.q[1])}" stroke="#aeb7c4" stroke-width="1"/>`});
 const labeled=new Set([...trace.observed.keys()].sort((a,b)=>trace.observed[b].intensity-trace.observed[a].intensity).slice(0,10));
 trace.observed.forEach((p,i)=>{let x=qx(p.q[0]),y=qy(p.q[1]),r=4+5*Math.sqrt(Math.max(0,p.intensity));svg+=`<circle cx="${x}" cy="${y}" r="${r}" fill="#fff" stroke="#246bce" stroke-width="2"/>${labeled.has(i)?`<text x="${x+8}" y="${y-8}">[${p.hkl.join(" ")}]</text>`:""}`});
 trace.predicted.forEach(p=>{let x=qx(p.q[0]),y=qy(p.q[1]);svg+=`<path d="M${x-4},${y-4}L${x+4},${y+4}M${x+4},${y-4}L${x-4},${y+4}" stroke="#e56b3f" stroke-width="2"/>`});
 $("v3-direct-plot").innerHTML=svg;
 $("v3-plot-metrics").innerHTML=[["输入峰",trace.observed.length],["模板峰",trace.predicted.length],["匹配率",pct(trace.observed_match_fraction)],["q-RMSE",fmt(trace.q_rmse_Ainv,6)+" Å⁻¹"],["ACOM → GT",fmt(trace.orientation_error_deg,3)+"°"]].map(x=>`<div class="metric"><span>${x[0]}</span><b>${x[1]}</b></div>`).join("");
}
function renderDetectorComparison(s,detector){
 const v=DATA.variants[variantKeyFor(detector)],peaks=v.peaks[state.sample],m=v.sample_peak_metrics[state.sample];
 const image=state.track==="expectation"?s.expectation_image:s.counted_images[`${state.dose}:${state.repeat}`];
 $(`compare-image-${detector}`).src=image;
 const [xmin,xmax,ymin,ymax]=DATA.q_bounds,px=q=>512*(q-xmin)/(xmax-xmin),py=q=>512*(ymax-q)/(ymax-ymin);
 let svg="";
 m.matches.forEach(match=>{let o=s.oracle_peaks[match.oracle_index],d=peaks[match.detected_index],ox=px(o.qx),oy=py(o.qy),dx=px(d.qx),dy=py(d.qy);svg+=`<line x1="${ox}" y1="${oy}" x2="${dx}" y2="${dy}" stroke="#42d392" stroke-width="1.4"/><circle cx="${ox}" cy="${oy}" r="${DATA.disk_radius_px+2}" fill="none" stroke="#42d392" stroke-width="1.5"/><circle cx="${dx}" cy="${dy}" r="2" fill="#42d392"/>`});
 m.false_negative_indices.forEach(i=>{let o=s.oracle_peaks[i];svg+=`<circle cx="${px(o.qx)}" cy="${py(o.qy)}" r="${DATA.disk_radius_px+3}" fill="none" stroke="#ffd166" stroke-width="2.4"/>`});
 m.false_positive_indices.forEach(i=>{let d=peaks[i],x=px(d.qx),y=py(d.qy);svg+=`<path d="M${x-6},${y-6}L${x+6},${y+6}M${x+6},${y-6}L${x-6},${y+6}" stroke="#ff5d5d" stroke-width="2.5"/>`});
 $(`compare-overlay-${detector}`).innerHTML=svg;
 $(`compare-metric-${detector}`).innerHTML=`TP ${m.tp} · FP ${m.false_positive_indices.length} · FN ${m.false_negative_indices.length} · Precision ${pct(m.precision)} · Recall ${pct(m.recall)} · RMSE ${fmt(m.rmse_px,3)} px · P95 ${fmt(m.p95_px,3)} px`;
}
function renderV3Trace(s){
 const trace=s.v3,rows=trace.observed,index=Math.min(state.v3Reflection,rows.length-1),r=rows[index];
 $("v3-reflection").innerHTML=rows.map((p,i)=>`<option value="${i}" ${i===index?"selected":""}>#${i+1} HKL [${p.hkl.join(", ")}] · I=${fmt(p.intensity,4)}</option>`).join("");
 $("trace-hkl").textContent=vector(r.hkl,0);$("trace-b").textContent=matrix(trace.reciprocal_lattice_matrix_B);$("trace-gcrystal").textContent=vector(r.g_crystal);
 $("trace-gsample-gt").textContent=vector(r.g_sample_gt);$("trace-gsample-v3").textContent=vector(r.g_sample_v3_acom_same_hkl);
 const delta=r.g_sample_v3_acom_same_hkl.map((x,i)=>x-r.g_sample_gt[i]);
 $("trace-equation").innerHTML=`h ${vector(r.hkl,0)} × B → g<sub>crystal</sub> ${vector(r.g_crystal)} Å⁻¹<br>g<sub>crystal</sub> × R<sup>GT</sup> → [q<sub>x</sub>,q<sub>y</sub>,q<sub>z</sub>] ${vector(r.g_sample_gt)} Å⁻¹<br>reported detector q = ${vector(r.q)} Å⁻¹ · same-HKL ‖Δg<sub>sample</sub>‖₂ = ${fmt(norm(delta),6)} Å⁻¹<br><b>v3 实际模板匹配：</b>observed match ${pct(trace.observed_match_fraction)} · matched-q RMSE ${fmt(trace.q_rmse_Ainv,6)} Å⁻¹。ACOM 可能返回晶体/Friedel 对称等价矩阵，因此“固定同一 HKL”的诊断差可以很大；它不等于最近邻模板峰匹配误差。`;
 $("v3-reflection-rows").innerHTML=rows.map((p,i)=>{let dv=p.g_sample_v3_acom_same_hkl.map((x,j)=>x-p.g_sample_gt[j]);return `<tr><td>${i+1}</td><td>[${p.hkl.join(", ")}]</td><td>${fmt(p.intensity,5)}</td><td>${vector(p.g_crystal)}</td><td>${vector(p.g_sample_gt)}</td><td>${vector(p.g_sample_v3_acom_same_hkl)}</td><td>${fmt(norm(dv),6)}</td></tr>`}).join("");
 $("v3-predicted-rows").innerHTML=trace.predicted.map((p,i)=>`<tr><td>${i+1}</td><td>[${p.hkl.join(", ")}]</td><td>${fmt(p.q[0],5)}</td><td>${fmt(p.q[1],5)}</td><td>${fmt(p.intensity,5)}</td><td>${vector(p.g_crystal)}</td><td>${vector(p.g_sample)}</td></tr>`).join("");
}
function redraw(){
 const s=DATA.samples[state.sample],v=DATA.variants[variantKey()],d=v.details[state.sample],m=v.sample_peak_metrics[state.sample];
 $("pattern").src=state.track==="expectation"?s.expectation_image:s.counted_images[`${state.dose}:${state.repeat}`];
 const preset=CASES.find(item=>item.id===activeCase);
 $("selection-title").textContent=preset?`${preset.label} · ${state.sample}`:`${s.label} · ${state.sample} · ${state.track==="expectation"?"Clean-E":`Clean-C ${state.dose.toLocaleString()} e⁻ / repeat ${state.repeat}`} · ${state.detector==="py4dstem"?"find_Bragg_disks":"AutoDisk"}`;
 $("sample-metrics").innerHTML=[
  ["Peak P/R",pct(m.precision)+" / "+pct(m.recall)],
  ["位置 RMSE",fmt(m.rmse_px,3)+" px"],
  ["ACOM → GT",fmt(d.gt_error_deg,3)+"°"],
  ["ACOM → Oracle",fmt(d.delta_oracle_deg,3)+"°"],
  ["v3 ACOM → GT",fmt(s.v3.orientation_error_deg,3)+"°"],
  ["v3 峰匹配率",pct(s.v3.observed_match_fraction)],
  ["v3 q-RMSE",fmt(s.v3.q_rmse_Ainv,6)+" Å⁻¹"],
  ["峰数 / Peaks",`${m.detected} / ${m.oracle}`],
  ["Correlation",fmt(d.correlation,4)]
 ].map(x=>`<div class="metric"><span>${x[0]}</span><b>${x[1]}</b></div>`).join("");
 $("gt-matrix").textContent=matrix(s.gt_matrix);$("v3-matrix").textContent=matrix(s.v3.acom_matrix);$("oracle-matrix").textContent=matrix(s.oracle_matrix);$("selected-matrix").textContent=matrix(d.matrix);
 $("seed-label").textContent=state.track==="counted"?`RNG seed: ${s.seeds[`${state.dose}:${state.repeat}`]}`:"确定性期望强度图 / deterministic expectation";
 const [xmin,xmax,ymin,ymax]=DATA.q_bounds,px=q=>512*(q-xmin)/(xmax-xmin),py=q=>512*(ymax-q)/(ymax-ymin);
 let svg=`<line x1="256" y1="0" x2="256" y2="512" stroke="#ffffff33"/><line x1="0" y1="256" x2="512" y2="256" stroke="#ffffff33"/>`;
 if(state.v3)s.v3.observed.forEach(p=>{let x=px(p.q[0]),y=py(p.q[1]);svg+=`<rect x="${x-5}" y="${y-5}" width="10" height="10" fill="none" stroke="#b99be8" stroke-width="1.5"/>`});
 if(state.oracle)s.oracle_peaks.forEach((p,i)=>{let show=i<12;svg+=`<circle cx="${px(p.qx)}" cy="${py(p.qy)}" r="${DATA.disk_radius_px+2}" fill="none" stroke="#58a6ff" stroke-width="1.5"/>${show?`<text x="${px(p.qx)+6}" y="${py(p.qy)-6}" fill="#d9ecff" font-size="9">[${p.hkl.join(" ")}]</text>`:""}`});
 if(state.detected)v.peaks[state.sample].forEach(p=>{let x=px(p.qx),y=py(p.qy);svg+=`<path d="M${x-4},${y-4}L${x+4},${y+4}M${x+4},${y-4}L${x-4},${y+4}" stroke="#ff8b5e" stroke-width="1.8"/>`});
 $("overlay").innerHTML=svg;
 $("peak-rows").innerHTML=s.oracle_peaks.map((p,i)=>`<tr><td>${i+1}</td><td>[${p.hkl.join(", ")}]</td><td>${fmt(p.qx,5)}</td><td>${fmt(p.qy,5)}</td><td>${fmt(p.intensity,5)}</td></tr>`).join("");
 renderV3Plot(s);renderDetectorComparison(s,"autodisk");renderDetectorComparison(s,"py4dstem");
 renderV3Trace(s);
 document.querySelectorAll(".counted").forEach(e=>e.classList.toggle("hidden",state.track!=="counted"));
 setButton("track-e",state.track==="expectation");setButton("track-c",state.track==="counted");setButton("v3-toggle",state.v3);setButton("oracle-toggle",state.oracle);setButton("detected-toggle",state.detected);
}
function showCaseDescription(text,label="场景说明 / Case note"){$("case-description").innerHTML=`<b>${label}：</b>${text}`}
function applyCase(id){
 const item=CASES.find(candidate=>candidate.id===id);if(!item)return;
 activeCase=id;state.sample=item.sample;state.track=item.track;state.dose=item.dose;state.repeat=item.repeat;state.detector=item.detector;state.v3Reflection=0;
 $("sample").value=id;$("dose").value=String(item.dose);$("repeat").value=String(item.repeat);$("detector").value=item.detector;showCaseDescription(item.description);redraw();
}
function markViewAdjusted(){
 activeCase=null;$("sample").value="";
 const imageLabel=state.track==="expectation"?"Clean-E 期望图":`Clean-C 计数图 · ${state.dose.toLocaleString()} e⁻ · repeat ${state.repeat}`;
 const detectorLabel=state.detector==="py4dstem"?"find_Bragg_disks":"AutoDisk";
 showCaseDescription(`<b>${imageLabel} · ${detectorLabel}</b>。下方主衍射图、橙色 Detected 标记、样本指标和“当前 ACOM”矩阵已经切换到这一组合。双检峰器对比区仍固定并列显示 AutoDisk 与 find_Bragg_disks 在同一张图上的结果。`,"已应用 / Applied");
}
function init(){
 initOverview();$("b-matrix").textContent=matrix(DATA.reciprocal_matrix_B);
 const example=DATA.samples[DATA.sample_order[0]];
 $("example-sample-id").textContent=DATA.sample_order[0];$("example-clean-e").src=example.expectation_image;$("example-clean-c-low").src=example.counted_images["10000:0"];$("example-clean-c-high").src=example.counted_images["1000000:0"];
 $("sample").innerHTML=`<option value="" disabled>选择一个案例预设 / Select a preset</option>`+CASES.map(item=>`<option value="${item.id}">${item.label} · ${item.sample}</option>`).join("");
 const parameterTable=rows=>`<table><thead><tr><th>参数 / Parameter</th><th>Code name</th><th>Value</th></tr></thead><tbody>${rows.map(r=>`<tr><td>${r[0]}</td><td><code>${r[1]}</code></td><td>${r[2]}</td></tr>`).join("")}</tbody></table>`;
 $("parameter-table").innerHTML=parameterTable(DATA.parameters);$("v3-parameter-table").innerHTML=parameterTable(DATA.parameter_tables.v3);$("image-parameter-table").innerHTML=parameterTable(DATA.parameter_tables.image);$("acom-parameter-table").innerHTML=parameterTable(DATA.parameter_tables.acom);
 $("sample").onchange=e=>applyCase(e.target.value);$("dose").onchange=e=>{state.dose=+e.target.value;markViewAdjusted();redraw()};$("repeat").onchange=e=>{state.repeat=+e.target.value;markViewAdjusted();redraw()};$("detector").onchange=e=>{state.detector=e.target.value;markViewAdjusted();redraw()};$("v3-reflection").onchange=e=>{state.v3Reflection=+e.target.value;renderV3Trace(DATA.samples[state.sample])};
 $("track-e").onclick=()=>{state.track="expectation";markViewAdjusted();redraw()};$("track-c").onclick=()=>{state.track="counted";markViewAdjusted();redraw()};$("v3-toggle").onclick=()=>{state.v3=!state.v3;redraw()};$("oracle-toggle").onclick=()=>{state.oracle=!state.oracle;redraw()};$("detected-toggle").onclick=()=>{state.detected=!state.detected;redraw()};applyCase(activeCase);
}
init();
</script></body></html>"""


def main() -> None:
    data = build_data()
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    OUTPUT.write_text(HTML.replace("__DATA__", payload), encoding="utf-8")
    print(f"Wrote {OUTPUT} ({OUTPUT.stat().st_size / 1024 / 1024:.1f} MiB)")


if __name__ == "__main__":
    main()
