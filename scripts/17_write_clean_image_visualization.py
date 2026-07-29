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
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports" / "CLEAN_IMAGE_ACOM_VISUALIZATION.html"
IMAGE_FILE = ROOT / "public" / "clean_images.h5"
COUNTED_FILE = ROOT / "public" / "clean_counted_images.h5"
ORACLE_FILE = ROOT / "private" / "clean_physical_oracle_reflections.h5"
GT_FILE = ROOT / "private" / "clean_ground_truth.jsonl"
TRACE_FILE = ROOT / "diagnostics" / "clean_coordinate_trace.jsonl.gz"
DETAILS_ORACLE = ROOT / "reports" / "acom_clean_details_physical_oracle.json"
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
    scaled = np.log1p(values)
    positive = scaled[scaled > 0]
    ceiling = float(np.percentile(positive, 99.9)) if positive.size else 1.0
    pixels = np.clip(scaled / max(ceiling, 1e-12), 0, 1)
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
    for distance2, oi, di in candidates:
        distance_px = distance2**0.5 / q_per_px
        if distance_px > 1.0:
            break
        if oi not in used_o and di not in used_d:
            used_o.add(oi)
            used_d.add(di)
            distances.append(distance_px)
    tp = len(distances)
    return {
        "oracle": len(oracle),
        "detected": len(detected),
        "tp": tp,
        "precision": tp / len(detected) if detected else 0.0,
        "recall": tp / len(oracle) if oracle else 0.0,
        "rmse_px": float(np.sqrt(np.mean(np.square(distances)))) if distances else None,
        "p95_px": float(np.percentile(distances, 95)) if distances else None,
    }


def reciprocal_matrix() -> list[list[float]]:
    with gzip.open(TRACE_FILE, "rt", encoding="utf-8") as handle:
        row = json.loads(next(handle))
    return rounded(row["reciprocal_lattice_matrix_B_Ainv"])


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
        }

    peak_e = load_json(PEAK_REPORT_E)
    peak_c = load_json(PEAK_REPORT_C)
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
.diag{display:grid;grid-template-columns:minmax(480px,1.1fr) minmax(410px,.9fr);gap:16px;margin-top:14px}.image-panel{position:relative;background:#081223;border-radius:10px;overflow:hidden;aspect-ratio:1}.image-panel img,.image-panel svg{position:absolute;inset:0;width:100%;height:100%}.image-panel img{image-rendering:auto}.image-panel svg{overflow:visible}.legend{display:flex;gap:18px;flex-wrap:wrap;font-size:13px;margin:9px 0}.dot{display:inline-block;width:11px;height:11px;border-radius:50%;border:2px solid var(--blue);margin-right:5px}.cross{color:var(--orange);font-size:18px;vertical-align:-1px}.matrix-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}.matrix{background:var(--wash);border-radius:8px;padding:10px}.matrix pre{font:12px/1.55 ui-monospace,SFMono-Regular,Menlo,monospace;margin:4px 0;white-space:pre-wrap}.metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin:11px 0}.metric{border:1px solid var(--line);border-radius:8px;padding:9px}.metric b{display:block;font-size:19px}.metric span{font-size:11px;color:var(--muted)}
.peak-table-wrap{max-height:360px;overflow:auto}.formula{font:14px/1.7 ui-monospace,SFMono-Regular,Menlo,monospace;background:#f6f8fb;border-radius:8px;padding:12px}.mini{font-size:12px;color:var(--muted)}details{border:1px solid var(--line);border-radius:10px;padding:11px 13px;margin-top:10px}summary{cursor:pointer;font-weight:670}.hidden{display:none!important}
@media(max-width:900px){main{padding:24px 16px}.cards,.flow{grid-template-columns:1fr 1fr}.summary-grid,.diag{grid-template-columns:1fr}.matrix-grid{grid-template-columns:1fr}.image-panel{min-width:0}}
</style>
</head>
<body><main>
<header>
  <h1>Clean 图像 Benchmark：衍射图 → 检峰 → ACOM</h1>
  <p class="lead">Full Clean image benchmark · expectation + counted electrons · AutoDisk vs py4DSTEM find_Bragg_disks</p>
  <span class="stamp">全量实际运行结果 / completed full run</span>
</header>
<div class="flow"><div><b>1</b> 运动学 CBED<br><span class="mini">physical expectation</span></div><div><b>2</b> 电子计数<br><span class="mini">10⁴ / 10⁵ / 10⁶ e⁻</span></div><div><b>3</b> 衍射盘检测<br><span class="mini">AutoDisk / find_Bragg_disks</span></div><div><b>4</b> ACOM + GT 评测<br><span class="mini">same 2° orientation plan</span></div></div>

<h2>总体结果 / Overview</h2>
<div class="cards" id="headline-cards"></div>
<div class="summary-grid" style="margin-top:14px">
 <section class="panel"><h3>剂量对检峰的影响 / Peak recovery vs dose</h3><svg id="peak-chart" class="chart" viewBox="0 0 620 280"></svg></section>
 <section class="panel"><h3>剂量对端到端取向的影响 / ACOM Acc@2° vs dose</h3><svg id="acom-chart" class="chart" viewBox="0 0 620 280"></svg></section>
</div>
<section class="panel" style="margin-top:14px"><h3>全量指标 / Full-run metrics</h3><div id="overview-table"></div><div class="verdict"><b>结论：</b>电子量增大有效。10⁴ e⁻ 时两种检峰器均明显丢峰；10⁵ e⁻ 已接近 oracle ACOM；10⁶ e⁻ 时 find_Bragg_disks 达到 100% Precision/Recall，且相对 oracle 的 ACOM P95 差异仅 0.0067°。当前正式推荐路径是 <b>10⁶ e⁻ + find_Bragg_disks</b>。</div></section>

<h2>样本诊断 / Pattern diagnostics</h2>
<div class="toolbar">
 <div class="control"><label>代表样本 / Representative</label><select id="sample"></select></div>
 <div class="control"><label>图像层 / Image track</label><div class="seg"><button id="track-e" class="active">Clean-E 期望图</button><button id="track-c">Clean-C 计数图</button></div></div>
 <div class="control counted"><label>电子剂量 / Dose</label><select id="dose"><option value="10000">10⁴ e⁻</option><option value="100000">10⁵ e⁻</option><option value="1000000" selected>10⁶ e⁻</option></select></div>
 <div class="control counted"><label>随机重复 / Repeat</label><select id="repeat"><option>0</option><option>1</option><option>2</option><option>3</option><option>4</option></select></div>
 <div class="control"><label>检峰器 / Detector</label><select id="detector"><option value="py4dstem">py4DSTEM find_Bragg_disks</option><option value="autodisk">AutoDisk</option></select></div>
 <div class="control"><label>图层 / Overlay</label><div class="seg"><button id="oracle-toggle" class="active">Oracle + HKL</button><button id="detected-toggle" class="active">Detected</button></div></div>
</div>
<div class="diag">
 <section>
  <div class="image-panel"><img id="pattern" alt="diffraction pattern"><svg id="overlay" viewBox="0 0 512 512"></svg></div>
  <div class="legend"><span><i class="dot"></i>物理 oracle + HKL</span><span><b class="cross">×</b>当前检峰</span><span id="seed-label" class="mini"></span></div>
 </section>
 <section class="panel">
  <h3 id="selection-title"></h3>
  <div class="metrics" id="sample-metrics"></div>
  <div class="matrix-grid">
   <div class="matrix"><b>Ground Truth</b><span class="mini">R<sub>sample→crystal</sub></span><pre id="gt-matrix"></pre></div>
   <div class="matrix"><b>Oracle-ACOM</b><span class="mini">R<sub>sample→crystal</sub></span><pre id="oracle-matrix"></pre></div>
   <div class="matrix"><b>当前 ACOM / Selected</b><span class="mini">R<sub>sample→crystal</sub></span><pre id="selected-matrix"></pre></div>
  </div>
  <p class="mini">三者均为同一语义的样品坐标→晶体坐标旋转矩阵。图像检峰误差与 ACOM 相对 GT 的取向误差分开报告。</p>
 </section>
</div>
<section class="panel" style="margin-top:14px"><h3>物理 Oracle 逐反射表 / All oracle reflections</h3><p class="mini">HKL 是倒易晶格反射索引，不是“样品只有一个晶面”。每个可见衍射盘对应一个满足当前探测范围和激发条件的 (h,k,l)。</p><div class="peak-table-wrap"><table><thead><tr><th>#</th><th>HKL</th><th>q<sub>x</sub> (Å⁻¹)</th><th>q<sub>y</sub> (Å⁻¹)</th><th>强度 / Intensity</th></tr></thead><tbody id="peak-rows"></tbody></table></div></section>

<h2>坐标与变量 / Coordinates and variables</h2>
<section class="panel">
 <div class="formula">B = [a*; b*; c*] (Å⁻¹, without 2π)<br>g<sub>crystal</sub> = [h k l] B<br>g<sub>sample</sub> = R<sub>sample→crystal</sub><sup>T</sup> g<sub>crystal</sub><sup>T</sup><br>[q<sub>x</sub>, q<sub>y</sub>] = detector-plane components of g<sub>sample</sub></div>
 <div class="matrix" style="margin-top:10px;max-width:520px"><b>B 倒易基矩阵 / Reciprocal basis</b><pre id="b-matrix"></pre></div>
 <p class="mini">A 的行是实空间基矢 a、b、c（Å）；B 的行是倒易基矢 a*、b*、c*（Å⁻¹）。本 benchmark 的 B 不含 2π。图像像素坐标仅用于检测；转换回 q 后才进入 ACOM。</p>
</section>

<h2>运行参数 / Runtime parameters</h2>
<details><summary>显示/隐藏中英文参数 · Show/hide bilingual parameters</summary><div id="parameter-table" style="margin-top:10px"></div></details>
<details><summary>结果来源与限制 / Provenance and limitation</summary><p>网页由 <code>scripts/17_write_clean_image_visualization.py</code> 从本地 HDF5 和 JSON 结果重新生成。汇总图使用全部 1081 个图样；headline 取向指标统计 1024 个 headline_core。Clean 图像与 ACOM 模板共享同一 CIF 和匹配的运动学模型，因此这里验证的是端到端自洽性，不代表真实实验数据或跨模拟器泛化能力。</p></details>
</main>
<script>
const DATA = __DATA__;
const $ = id => document.getElementById(id);
let state={sample:DATA.sample_order[0],track:"expectation",dose:1000000,repeat:0,detector:"py4dstem",oracle:true,detected:true};
const fmt=(x,n=3)=>x==null?"—":Number(x).toFixed(n);
const pct=x=>fmt(100*x,2)+"%";
const matrix=m=>m.map(r=>"["+r.map(x=>(x>=0?" ":"")+Number(x).toFixed(5)).join(", ")+"]").join("\n");
function variantKey(){return state.track==="expectation"?`${state.detector}_expectation`:`counted_dose${state.dose}_repeat${state.repeat}_${state.detector}`}
function svgChart(id,rows,field,yLabel,percent=false){
 const svg=$(id),W=620,H=280,L=52,R=18,T=22,B=42, xs=[10000,100000,1000000];
 const vals=rows.map(r=>r[field]); let ymin=Math.min(...vals),ymax=Math.max(...vals); if(percent){ymin=Math.max(0,ymin-.05);ymax=Math.min(1,ymax+.03)} else {ymin*=.92;ymax*=1.08}
 const x=i=>L+i*(W-L-R)/2, y=v=>T+(ymax-v)/(ymax-ymin||1)*(H-T-B);
 let h=""; for(let i=0;i<5;i++){let v=ymin+(ymax-ymin)*i/4,yy=y(v);h+=`<line class="grid" x1="${L}" y1="${yy}" x2="${W-R}" y2="${yy}"/><text x="${L-7}" y="${yy+4}" text-anchor="end">${percent?(v*100).toFixed(0)+"%":v.toFixed(2)}</text>`}
 h+=`<line class="axis" x1="${L}" y1="${H-B}" x2="${W-R}" y2="${H-B}"/><text x="12" y="15">${yLabel}</text>`;
 xs.forEach((v,i)=>h+=`<text x="${x(i)}" y="${H-17}" text-anchor="middle">10${["⁴","⁵","⁶"][i]} e⁻</text>`);
 ["autodisk","py4dstem"].forEach((det,di)=>{let rs=xs.map(d=>rows.find(r=>r.detector===det&&r.dose_electrons===d));let cls=det==="autodisk"?"auto":"py";h+=`<polyline class="${cls}" points="${rs.map((r,i)=>x(i)+","+y(r[field])).join(" ")}"/>`;rs.forEach((r,i)=>h+=`<circle class="${cls}" cx="${x(i)}" cy="${y(r[field])}" r="4"/>`)});
 h+=`<line class="auto" x1="405" y1="13" x2="430" y2="13"/><text x="436" y="17">AutoDisk</text><line class="py" x1="510" y1="13" x2="535" y2="13"/><text x="541" y="17">find_Bragg_disks</text>`;svg.innerHTML=h;
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
}
function setButton(id,on){$(id).classList.toggle("active",on)}
function redraw(){
 const s=DATA.samples[state.sample],v=DATA.variants[variantKey()],d=v.details[state.sample],m=v.sample_peak_metrics[state.sample];
 $("pattern").src=state.track==="expectation"?s.expectation_image:s.counted_images[`${state.dose}:${state.repeat}`];
 $("selection-title").textContent=`${s.label} · ${state.sample} · ${state.track==="expectation"?"Clean-E":`Clean-C ${state.dose.toLocaleString()} e⁻ / repeat ${state.repeat}`} · ${state.detector==="py4dstem"?"find_Bragg_disks":"AutoDisk"}`;
 $("sample-metrics").innerHTML=[
  ["Peak P/R",pct(m.precision)+" / "+pct(m.recall)],
  ["位置 RMSE",fmt(m.rmse_px,3)+" px"],
  ["ACOM → GT",fmt(d.gt_error_deg,3)+"°"],
  ["ACOM → Oracle",fmt(d.delta_oracle_deg,3)+"°"],
  ["峰数 / Peaks",`${m.detected} / ${m.oracle}`],
  ["Correlation",fmt(d.correlation,4)]
 ].map(x=>`<div class="metric"><span>${x[0]}</span><b>${x[1]}</b></div>`).join("");
 $("gt-matrix").textContent=matrix(s.gt_matrix);$("oracle-matrix").textContent=matrix(s.oracle_matrix);$("selected-matrix").textContent=matrix(d.matrix);
 $("seed-label").textContent=state.track==="counted"?`RNG seed: ${s.seeds[`${state.dose}:${state.repeat}`]}`:"确定性期望强度图 / deterministic expectation";
 const [xmin,xmax,ymin,ymax]=DATA.q_bounds,px=q=>512*(q-xmin)/(xmax-xmin),py=q=>512*(ymax-q)/(ymax-ymin);
 let svg=`<line x1="256" y1="0" x2="256" y2="512" stroke="#ffffff33"/><line x1="0" y1="256" x2="512" y2="256" stroke="#ffffff33"/>`;
 if(state.oracle)s.oracle_peaks.forEach((p,i)=>{let show=i<12;svg+=`<circle cx="${px(p.qx)}" cy="${py(p.qy)}" r="${DATA.disk_radius_px+2}" fill="none" stroke="#58a6ff" stroke-width="1.5"/>${show?`<text x="${px(p.qx)+6}" y="${py(p.qy)-6}" fill="#d9ecff" font-size="9">[${p.hkl.join(" ")}]</text>`:""}`});
 if(state.detected)v.peaks[state.sample].forEach(p=>{let x=px(p.qx),y=py(p.qy);svg+=`<path d="M${x-4},${y-4}L${x+4},${y+4}M${x+4},${y-4}L${x-4},${y+4}" stroke="#ff8b5e" stroke-width="1.8"/>`});
 $("overlay").innerHTML=svg;
 $("peak-rows").innerHTML=s.oracle_peaks.map((p,i)=>`<tr><td>${i+1}</td><td>[${p.hkl.join(", ")}]</td><td>${fmt(p.qx,5)}</td><td>${fmt(p.qy,5)}</td><td>${fmt(p.intensity,5)}</td></tr>`).join("");
 document.querySelectorAll(".counted").forEach(e=>e.classList.toggle("hidden",state.track!=="counted"));
 setButton("track-e",state.track==="expectation");setButton("track-c",state.track==="counted");setButton("oracle-toggle",state.oracle);setButton("detected-toggle",state.detected);
}
function init(){
 initOverview();$("b-matrix").textContent=matrix(DATA.reciprocal_matrix_B);
 $("sample").innerHTML=DATA.sample_order.map(id=>`<option value="${id}">${DATA.samples[id].label} · ${id}</option>`).join("");
 $("parameter-table").innerHTML=`<table><thead><tr><th>参数 / Parameter</th><th>Code name</th><th>Value</th></tr></thead><tbody>${DATA.parameters.map(r=>`<tr><td>${r[0]}</td><td><code>${r[1]}</code></td><td>${r[2]}</td></tr>`).join("")}</tbody></table>`;
 $("sample").onchange=e=>{state.sample=e.target.value;redraw()};$("dose").onchange=e=>{state.dose=+e.target.value;redraw()};$("repeat").onchange=e=>{state.repeat=+e.target.value;redraw()};$("detector").onchange=e=>{state.detector=e.target.value;redraw()};
 $("track-e").onclick=()=>{state.track="expectation";redraw()};$("track-c").onclick=()=>{state.track="counted";redraw()};$("oracle-toggle").onclick=()=>{state.oracle=!state.oracle;redraw()};$("detected-toggle").onclick=()=>{state.detected=!state.detected;redraw()};redraw();
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
