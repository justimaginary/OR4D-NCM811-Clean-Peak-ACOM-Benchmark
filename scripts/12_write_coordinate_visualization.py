#!/usr/bin/env python3
from __future__ import annotations

import gzip
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "benchmark.yaml"
TRACE_PATH = ROOT / "diagnostics" / "clean_coordinate_trace.jsonl.gz"
DETAILS_PATH = ROOT / "reports" / "acom_clean_details.json"
PLAN_AUDIT_PATH = ROOT / "reports" / "acom_plan_audit.json"
OUTPUT_PATH = ROOT / "reports" / "ACOM_COORDINATE_VISUALIZATION.html"


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
    return [rows[position]["sample_id"] for position in positions]


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
        ["数据集", str(dataset["id"]), "本次结果对应的数据版本"],
        [
            "样本组成",
            (
                f"{sum(role_counts.values())} = "
                f"{role_counts['legacy_smoke']} legacy + "
                f"{role_counts['headline_core']} headline + "
                f"{role_counts['acom_grid_probe']} probe"
            ),
            "headline 指标只统计 headline_core",
        ],
        [
            "headline 取向采样",
            f"{sampling['method']}; seed={sampling['seed']}; scramble={sampling['scramble']}",
            "独立于 ACOM 搜索网格的 SO(3) 采样",
        ],
        [
            "加速电压",
            f"{float(common['accelerating_voltage_V']) / 1000:g} kV",
            "电子波长与衍射几何",
        ],
        ["Kmax", f"{common['k_max_Ainv']} Å⁻¹", "保留的最大探测器倒空间半径"],
        [
            "中心束排除半径",
            f"{common['central_beam_exclusion_Ainv']} Å⁻¹",
            "去掉透射中心束附近峰",
        ],
        [
            "激发误差 σ",
            f"{clean['sigma_excitation_error_Ainv']} Å⁻¹",
            "衍射峰沿 Ewald 球偏离的权重尺度",
        ],
        [
            "激发误差截断",
            (
                f"{clean['tol_excitation_error_mult']}σ = "
                f"{float(clean['tol_excitation_error_mult']) * float(clean['sigma_excitation_error_Ainv']):g} Å⁻¹"
            ),
            "超出该范围的反射不进入图样",
        ],
        ["结构因子阈值", str(clean["tol_structure_factor"]), "生成候选反射时过滤弱结构因子"],
        ["峰强度阈值", str(clean["tol_intensity"]), "生成图样时过滤弱峰"],
        ["强度归一化", str(common["normalize_peak_intensity"]), "每个样本以最强峰归一化为 1"],
    ]
    acom_rows = [
        ["py4DSTEM", str(audit["py4DSTEM_version"]), "ACOM 实现版本"],
        ["zone-axis 范围", str(acom["zone_axis_range"]), "取向搜索覆盖范围"],
        [
            "标准运行角步长",
            f"zone={acom['angle_step_zone_axis_deg']}°; in-plane={acom['angle_step_in_plane_deg']}°",
            "当前报告的 canonical ACOM 网格",
        ],
        [
            "对照扫描角步长",
            ", ".join(f"{value}°" for value in acom["sweep_angle_steps_deg"]),
            "同一方法分别运行，不混用检测规则",
        ],
        [
            "实际搜索网格",
            (
                f"{plan['num_zone_axes']} zone axes × "
                f"{plan['num_in_plane_steps']} in-plane; "
                f"{plan['num_discrete_seeds_including_mirror']} seeds"
            ),
            "由 canonical 2° 参数生成",
        ],
        ["相关核半径", f"{acom['corr_kernel_size_Ainv']} Å⁻¹", "峰位置相关的空间尺度"],
        ["ACOM 激发误差 σ", f"{acom['sigma_excitation_error_Ainv']} Å⁻¹", "模拟模板中的激发误差尺度"],
        ["径向权重指数", str(acom["power_radial"]), "相关评分中的径向权重"],
        [
            "强度权重指数",
            f"sim={acom['power_intensity_simulated']}; exp={acom['power_intensity_experiment']}",
            "模拟峰与观测峰的强度加权",
        ],
        ["ACOM 峰距离容差", f"{acom['tol_distance_Ainv']} Å⁻¹", "orientation_plan 内部峰相关容差"],
        ["最少峰数", str(acom["min_number_peaks"]), "少于该数量不做可靠取向匹配"],
        ["反演对称", str(acom["inversion_symmetry"]), "搜索时包含 Friedel / mirror 等价"],
        ["返回候选数", str(acom["num_matches_return"]), "每个样本保留的 ACOM 取向候选"],
        ["CUDA", str(acom["cuda"]), "本次运行是否使用 GPU"],
        [
            "评估峰匹配容差",
            f"{evaluation['coordinate_match_tolerance_Ainv']} Å⁻¹",
            "GT 与 ACOM 峰在探测器二维 q 空间的一对一匹配阈值",
        ],
    ]
    return {"benchmark": benchmark, "acom": acom_rows}


def compact_trace(row: dict, label: str) -> dict:
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
            "q_acom_same_hkl": reflection["acom_same_hkl_g_sample_Ainv"],
        }
        for _, reflection in observed_ranked
    ]
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
            "orientation_error_deg": row["acom_result"][
                "friedel_equivalent_misorientation_deg"
            ],
            "observed_match_fraction": row["comparison_summary"][
                "observed_match_fraction"
            ],
            "q_rmse_Ainv": row["comparison_summary"]["q_distance_rmse_Ainv"],
            "standard_matrix": row[
                "standard_orientation_matrix_sample_to_crystal"
            ],
            "acom_matrix": row["acom_orientation_matrix_sample_to_crystal"],
            "reciprocal_matrix": row["reciprocal_lattice_matrix_B_Ainv"],
            "observed": observed,
            "predicted": predicted,
            "matches": matches,
        }
    )


def load_samples() -> list[dict]:
    selected_ids = representative_ids()
    labels = dict(zip(selected_ids, ("Best", "Median", "P95", "Worst")))
    selected: dict[str, dict] = {}
    with gzip.open(TRACE_PATH, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            sample_id = row["sample_id"]
            if sample_id in labels:
                selected[sample_id] = compact_trace(row, labels[sample_id])
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
.parameters > h2 { margin: 0 0 9px; font-size: 17px; }
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
  .plots, .matrix-wrap, .metrics, .equation-grid, .matrix-equation, .process-grid, .parameter-grid { grid-template-columns: 1fr; }
  .operator { display: none; }
}
@media print {
  @page { size: landscape; margin: 10mm; }
  body { color: #000000; background: #ffffff; }
  main { max-width: none; padding: 0; }
  button, select, pre, th { color: #000000; background: #ffffff; }
  .process-grid, .parameters, .notation, .plots, .transform, .all-reflections, .matrix-wrap { break-inside: avoid; }
  .all-reflections { break-before: page; }
}
</style>
</head>
<body>
<main>
  <h1>Clean v3 坐标与倒空间中间变量</h1>
  <div class="controls">
    <strong>代表样本</strong>
    <div id="tabs" class="tabs"></div>
  </div>
  <div class="metrics">
    <div class="metric"><span>取向误差：R<sup>ACOM</sup> 对 R<sup>GT</sup></span><strong id="error"></strong></div>
    <div class="metric"><span>探测器观测峰匹配率</span><strong id="match"></strong></div>
    <div class="metric"><span>探测器峰匹配 q-RMSE（二维）</span><strong id="rmse"></strong></div>
  </div>
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
  <section class="parameters">
    <h2>本次运行参数（从 config/benchmark.yaml 与实际 ACOM plan 读取）</h2>
    <div class="parameter-grid">
      <article>
        <h3>过程 A｜Benchmark 生成参数</h3>
        <div class="table-wrap">
          <table><thead><tr><th>参数</th><th>本次取值</th><th>作用</th></tr></thead><tbody id="benchmark-parameters"></tbody></table>
        </div>
      </article>
      <article>
        <h3>过程 B｜ACOM 预测与评估参数</h3>
        <div class="table-wrap">
          <table><thead><tr><th>参数</th><th>本次取值</th><th>作用</th></tr></thead><tbody id="acom-parameters"></tbody></table>
        </div>
      </article>
    </div>
  </section>
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
      <h3>④ 诊断：g<sub>s</sub><sup>ACOM|same h</sup> = g<sub>c</sub> R<sup>ACOM</sup></h3>
      <div class="matrix-equation"><pre id="acom-matrix"></pre><div class="operator">→</div><pre id="acom-q"></pre></div>
      <p class="equation-note">固定同一个 GT HKL 后的坐标差；仅用于解释误差，不是 ACOM 的输入，也不等同于 ACOM 最近邻匹配峰。</p>
    </section>
  </div>
  <section class="all-reflections">
    <h2 id="all-reflections-title"></h2>
    <div class="table-wrap">
      <table>
        <thead><tr><th>#</th><th>GT HKL</th><th>I<sup>GT</sup></th><th>g<sub>c</sub> (Å⁻¹)</th><th>g<sub>s</sub><sup>GT</sup> = [qₓ,qᵧ,qz] (Å⁻¹)</th><th>g<sub>s</sub><sup>ACOM|same h</sup> (Å⁻¹)</th><th>‖Δg<sub>s</sub>‖₂ (Å⁻¹)</th></tr></thead>
        <tbody id="reflection-rows"></tbody>
      </table>
    </div>
  </section>
</main>
<script>
const reportData = __DATA__;
const samples = reportData.samples;
const runtimeParameters = reportData.runtime_parameters;
const kMaxAinv = Number(reportData.k_max_Ainv);
const tabs = document.getElementById("tabs");
const viewTabs = document.getElementById("view-tabs");
const detector = document.getElementById("detector");
const axes = document.getElementById("axes");
const reflection = document.getElementById("reflection");
let sampleIndex = 1;
let reflectionIndex = 0;
let displayMode = "overlay";
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
    `${showACOM ? '<i class="line acom"></i>R<sup>ACOM</sup>' : ''}`;
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
    {matrix: sample.acom_matrix, key: "acom"},
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
        ? "仅显示 R<sup>ACOM</sup>；虚线表示算法估计的样品坐标轴"
        : "叠加 R<sup>GT</sup>（实线）与 R<sup>ACOM</sup>（虚线）；紫色为所选 GT 反射的 g<sub>c</sub> 方向";
}

function renderTransform(sample) {
  const selected = sample.observed[reflectionIndex];
  document.getElementById("hkl-vector").textContent = `[${selected.hkl.join(", ")}]`;
  document.getElementById("reciprocal-matrix").textContent = matrix(sample.reciprocal_matrix);
  document.getElementById("g-vector").textContent = vector(selected.g_crystal);
  document.getElementById("standard-matrix").textContent = matrix(sample.standard_matrix);
  document.getElementById("acom-matrix").textContent = matrix(sample.acom_matrix);
  document.getElementById("standard-q").textContent = vector(selected.q_standard);
  document.getElementById("acom-q").textContent = vector(selected.q_acom_same_hkl);
}

function renderReflectionTable(sample) {
  document.getElementById("all-reflections-title").innerHTML =
    `当前 GT 图样中的 ${sample.observed.length} 个有效反射（Kmax = ${kMaxAinv} Å⁻¹）`;
  document.getElementById("reflection-rows").innerHTML = sample.observed.map((peak, index) => {
    const delta = Math.hypot(
      peak.q_standard[0] - peak.q_acom_same_hkl[0],
      peak.q_standard[1] - peak.q_acom_same_hkl[1],
      peak.q_standard[2] - peak.q_acom_same_hkl[2]
    );
    return `<tr class="${index === reflectionIndex ? "is-selected" : ""}">` +
      `<td>${index + 1}</td>` +
      `<td>[${esc(peak.hkl.join(", "))}]</td>` +
      `<td>${peak.intensity.toFixed(4)}</td>` +
      `<td>${esc(vector(peak.g_crystal))}</td>` +
      `<td>${esc(vector(peak.q_standard))}</td>` +
      `<td>${esc(vector(peak.q_acom_same_hkl))}</td>` +
      `<td>${delta.toFixed(5)}</td></tr>`;
  }).join("");
}

function render() {
  const sample = samples[sampleIndex];
  renderTabs();
  renderViewTabs();
  document.getElementById("error").textContent = `${sample.orientation_error_deg.toFixed(3)}°`;
  document.getElementById("match").textContent = `${(sample.observed_match_fraction * 100).toFixed(1)}%`;
  document.getElementById("rmse").textContent = `${sample.q_rmse_Ainv.toFixed(4)} Å⁻¹`;
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
