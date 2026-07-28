#!/usr/bin/env python3
from __future__ import annotations

import gzip
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRACE_PATH = ROOT / "diagnostics" / "clean_coordinate_trace.jsonl.gz"
DETAILS_PATH = ROOT / "reports" / "acom_clean_details.json"
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


def compact_trace(row: dict, label: str) -> dict:
    observed_ranked = sorted(
        enumerate(row["standard_observed_reflections"]),
        key=lambda item: item[1]["reported_intensity_normalized"],
        reverse=True,
    )[:12]
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
  color-scheme: light dark;
  --background: #ffffff;
  --foreground: #172033;
  --muted: #eef1f6;
  --muted-foreground: #627087;
  --border: #c9d0dc;
  --series-1: #2774d8;
  --series-2: #d45c37;
  --series-3: #7b52ab;
}
@media (prefers-color-scheme: dark) {
  :root {
    --background: #111722;
    --foreground: #edf2fa;
    --muted: #202938;
    --muted-foreground: #aab5c6;
    --border: #3b4658;
    --series-1: #62a8ff;
    --series-2: #ff8a65;
    --series-3: #c398ff;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--background);
  color: var(--foreground);
  font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
main { max-width: 1280px; margin: 0 auto; padding: 24px; }
h1 { margin: 0 0 6px; font-size: 24px; font-weight: 600; }
.subtitle { margin: 0 0 20px; color: var(--muted-foreground); }
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
.plots { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 20px; }
.plot-head { display: flex; justify-content: space-between; gap: 12px; flex-wrap: wrap; min-height: 30px; font-weight: 600; }
.legend { color: var(--muted-foreground); font-weight: 400; }
.dot { width: 10px; height: 10px; border: 2px solid var(--series-1); border-radius: 50%; }
.cross { color: var(--series-2); font-size: 20px; line-height: 1; }
.line { width: 20px; border-top: 2px solid var(--series-1); }
.line.acom { border-top-color: var(--series-2); border-top-style: dashed; }
svg { display: block; width: 100%; height: auto; color: var(--foreground); }
.reflection-control { margin-top: 14px; }
.flow {
  margin-top: 10px; padding: 12px 0; overflow-wrap: anywhere;
  border-top: 1px solid var(--border); color: var(--foreground);
}
.flow .arrow, .flow .dim { color: var(--muted-foreground); }
.matrix-wrap { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 18px; margin-top: 18px; }
.matrix-wrap h2 { font-size: 15px; margin: 0 0 6px; }
pre { margin: 0; padding: 11px; border: 1px solid var(--border); border-radius: 8px; background: var(--muted); overflow-x: auto; }
@media (max-width: 720px) {
  main { padding: 16px; }
  .plots, .matrix-wrap, .metrics { grid-template-columns: 1fr; }
}
</style>
</head>
<body>
<main>
  <h1>Clean v3 坐标与倒空间中间变量</h1>
  <p class="subtitle">直接打开本文件即可使用；数据内嵌，不依赖网络或 Codex。</p>
  <div class="controls">
    <strong>代表样本</strong>
    <div id="tabs" class="tabs"></div>
  </div>
  <div class="metrics">
    <div class="metric"><span>取向误差</span><strong id="error"></strong></div>
    <div class="metric"><span>观测峰匹配率</span><strong id="match"></strong></div>
    <div class="metric"><span>q-RMSE</span><strong id="rmse"></strong></div>
  </div>
  <div class="plots">
    <section>
      <div class="plot-head">
        <span>探测器倒空间</span>
        <span class="legend"><i class="dot"></i>标准 <i class="cross">×</i>ACOM</span>
      </div>
      <svg id="detector" viewBox="0 0 560 440" role="img" aria-label="标准与 ACOM 衍射峰"></svg>
    </section>
    <section>
      <div class="plot-head">
        <span>样品坐标轴在晶体 XY 平面的投影</span>
        <span class="legend"><i class="line"></i>标准 <i class="line acom"></i>ACOM</span>
      </div>
      <svg id="axes" viewBox="0 0 560 440" role="img" aria-label="标准与 ACOM 样品坐标轴"></svg>
    </section>
  </div>
  <div class="reflection-control">
    <label for="reflection"><strong>高亮标准反射</strong></label>
    <select id="reflection"></select>
  </div>
  <div id="flow" class="flow"></div>
  <div class="matrix-wrap">
    <section><h2>标准 R_sample_to_crystal</h2><pre id="standard-matrix"></pre></section>
    <section><h2>ACOM R_sample_to_crystal</h2><pre id="acom-matrix"></pre></section>
  </div>
</main>
<script>
const samples = __DATA__;
const tabs = document.getElementById("tabs");
const detector = document.getElementById("detector");
const axes = document.getElementById("axes");
const reflection = document.getElementById("reflection");
let sampleIndex = 1;
let reflectionIndex = 0;
const esc = value => String(value).replace(/[&<>"']/g, character => {
  if (character === "&") return "&amp;";
  if (character === "<") return "&lt;";
  if (character === ">") return "&gt;";
  if (character === "'") return "&#39;";
  return "&quot;";
});
const vector = values => `[${values.map(value => Number(value).toFixed(4)).join(", ")}]`;
const qScale = value => 280 + value * 125;
const axisScale = value => value * 135;

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

function renderDetector(sample) {
  let markup = `
    <line x1="80" y1="220" x2="500" y2="220" stroke="var(--border)"/>
    <line x1="280" y1="20" x2="280" y2="420" stroke="var(--border)"/>
    <circle cx="280" cy="220" r="187.5" fill="none" stroke="var(--border)"/>
    <text x="495" y="242" text-anchor="end" fill="var(--muted-foreground)">qₓ (Å⁻¹)</text>
    <text x="290" y="34" fill="var(--muted-foreground)">qᵧ (Å⁻¹)</text>`;
  sample.matches.forEach(match => {
    const observed = sample.observed[match.observed];
    const predicted = sample.predicted[match.predicted];
    markup += `<line x1="${qScale(observed.q[0])}" y1="${qScale(-observed.q[1])}" x2="${qScale(predicted.q[0])}" y2="${qScale(-predicted.q[1])}" stroke="var(--border)"/>`;
  });
  sample.observed.forEach((peak, index) => {
    const radius = 4 + 7 * Math.sqrt(peak.intensity);
    const selected = index === reflectionIndex;
    markup += `<circle cx="${qScale(peak.q[0])}" cy="${qScale(-peak.q[1])}" r="${selected ? radius + 3 : radius}" fill="none" stroke="${selected ? "var(--series-3)" : "var(--series-1)"}" stroke-width="${selected ? 3 : 1.8}"><title>标准 HKL ${peak.hkl.join(",")} · I=${peak.intensity}</title></circle>`;
  });
  sample.predicted.forEach(peak => {
    const x = qScale(peak.q[0]);
    const y = qScale(-peak.q[1]);
    const size = 4 + 6 * Math.sqrt(peak.intensity);
    markup += `<path d="M ${x-size} ${y-size} L ${x+size} ${y+size} M ${x-size} ${y+size} L ${x+size} ${y-size}" stroke="var(--series-2)" stroke-width="2"><title>ACOM HKL ${peak.hkl.join(",")} · I=${peak.intensity}</title></path>`;
  });
  markup += `<text x="88" y="410" fill="var(--muted-foreground)">按强度显示前 12 个峰 · 高亮 HKL [${esc(sample.observed[reflectionIndex].hkl.join(", "))}]</text>`;
  detector.innerHTML = markup;
}

function renderAxes(sample) {
  const centerX = 280;
  const centerY = 210;
  const names = ["x", "y", "z beam"];
  let markup = `
    <circle cx="${centerX}" cy="${centerY}" r="140" fill="none" stroke="var(--border)"/>
    <line x1="80" y1="${centerY}" x2="480" y2="${centerY}" stroke="var(--border)"/>
    <line x1="${centerX}" y1="20" x2="${centerX}" y2="400" stroke="var(--border)"/>
    <text x="475" y="${centerY + 22}" text-anchor="end" fill="var(--muted-foreground)">crystal X</text>
    <text x="${centerX + 10}" y="34" fill="var(--muted-foreground)">crystal Y</text>`;
  [sample.standard_matrix, sample.acom_matrix].forEach((matrix, matrixIndex) => {
    names.forEach((name, column) => {
      const x = centerX + axisScale(matrix[0][column]);
      const y = centerY - axisScale(matrix[1][column]);
      const color = matrixIndex === 0 ? "var(--series-1)" : "var(--series-2)";
      const dash = matrixIndex === 0 ? "" : `stroke-dasharray="7 5"`;
      markup += `<line x1="${centerX}" y1="${centerY}" x2="${x}" y2="${y}" stroke="${color}" stroke-width="2.5" ${dash}/><circle cx="${x}" cy="${y}" r="3.5" fill="${color}"/><text x="${x+6}" y="${y-6}" fill="var(--foreground)">${matrixIndex === 0 ? "S" : "A"}:${name}</text>`;
    });
  });
  const selected = sample.observed[reflectionIndex];
  const norm = Math.hypot(selected.g_crystal[0], selected.g_crystal[1]) || 1;
  const gx = centerX + 105 * selected.g_crystal[0] / norm;
  const gy = centerY - 105 * selected.g_crystal[1] / norm;
  markup += `<line x1="${centerX}" y1="${centerY}" x2="${gx}" y2="${gy}" stroke="var(--series-3)" stroke-width="4"/><circle cx="${gx}" cy="${gy}" r="4" fill="var(--series-3)"/><text x="88" y="418" fill="var(--muted-foreground)">紫色为所选 g_crystal 方向 · S=标准，A=ACOM</text>`;
  axes.innerHTML = markup;
}

function renderFlow(sample) {
  const selected = sample.observed[reflectionIndex];
  document.getElementById("flow").innerHTML =
    `<strong>HKL [${esc(selected.hkl.join(", "))}]</strong> <span class="arrow">→</span> ` +
    `g<sub>crystal</sub> ${esc(vector(selected.g_crystal))} Å⁻¹ <span class="arrow">→ Rᵀ →</span> ` +
    `q<sub>standard</sub> ${esc(vector(selected.q_standard))} Å⁻¹ <span class="dim">对比</span> ` +
    `q<sub>ACOM,same HKL</sub> ${esc(vector(selected.q_acom_same_hkl))} Å⁻¹`;
}

function render() {
  const sample = samples[sampleIndex];
  renderTabs();
  document.getElementById("error").textContent = `${sample.orientation_error_deg.toFixed(3)}°`;
  document.getElementById("match").textContent = `${(sample.observed_match_fraction * 100).toFixed(1)}%`;
  document.getElementById("rmse").textContent = `${sample.q_rmse_Ainv.toFixed(4)} Å⁻¹`;
  reflection.innerHTML = sample.observed.map((peak, index) =>
    `<option value="${index}" ${index === reflectionIndex ? "selected" : ""}>HKL [${esc(peak.hkl.join(", "))}] · I=${peak.intensity.toFixed(3)}</option>`
  ).join("");
  document.getElementById("standard-matrix").textContent = sample.standard_matrix.map(vector).join("\\n");
  document.getElementById("acom-matrix").textContent = sample.acom_matrix.map(vector).join("\\n");
  renderDetector(sample);
  renderAxes(sample);
  renderFlow(sample);
}

reflection.addEventListener("change", event => {
  reflectionIndex = Number(event.target.value);
  render();
});
render();
</script>
</body>
</html>
"""


def main() -> None:
    samples = load_samples()
    payload = json.dumps(samples, ensure_ascii=False, separators=(",", ":"))
    output = HTML_TEMPLATE.replace("__DATA__", payload)
    OUTPUT_PATH.write_text(output, encoding="utf-8")
    print(f"Standalone coordinate visualization: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
