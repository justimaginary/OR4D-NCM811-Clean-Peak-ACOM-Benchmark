#!/usr/bin/env python3
"""Build a dependency-free local HTML report from the compact [001] audit."""

from __future__ import annotations

import argparse
import html
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=ROOT / "reports" / "v5" / "study_001" / "acom_audit",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT
            / "reports"
            / "v5"
            / "study_001"
            / "acom_audit"
            / "ACOM_001_CORRELATION_AUDIT.html"
        ),
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def median(values) -> float:
    return float(statistics.median(values))


def pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


COLORS = {
    "physical_oracle": "#2563eb",
    "physical_uniform_intensity": "#9333ea",
    "acom_exact_gt": "#ea580c",
    "acom_discrete_seed": "#15803d",
}

LABELS = {
    "physical_oracle": "V5 physical oracle",
    "physical_uniform_intensity": "Physical positions + uniform intensity",
    "acom_exact_gt": "ACOM peaks at exact GT",
    "acom_discrete_seed": "ACOM peaks at nearest plan seed",
}


def svg_line_chart(
    series: dict[str, list[tuple[float, float]]],
    *,
    title: str,
    y_label: str,
    y_min: float,
    y_max: float,
    percent: bool,
    log_y: bool = False,
) -> str:
    width, height = 760, 430
    left, right, top = 82, 24, 52
    bottom = 112 if len(series) > 2 else 82
    plot_w = width - left - right
    plot_h = height - top - bottom
    all_x = sorted({x for values in series.values() for x, _ in values})
    x_min, x_max = min(all_x), max(all_x)

    def xp(value: float) -> float:
        return left + (value - x_min) / max(x_max - x_min, 1e-12) * plot_w

    def transform(value: float) -> float:
        if log_y:
            import math

            return math.log10(max(value, 1.0))
        return value

    ty_min, ty_max = transform(y_min), transform(y_max)

    def yp(value: float) -> float:
        return top + (ty_max - transform(value)) / max(ty_max - ty_min, 1e-12) * plot_h

    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">',
        f'<text x="{left}" y="28" class="chart-title">{html.escape(title)}</text>',
    ]
    tick_values = (
        [1, 10, 100, 1000]
        if log_y
        else [y_min + (y_max - y_min) * index / 4.0 for index in range(5)]
    )
    for value in tick_values:
        if value < y_min or value > y_max:
            continue
        y = yp(value)
        label = pct(value) if percent else f"{value:g}"
        parts.extend(
            [
                f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" class="grid"/>',
                f'<text x="{left-12}" y="{y+5:.1f}" text-anchor="end" class="tick">{label}</text>',
            ]
        )
    for value in all_x:
        x = xp(value)
        parts.extend(
            [
                f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{height-bottom}" class="grid vertical"/>',
                f'<text x="{x:.1f}" y="{height-bottom+26}" text-anchor="middle" class="tick">{value:g}°</text>',
            ]
        )
    parts.append(
        f'<text transform="translate(20 {top+plot_h/2}) rotate(-90)" text-anchor="middle" class="axis-label">{html.escape(y_label)}</text>'
    )
    for key, values in series.items():
        points = " ".join(f"{xp(x):.1f},{yp(y):.1f}" for x, y in values)
        color = COLORS.get(key, "#334155")
        parts.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3"/>'
        )
        for x_value, y_value in values:
            x, y = xp(x_value), yp(y_value)
            label = pct(y_value) if percent else f"{y_value:g}"
            parts.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{color}"/>'
            )
            if len(series) <= 2:
                parts.append(
                    f'<text x="{x:.1f}" y="{y-10:.1f}" text-anchor="middle" class="value">{label}</text>'
                )
    for legend_index, key in enumerate(series):
        color = COLORS.get(key, "#334155")
        label = LABELS.get(key, key)
        if len(series) > 2:
            legend_x = left + (legend_index % 2) * 330
            legend_y = height - 52 + (legend_index // 2) * 24
        else:
            legend_x = left + legend_index * 290
            legend_y = height - 25
        parts.extend(
            [
                f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x+24}" y2="{legend_y}" stroke="{color}" stroke-width="4"/>',
                f'<text x="{legend_x+31}" y="{legend_y+5}" class="legend">{html.escape(label)}</text>',
            ]
        )
    parts.append("</svg>")
    return "".join(parts)


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    summary = json.loads(
        (input_dir / "acom_001_audit_summary.json").read_text(encoding="utf-8")
    )
    rows = load_jsonl(input_dir / "acom_001_correlation_audit.jsonl")
    diagnostics = load_jsonl(input_dir / "acom_001_projection_diagnostics.jsonl")
    tilts = sorted({float(row["tilt_deg"]) for row in rows})
    variants = list(COLORS)

    grouped_rows: dict[tuple[float, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped_rows[(float(row["tilt_deg"]), row["variant"])].append(row)
    grouped_diagnostics: dict[float, list[dict]] = defaultdict(list)
    for row in diagnostics:
        grouped_diagnostics[float(row["tilt_deg"])].append(row)

    merge_series = {
        "projection_merge": [
            (
                tilt,
                median(
                    row["collision"]["renderer_merge_fraction"]
                    for row in grouped_diagnostics[tilt]
                ),
            )
            for tilt in tilts
        ]
    }
    top5_series = {
        variant: [
            (
                tilt,
                sum(row["raw_top5_min_error_deg"] <= 2.0 for row in grouped_rows[(tilt, variant)])
                / len(grouped_rows[(tilt, variant)]),
            )
            for tilt in tilts
        ]
        for variant in variants
    }
    rank_series = {
        variant: [
            (
                tilt,
                median(row["best_correct_seed_rank"] for row in grouped_rows[(tilt, variant)]),
            )
            for tilt in tilts
        ]
        for variant in variants
    }
    merge_chart = svg_line_chart(
        merge_series,
        title="Reflection projection overlap decreases away from [001]",
        y_label="Merged reflections / candidate reflections",
        y_min=0.0,
        y_max=1.0,
        percent=True,
    ).replace("#334155", "#dc2626").replace("projection_merge", "Renderer merge fraction")
    top5_chart = svg_line_chart(
        top5_series,
        title="Correct orientation among raw correlation Top-5",
        y_label="Top-5 Acc@2°",
        y_min=0.0,
        y_max=1.0,
        percent=True,
    )
    rank_chart = svg_line_chart(
        rank_series,
        title="Global rank of the best correct ACOM correlation cell",
        y_label="Median rank (log scale; lower is better)",
        y_min=1.0,
        y_max=1000.0,
        percent=False,
        log_y=True,
    )

    table_rows = []
    for tilt in tilts:
        merge = median(
            row["collision"]["renderer_merge_fraction"]
            for row in grouped_diagnostics[tilt]
        )
        multiplicity = median(
            row["collision"]["greedy_max_multiplicity"]
            for row in grouped_diagnostics[tilt]
        )
        physical = grouped_rows[(tilt, "physical_oracle")]
        table_rows.append(
            "<tr>"
            f"<td>{tilt:g}°</td>"
            f"<td>{len(physical)}</td>"
            f"<td>{pct(merge)}</td>"
            f"<td>{multiplicity:g}</td>"
            f"<td>{pct(sum(row['native_error_deg'] <= 2 for row in physical)/len(physical))}</td>"
            f"<td>{pct(sum(row['raw_top5_min_error_deg'] <= 2 for row in physical)/len(physical))}</td>"
            f"<td>{median(row['best_correct_seed_rank'] for row in physical):g}</td>"
            "</tr>"
        )

    aggregate = summary["aggregate"]
    exact_physical = aggregate["exact_001/physical_oracle"]
    exact_closed = aggregate["exact_001/acom_discrete_seed"]
    transition_physical = aggregate["transition_001/physical_oracle"]
    output_html = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>V5 exact [001] ACOM correlation audit</title>
<style>
:root{{--ink:#172033;--muted:#64748b;--line:#dbe3ef;--panel:#fff;--bg:#f5f7fb;--accent:#2563eb}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font:16px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans SC",sans-serif}}
main{{width:min(1500px,100%);margin:auto;padding:34px 24px 70px}} h1{{font-size:34px;margin:0 0 8px}} h2{{font-size:25px;margin:0 0 16px}} h3{{margin:0 0 8px}} p{{margin:8px 0}} .muted{{color:var(--muted)}}
.cards{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px;margin:24px 0}} .card,.section{{background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:20px;box-shadow:0 4px 18px #1e293b0b}} .card b{{display:block;font-size:28px;color:var(--accent)}}
.section{{margin-top:18px}} .charts{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}} .chart{{border:1px solid var(--line);border-radius:14px;background:#fff;padding:8px;min-width:0}} svg{{display:block;width:100%;height:auto}} .chart-title{{font-size:18px;font-weight:700;fill:#172033}} .grid{{stroke:#dbe3ef;stroke-width:1}} .vertical{{stroke-dasharray:3 5}} .tick,.legend{{font-size:12px;fill:#64748b}} .value{{font-size:11px;font-weight:700;fill:#334155}} .axis-label{{font-size:13px;fill:#475569}}
table{{width:100%;border-collapse:collapse;table-layout:fixed}} th,td{{border-bottom:1px solid var(--line);padding:10px 8px;text-align:left;overflow-wrap:anywhere}} th{{background:#f8fafc;font-size:14px}} code{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;background:#eef2f7;padding:2px 5px;border-radius:5px}} .finding{{border-left:5px solid var(--accent);padding:12px 16px;background:#eff6ff;border-radius:8px;margin:12px 0}}
@media(max-width:900px){{.cards,.charts{{grid-template-columns:1fr 1fr}}}} @media(max-width:620px){{main{{padding:22px 12px 50px}}.cards,.charts{{grid-template-columns:1fr}}h1{{font-size:28px}}}}
</style></head><body><main>
<header><div class="muted">OR4D Clean V5 · independent diagnostic run</div><h1>Exact [001] ACOM correlation audit</h1><p>该审计重放 32 个 exact [001] 与 96 个 3°/4°/6° transition 样本。正式 V5 结果未被覆盖。所有候选排名均来自 normal 与 Friedel/inverse 两个分支的完整相关面，发生在 zone-axis Top-K 抑制之前。</p></header>
<div class="cards"><div class="card"><span>Exact [001] projection merge</span><b>92.2%</b><small>696 candidate reflections → median 54 disks</small></div><div class="card"><span>Exact physical oracle raw Top-5</span><b>{pct(exact_physical['raw_cell_acc_at_2deg_top5'])}</b><small>best correct cell median rank {exact_physical['best_correct_seed_rank_median']:g}</small></div><div class="card"><span>Exact ACOM-seed closed loop</span><b>{pct(exact_closed['native_acc_at_2deg'])}</b><small>median correct rank {exact_closed['best_correct_seed_rank_median']:g}</small></div><div class="card"><span>Transition physical oracle raw Top-5</span><b>{pct(transition_physical['raw_cell_acc_at_2deg_top5'])}</b><small>96 samples pooled</small></div></div>
<section class="section"><h2>结论</h2><div class="finding"><b>检峰不是 exact [001] 失败的主因。</b>审计直接使用保存的 image-matched physical oracle 峰；此前 AutoDisk 和 find_Bragg_disks 在 exact [001] 的峰召回均为 100%。</div><div class="finding"><b>现有 zone-axis Top-K 去重也不是主因。</b>完整相关面中，exact [001] 正确解的中位排名为第 82，原始 Top-5 仍为 0%。</div><div class="finding"><b>问题在 ACOM 相关表示与 [001] 投影简并。</b>即使输入最近离散 ACOM seed 自己生成的峰，32 个 exact 样本仍全部失败；696 个候选反射在 [001] 投影后只形成 54 个盘，最大单盘包含 16 个 HKL 反射。偏离带轴后投影重合减少，但物理 oracle 的恢复并非严格单调，说明强度模型失配仍然存在。</div></section>
<section class="section"><h2>随倾角变化</h2><div class="charts"><div class="chart">{merge_chart}</div><div class="chart">{top5_chart}</div><div class="chart">{rank_chart}</div></div></section>
<section class="section"><h2>V5 physical oracle 分层结果</h2><table><thead><tr><th>Tilt</th><th>n</th><th>Merge fraction</th><th>Max HKL multiplicity</th><th>Native Top-1 Acc@2°</th><th>Raw Top-5 Acc@2°</th><th>Median correct rank</th></tr></thead><tbody>{''.join(table_rows)}</tbody></table></section>
<section class="section"><h2>四个控制输入</h2><table><thead><tr><th>Label</th><th>Input</th><th>Controlled factor</th></tr></thead><tbody><tr><td>physical_oracle</td><td>V5 最终期望图积分得到的真实盘中心与强度</td><td>完整 V5 图像形成后的峰列表</td></tr><tr><td>physical_uniform_intensity</td><td>相同物理盘位置，所有强度设为 1</td><td>隔离强度分布；该控制在 transition 上整体变差，因此不能把强度全部丢弃</td></tr><tr><td>acom_exact_gt</td><td>py4DSTEM 在精确 GT 矩阵生成的峰</td><td>移除 coherent First-Born 图像模型差异</td></tr><tr><td>acom_discrete_seed</td><td>py4DSTEM 在最近 ACOM 搜索网格节点生成的峰</td><td>最严格的离散计划闭环</td></tr></tbody></table></section>
<section class="section"><h2>可复现记录</h2><p>运行时间 {summary['duration_seconds']:.1f} s；GPU <code>{html.escape(str(summary['cuda_visible_devices']))}</code>；Kmax <code>{summary['acom']['kmax_Ainv']} Å⁻¹</code>；搜索角步长 <code>{summary['acom']['angle_step_zone_axis_deg']}° / {summary['acom']['angle_step_in_plane_deg']}°</code>；完整相关单元 <code>{summary['acom']['num_scored_cells']:,}</code>。</p><p>小结果：<code>acom_001_audit_summary.json</code>、<code>acom_001_correlation_audit.jsonl</code>、<code>acom_001_projection_diagnostics.jsonl</code>。194 MB 完整相关面保存在服务器 <code>/mnt/data/xietianhong/or4d-clean-v5/results/v5_001_audit/correlation_surfaces.h5</code>，不进入 Git。</p></section>
</main></body></html>"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output_html, encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
