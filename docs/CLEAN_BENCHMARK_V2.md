# Clean-Peak Benchmark v3 设计

## 目标

Clean-Peak v3 仍然是“已知 NCM811 晶体结构时，从理想运动学衍射峰集恢复取向”的自洽测试。它用于验证取向约定、对称性、Friedel 等价处理和 ACOM 基线，不用于声称对真实实验数据的泛化能力。

公开输入为每个样本的变长 `{qx, qy, intensity}` 峰集；输出为 `orientation_matrix_sample_to_crystal`。Clean 主指标是在 proper crystal point-group rotations 和 Friedel 等价下的误取向角。

v3 的公开峰集和 ACOM 模板统一使用 `k_max = 1.5 Å⁻¹`；中心束排除半径仍为 `0.03 Å⁻¹`。旧版 `2.0 Å⁻¹` 结果只保留在升级前备份中，不与 v3 指标混合。

## 样本分层

| `sample_role` | 数量 | 生成方式 | 用途 |
| --- | ---: | --- | --- |
| `legacy_smoke` | 17 | 保留 v1 的 9 个手工取向与 8 个确定性 SO(3) 取向 | 回归、坐标约定和 mirror/Friedel 检查 |
| `headline_core` | 1024 | 固定 seed 的 scrambled Sobol(3) 经 Shoemake 映射到 SO(3) | Clean headline 指标 |
| `acom_grid_probe` | 40 | 5 个低指数区轴 × 8 个 2°面内网格子偏移 | 单独诊断 ACOM 面内亚网格行为 |

总样本数为 1081。只有 `headline_core` 进入 headline 指标；其余样本始终单独报告。

### Headline core

- 样本生成不读取 ACOM orientation plan、相关分数或误差。
- 1024 是 Sobol 序列适合的 2 次幂规模；它是 v2 headline core 的确定性扩展，前 512 个样本保持不变。
- 不用大角度拒绝采样改变均匀分布；只审计并拒绝数值意义上的 crystal/Friedel 等价重复。
- 固定 seed 和生成算法用于本地可复现。若将来建设外部盲测，应把 eval manifest/seed 留在评测端。

### ACOM grid probes

Probe 允许显式依赖 canonical 2° ACOM 网格，因为它们不参与公平排名。每个选定低指数区轴使用
`0.125°、0.25°、0.375°、0.5°、0.625°、0.75°、0.875°、1.0°`
面内偏移，覆盖一个网格单元到半步边界。

ACOM 会对面内相关峰做抛物线亚网格拟合，所以“最近离散 plan 节点距离”只能称为 discrete seed distance，不能解释成理论误差下界。报告同时给出最近区轴节点距离，用于分离区轴离散误差。

## ACOM 报告

Canonical baseline 固定为 2°区轴步长和 2°面内步长，并额外运行
4°、3°、2°扫描，以量化加密 orientation plan 的收益与代价。报告必须包含：

- headline core 的 Friedel-aware mean、median、p90、p95、max 和 Acc@1/2/5°；
- strict 指标及 strict/Friedel 分歧率；
- 按 `sample_role`、最近区轴节点距离分箱、峰数分箱的结果；
- plan 构建时间、匹配耗时 p50/p90/p99 和吞吐量；
- dataset/config/manifest hash 与软件版本；
- 可扩展到数百样本的 ECDF、距离—误差散点图和固定数量代表样本 overlay。

`reports/acom_clean_details.json` 是 ACOM 结果的唯一事实源；Markdown 和图片由脚本自动生成。旧 PDF 只代表 v1 的 17 样本结果。

## 坐标与逐反射追踪

`scripts/10_trace_clean_coordinates.py` 输出从实空间晶格与取向到倒空间峰坐标的完整链路。对每个样本，它记录直接晶格矩阵 `A`、晶体学倒易晶格矩阵 `B`、标准取向和 ACOM 取向、样品坐标轴在晶体笛卡尔系中的表达，以及每个 `(h,k,l)` 的

`g_crystal = [h,k,l] @ B`，
`g_sample = R_sample_to_crystal.T @ g_crystal`，
`qx = g_sample[0]`、`qy = g_sample[1]`、`qz = g_sample[2]`。

完整逐样本逐反射记录写入压缩 JSONL；Markdown 报告汇总标准坐标恒等式残差、ACOM 峰集匹配率、`q` 偏差、原始 HKL 对照和最差样本。HKL 是晶体基底标签，在两个对称等价取向之间可能改变，因此模型优劣以 detector-plane `q` 匹配与 symmetry/Friedel-aware 误取向为主，原始 HKL 相同率只作诊断。

## 当前边界

Clean 生成器和 ACOM 都使用同一 CIF 与 py4DSTEM 运动学模型，属于 matched-model 自洽测试。后续真实性工作应建立独立生成器的 `clean_crossgen`，再进入 dynamical 与真实实验数据轨道，并与本轨结果分组报告。
