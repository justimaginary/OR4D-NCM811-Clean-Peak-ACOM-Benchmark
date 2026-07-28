# Clean-Peak Benchmark v2 设计

## 目标

Clean-Peak v2 仍然是“已知 NCM811 晶体结构时，从理想运动学衍射峰集恢复取向”的自洽测试。它用于验证取向约定、对称性、Friedel 等价处理和 ACOM 基线，不用于声称对真实实验数据的泛化能力。

公开输入为每个样本的变长 `{qx, qy, intensity}` 峰集；输出为 `orientation_matrix_sample_to_crystal`。Clean 主指标是在 proper crystal point-group rotations 和 Friedel 等价下的误取向角。

## 样本分层

| `sample_role` | 数量 | 生成方式 | 用途 |
| --- | ---: | --- | --- |
| `legacy_smoke` | 17 | 保留 v1 的 9 个手工取向与 8 个确定性 SO(3) 取向 | 回归、坐标约定和 mirror/Friedel 检查 |
| `headline_core` | 512 | 固定 seed 的 scrambled Sobol(3) 经 Shoemake 映射到 SO(3) | Clean headline 指标 |
| `acom_grid_probe` | 40 | 5 个低指数区轴 × 8 个 4°面内网格子偏移 | 单独诊断 ACOM 面内亚网格行为 |

总样本数为 569。只有 `headline_core` 进入 headline 指标；其余样本始终单独报告。

### Headline core

- 样本生成不读取 ACOM orientation plan、相关分数或误差。
- 512 是 Sobol 序列适合的 2 次幂规模。
- 不用大角度拒绝采样改变均匀分布；只审计并拒绝数值意义上的 crystal/Friedel 等价重复。
- 固定 seed 和生成算法用于本地可复现。若将来建设外部盲测，应把 eval manifest/seed 留在评测端。

### ACOM grid probes

Probe 允许显式依赖 canonical 4° ACOM 网格，因为它们不参与公平排名。每个选定低指数区轴使用
`0.25°、0.5°、0.75°、1.0°、1.25°、1.5°、1.75°、2.0°`
面内偏移，覆盖一个网格单元到半步边界。

ACOM 会对面内相关峰做抛物线亚网格拟合，所以“最近离散 plan 节点距离”只能称为 discrete seed distance，不能解释成理论误差下界。报告同时给出最近区轴节点距离，用于分离区轴离散误差。

## ACOM 报告

Canonical baseline 固定为 4°区轴步长和 4°面内步长。报告必须包含：

- headline core 的 Friedel-aware mean、median、p90、p95、max 和 Acc@1/2/5°；
- strict 指标及 strict/Friedel 分歧率；
- 按 `sample_role`、最近区轴节点距离分箱、峰数分箱的结果；
- plan 构建时间、匹配耗时 p50/p90/p99 和吞吐量；
- dataset/config/manifest hash 与软件版本；
- 可扩展到数百样本的 ECDF、距离—误差散点图和固定数量代表样本 overlay。

`reports/acom_clean_details.json` 是 ACOM 结果的唯一事实源；Markdown 和图片由脚本自动生成。旧 PDF 只代表 v1 的 17 样本结果。

## 当前边界

Clean 生成器和 ACOM 都使用同一 CIF 与 py4DSTEM 运动学模型，属于 matched-model 自洽测试。后续真实性工作应建立独立生成器的 `clean_crossgen`，再进入 dynamical 与真实实验数据轨道，并与本轨结果分组报告。
