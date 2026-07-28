# OR4D NCM811 Clean-Peak ACOM Benchmark v3

本仓库构建并评测一个取向恢复基准：在参考 NCM811 晶体结构已知的条件下，算法仅根据理想运动学电子衍射峰集合 `{qx, qy, intensity}`，预测完整的 `3×3 orientation_matrix_sample_to_crystal`。

当前正式结果来自 Clean-Peak v3。Dynamical 轨道仍是后续工作，不应把本页结果解释为真实实验数据或动力学衍射上的泛化性能。

## 当前结果

正式 headline cohort 包含 1024 个与 ACOM 搜索网格独立生成的 scrambled Sobol-SO(3) 取向。主指标为 proper crystal symmetry 和 Friedel 等价下的取向误差。

| ACOM 角步长 | Mean | Median | P95 | Acc@1° | Acc@2° | Acc@5° | 吞吐量 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4° | 2.991° | 1.625° | 5.311° | 20.8% | 68.3% | 94.6% | 70.4 samples/s |
| 3° | 2.581° | 1.283° | 5.069° | 32.3% | 84.7% | 94.8% | 32.9 samples/s |
| **2° canonical** | **1.625°** | **0.977°** | **3.525°** | **52.1%** | **90.5%** | **96.7%** | **9.8 samples/s** |

2°配置的最大误差为 `89.850°`，其中 34/1024 个样本误差大于 5°，12/1024 个样本误差大于 10°。因此 Median 和 Acc@2°较好，但仍存在少量模式级灾难性匹配。

## 数据集组成

| `sample_role` | 数量 | 生成方式 | 用途 |
| --- | ---: | --- | --- |
| `legacy_smoke` | 17 | 手工控制取向与确定性旧样本 | 回归、坐标和 Friedel/mirror 检查 |
| `headline_core` | 1024 | 固定 seed 的 scrambled Sobol(3) 经 Shoemake 映射到 SO(3) | 唯一正式主指标 |
| `acom_grid_probe` | 40 | 5 个区轴 × 8 个 2°网格内偏移 | ACOM 亚网格诊断 |

总样本数为 1081。生成使用 `Kmax = 1.5 Å⁻¹`，公开峰表不包含 HKL、区轴、欧拉角或标准取向。

## 输入输出契约

公开输入：

```text
sample_id
peaks = {qx, qy, intensity}
```

算法输出：

```text
sample_id
orientation_matrix_sample_to_crystal
```

坐标约定：

```text
v_crystal = R_sample_to_crystal @ v_sample
```

对 Miller 指数列向量 `h = [h,k,l]ᵀ`，若倒易基矩阵 `B` 的三行是 `a*、b*、c*`：

```text
g_crystal = B.T @ h
g_sample  = R_sample_to_crystal.T @ g_crystal
q = [g_sample[0], g_sample[1]]
```

`qz = g_sample[2]` 沿电子束方向，不作为公开二维峰坐标。NumPy 代码中的一维数组写法 `hkl @ B` 是上述列向量公式的转置等价形式。

## 环境

Clean 与 Dynamical 使用两个环境，因为 py4DSTEM 0.14.18 需要 NumPy `<2`，而 abTEM 1.0.10 需要 NumPy `>=2`。

```bash
conda env create -f environment-clean.yml
conda env create -f environment-dynamical.yml
```

当前 Clean-Peak + ACOM 流程使用：

```bash
conda activate or4d-clean
```

## 完整运行

```bash
cd /Users/xie/code/OR4D-NCM811-smoke-v0
./run_clean_acom.sh
```

脚本依次执行：

```bash
python scripts/01_generate_orientations.py
python scripts/02_generate_clean.py
python scripts/04_validate_dataset.py
python scripts/11_run_acom_sweep.py
python scripts/08_visualize_acom_results.py
python scripts/09_write_acom_report.py
python scripts/10_trace_clean_coordinates.py --all
python scripts/12_write_coordinate_visualization.py
```

`scripts/11_run_acom_sweep.py` 内部针对 4°、3°、2°分别运行 ACOM baseline 和评价脚本，不需要在完整流程中再次单独执行 07、06。

## 本地 HTML 使用方式

交互式离线报告位于：

[`reports/ACOM_COORDINATE_VISUALIZATION.html`](reports/ACOM_COORDINATE_VISUALIZATION.html)

macOS 可直接打开：

```bash
open reports/ACOM_COORDINATE_VISUALIZATION.html
```

也可以在文件管理器或浏览器中直接打开。页面不依赖网络和本地服务器，数据已嵌入 HTML。

页面提供：

- 总体 Overview、4°/3°/2°结果表和误差/准确率对比图；
- Benchmark 生成流程与 ACOM 预测流程；
- 中英双语、可折叠的实际运行参数；
- Best、Median、P95、Worst 四个代表样本；
- 仅 GT、仅 ACOM、GT + ACOM 叠加三种图层；
- 探测器倒空间峰、取向矩阵三列和所选 `g_crystal` 方向；
- `HKL → B → g_crystal → g_sample → q` 的完整数值链；
- 当前 GT 图样中全部有效 HKL、强度、标准/ACOM 坐标和差值。

图中的紫色圆圈和紫色方向线只表示下拉框当前选中的 GT 反射，不是额外峰、异常值或第三种算法结果。

### HTML 是否实时更新

HTML 是最近一次结果的离线快照：

- 打开页面和切换控件时，浏览器使用已嵌入的数据即时重画；
- 页面不会重新读取 JSON/HDF5，也不会重新运行 ACOM；
- 结果文件变化后必须重新生成 HTML。

只重新生成 HTML：

```bash
python scripts/12_write_coordinate_visualization.py
```

完整运行 `./run_clean_acom.sh` 时，最后会自动更新该 HTML。

## PDF 报告

完整静态报告：

[`OR4D_NCM811_CleanPeak_ACOM_Benchmark_Report_v3.pdf`](OR4D_NCM811_CleanPeak_ACOM_Benchmark_Report_v3.pdf)

报告共 44 页，内容包括：

1. Benchmark 任务、公开输入和隐藏标准答案；
2. 实空间、倒空间、探测器坐标与矩阵约定；
3. legacy、headline、grid probe 三层采样设计；
4. Clean-Peak 生成参数和峰过滤；
5. py4DSTEM、pymatgen、NumPy、SciPy、h5py 等依赖及关键接口；
6. ACOM Orientation Plan、单样本匹配和 4°/3°/2°扫描；
7. proper symmetry、Strict 与 Friedel-equivalent 评价协议；
8. headline、sample role、区轴、峰数和运行性能结果；
9. HKL、`g_crystal`、`g_sample`、q 和峰级匹配审计；
10. 误差分布、区轴距离和输入/预测峰叠加图；
11. 软件版本、Git revision、SHA256 和完整运行流程；
12. 结论、限制和中英文术语。

该 PDF 是已生成的静态报告；仓库当前未包含其 Typst 源文件。

## 主要文件

### 数据与标准答案

- `config/benchmark.yaml`：数据、Clean、ACOM 和评价参数；
- `data/structure/NCM811_113035.cif`：固定 NCM811 平均占位结构；
- `private/orientations.jsonl`：完整取向 manifest；
- `public/clean_peaks.h5`：公开变长峰集合；
- `private/clean_ground_truth.jsonl`：隐藏取向标准答案；
- `diagnostics/clean_reflections.h5`：带 HKL 的内部诊断反射。

### ACOM 预测与评价

- `submissions/acom_clean_predictions*.jsonl`：2°/3°/4° ACOM 预测；
- `reports/acom_plan_audit*.json`：Orientation Plan 和最近网格距离；
- `reports/acom_clean_details*.json`：相关分数、模板索引、运行时间和逐样本诊断；
- `reports/acom_clean_evaluation*.json`：对称性/Friedel-aware 评价结果；
- `reports/ACOM_CLEAN_REPORT.md`：由 JSON 自动生成的结果报告。

### 坐标审计与可视化

- `diagnostics/clean_coordinate_trace.jsonl.gz`：1081 个样本的逐反射完整坐标链；
- `reports/ACOM_COORDINATE_ANALYSIS.md`：坐标、HKL、峰匹配和严重错误摘要；
- `reports/ACOM_COORDINATE_VISUALIZATION.html`：离线交互式报告；
- `reports/acom_error_comparison.png`：误差分布；
- `reports/acom_offgrid_vs_error.png`：网格距离与误差；
- `reports/acom_peak_overlay.png`：输入峰与预测峰叠加。

## 打印中间变量

打印一个样本：

```bash
python scripts/10_trace_clean_coordinates.py \
  --sample-id clean_core_0000 --stdout
```

打印全部样本：

```bash
python scripts/10_trace_clean_coordinates.py --all --stdout
```

完整输出量很大。通常应查询压缩 JSONL，而不是把全部内容打印到终端。

## 评价已有提交

```bash
python scripts/06_evaluate_submission.py \
  submissions/acom_clean_predictions.jsonl \
  --track clean \
  --output reports/acom_clean_evaluation.json
```

提交矩阵必须为有限数值的 `3×3` proper rotation：正交且行列式为 `+1`。

## 测试

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

测试覆盖：

- Sobol-SO(3) 采样确定性、前 512 个样本继承和重复检查；
- HKL、倒易矩阵、标准取向到探测器 q 的坐标闭环；
- 提交矩阵形状、正交性和评价契约。

## 已知限制

- Clean 输入和 ACOM 模板使用同一 CIF 与同一 py4DSTEM 运动学模型；这是 matched-model self-consistency，不是独立生成器或真实数据泛化测试。
- `[001]` 取向族在 `Kmax = 1.5 Å⁻¹` 下存在稳定约 72°错误模式，是当前最重要的科学问题。
- 2°计划比 4°计划精度更高，但搜索种子和运行时间约增加 8 倍。
- `acom_grid_probe` 只能作为诊断，不能混入正式 headline 指标。
- 仓库同时包含 public、private 和评价文件，适合本地复现，不构成服务器端盲测。

更详细的设计协议见 [`docs/CLEAN_BENCHMARK_V3.md`](docs/CLEAN_BENCHMARK_V3.md)。
