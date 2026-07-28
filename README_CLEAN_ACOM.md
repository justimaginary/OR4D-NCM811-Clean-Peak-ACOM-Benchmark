# Clean-Peak v2 + ACOM

Clean v2 将样本分为三个互不混算的角色：

- `legacy_smoke`：保留原 17 个样本，仅用于回归；
- `headline_core`：512 个固定 seed 的 scrambled Sobol-SO(3) 样本，作为主指标；
- `acom_grid_probe`：40 个面内子网格样本，仅用于诊断 ACOM 4°网格。

主样本生成完全不读取 ACOM plan、相关分数或误差。ACOM Orientation
Plan 固定使用 4°区轴步长和 4°面内角步长。报告区分最近离散 seed
距离与最近区轴节点距离；离散 seed 距离不是误差下界，因为 ACOM 会做面内亚网格拟合。

详细设计见 `docs/CLEAN_BENCHMARK_V2.md`。

## 运行

```bash
conda activate or4d-clean
cd /Users/xie/code/OR4D-NCM811-smoke-v0

python scripts/01_generate_orientations.py
python scripts/02_generate_clean.py
python scripts/04_validate_dataset.py
python scripts/07_run_acom_baseline.py
python scripts/06_evaluate_submission.py \
  submissions/acom_clean_predictions.jsonl \
  --track clean \
  --output reports/acom_clean_evaluation.json
```

或者：

```bash
./run_clean_acom.sh
```

## 主要输出

- `private/orientations.jsonl`：完整取向真值库；
- `public/clean_peaks.h5`：公开峰集合输入；
- `private/clean_ground_truth.jsonl`：Clean 隐藏答案；
- `reports/acom_plan_audit.json`：测试取向与 Orientation Plan 节点的最近距离；
- `submissions/acom_clean_predictions.jsonl`：ACOM 预测矩阵；
- `reports/acom_clean_details.json`：相关分数、模板索引、单样本耗时和逐样本诊断；
- `reports/acom_clean_evaluation.json`：对称性感知取向误差。
- `reports/ACOM_CLEAN_REPORT.md`：由 JSON 自动生成的 canonical ACOM 报告；
- 三张 PNG：可扩展的误差 ECDF、区轴距离—误差图和代表样本峰叠加。

## ACOM 固定参数

- `zone_axis_range = auto`：依据 NCM811 点群自动选择对称性约化区轴范围；
- `angle_step_zone_axis = 4°`：区轴采样步长；
- `angle_step_in_plane = 4°`：面内旋转模板步长；
- `corr_kernel_size = 0.08 Å⁻¹`：峰位置相关核宽度；
- `sigma_excitation_error = 0.02 Å⁻¹`：理论模板激发误差宽度；
- `power_radial = 1.0`：散射矢量半径权重指数；
- `power_intensity_simulated = 0.25`：模拟峰强度权重指数；
- `power_intensity_experiment = 0.25`：输入峰强度权重指数；
- `tol_distance = 0.01 Å⁻¹`：径向壳层归并容差；
- `inversion_symmetry = true`：匹配时检查投影反演分支；
- `CUDA = false`：Mac CPU 执行。
