# Clean-Peak + ACOM 扩展测试

本扩展把 Clean 测试从 5 个低指数、零面内旋转取向扩展到 17 个取向：

- 5 个原始低指数控制取向；
- 4 个非零面内旋转取向，角度分别为 6.1°、14.2°、22.3°、30.4°；
- 8 个确定性低差异 SO(3) 取向，经过 NCM811 点群对称性去近邻筛选。

ACOM Orientation Plan（取向模板库）使用 4° 区轴步长和 4° 面内角步长。新增测试取向独立于模板库生成。`07_run_acom_baseline.py` 会枚举实际 Orientation Plan 的完整取向节点，并计算测试真值到最近模板节点的对称性感知取向差。新增样本与模板节点距离小于 0.25° 时，脚本会停止并报告具体样本。

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
- `reports/acom_clean_details.json`：相关分数、模板索引、单样本耗时、逐样本取向差和汇总指标；
- `reports/acom_clean_evaluation.json`：对称性感知取向误差。

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
