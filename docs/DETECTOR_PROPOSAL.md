# Proposed detector: variance-stabilized multi-scale matched filtering

当前 Clean 的 AutoDisk、DoG-RGM 和 `find_Bragg_disks` 都依赖单一尺度的
盘形或 LoG 候选。低剂量时，弱盘的峰值会被 Poisson 散粒噪声和读出噪声压低，
固定阈值会同时损失低强度、高角度和略有盘形变化的反射。

建议新增一个不读取真值和 ACOM 结果的无训练检峰器，暂命名为
**VS-MSFM-Friedel**（variance-stabilized multi-scale matched filter with
Friedel-pair rescue）：

```text
二维图像
  → Poisson+read-noise 方差稳定化
  → 中心束/低频背景估计与扣除
  → 多半径、 多盘径模板的 FFT 匹配
  → 局部噪声自适应候选阈值
  → 亚像素 RGM 拟合与积分
  → ±q Friedel 配对证据补救弱盘
  → NMS、中心束排除、统一 PointList
```

## 物理和统计假设

对计数图使用已知剂量和读出噪声参数构造方差稳定变换。Poisson-only
使用 Anscombe 变换；带 EMPAD-G2 读出噪声时使用广义 Anscombe 形式，参数来自
`config/benchmark_v5.yaml` 的 `instrument_noise`，而不是从测试标签估计。变换
只用于候选分数，盘强度仍从原始图像积分。

模板由 `vacuum_probe` 和三个相邻半径尺度构造，半径覆盖会聚半角和一个像素
的亚像素偏差。模板匹配在倒空间图像上使用 FFT，候选阈值使用局部 MAD/方差，
避免用一个全图绝对阈值压掉高角度弱盘。

Friedel 配对只作为候选置信度项：以估计 beam center 为原点寻找 `q` 与 `-q`
的近邻，要求位置容差和盘形相似度满足固定条件。配对项不能凭空生成强度，
只允许把已经通过低阈值模板检验的弱候选提升到精修队列。中心盘和边界盘不参与
配对补救。这样不会把晶体对称性或 ACOM orientation plan 泄漏进检峰器。

## 输出和评测

输出继续使用现有统一格式：

```text
sample_id
qx
qy
integrated intensity
peak_diagnostics
sample_metadata
```

必须与三个现有检峰器使用同一份 image-matched oracle、同一匹配容差和同一
`scripts/06_evaluate_submission.py` 评测。至少报告：

- Recall、Precision、high-angle Recall；
- position RMSE/P95；
- false positives/negatives per sample；
- 运行时间和失败样本；
- Clean-E 与每个 Clean-C 剂量/噪声条件；
- ACOM Top-1…Top-5 相对 oracle 的变化。

先在 17 个 `legacy_smoke` 和独立 `[001]` 集合上验证，再跑 2,048 headline
条件。当前没有该方法的实测结果；在实现和全量运行完成前，不应把它写入 V5
结果图或 headline 指标。
