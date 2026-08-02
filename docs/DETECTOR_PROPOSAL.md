# Proposed detector: noise-adaptive segmentation with parametric disk fitting

当前的 AutoDisk 已经明确使用了：

```text
sqrt intensity → ring-template cross-correlation → LoG blobs
→ modified-RGM radial refinement → aperture integration
```

因此，另一个“多尺度模板匹配 + RGM”的变体不能算作有区分度的新方法。建议
采用完全不同的候选生成路线，暂命名为 **NAS-PSF**（Noise-Adaptive
Segmentation and Parametric Shape Fitting）。它不使用环形互相关、LoG 或 RGM，
也不读取 ACOM orientation plan 和真值取向。

```text
二维衍射图
  → Poisson/读出噪声方差建模
  → 大尺度背景与中心束遮罩
  → 像素显著性分割（局部 FDR）
  → 连通域 + 距离变换 watershed 分裂重叠盘
  → 圆盘/椭圆盘形状筛选
  → 2D 探测器响应模型的 Poisson-Gaussian 似然拟合
  → 联合亚像素中心、半径、幅度、背景拟合
  → 非极大值抑制与边界/中心束排除
```

## 与 AutoDisk 的结构差异

| 环节 | AutoDisk | NAS-PSF |
| --- | --- | --- |
| 候选生成 | 环形模板卷积 | 像素显著性分割和连通域 |
| 多盘分离 | LoG blob overlap | 距离变换 watershed |
| 中心精修 | RGM 内外环响应网格搜索 | 全二维盘形状似然拟合 |
| 强度 | 精修后固定半径积分 | 拟合幅度并估计背景不确定度 |
| Friedel | 可作为后处理诊断 | 主方法不使用；只做独立消融 |
| 依赖 | vacuum probe 的环形模板 | vacuum probe 只进入形状/响应模型 |

关键区别是：NAS-PSF 先在像素层判断“哪些区域显著”，再对候选区域进行物理
形状拟合；它不是在整幅图上寻找相关峰。

## 物理和统计模型

对每个像素使用计数和读出噪声的方差：

\[
V(I)=I/g+\sigma_{read}^{2}.
\]

候选分割使用 Pearson residual 或广义 Anscombe 变换后的显著性图。背景由大于
衍射盘直径的 rolling-ball/大尺度中值估计，并对中心透射盘单独遮罩；背景估计
不使用任何 HKL 或取向信息。

对每个连通域，以真空探针测得的径向响应作为**拟合模型**，而不是候选检测模板：

\[
I(x,y)=b+A\,[P_{detector}*D_r](x-c_x,y-c_y),
\]

拟合参数为 \((c_x,c_y,r,A,b)\)，必要时增加椭圆率和方向。目标函数是
Poisson-Gaussian 负对数似然；中心通过连续参数优化得到，不进行 RGM 网格搜索。
邻近连通域可以在同一个局部窗口中联合拟合，避免两个相邻衍射盘互相吞并。

Friedel 配对不进入 NAS-PSF 主检测结果。可以单独增加 `+Friedel-QC` 消融分支，
仅报告配对后置信度变化，不能借助 \(-q\) 位置凭空生成一个未被图像分割发现的
衍射盘。这样能把“检测器本身的召回率”和“Friedel 先验带来的收益”分开。

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

NAS-PSF 与 AutoDisk、DoG-RGM、`find_Bragg_disks` 使用同一份 image-matched
oracle、同一匹配容差和同一 `scripts/06_evaluate_submission.py` 评测。报告至少
包含：

- Recall、Precision、high-angle Recall；
- position RMSE/P95；
- false positives/negatives per sample；
- 运行时间、失败样本和连通域合并/分裂次数；
- Clean-E 及每个 Clean-C 剂量/噪声条件；
- ACOM Top-1…Top-5 相对 oracle 的变化。

实现后先在已有小规模案例和独立 `[001]` 集合上验证，再运行 2,048 headline
条件。当前没有 NAS-PSF 的实测结果；在实现和全量运行完成前，不把它写入 V5
结果图或 headline 指标。
