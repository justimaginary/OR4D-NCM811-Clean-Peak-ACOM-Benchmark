# Clean v3 坐标链路与 ACOM 对比

- 数据集：`OR4D-NCM811-CleanPeak-acom-v3`
- 已追踪样本：1081
- `k_max`：1.500 Å⁻¹
- 公共输入：每个样本一个变长峰表 `{qx, qy, intensity}`，HKL 不公开给算法。
- 标准输出与 ACOM 输出：`3×3 orientation_matrix_sample_to_crystal`。
- 完整逐样本逐反射数据：`clean_coordinate_trace.jsonl.gz`

## 坐标是怎么来的

直接晶格矩阵 `A` 的行是实空间 `a,b,c`（Å）；晶体学倒易矩阵 `B` 的行是 `a*,b*,c*`（Å⁻¹，不含 `2π`）。对反射 `(h,k,l)`：

```text
g_crystal = [h, k, l] @ B
g_sample  = R_sample_to_crystal.T @ g_crystal
qx = g_sample[0], qy = g_sample[1], qz = g_sample[2]
```

`R` 的三列分别是样品 `x,y,z` 轴在晶体笛卡尔坐标中的方向；因此 `qx` 也等于 `g_crystal · x_crystal`。`qz` 沿电子束方向，公开峰表只保留探测器平面的 `qx,qy`。

## 标准结果与 ACOM 结果

- 标准 HKL→q 恒等式的全局最大残差：8.074e-08 Å⁻¹。
- 全部样本 ACOM 峰的中位观测匹配率：88.24%；平均 86.69%。
- Headline ACOM 峰的中位观测匹配率：87.50%；中位 q-RMSE 0.0006 Å⁻¹。
- Headline 已匹配峰的中位强度 MAE：0.0547；逐样本原始 HKL 相同率的中位数为 0.0%，计入 `(h,k,l)↔(-h,-k,-l)` 后为 15.9%。
- Headline 取向误差：median 0.977°，p95 3.525°，max 89.850°。
- 大于 5°：34/1024；大于 10°：12/1024。

峰匹配是在探测器 `q` 平面做一对一最小距离分配，并以 0.080 Å⁻¹ 截断。原始 HKL 标签可能因晶体对称操作或 Friedel 等价而改变，所以 HKL 相同率只是诊断；主判断依据仍是 q 匹配与 symmetry/Friedel-aware 取向误差。

## 数据和结果长什么样

公共输入不含 HKL，下面只截取一个样本的前三个峰：

```json
{
  "sample_id": "clean_core_0198",
  "peaks": [
    {
      "qx": -0.06510347127914429,
      "qy": -0.19595740735530853,
      "intensity": 0.02972974255681038
    },
    {
      "qx": 0.06510347127914429,
      "qy": 0.19595740735530853,
      "intensity": 0.027417318895459175
    },
    {
      "qx": -0.35129889845848083,
      "qy": 0.20111511647701263,
      "intensity": 0.02341185137629509
    }
  ]
}
```

标准答案与 ACOM 都是 sample→crystal 的 3×3 旋转矩阵：

```json
{
  "standard_orientation_matrix_sample_to_crystal": [
    [
      0.7055796128032421,
      -0.3593861650177689,
      -0.6107364361082835
    ],
    [
      0.63746631582821,
      -0.054526446710509026,
      0.7685463959928086
    ],
    [
      -0.30950622963170754,
      -0.9315945743739317,
      0.1906238254159688
    ]
  ],
  "acom_orientation_matrix_sample_to_crystal": [
    [
      0.15305999842369666,
      0.09401770209636023,
      0.9837343689101533
    ],
    [
      -0.9354727920739748,
      0.33465372535685356,
      0.11356733418594203
    ],
    [
      -0.31853303152398454,
      -0.9376393527349944,
      0.13917310096006538
    ]
  ],
  "friedel_equivalent_misorientation_deg": 3.522181236755039
}
```

诊断文件才把 HKL 和完整坐标链路连起来；同一条反射的实际记录例如：

```json
{
  "hkl": [
    0,
    0,
    3
  ],
  "g_crystal_cartesian_Ainv": [
    -1.2879991857644891e-17,
    -2.2308800298514395e-17,
    0.21034622989440618
  ],
  "standard_g_sample_Ainv": [
    -0.06510346853186205,
    -0.1959574065096405,
    0.04009700300429851
  ],
  "reported_qx_Ainv": -0.06510347127914429,
  "reported_qy_Ainv": -0.19595740735530853,
  "acom_same_hkl_g_sample_Ainv": [
    -0.06700222227790617,
    -0.19722890284843736,
    0.0292745370896633
  ]
}
```

差异主要有三类：取向矩阵/样品坐标轴不同，使同一 HKL 的 `qx,qy,qz` 改变；激发误差筛选使两套峰表出现缺峰或多峰；晶体对称和 Friedel 等价会让几乎相同的 detector `q` 使用不同原始 HKL 标签。

## 代表样本

| sample | role | 取向误差 | observed/predicted/matched | observed match | q-RMSE | raw HKL 相同 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `clean_core_0970` | headline_core | 0.022° | 21/21/21 | 100.0% | 0.0001 | 0.000 |
| `clean_core_0530` | headline_core | 0.977° | 22/21/19 | 86.4% | 0.0005 | 1.000 |
| `clean_core_0198` | headline_core | 3.522° | 19/17/15 | 78.9% | 0.0018 | 0.133 |
| `clean_core_0037` | headline_core | 89.850° | 23/17/17 | 73.9% | 0.0071 | 0.294 |

## Headline 差异最大的 20 个样本

| sample | 取向误差 | corr | peaks observed/predicted/matched | observed match | q-RMSE | mirror |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `clean_core_0037` | 89.850° | 2.8631 | 23/17/17 | 73.9% | 0.0071 | False |
| `clean_core_0211` | 89.848° | 2.8577 | 22/17/17 | 77.3% | 0.0066 | False |
| `clean_core_0722` | 71.329° | 2.8864 | 25/20/12 | 48.0% | 0.0066 | False |
| `clean_core_0347` | 70.131° | 2.7978 | 28/20/15 | 53.6% | 0.0080 | True |
| `clean_core_0862` | 69.839° | 2.9680 | 21/21/20 | 95.2% | 0.0063 | True |
| `clean_core_0694` | 13.121° | 2.6319 | 17/15/15 | 88.2% | 0.0173 | False |
| `clean_core_0671` | 11.690° | 2.6753 | 22/17/17 | 77.3% | 0.0009 | False |
| `clean_core_0982` | 11.460° | 2.5662 | 19/17/16 | 84.2% | 0.0016 | False |
| `clean_core_0416` | 11.329° | 2.7334 | 22/17/17 | 77.3% | 0.0006 | False |
| `clean_core_0018` | 11.180° | 2.6794 | 18/20/15 | 83.3% | 0.0005 | False |
| `clean_core_0402` | 10.147° | 2.9490 | 22/21/20 | 90.9% | 0.0003 | True |
| `clean_core_0034` | 10.045° | 2.7372 | 22/19/17 | 77.3% | 0.0006 | True |
| `clean_core_0340` | 9.870° | 2.9710 | 18/19/16 | 88.9% | 0.0007 | True |
| `clean_core_0996` | 9.593° | 2.9017 | 23/21/21 | 91.3% | 0.0004 | True |
| `clean_core_0388` | 9.510° | 3.2087 | 18/22/18 | 100.0% | 0.0004 | True |
| `clean_core_0969` | 9.127° | 3.1468 | 20/21/18 | 90.0% | 0.0007 | True |
| `clean_core_0596` | 8.056° | 2.3719 | 18/17/15 | 83.3% | 0.0006 | False |
| `clean_core_0269` | 7.546° | 3.9315 | 28/22/21 | 75.0% | 0.0007 | False |
| `clean_core_0917` | 7.217° | 2.3094 | 16/20/16 | 100.0% | 0.0007 | False |
| `clean_core_0116` | 7.046° | 2.8424 | 26/24/23 | 88.5% | 0.0007 | False |

## 如何打印所有中间变量

```bash
# 打印一个样本（含矩阵、每个 HKL、g_crystal、qx/qy/qz、ACOM 峰和差异）
conda run -n or4d-clean python scripts/10_trace_clean_coordinates.py \
  --sample-id clean_core_0000 --stdout

# 打印全部样本；输出量很大
conda run -n or4d-clean python scripts/10_trace_clean_coordinates.py \
  --all --stdout
```

压缩 JSONL 中每行对应一个样本，可直接解压后按 `sample_id`、`standard_observed_reflections[].hkl` 或 `detector_q_assignment.matches[]` 检索。
