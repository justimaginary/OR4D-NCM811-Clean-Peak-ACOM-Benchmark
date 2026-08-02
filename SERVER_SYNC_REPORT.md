# 服务器实验运行与代码同步检查报告

**检查时间**: 2026-07-31 13:10 (服务器时间)  
**服务器**: 124.236.79.6:20009  
**项目分支**: codex/v5-clean-low-dose-friedel

## 1. GPU状态与运行情况

### GPU使用情况
- **所有GPU (0-7) 均已空闲**，显存占用均为 1MiB
- **无运行中的进程**
- 结论: ✅ 实验已全部完成

### 最近完成的实验

根据文件时间戳，服务器最近运行了以下实验：

1. **Pyxem模板匹配评估** (7月31日 00:51完成)
   - `V5_PYXEM_CLEAN_E_FULL_EVALUATION.json` (1.4K)
   - `pyxem_clean_e_full_details.jsonl` (996K)

2. **ACOM Clean C 全量测试** (7月31日 01:00-01:28完成)
   - `acom_clean_c_full/py4dstem/` 目录
   - `acom_clean_c_full/py4dstem_noiseless/` 目录

3. **Clean C Py4DSTEM检测** (7月31日 08:03完成)
   - `clean_c_full_py4dstem_detection.json` (181K)
   - `clean_c_full_py4dstem_detection_high_dose_shard.json` (71K)
   - `clean_c_dose_noiseless_full_py4dstem_detection.json` (6.2K)

## 2. 代码文件同步情况

### 服务器上已修改但未提交的tracked文件

| 文件 | 状态 | 本地是否同步 |
|------|------|------------|
| `.gitignore` | Modified | ❌ 本地未修改 |
| `config/benchmark_v5.yaml` | Modified | ❌ 本地未修改 |
| `environment-clean.yml` | Modified | ❌ 本地未修改 |
| `scripts/03_extract_clean_disks.py` | Modified | ❌ 本地未修改 |
| `src/py4dstem_disk_adapter.py` | Modified | ❌ 本地未修改 |

### 服务器上新增的untracked代码文件

| 文件 | 本地是否存在 |
|------|------------|
| `scripts/25_run_v5_pyxem_template_matching.py` | ✅ 存在 |
| `scripts/26_evaluate_v5_pyxem.py` | ✅ 存在 (但本地有修改) |
| `scripts/27_run_v5_clean_c_acom_full.py` | ✅ 存在 |
| `src/pyxem_template_adapter.py` | ✅ 存在 |
| `tests/test_pyxem_template_adapter.py` | ✅ 存在 |

### 主要代码差异说明

#### 1. `.gitignore`
- 新增: `pyxem_templates/` 和 `*pyxem_template_library*.pickle` 忽略规则

#### 2. `config/benchmark_v5.yaml`
- 新增: `pyxem_template_matching` 配置块，包含模板匹配所有参数

#### 3. `environment-clean.yml`
- 新增依赖: `pyxem=0.21`

#### 4. `scripts/03_extract_clean_disks.py`
- 新增: `--progress-every` 参数，控制进度输出频率
- 改进: 批处理时的异常处理逻辑

#### 5. `src/py4dstem_disk_adapter.py`
- 改进: 批处理函数增加异常捕获，单个样本失败不影响整批

## 3. 数据文件分布情况

### 服务器数据存储
- **数据目录**: `/mnt/data/xietianhong/or4d-clean-v5` (5.9G)
  - datasets: 4.2G
  - intermediates: 1.6G
  - results: 46M
  - diagnostics: 28M
  - logs: 4.6M

- **报告目录**: `/home/xietianhong/OR4D-NCM811-smoke-v0/reports/v5` (100M)

### 本地数据存储
- **报告目录**: `reports/v5` (29M)
- **缺失的服务器大型结果**:
  - ❌ `acom_clean_c_full/` 目录 (服务器独有，本地不存在)
  - ❌ `clean_c_full_py4dstem_detection.json` (181K，服务器独有)
  - ❌ `clean_c_full_py4dstem_detection_high_dose_shard.json` (71K)
  - ❌ `clean_c_dose_noiseless_full_py4dstem_detection.json` (6.2K)

### 已同步到本地的小结果文件
- ✅ `V5_PYXEM_CLEAN_E_FULL_EVALUATION.json` (1.4K)
- ✅ `pyxem_clean_e_full_details.jsonl` (996K)

## 4. 总结

### ✅ 正常情况
1. 服务器GPU全部空闲，实验已完成
2. 新增的代码文件 (scripts 25/26/27, pyxem_template_adapter) 本地已同步
3. 小型结果文件已同步到本地
4. 大型数据集保留在服务器的 `/mnt/data` 目录 (符合预期)

### ⚠️ 需要同步的内容
1. **代码修改**: 5个已修改的tracked文件需要从服务器拉取到本地
   - `.gitignore`
   - `config/benchmark_v5.yaml`
   - `environment-clean.yml`
   - `scripts/03_extract_clean_disks.py`
   - `src/py4dstem_disk_adapter.py`

2. **大型实验结果**: 服务器独有的Clean C全量检测结果
   - `clean_c_full_py4dstem_detection.json` (181K)
   - `clean_c_full_py4dstem_detection_high_dose_shard.json` (71K)
   - `clean_c_dose_noiseless_full_py4dstem_detection.json` (6.2K)
   - `acom_clean_c_full/` 目录

3. **本地修改**: `scripts/26_evaluate_v5_pyxem.py` 本地有修改，需确认是否与服务器版本冲突

### 📊 数据分布符合规范
- ✅ 大型数据集 (5.9G) 保留在服务器 `/mnt/data/xietianhong/or4d-clean-v5`
- ✅ 代码文件在 `/home/xietianhong/OR4D-NCM811-smoke-v0`
- ✅ 只有必要的小型结果文件需要同步到本地
