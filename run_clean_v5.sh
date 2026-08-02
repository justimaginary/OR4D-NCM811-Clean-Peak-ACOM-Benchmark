#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")" && pwd)
stage="${1:-check}"
study="${2:-all}"
if [[ -n "${OR4D_V5_DATA_ROOT:-}" ]]; then
  data_root="${OR4D_V5_DATA_ROOT}"
elif [[ -d "/mnt/data/${USER:-xietianhong}/or4d-clean-v5" ]]; then
  data_root="/mnt/data/${USER:-xietianhong}/or4d-clean-v5"
else
  data_root="${ROOT}/../or4d-clean-v5"
fi
cpu_threads="${OR4D_CPU_THREADS:-8}"
detector_backend="${OR4D_DETECTOR_BACKEND:-cpu}"
acom_cuda_args=()

if ! [[ "${cpu_threads}" =~ ^[0-9]+$ ]] || (( cpu_threads < 1 || cpu_threads > 16 )); then
  echo "OR4D_CPU_THREADS must be an integer in [1, 16]" >&2
  exit 2
fi
if [[ "${detector_backend}" != "cpu" && "${detector_backend}" != "cuda" ]]; then
  echo "OR4D_DETECTOR_BACKEND must be cpu or cuda" >&2
  exit 2
fi
if [[ "${OR4D_ACOM_CUDA:-0}" == "1" ]]; then
  acom_cuda_args=(--cuda)
elif [[ "${OR4D_ACOM_CUDA:-0}" != "0" ]]; then
  echo "OR4D_ACOM_CUDA must be 0 or 1" >&2
  exit 2
fi

if [[ "${stage}" != "check" && "${stage}" != "detect-e" && "${stage}" != "acom-e" && "${stage}" != "pyxem" && "${stage}" != "clean-c" && "${stage}" != "report" ]]; then
  echo "usage: $0 [check|detect-e|acom-e|pyxem|clean-c|report] [main|001|all]" >&2
  exit 2
fi
if [[ "${study}" != "main" && "${study}" != "001" && "${study}" != "all" ]]; then
  echo "study must be main, 001, or all" >&2
  exit 2
fi

if [[ -n "${OR4D_CLEAN_PYTHON:-}" ]]; then
  clean_python=("${OR4D_CLEAN_PYTHON}")
elif command -v conda >/dev/null 2>&1; then
  clean_python=(conda run --no-capture-output -n or4d-clean python)
else
  clean_python=(python3)
fi

export OR4D_CONFIG="${ROOT}/config/benchmark_v5.yaml"
mkdir -p "${ROOT}/reports/v5"

case "${stage}" in
  check)
    # V5 was generated on the server. This stage only checks that the existing
    # server artifact root matches the tracked manifest; it never writes data.
    "${clean_python[@]}" "${ROOT}/scripts/20_check_v5_artifacts.py" \
      --data-root "${data_root}" --study "${study}"
    ;;
  detect-e)
    if [[ "${study}" == "main" || "${study}" == "all" ]]; then
      [[ -f "${data_root}/datasets/clean_v5_first_born_expectation_2048.h5" ]] || { echo "existing V5 server artifact is missing; run '$0 check main'" >&2; exit 1; }
      "${clean_python[@]}" "${ROOT}/scripts/03_extract_clean_disks.py" \
        --image-file "${data_root}/datasets/clean_v5_first_born_expectation_2048.h5" \
        --track expectation --detector autodisk --detector dog_rgm \
        --output-dir "${data_root}/intermediates/detected_expectation_2048" \
        --report-output "${data_root}/reports/first_born_disk_detection_expectation_2048_cpu.json"
      "${clean_python[@]}" "${ROOT}/scripts/03_extract_clean_disks.py" \
        --image-file "${data_root}/datasets/clean_v5_first_born_expectation_2048.h5" \
        --track expectation --detector py4dstem --compute-backend "${detector_backend}" \
        --output-dir "${data_root}/intermediates/detected_expectation_2048" \
        --report-output "${data_root}/reports/first_born_disk_detection_expectation_2048_py4dstem.json"
    fi
    if [[ "${study}" == "001" || "${study}" == "all" ]]; then
      [[ -f "${data_root}/datasets/clean_v5_001_first_born_expectation_512.h5" ]] || { echo "existing V5 [001] server artifact is missing; run '$0 check 001'" >&2; exit 1; }
      "${clean_python[@]}" "${ROOT}/scripts/03_extract_clean_disks.py" \
        --image-file "${data_root}/datasets/clean_v5_001_first_born_expectation_512.h5" \
        --track expectation --detector autodisk --detector dog_rgm \
        --output-dir "${data_root}/intermediates/detected_001_expectation_512" \
        --report-output "${data_root}/reports/first_born_001_disk_detection_expectation_512_cpu.json"
      "${clean_python[@]}" "${ROOT}/scripts/03_extract_clean_disks.py" \
        --image-file "${data_root}/datasets/clean_v5_001_first_born_expectation_512.h5" \
        --track expectation --detector py4dstem --compute-backend "${detector_backend}" \
        --output-dir "${data_root}/intermediates/detected_001_expectation_512" \
        --report-output "${data_root}/reports/first_born_001_disk_detection_expectation_512_py4dstem.json"
    fi
    ;;
  acom-e)
    "${clean_python[@]}" "${ROOT}/scripts/23_run_v5_acom_suite.py" \
      --data-root "${data_root}" --study "${study}" \
      --output-root "${data_root}/results/acom_suite" --cpu-threads "${cpu_threads}" --resume \
      "${acom_cuda_args[@]}"
    ;;
  pyxem)
    target="${OR4D_PYXEM_TARGET:-gpu}"
    if [[ "${study}" == "main" || "${study}" == "all" ]]; then
      "${clean_python[@]}" "${ROOT}/scripts/25_run_v5_pyxem_template_matching.py" \
        --data-root "${data_root}" --study main --track all --target "${target}" \
        --output-file "${data_root}/results/pyxem_top5_merged.h5"
    fi
    if [[ "${study}" == "001" || "${study}" == "all" ]]; then
      "${clean_python[@]}" "${ROOT}/scripts/25_run_v5_pyxem_template_matching.py" \
        --data-root "${data_root}" --study 001 --track all --target "${target}" \
        --output-file "${data_root}/results/pyxem_001_top5.h5"
    fi
    ;;
  clean-c)
    devices="${CUDA_VISIBLE_DEVICES:-}"
    [[ -n "${devices}" ]] || { echo "clean-c full requires CUDA_VISIBLE_DEVICES (one physical GPU by default)" >&2; exit 1; }
    if [[ "${devices}" == *,* && "${OR4D_ALLOW_TWO_GPUS:-0}" != "1" ]]; then
      echo "Two GPUs require explicit OR4D_ALLOW_TWO_GPUS=1 for this run" >&2
      exit 1
    fi
    cpu_worker_limit="${OR4D_CPU_WORKERS:-16}"
    if ! [[ "${cpu_worker_limit}" =~ ^[0-9]+$ ]]; then
      echo "OR4D_CPU_WORKERS must be an integer" >&2
      exit 2
    fi
    if [[ "${devices}" == *,* ]]; then
      (( cpu_worker_limit <= 32 )) || { echo "OR4D_CPU_WORKERS must be <= 32 for two GPUs" >&2; exit 2; }
    else
      (( cpu_worker_limit <= 16 )) || { echo "OR4D_CPU_WORKERS must be <= 16 for one GPU" >&2; exit 2; }
    fi
    if [[ "${study}" == "main" || "${study}" == "all" ]]; then
      "${clean_python[@]}" "${ROOT}/scripts/31_run_v5_clean_c_autodisk_dog_full.py" \
        --data-root "${data_root}" \
        --ground-truth-file "${data_root}/manifests/clean_v5_orientations.jsonl" \
        --study main --cuda-visible-device "${devices}" \
        --detection-workers "${OR4D_DETECTION_WORKERS:-8}" \
        --acom-workers "${OR4D_ACOM_WORKERS:-4}" \
        --cpu-worker-limit "${cpu_worker_limit}" --resume
    fi
    if [[ "${study}" == "001" || "${study}" == "all" ]]; then
      "${clean_python[@]}" "${ROOT}/scripts/31_run_v5_clean_c_autodisk_dog_full.py" \
        --data-root "${data_root}" \
        --ground-truth-file "${data_root}/manifests/clean_v5_001_orientations.jsonl" \
        --study 001 --cuda-visible-device "${devices}" \
        --detection-workers "${OR4D_DETECTION_WORKERS:-8}" \
        --acom-workers "${OR4D_ACOM_WORKERS:-4}" \
        --cpu-worker-limit "${cpu_worker_limit}" --resume
    fi
    ;;
  report)
    echo "Compact V5 summaries are produced by scripts/29_finalize_v5_top5.py and scripts/32_summarize_v5_clean_c_disk_recovery.py." >&2
    echo "The canonical server artifact paths and hashes are recorded in reports/v5/MANIFEST.json." >&2
    ;;
esac
