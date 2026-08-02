#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")" && pwd)
cd "${ROOT}"

if [[ -n "${OR4D_CLEAN_PYTHON:-}" ]]; then
  clean_python=("${OR4D_CLEAN_PYTHON}")
elif command -v conda >/dev/null 2>&1; then
  clean_python=(conda run --no-capture-output -n or4d-clean python)
else
  clean_python=(python3)
fi
export OR4D_CONFIG="${ROOT}/config/benchmark.yaml"

mode="${1:-smoke}"
stage="${2:-all}"
if [[ "${mode}" != "smoke" && "${mode}" != "full" ]]; then
  echo "usage: $0 [smoke|full] [all|peaks|acom]" >&2
  exit 2
fi
if [[ "${stage}" != "all" && "${stage}" != "peaks" && "${stage}" != "acom" ]]; then
  echo "usage: $0 [smoke|full] [all|peaks|acom]" >&2
  exit 2
fi

if [[ ! -f private/orientations.jsonl || ! -f public/clean_peaks.h5 ]]; then
  "${clean_python[@]}" scripts/01_generate_orientations.py
  "${clean_python[@]}" scripts/02_generate_clean.py
fi
if [[ ! -f private/clean_oracle_peaks.h5 ]]; then
  cp public/clean_peaks.h5 private/clean_oracle_peaks.h5
fi

if [[ "${mode}" == "smoke" ]]; then
  role_args=(--role legacy_smoke)
  suffix="_smoke"
  subset_args=(--allow-subset)
else
  role_args=()
  suffix=""
  subset_args=()
fi

expectation_file="public/clean_images${suffix}.h5"
counted_file="public/clean_counted_images${suffix}.h5"
oracle_file="private/clean_physical_oracle_peaks${suffix}.h5"

if [[ "${stage}" == "all" || "${stage}" == "peaks" ]]; then
  "${clean_python[@]}" scripts/02b_generate_clean_images.py "${role_args[@]}"
  "${clean_python[@]}" scripts/02c_generate_clean_counted_images.py \
    --expectation-file "${expectation_file}" \
    --output "${counted_file}"

  "${clean_python[@]}" scripts/03_extract_clean_disks.py \
    --image-file "${expectation_file}" \
    --track expectation
  "${clean_python[@]}" scripts/13_evaluate_clean_image_pipeline.py \
    --image-file "${expectation_file}" \
    --oracle-file "${oracle_file}" \
    --detection-report "reports/clean_disk_detection_expectation${suffix}.json" \
    --output "reports/clean_image_pipeline_evaluation${suffix}.json" \
    --overlay-dir "diagnostics/clean_image_overlays${suffix}"

  "${clean_python[@]}" scripts/03_extract_clean_disks.py \
    --image-file "${counted_file}" \
    --track counted
  "${clean_python[@]}" scripts/13_evaluate_clean_image_pipeline.py \
    --image-file "${counted_file}" \
    --oracle-file "${oracle_file}" \
    --detection-report "reports/clean_disk_detection_counted${suffix}.json" \
    --output "reports/clean_counted_pipeline_evaluation${suffix}.json" \
    --overlay-dir "diagnostics/clean_counted_overlays${suffix}" \
    --overlay-count 0
fi

if [[ "${stage}" == "all" || "${stage}" == "acom" ]]; then
  "${clean_python[@]}" scripts/07_run_acom_baseline.py \
    --peak-file "${oracle_file}" \
    --output-tag "physical_oracle${suffix}" \
    --prediction-file "submissions/acom_clean_physical_oracle${suffix}.jsonl" \
    "${subset_args[@]}"
  for detector in autodisk py4dstem; do
    "${clean_python[@]}" scripts/07_run_acom_baseline.py \
      --peak-file "diagnostics/clean_expectation_${detector}_peaks${suffix}.h5" \
      --output-tag "expectation_${detector}${suffix}" \
      --prediction-file "submissions/acom_clean_expectation_${detector}${suffix}.jsonl" \
      "${subset_args[@]}"
  done

  "${clean_python[@]}" scripts/14_compare_clean_acom.py \
    --baseline-details "reports/acom_clean_details_physical_oracle${suffix}.json" \
    --candidate "autodisk_expectation=reports/acom_clean_details_expectation_autodisk${suffix}.json" \
    --candidate "py4dstem_expectation=reports/acom_clean_details_expectation_py4dstem${suffix}.json" \
    --output "reports/clean_acom_comparison${suffix}.json"

  "${clean_python[@]}" scripts/15_run_clean_counted_acom.py \
    --detection-report "reports/clean_disk_detection_counted${suffix}.json" \
    --baseline-details "reports/acom_clean_details_physical_oracle${suffix}.json" \
    --output "reports/clean_counted_acom_comparison${suffix}.json" \
    --max-workers 2 \
    "${subset_args[@]}"
fi
