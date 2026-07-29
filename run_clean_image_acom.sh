#!/usr/bin/env bash
set -euo pipefail

mode="${1:-smoke}"
if [[ "${mode}" != "smoke" && "${mode}" != "full" ]]; then
  echo "usage: $0 [smoke|full]" >&2
  exit 2
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

python scripts/02b_generate_clean_images.py "${role_args[@]}"
python scripts/02c_generate_clean_counted_images.py \
  --expectation-file "${expectation_file}" \
  --output "${counted_file}"

python scripts/03_extract_clean_disks.py \
  --image-file "${expectation_file}" \
  --track expectation
python scripts/13_evaluate_clean_image_pipeline.py \
  --image-file "${expectation_file}" \
  --oracle-file "${oracle_file}" \
  --detection-report "reports/clean_disk_detection_expectation${suffix}.json" \
  --output "reports/clean_image_pipeline_evaluation${suffix}.json" \
  --overlay-dir "diagnostics/clean_image_overlays${suffix}"

python scripts/03_extract_clean_disks.py \
  --image-file "${counted_file}" \
  --track counted
python scripts/13_evaluate_clean_image_pipeline.py \
  --image-file "${counted_file}" \
  --oracle-file "${oracle_file}" \
  --detection-report "reports/clean_disk_detection_counted${suffix}.json" \
  --output "reports/clean_counted_pipeline_evaluation${suffix}.json" \
  --overlay-dir "diagnostics/clean_counted_overlays${suffix}" \
  --overlay-count 0

python scripts/07_run_acom_baseline.py \
  --peak-file "${oracle_file}" \
  --output-tag "physical_oracle${suffix}" \
  --prediction-file "submissions/acom_clean_physical_oracle${suffix}.jsonl" \
  "${subset_args[@]}"
for detector in autodisk py4dstem; do
  python scripts/07_run_acom_baseline.py \
    --peak-file "diagnostics/clean_expectation_${detector}_peaks${suffix}.h5" \
    --output-tag "expectation_${detector}${suffix}" \
    --prediction-file "submissions/acom_clean_expectation_${detector}${suffix}.jsonl" \
    "${subset_args[@]}"
done

python scripts/14_compare_clean_acom.py \
  --baseline-details "reports/acom_clean_details_physical_oracle${suffix}.json" \
  --candidate "autodisk_expectation=reports/acom_clean_details_expectation_autodisk${suffix}.json" \
  --candidate "py4dstem_expectation=reports/acom_clean_details_expectation_py4dstem${suffix}.json" \
  --output "reports/clean_acom_comparison${suffix}.json"

python scripts/15_run_clean_counted_acom.py \
  --detection-report "reports/clean_disk_detection_counted${suffix}.json" \
  --baseline-details "reports/acom_clean_details_physical_oracle${suffix}.json" \
  --output "reports/clean_counted_acom_comparison${suffix}.json" \
  --max-workers 2 \
  "${subset_args[@]}"
