#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

conda run --no-capture-output -n or4d-clean python scripts/01_generate_orientations.py
conda run --no-capture-output -n or4d-clean python scripts/02_generate_clean.py
conda run --no-capture-output -n or4d-clean python scripts/04_validate_dataset.py
conda run --no-capture-output -n or4d-clean python scripts/07_run_acom_baseline.py
conda run --no-capture-output -n or4d-clean python scripts/06_evaluate_submission.py \
  submissions/acom_clean_predictions.jsonl \
  --track clean \
  --output reports/acom_clean_evaluation.json
conda run --no-capture-output -n or4d-clean python scripts/08_visualize_acom_results.py
conda run --no-capture-output -n or4d-clean python scripts/09_write_acom_report.py
