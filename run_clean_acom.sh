#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

conda run --no-capture-output -n or4d-clean python scripts/01_generate_orientations.py
conda run --no-capture-output -n or4d-clean python scripts/02_generate_clean.py
conda run --no-capture-output -n or4d-clean python scripts/04_validate_dataset.py
conda run --no-capture-output -n or4d-clean python scripts/11_run_acom_sweep.py
conda run --no-capture-output -n or4d-clean python scripts/08_visualize_acom_results.py
conda run --no-capture-output -n or4d-clean python scripts/09_write_acom_report.py
conda run --no-capture-output -n or4d-clean python \
  scripts/10_trace_clean_coordinates.py --all
