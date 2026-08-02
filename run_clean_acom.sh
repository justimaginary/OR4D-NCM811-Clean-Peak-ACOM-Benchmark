#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")" && pwd)
cd "${ROOT}"

if [[ -n "${OR4D_CLEAN_PYTHON:-}" ]]; then
  clean_python=("${OR4D_CLEAN_PYTHON}")
elif command -v conda >/dev/null 2>&1; then
  clean_python=(conda run --no-capture-output -n or4d-clean python)
elif [[ -x /opt/anaconda3/envs/or4d-clean/bin/python ]]; then
  clean_python=(/opt/anaconda3/envs/or4d-clean/bin/python)
else
  clean_python=(python3)
fi

export OR4D_CONFIG="${ROOT}/config/benchmark.yaml"

"${clean_python[@]}" scripts/01_generate_orientations.py
"${clean_python[@]}" scripts/02_generate_clean.py
"${clean_python[@]}" scripts/04_validate_dataset.py
"${clean_python[@]}" scripts/11_run_acom_sweep.py
"${clean_python[@]}" scripts/08_visualize_acom_results.py
"${clean_python[@]}" scripts/09_write_acom_report.py
"${clean_python[@]}" scripts/10_trace_clean_coordinates.py --all
"${clean_python[@]}" scripts/12_write_coordinate_visualization.py
