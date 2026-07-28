#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [[ -n "${OR4D_CLEAN_PYTHON:-}" ]]; then
  clean_python="$OR4D_CLEAN_PYTHON"
elif [[ -x /opt/anaconda3/envs/or4d-clean/bin/python ]]; then
  clean_python=/opt/anaconda3/envs/or4d-clean/bin/python
else
  clean_python=python
fi

"$clean_python" scripts/01_generate_orientations.py
"$clean_python" scripts/02_generate_clean.py
"$clean_python" scripts/04_validate_dataset.py
"$clean_python" scripts/11_run_acom_sweep.py
"$clean_python" scripts/08_visualize_acom_results.py
"$clean_python" scripts/09_write_acom_report.py
"$clean_python" scripts/10_trace_clean_coordinates.py --all
"$clean_python" scripts/12_write_coordinate_visualization.py
