#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")" && pwd)
cd "${ROOT}"

resolver_python="${OR4D_BOOTSTRAP_PYTHON:-python3}"
clean_python=$("${resolver_python}" scripts/00_resolve_runtime_paths.py clean-python)
report_dir=$("${resolver_python}" scripts/00_resolve_runtime_paths.py report-dir --version v3)

export OR4D_CONFIG="${ROOT}/config/benchmark.yaml"
export OR4D_REPORT_V3_DIR="${report_dir}"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${report_dir}"

"${clean_python}" scripts/01_generate_orientations.py
"${clean_python}" scripts/02_generate_clean.py
"${clean_python}" scripts/04_validate_dataset.py
"${clean_python}" scripts/11_run_acom_sweep.py
"${clean_python}" scripts/08_visualize_acom_results.py
"${clean_python}" scripts/09_write_acom_report.py
"${clean_python}" scripts/10_trace_clean_coordinates.py --all
"${clean_python}" scripts/12_write_coordinate_visualization.py
