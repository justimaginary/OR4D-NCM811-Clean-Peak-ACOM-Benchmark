#!/usr/bin/env bash
set -euo pipefail

conda run -n or4d-clean python scripts/01_generate_orientations.py
conda run -n or4d-clean python scripts/02_generate_clean.py
conda run -n or4d-dynamical python scripts/03_generate_dynamical.py
conda run -n or4d-clean python scripts/04_validate_dataset.py
conda run -n or4d-clean python scripts/05_make_submission_template.py
