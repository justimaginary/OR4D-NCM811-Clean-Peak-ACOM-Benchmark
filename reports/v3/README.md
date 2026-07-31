# V3 — direct-peak ACOM baseline

Input is a peak list `(qx, qy, intensity)`, not a diffraction image.

## Canonical 2° result

Headline set: 1024 samples. The symmetry- and Friedel-equivalent orientation
error has median `0.977°`, P95 `3.525°`, Acc@1° `52.15%`, Acc@2° `90.53%`,
and Acc@5° `96.68%`. These values are copied from
`acom_clean_evaluation.json`; they are not recomputed or rounded inside the
HTML.

- `ACOM_COORDINATE_VISUALIZATION.html`: interactive coordinate, symmetry and
  Friedel diagnostics.
- `acom_clean_evaluation*.json`: 2°, 3° and 4° orientation-plan evaluations.
- `acom_clean_details*.json`: sample-level ACOM results, including all 40
  `acom_grid_probe` cases.
- `acom_plan_audit*.json`: exact orientation-plan construction audit.
- PNG files and the Markdown reports are static supporting summaries.
