# OR4D NCM811 Clean-Peak Benchmark v2

> **Note**
>
> Currently, only the Clean-Peak track has been evaluated. The generated v2
> report is `reports/ACOM_CLEAN_REPORT.md`; the root PDF is the legacy v1 report.
> Other v1 smoke artifacts are kept under `reports/legacy/` and
> `submissions/legacy/`.


This project creates two tracks with one shared input/output contract.

- Public input: variable-length peak sets `{qx, qy, intensity}` in `public/*.h5`.
- Private ground truth: `orientation_matrix_sample_to_crystal` in `private/*_ground_truth.jsonl`.
- Clean: py4DSTEM kinematical peak generation from the average-occupancy CIF.
- Dynamical: explicit random occupancy, abTEM multislice CBED/PACBED generation, then fixed peak extraction.

Two conda environments are used because py4DSTEM 0.14.18 requires NumPy `<2`, while abTEM 1.0.10 requires NumPy `>=2`.

Clean v2 contains 17 legacy smoke samples, 512 ACOM-independent Sobol-SO(3)
headline samples, and 40 ACOM grid probes. See
`docs/CLEAN_BENCHMARK_V2.md` for the sampling and reporting contract.

## Run

```bash
conda env create -f environment-clean.yml
conda env create -f environment-dynamical.yml

conda run -n or4d-clean python scripts/01_generate_orientations.py
conda run -n or4d-clean python scripts/02_generate_clean.py
conda run -n or4d-dynamical python scripts/03_generate_dynamical.py
conda run -n or4d-clean python scripts/04_validate_dataset.py
conda run -n or4d-clean python scripts/05_make_submission_template.py
conda run -n or4d-clean python scripts/06_evaluate_submission.py submissions/submission_example.jsonl
```
