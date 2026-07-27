# OR4D NCM811 Peak Benchmark — five-orientation smoke test

> **Note**
>
> Currently, only the Clean-Peak track has been evaluated. For more details,
> see `OR4D_NCM811_CleanPeak_ACOM_Benchmark_Report_v2.pdf`.


This project creates two tracks with one shared input/output contract.

- Public input: variable-length peak sets `{qx, qy, intensity}` in `public/*.h5`.
- Private ground truth: `orientation_matrix_sample_to_crystal` in `private/*_ground_truth.jsonl`.
- Clean: py4DSTEM kinematical peak generation from the average-occupancy CIF.
- Dynamical: explicit random occupancy, abTEM multislice CBED/PACBED generation, then fixed peak extraction.

Two conda environments are used because py4DSTEM 0.14.18 requires NumPy `<2`, while abTEM 1.0.10 requires NumPy `>=2`.

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

