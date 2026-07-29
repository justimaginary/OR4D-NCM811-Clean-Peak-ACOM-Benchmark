# OR4D NCM811 Clean-Peak ACOM Benchmark

> **Note**
>
> Currently, only the Clean-Peak track has been evaluated. For more details,
> see [OR4D NCM811 Clean-Peak ACOM Benchmark Report v3](OR4D_NCM811_CleanPeak_ACOM_Benchmark_Report_v3.pdf).

> **Status**
>
> Clean-Peak v3 is the only evaluated track. It tests matched-model
> orientation recovery from ideal kinematical diffraction peaks; it does not
> establish performance on experimental data or dynamical diffraction.

This repository builds a synthetic NCM811 orientation benchmark and evaluates a
py4DSTEM ACOM baseline using one shared contract:

- public input: variable-length peak sets `{qx, qy, intensity}`;
- private ground truth: a `3×3 orientation_matrix_sample_to_crystal`;
- prediction: one matrix with the same convention for every sample.

The v3 dataset contains 1,081 samples: 17 legacy regression cases, 1,024
scrambled Sobol-SO(3) headline cases, and 40 ACOM-grid probes. Only the 1,024
`headline_core` samples contribute to the main score. Peak generation uses
`Kmax = 1.5 Å⁻¹`.

## Results

Errors are reduced by proper crystal symmetry and Friedel equivalence.

| ACOM step | Mean | Median | P95 | Acc@1° | Acc@2° | Acc@5° | Throughput |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4° | 2.991° | 1.625° | 5.311° | 20.8% | 68.3% | 94.6% | 70.4 samples/s |
| 3° | 2.581° | 1.283° | 5.069° | 32.3% | 84.7% | 94.8% | 32.9 samples/s |
| **2° (canonical)** | **1.625°** | **0.977°** | **3.525°** | **52.1%** | **90.5%** | **96.7%** | **9.8 samples/s** |

The canonical run still has a small catastrophic-error tail: 34/1,024 samples
exceed 5°, 12/1,024 exceed 10°, and the maximum error is 89.850°.

## Run

The Clean and Dynamical tracks use separate environments because py4DSTEM
0.14.18 requires NumPy `<2`, while abTEM 1.0.10 requires NumPy `>=2`.

```bash
conda env create -f environment-clean.yml
conda env create -f environment-dynamical.yml
```

Run the complete Clean-Peak benchmark and ACOM sweep:

```bash
conda run -n or4d-clean ./run_clean_acom.sh
```

The pipeline runs:

```text
01_generate_orientations.py
02_generate_clean.py
04_validate_dataset.py
11_run_acom_sweep.py
08_visualize_acom_results.py
09_write_acom_report.py
10_trace_clean_coordinates.py --all
12_write_coordinate_visualization.py
```

`11_run_acom_sweep.py` runs and evaluates the 4°, 3°, and 2° ACOM configurations
internally. Do not run scripts 07 and 06 again after the sweep.

### Experimental Clean image-input pipeline

The repository now also contains a separate, Clean-only image path:

```text
CIF + orientation
→ ACOM-matched kinematical finite-disk expectation (Clean-E)
→ fixed-total multinomial electron counts (Clean-C)
→ AutoDisk and py4DSTEM find_Bragg_disks
→ the same 2° ACOM
→ peak, runtime, dose, and orientation comparisons
```

This path does not overwrite the accepted Clean-Peak v3 files. Run its 17-case
regression set first:

```bash
conda run -n or4d-clean ./run_clean_image_acom.sh smoke
```

The corrected 17-case smoke passes. The 1,081-pattern peak stage has also been
run. Run or reproduce only that stage with:

```bash
conda run -n or4d-clean ./run_clean_image_acom.sh full peaks
```

After reviewing the peak report, continue from the existing detector outputs
without regenerating the images:

```bash
conda run -n or4d-clean ./run_clean_image_acom.sh full acom
```

The image run freezes the old analytic input locally as
`private/clean_oracle_peaks.h5`, writes reproducible generated HDF5 files, and
runs both disk detectors on exactly the same images. `find_Bragg_disks` is the
real py4DSTEM 0.14.18 public function, not an alias for the local AutoDisk
implementation. Both outputs are converted to the same physical
`(qx, qy, integrated intensity)` contract before ACOM.

Current 1,081-pattern status: both detectors recover Clean-E with 100%
precision and recall. At \(10^5\) electrons, AutoDisk reaches 95.66%/95.66%
precision/recall and `find_Bragg_disks` reaches 99.38%/99.38%. At \(10^6\),
AutoDisk reaches 99.95% and `find_Bragg_disks` reaches 100%. Full ACOM is also
complete: the image-matched oracle reaches 90.19% Acc@2°, while \(10^6\)
`find_Bragg_disks` matches that mean with a 0.0067° P95 orientation delta.
AutoDisk saturates near a 0.076–0.079° P95 delta and consistently introduces
one new >5° case. The separate finite-thickness First-Born mode remains
diagnostic because its sinc excitation envelope is not matched by the current
ACOM orientation plan.

See [`docs/CLEAN_IMAGE_PIPELINE.md`](docs/CLEAN_IMAGE_PIPELINE.md) for the
formulas, HDF5 schema, detector comparison, commands, and current measured
limitations.

Run the tests:

```bash
conda run -n or4d-clean \
  python -m unittest discover -s tests -p 'test_*.py'
```

Print one complete coordinate trace:

```bash
conda run -n or4d-clean \
  python scripts/10_trace_clean_coordinates.py \
  --sample-id clean_core_0000 --stdout
```

Use `--all --stdout` to print every sample. The compressed full trace is stored
in `diagnostics/clean_coordinate_trace.jsonl.gz`.

## Interactive HTML

Open
[`reports/ACOM_COORDINATE_VISUALIZATION.html`](reports/ACOM_COORDINATE_VISUALIZATION.html)
directly in a browser:

```bash
open reports/ACOM_COORDINATE_VISUALIZATION.html
```

The self-contained page includes:

- an overview and 4°/3°/2° result comparison;
- separate Benchmark/GT and ACOM prediction layers, plus an overlay;
- bilingual, collapsible run parameters;
- Best, Median, P95, and Worst representative samples;
- detector reciprocal-space peaks and orientation-matrix axes;
- every valid HKL for the selected GT pattern;
- the complete `HKL → B → g_crystal → g_sample → q` trace;
- GT/ACOM coordinates, peak matches, and coordinate differences.

The purple circle and vector mark the currently selected GT reflection. They
are not an extra peak or a third result.

The HTML is an offline snapshot. Its controls redraw embedded data in real
time, but the page does not reread JSON/HDF5 files or rerun ACOM. Regenerate it
after the underlying results change:

```bash
conda run -n or4d-clean \
  python scripts/12_write_coordinate_visualization.py
```

## Report

The 44-page static report is
[`OR4D_NCM811_CleanPeak_ACOM_Benchmark_Report_v3.pdf`](OR4D_NCM811_CleanPeak_ACOM_Benchmark_Report_v3.pdf).

It documents the task contract, sampling design, simulation parameters, ACOM
search, symmetry-aware evaluation, 4°/3°/2° results, failure cases, coordinate
and HKL audit, figures, software versions, file hashes, and reproduction
commands. The generated PDF is tracked; its Typst source is not currently in
this repository.

Machine-readable results and the generated Markdown report are under
[`reports/`](reports/), including:

- `ACOM_CLEAN_REPORT.md`: generated numerical report;
- `ACOM_COORDINATE_ANALYSIS.md`: coordinate and peak-level audit;
- `acom_clean_evaluation*.json`: symmetry/Friedel-aware metrics;
- `acom_clean_details*.json`: per-sample ACOM diagnostics;
- `acom_plan_audit*.json`: orientation-plan metadata.


## Repository layout

```text
config/       benchmark, Clean, ACOM, and evaluation parameters
data/         NCM811 CIF and generated metadata
public/       participant-visible peak sets
private/      orientations and hidden ground truth
submissions/  ACOM predictions and submission examples
diagnostics/  HKL reflections and complete coordinate traces
reports/      metrics, figures, analyses, and interactive HTML
scripts/      generation, prediction, evaluation, and reporting tools
```

See [`docs/CLEAN_BENCHMARK_V3.md`](docs/CLEAN_BENCHMARK_V3.md) for the detailed
sampling and evaluation contract.

## Limitations

- Clean inputs and ACOM templates share the same CIF and py4DSTEM kinematical
  model; this is a matched-model self-consistency benchmark.
- The `[001]` orientation family has a stable approximately 72° failure mode at
  `Kmax = 1.5 Å⁻¹`.
- A 2° plan is more accurate than a 4° plan but uses approximately eight times
  as many orientation seeds and substantially more runtime.
- Grid probes are diagnostic only and must not be mixed into headline metrics.
- Public, private, and evaluation files coexist for local reproducibility; this
  repository is not a server-side blind benchmark.
