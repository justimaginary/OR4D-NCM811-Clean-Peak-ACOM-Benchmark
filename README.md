# OR4D NCM811 Clean-Peak ACOM Benchmark

> **Note**
>
> V3 is the direct peak-input baseline. V4 and V5 add the Clean diffraction
> image and dose/noise observation tracks; their compact results and pages are
> included below. For the original benchmark contract, see [OR4D NCM811
> Clean-Peak ACOM Benchmark Report v3](OR4D_NCM811_CleanPeak_ACOM_Benchmark_Report_v3.pdf).

> **Status**
>
> All three versions are Clean, matched-model synthetic benchmarks. They test
> kinematical orientation recovery and detector robustness; they do not
> establish performance on experimental data or dynamical diffraction.

## Clone, data, and reproducibility

The Git checkout is intentionally code-and-results only. It contains the
source CIF, YAML configurations, generation/detection/indexing scripts, tests,
compact JSON/JSONL summaries, figures, PDFs, and self-contained HTML
snapshots. Generated peak tables, orientation manifests, diffraction images,
count images, detector-peak HDF5 files, and per-sample caches are not stored in
Git. They are written to an external data root (on the server use
`/mnt/data/$USER/or4d-clean-v5`; locally set `OR4D_V5_DATA_ROOT` to another
directory).

The small V3 public peak table (`public/clean_peaks.h5`) and the compact
reflection table (`diagnostics/clean_reflections.h5`) are deliberate exceptions:
they are tracked because they are part of the direct-peak baseline and are
useful when rebuilding the V3 coordinate page. V4/V5 raw image stacks and
full detector caches remain external.

Small summaries, audit tables, representative overlays, and HTML payloads
remain under `reports/` and are tracked because they are needed to inspect and
regenerate the pages. The external run manifest records the Git commit,
configuration, commands, backend, CPU limit, output hashes, and timing, so a
generated data root can be checked against the exact checkout that produced it.

After cloning, create the single Clean environment. The tracked reports and
self-contained HTML files can be inspected without generated HDF5 files.
Re-running V3/V4 generates their ignored working data; a V5 full rerun reuses
the existing external server data root described below.

```bash
git clone <repository-url>
cd OR4D-NCM811-smoke-v0
conda env create -f environment-clean.yml   # environment name: or4d-clean
```

Machine-dependent paths are centralized in
[`config/runtime_paths.json`](config/runtime_paths.json). The versioned shell
entrypoints resolve the `or4d-clean` Python executable from that file instead
of assuming whichever `conda` happens to be first on `PATH`. They also resolve
the V3/V4/V5 report directories from the repository root. The following
environment variables override the checked-in defaults without editing code:

```text
OR4D_CLEAN_PYTHON       absolute path to or4d-clean/bin/python
OR4D_V5_DATA_ROOT       V5 external data root on a non-server machine
OR4D_REPORT_V3_DIR      optional V3 report-directory override
OR4D_REPORT_V4_DIR      optional V4 report-directory override
OR4D_REPORT_V5_DIR      optional V5 report-directory override
OR4D_RUNTIME_PATHS_FILE alternate runtime_paths.json
```

Inspect the paths that will be used before a run:

```bash
python3 scripts/00_resolve_runtime_paths.py show
```

The dynamical environment is retained as historical project scaffolding; the
current reproducible V3/V4/V5 experiments are Clean-only and do not run
multislice.

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
./run_clean_acom.sh
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
./run_clean_image_acom.sh smoke
```

The corrected 17-case smoke passes. The 1,081-pattern peak stage has also been
run. Run or reproduce only that stage with:

```bash
./run_clean_image_acom.sh full peaks
```

After reviewing the peak report, continue from the existing detector outputs
without regenerating the images:

```bash
./run_clean_image_acom.sh full acom
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

### V5 Clean-E/C and the existing server run

V5 keeps the expectation image and the counted observation as separate tracks:

- **Clean-E (expectation)** is the deterministic float32 kinematical
  First-Born expectation image before electron counting.
- **Clean-C (counted)** samples that same expectation with the configured
  electron dose and noise model. It is an observation layer, not a different
  scattering model.

The V5 input images, manifests, detector outputs and full candidate files were
already generated on the server. The canonical data root is
`/mnt/data/xietianhong/or4d-clean-v5`; the tracked
[`reports/v5/MANIFEST.json`](reports/v5/MANIFEST.json) records its
server-relative paths, sizes, hashes, and producing commit. Compact summaries,
plots, run records and HTML are already retained under `reports/v5`.

Resolve that location rather than embedding it in a command:

```bash
v5_data_root=$(python3 scripts/00_resolve_runtime_paths.py \
  v5-data-root --must-exist)
```

On another machine, set `OR4D_V5_DATA_ROOT` to a verified copy of the same
directory. The artifact sizes and SHA-256 values in `reports/v5/MANIFEST.json`
identify the canonical data. Its `git_commit` and `config_files` hashes record
the exact data-generation state; later Top-5 and Pyxem settings extend the
evaluation configuration without changing those frozen image artifacts.

No replacement V5 data-preparation wrapper is required. If a server rerun is
explicitly needed, use the existing project scripts in their original stages:

```text
01_generate_orientations.py / 01b_generate_001_study.py
02b_generate_clean_images.py
02c_generate_clean_counted_images.py
02d_generate_clean_dose_expectations.py
02e_generate_clean_noise_manifest.py
03_extract_clean_disks.py
18_write_v5_manifest.py
23_run_v5_acom_suite.py
25_run_v5_pyxem_template_matching.py
31_run_v5_clean_c_autodisk_dog_full.py
33_run_v5_001_py4_pyxem.py
```

These scripts read and write the pre-existing server data root; they are not
replaced by a second local generator. GPU stages must still be launched only on
an explicitly verified idle card, with the CPU limit required by `AGENTS.md`.

To rebuild the tracked V5 HTML from the existing server results, use the
original report producer:

```bash
clean_python=$(python3 scripts/00_resolve_runtime_paths.py clean-python)
v5_data_root=$(python3 scripts/00_resolve_runtime_paths.py \
  v5-data-root --must-exist)
export OR4D_REPORT_V5_DIR=$(python3 scripts/00_resolve_runtime_paths.py \
  report-dir --version v5)
"${clean_python}" scripts/30_write_v5_clean_visualization.py \
  --data-root "${v5_data_root}"
```

Run the tests:

```bash
clean_python=$(python3 scripts/00_resolve_runtime_paths.py clean-python)
PYTHONPATH=src "${clean_python}" -m unittest discover \
  -s tests -p 'test_*.py'
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

The frozen v3 coordinate page remains available. Open
[`reports/v3/ACOM_COORDINATE_VISUALIZATION.html`](reports/v3/ACOM_COORDINATE_VISUALIZATION.html)
directly in a browser:

```bash
open reports/v3/ACOM_COORDINATE_VISUALIZATION.html
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

The full image-first comparison is
[`reports/v4/CLEAN_IMAGE_ACOM_VISUALIZATION.html`](reports/v4/CLEAN_IMAGE_ACOM_VISUALIZATION.html):

```bash
open reports/v4/CLEAN_IMAGE_ACOM_VISUALIZATION.html
```

It keeps the v3 direct-peak results, parameters, matrices, HKL tables, and
`HKL → B → g_crystal → g_sample → q` trace, then compares them with the new
Clean-E/C diffraction-image paths. It also embeds the actual representative
512×512 images, a step-by-step explanation of image formation, explicit
definitions of v3/Clean-E/C, a separate v3 direct-input reciprocal-space
diagnostic, and side-by-side AutoDisk versus `find_Bragg_disks` TP/FP/FN
overlays with position-error lines. Dose curves and all completed ACOM
aggregate results are included as well.
Like the v3 page, it is a self-contained offline snapshot; regenerate it after
rerunning the benchmark:

```bash
conda run -n or4d-clean \
  python scripts/17_write_clean_image_visualization.py
```

The v5 dose/noise/Top-5 comparison is
[`reports/v5/ACOM_CLEAN_V5_VISUALIZATION.html`](reports/v5/ACOM_CLEAN_V5_VISUALIZATION.html):

```bash
open reports/v5/ACOM_CLEAN_V5_VISUALIZATION.html
```

It shows the full Clean-E and Clean-C result grid, Top-1 through Top-5
success curves, Acc@1°/2°/5°, median/P95 error, indexing coverage,
ACOM-versus-Pyxem comparisons, real input images and peak overlays, and
per-sample `HKL → B → g_crystal → g_sample → q` intermediate variables.
The three HTML pages link to one another. See
[`reports/README.md`](reports/README.md) for the result-directory layout.

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

- `reports/v3/ACOM_CLEAN_REPORT.md`: generated numerical report;
- `reports/v3/ACOM_COORDINATE_ANALYSIS.md`: coordinate and peak-level audit;
- `reports/v3/acom_clean_evaluation*.json`: symmetry/Friedel-aware metrics;
- `reports/v3/acom_clean_details*.json`: per-sample ACOM diagnostics;
- `reports/v3/acom_plan_audit*.json`: orientation-plan metadata.


## Repository layout

```text
config/       benchmark, Clean, ACOM, and evaluation parameters
data/         tracked NCM811 CIF and small source metadata
public/       tracked V3 peaks; generated V4/V5 image HDF5 is ignored
private/      generated orientations and hidden ground truth (ignored)
submissions/  ACOM predictions and submission examples
diagnostics/  tracked compact reflections/traces/overlays; large HDF5 ignored
reports/      metrics, figures, analyses, and interactive HTML
scripts/      generation, prediction, evaluation, and reporting tools
<external root>/datasets      generated V4/V5 images and count images
<external root>/intermediates generated detector peaks and traces
<external root>/run_records   command, hash, and environment manifests
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
- The proposed next detector is documented in
  [`docs/DETECTOR_PROPOSAL.md`](docs/DETECTOR_PROPOSAL.md). It is a design
  proposal only; no performance number is claimed until it is run through the
  same frozen evaluation contract.
- Public, private, and evaluation files coexist for local reproducibility; this
  repository is not a server-side blind benchmark.
