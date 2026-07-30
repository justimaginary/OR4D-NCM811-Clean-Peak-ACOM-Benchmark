# V5 Clean ACOM results

## What was actually run

All results below were produced on the server in the `or4d-clean` Conda
environment. A single empty GPU was exposed with `CUDA_VISIBLE_DEVICES=2`;
OpenMP and BLAS thread counts were limited to 8. No dynamical or multislice
simulation was used.

Four inputs were evaluated with the same py4DSTEM ACOM configuration:

1. `oracle`: the physical First-Born peak list `(qx, qy, intensity)` was passed
   directly to ACOM. It does **not** pass through a 2D diffraction image or a
   disk detector. Each diffraction pattern still contains many reflections;
   “direct peak input” does not mean that only one reflection is supplied.
2. `autodisk`: the Clean-E 2D expectation image was processed by AutoDisk, and
   its recovered peak list was passed to ACOM.
3. `py4dstem`: the same image was processed by py4DSTEM
   `find_Bragg_disks`, then passed to ACOM.
4. `dog_rgm`: the same image was processed by the DoG-RGM detector, then
   passed to ACOM.

The main dataset has 2,048 orientations. The separate `[001]` study has 512
additional orientations and is not part of the 2,048 allocation.

## Frozen ACOM parameters

| Parameter | Value |
|---|---:|
| Accelerating voltage | 300 kV |
| `k_max` | 1.5 Å⁻¹ |
| Zone-axis angular step | 2° |
| In-plane angular step | 2° |
| Correlation kernel size | 0.08 Å⁻¹ |
| Excitation-error sigma | 0.02 Å⁻¹ |
| Radial power | 1.0 |
| Simulated intensity power | 0.25 |
| Experimental intensity power | 0.25 |
| Peak-distance tolerance | 0.01 Å⁻¹ |
| Minimum number of peaks | 3 |
| Inversion/Friedel search branch | enabled |
| Returned matches | 1 |

Evaluation uses proper crystal point-group operations and the Clean Friedel
equivalence. The primary metric is the resulting minimum misorientation.

## Main dataset: 2,048 orientations

| ACOM input | Median | P95 | Max | Acc@1° | Acc@2° | Acc@5° | >5° |
|---|---:|---:|---:|---:|---:|---:|---:|
| Direct physical peaks | 1.174° | 9.167° | 89.934° | 39.16% | 78.86% | 91.26% | 179 |
| Clean-E → AutoDisk peaks | 1.146° | 7.586° | 89.964° | 40.77% | 80.96% | 92.43% | 155 |
| Clean-E → `find_Bragg_disks` peaks | 1.142° | 7.586° | 89.969° | 41.06% | 80.91% | 92.48% | 154 |
| Clean-E → DoG-RGM peaks | 1.145° | 7.662° | 89.962° | 40.97% | 80.81% | 92.09% | 162 |

All four calls completed 2,048/2,048 samples. ACOM matching itself took
24.4–26.0 seconds per path; process startup, plan construction and evaluation
are additional.

The detected-peak paths are slightly better than the direct physical peak
path. This must not be interpreted as a detector creating information.
Plausible causes are removal of weak reflections and changes in the intensity
weights seen by ACOM. The median absolute change in per-sample GT error is
below 0.0005° for every detector, while a small set of branch changes produces
large outliers. This is therefore primarily a tail/branch-selection effect and
requires per-sample analysis.

## Separate `[001]` study: 512 orientations

| ACOM input | Median | P95 | Max | Acc@1° | Acc@2° | Acc@5° |
|---|---:|---:|---:|---:|---:|---:|
| Direct physical peaks | 71.715° | 73.058° | 73.625° | 0.39% | 2.73% | 15.23% |
| Clean-E → AutoDisk peaks | 71.647° | 72.814° | 73.625° | 0.78% | 3.13% | 15.63% |
| Clean-E → `find_Bragg_disks` peaks | 71.672° | 73.058° | 73.625° | 3.13% | 6.25% | 18.75% |
| Clean-E → DoG-RGM peaks | 71.672° | 73.058° | 73.625° | 1.17% | 4.30% | 16.80% |

The failure is present in the direct-peak path, so the 2D image renderer and
disk detectors are not its root cause. The dominant error near 72° points to a
systematic orientation-representation, search-coverage, or equivalence issue
near `[001]`, not random peak-location noise. This dataset should be analyzed
by `study_group`, tilt, azimuth and in-plane angle before changing ACOM
parameters.

## Clean-C counted smoke: dose and instrument noise

The current counted study uses the same fixed 8 orientations and repeat 0.
These results diagnose trends and call behavior; they are not 2,048-orientation
headline statistics.

Acc@2° across the independent Poisson dose ladder is:

| Dose (e⁻/pattern) | AutoDisk | `find_Bragg_disks` | DoG-RGM |
|---:|---:|---:|---:|
| 100 | 33.3% (6/8 indexed) | 42.9% (7/8 indexed) | 37.5% (8/8 indexed) |
| 300 | 50.0% | 62.5% | 50.0% |
| 1,000 | 62.5% | 62.5% | 62.5% |
| 3,000 | 75.0% | 75.0% | 75.0% |
| 10,000 | 75.0% | 75.0% | 87.5% |
| 30,000 | 75.0% | 75.0% | 75.0% |
| 100,000 | 75.0% | 75.0% | 75.0% |
| 300,000 | 87.5% | 75.0% | 87.5% |
| 1,000,000 | 75.0% | 75.0% | 75.0% |

Increasing dose clearly improves the low-dose regime, but the 8-sample metric
is quantized in 12.5 percentage-point steps and is not monotonic after
saturation. At 100 e⁻, patterns with fewer than three detected peaks are
reported as `insufficient_detected_peaks`; they are not assigned fabricated
orientations. Evaluation reports both accuracy on indexed patterns and total
prediction coverage.

At fixed `10⁴ e⁻`, the P95 misorientation after adding independent EMPAD-G2
readout noise is:

| Readout-noise exposure | AutoDisk | `find_Bragg_disks` | DoG-RGM |
|---:|---:|---:|---:|
| No added readout noise | 7.12° | 7.12° | 3.54° |
| 1 frame | 8.26° | 8.28° | 8.05° |
| 4 frames | 8.05° | 8.26° | 64.92° |
| 16 frames | 14.46° | 61.22° | 64.78° |
| 64 frames | 64.79° | 71.23° | 85.83° |

The degradation is not caused by ACOM call failure: all twelve noise
conditions indexed 8/8 patterns. The detectors return many noise-induced
peaks, which can move ACOM to a wrong orientation branch. Peak count alone is
therefore an invalid quality criterion under readout noise.

## Reproducibility and saved outputs

The complete suite can be rerun with:

```bash
CUDA_VISIBLE_DEVICES=2 \
python scripts/23_run_v5_acom_suite.py \
  --data-root /mnt/data/xietianhong/or4d-clean-v5 \
  --output-root reports/v5 \
  --study all \
  --cuda \
  --cpu-threads 8
```

The runner refuses CUDA execution unless exactly one physical GPU is exposed
and that GPU is empty at launch. The actual GPU must be selected after checking
`nvidia-smi`; index 2 is only the GPU used for this recorded run.

For every method and study, the repository retains:

- `*_predictions.jsonl`: predicted orientation matrices;
- `*_details.json`: per-sample scores, plan indices, branch, errors, matrices,
  runtimes, versions, source paths and hashes;
- `*_audit.json`: orientation-plan coverage diagnostics;
- `*_evaluation.json`: aggregate and per-sample evaluation.

The source HDF5 images and peak intermediates remain only under the server data
root and are reproducible from the tracked code and manifests.

The 39 counted/noise runs are retained under
`reports/v5/acom_counted_smoke/`; their compact aggregate is
`reports/v5/V5_ACOM_COUNTED_SUMMARY.json`.

## Other orientation-indexing methods

### Recommended next comparator: Pyxem accelerated template matching

Pyxem provides an independent open-source template-matching implementation for
NBED/PED diffraction patterns. It correlates experimental 2D diffraction
signals with a simulated diffraction library, optimizes in-plane rotation, and
supports CPU or GPU execution. It is the best next comparator because it can
consume the same Clean-E/Clean-C images without first using our AutoDisk peak
adapter. See the
[Pyxem accelerated indexer documentation](https://www.pyxem.org/en/stable/reference/generated/pyxem.generators.AcceleratedIndexationGenerator.html)
and the primary
[Pyxem orientation-mapping paper](https://doi.org/10.1016/j.ultramic.2022.113517).

Pyxem and `diffsims` are not currently installed in `or4d-clean`. They should
be added only after an isolated compatibility solve and a 17/64-pattern smoke
test; the current ACOM results must not be overwritten.

### Useful transparent baseline: geometric peak voting

A small independent baseline can match pairwise peak lengths and angles against
the reciprocal lattice, use RANSAC/voting to propose rotations, and refine the
best rotation by weighted least squares. It would consume the same direct or
detected peak lists but would not share ACOM's polar sparse-correlation
implementation. This is not yet implemented and must be labeled a project
baseline rather than an external published package.

### Relevant but not a drop-in benchmark method

- ASTAR/NanoMEGAS is a widely used commercial template-matching reference, but
  it is not an open, automatable dependency for this benchmark.
- [`problematic`](https://github.com/stefsmeets/problematic) contains
  orientation-finding code for serial electron diffraction. Its target
  workflow and maintenance state differ from 4D-STEM orientation mapping, so it
  is lower priority than Pyxem.
- Adaptive sub-pixel reconstruction can improve the image supplied to a
  template matcher, but it is a preprocessing variant rather than an
  independent orientation-indexing algorithm. See the
  [open-access study](https://pmc.ncbi.nlm.nih.gov/articles/PMC11137144/).
- EBSD/TKD Hough and dictionary indexers operate on Kikuchi-band patterns, not
  the current Bragg-disk Clean input, and should not be mixed into this
  benchmark track.

## Recorded warnings

Pymatgen emitted an “Incorrect stoichiometry” warning while reading the
partially occupied NCM811 CIF. It did not abort any run, and all four paths used
the same parsed structure. Nevertheless, the parsed occupancies and resulting
structure factors should be audited separately before this benchmark is
presented as cross-simulator or experimental validation.
