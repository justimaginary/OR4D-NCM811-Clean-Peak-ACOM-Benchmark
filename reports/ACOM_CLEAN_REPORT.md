# NCM811 Clean-Peak ACOM Report

- Dataset: `OR4D-NCM811-CleanPeak-acom-v2`
- Headline cohort: `headline_core` (512 samples)
- Primary metric: Friedel-equivalent misorientation under proper crystal point-group rotations
- Baseline: py4DSTEM ACOM, 4° zone-axis step and 4° in-plane step

## Headline result

| Metric | n | Mean | Median | P90 | P95 | Max | Acc@1° | Acc@2° | Acc@5° |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Friedel-equivalent | 512 | 2.105 | 1.548 | 2.351 | 2.811 | 89.860 | 23.6% | 75.0% | 97.9% |
| Strict | 512 | 6.602 | 1.555 | 2.581 | 66.430 | 97.943 | 23.4% | 73.6% | 93.2% |

Strict/Friedel disagreement rate: 6.25%.
Catastrophic mismatches: 11/512 above 5°; 4/512 above 10°.

## Result by sample role

| Role | n | Mean | Median | P90 | P95 | Max | Acc@1° | Acc@2° | Acc@5° |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| acom_grid_probe | 40 | 1.535 | 2.225 | 2.728 | 2.729 | 2.731 | 40.0% | 40.0% | 100.0% |
| headline_core | 512 | 2.105 | 1.548 | 2.351 | 2.811 | 89.860 | 23.6% | 75.0% | 97.9% |
| legacy_smoke | 17 | 1.581 | 2.005 | 2.593 | 2.729 | 2.729 | 29.4% | 47.1% | 100.0% |

## Headline result by nearest zone-axis node distance

| Distance bin | n | Mean | Median | P90 | P95 | Max | Acc@1° | Acc@2° | Acc@5° |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| [0, 0.5)° | 23 | 0.398 | 0.421 | 0.492 | 0.500 | 0.562 | 100.0% | 100.0% | 100.0% |
| [0.5, 1)° | 102 | 0.839 | 0.786 | 0.962 | 0.993 | 6.166 | 96.1% | 98.0% | 99.0% |
| [1, 2)° | 294 | 1.735 | 1.583 | 2.028 | 2.740 | 7.491 | 0.0% | 88.8% | 98.0% |
| [2, +∞)° | 93 | 5.086 | 2.228 | 2.861 | 3.628 | 89.860 | 0.0% | 0.0% | 95.7% |

## ACOM diagnostics

- Zone-axis node distance/error Pearson correlation: 0.215
- Discrete seed distance/error Pearson correlation: 0.212
- Discrete seed distance is diagnostic only. ACOM performs a parabolic sub-grid fit of the in-plane correlation peak, so it is not an error lower bound.
- Plan nodes: 276 zone axes × 90 in-plane steps; 49680 seeds including mirror.
- Plan build: 0.034 s.
- Matching: 12.418 s total, 0.0216 s p50, 0.0227 s p90, 0.0246 s p99, 45.8 samples/s.

## Reproducibility

- Source git revision: `60c2c43014122652a92f453de1a99f64442e044d`
- Versions: Python 3.11.15, NumPy 1.26.4, py4DSTEM 0.14.18, pymatgen 2024.7.18, h5py 3.16.0
- SHA256 `cif`: `60f28f9d4ace01baeab707d22c8de267ce0c7759c3574598fb1abe84e4fef94f`
- SHA256 `config`: `cc403748a4c568d967c9ac22eec94bc2d6516ef5334963ef49466d74582500f1`
- SHA256 `ground_truth`: `cd6bbf95a550545b44dfc4a502df643d3c4d7547c2353402bf791cbb96a187c7`
- SHA256 `orientation_manifest`: `9f16da1b53cd87480f7a329829ddec5a85f974ad47ae581cc61321cc12a5a8d6`
- SHA256 `public_peaks`: `7cdbbcc94cb798b2cd761c86566afedce32aba4aeee0ae59530dd6a425023d62`

## Interpretation boundary

Clean inputs and ACOM templates use the same CIF and py4DSTEM kinematical model; this measures self-consistency, not real-data or cross-simulator generalization.

![ACOM error distributions](acom_error_comparison.png)

![Zone-axis distance versus error](acom_offgrid_vs_error.png)

![Representative peak overlays](acom_peak_overlay.png)
