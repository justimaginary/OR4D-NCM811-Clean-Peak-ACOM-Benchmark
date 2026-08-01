# V5 Clean Top-1 / Top-5 results

This report is generated from the saved per-condition candidate files. No failed or inaccurate scientific result is removed.

## Scope

- Dataset: 2,048 orientations per condition; `kmax = 1.5 Å⁻¹`.
- Clean-E: deterministic expectation-intensity diffraction images.
- Clean-C: electron-counted images at 9 independent dose levels. Each dose has a noiseless condition plus Poisson-only and EMPAD-G2 detector noise levels; the counted noise conditions have 5 repeats.
- ACOM: py4DSTEM ACOM using saved peak lists. Clean-E compares oracle, AutoDisk, DoG-RGM, and find_Bragg_disks inputs; Clean-C runs all three automatic disk detectors independently.
- Pyxem: accelerated template matching from the diffraction image.
- Error: minimum misorientation over proper crystal point-group symmetry and the detector-plane Friedel branch.
- Top-K: the best symmetry/Friedel-equivalent error among the first K saved candidates. Accuracy uses all input samples as the denominator; an indexing failure therefore counts as incorrect.

## Clean-E

| Method / input | Coverage | Top-1 Acc@2° | Top-5 Acc@2° | Top-1 median | Top-1 P95 |
|---|---:|---:|---:|---:|---:|
| ACOM / oracle | 100.00% | 78.86% | 97.22% | 1.174° | 9.167° |
| ACOM / autodisk | 100.00% | 80.96% | 97.90% | 1.146° | 7.586° |
| ACOM / dog_rgm | 100.00% | 80.81% | 97.90% | 1.145° | 7.662° |
| ACOM / py4dstem | 100.00% | 80.91% | 97.85% | 1.142° | 7.586° |
| Pyxem / expectation image | 100.00% | 31.05% | 43.21% | 9.089° | 59.758° |

## Clean-C: counted conditions grouped by electron dose

| Method / input | Electrons / pattern | Coverage | Top-1 Acc@2° | Top-5 Acc@2° | Top-1 median | Top-1 P95 |
|---|---:|---:|---:|---:|---:|---:|
| ACOM / AutoDisk | 100 | 98.54% | 8.29% | 14.08% | 61.286° | 91.793° |
| ACOM / DoG-RGM | 100 | 99.08% | 8.55% | 14.21% | 61.966° | 91.853° |
| ACOM / find_Bragg_disks | 100 | 91.33% | 3.97% | 7.30% | 64.695° | 92.232° |
| Pyxem / image | 100 | 100.00% | 0.21% | 0.68% | 66.044° | 91.752° |
| ACOM / AutoDisk | 300 | 99.96% | 16.39% | 26.07% | 55.819° | 91.289° |
| ACOM / DoG-RGM | 300 | 99.96% | 13.06% | 19.23% | 59.958° | 91.514° |
| ACOM / find_Bragg_disks | 300 | 99.51% | 11.67% | 18.93% | 60.226° | 91.959° |
| Pyxem / image | 300 | 100.00% | 0.57% | 1.44% | 65.425° | 92.010° |
| ACOM / AutoDisk | 1,000 | 100.00% | 37.76% | 57.44% | 4.820° | 88.987° |
| ACOM / DoG-RGM | 1,000 | 100.00% | 18.74% | 28.28% | 54.091° | 91.272° |
| ACOM / find_Bragg_disks | 1,000 | 100.00% | 26.71% | 42.48% | 38.858° | 90.807° |
| Pyxem / image | 1,000 | 100.00% | 1.73% | 3.93% | 62.229° | 91.910° |
| ACOM / AutoDisk | 3,000 | 100.00% | 61.19% | 84.17% | 1.553° | 75.291° |
| ACOM / DoG-RGM | 3,000 | 100.00% | 31.67% | 49.27% | 10.511° | 89.344° |
| ACOM / find_Bragg_disks | 3,000 | 100.00% | 50.11% | 73.13% | 1.990° | 85.372° |
| Pyxem / image | 3,000 | 100.00% | 4.96% | 8.81% | 58.394° | 91.087° |
| ACOM / AutoDisk | 10,000 | 100.00% | 74.39% | 94.97% | 1.258° | 16.534° |
| ACOM / DoG-RGM | 10,000 | 100.00% | 53.92% | 78.84% | 1.800° | 75.108° |
| ACOM / find_Bragg_disks | 10,000 | 100.00% | 67.11% | 90.18% | 1.418° | 57.243° |
| Pyxem / image | 10,000 | 100.00% | 10.75% | 15.68% | 50.620° | 89.131° |
| ACOM / AutoDisk | 30,000 | 100.00% | 78.74% | 97.08% | 1.189° | 8.331° |
| ACOM / DoG-RGM | 30,000 | 100.00% | 70.23% | 92.81% | 1.352° | 29.575° |
| ACOM / find_Bragg_disks | 30,000 | 100.00% | 73.11% | 94.07% | 1.292° | 11.551° |
| Pyxem / image | 30,000 | 100.00% | 14.25% | 19.33% | 45.265° | 85.317° |
| ACOM / AutoDisk | 100,000 | 100.00% | 80.52% | 97.77% | 1.148° | 7.605° |
| ACOM / DoG-RGM | 100,000 | 100.00% | 78.70% | 97.14% | 1.181° | 8.432° |
| ACOM / find_Bragg_disks | 100,000 | 100.00% | 75.54% | 95.34% | 1.241° | 10.164° |
| Pyxem / image | 100,000 | 100.00% | 16.55% | 22.01% | 40.689° | 81.452° |
| ACOM / AutoDisk | 300,000 | 100.00% | 81.20% | 98.01% | 1.136° | 7.217° |
| ACOM / DoG-RGM | 300,000 | 100.00% | 82.12% | 98.20% | 1.118° | 6.648° |
| ACOM / find_Bragg_disks | 300,000 | 100.00% | 76.55% | 96.23% | 1.222° | 9.795° |
| Pyxem / image | 300,000 | 100.00% | 18.49% | 24.27% | 37.056° | 76.396° |
| ACOM / AutoDisk | 1,000,000 | 100.00% | 81.71% | 98.11% | 1.129° | 6.919° |
| ACOM / DoG-RGM | 1,000,000 | 100.00% | 82.42% | 98.33% | 1.120° | 6.685° |
| ACOM / find_Bragg_disks | 1,000,000 | 100.00% | 79.41% | 97.38% | 1.168° | 8.432° |
| Pyxem / image | 1,000,000 | 100.00% | 20.61% | 27.13% | 32.132° | 70.173° |

## Clean-C: independent noise-model groups

These rows aggregate all 9 dose levels. Dose and noise remain separate experimental variables in the stored condition table.

| Method / input | Noise model | Conditions | Top-1 Acc@2° | Top-5 Acc@2° |
|---|---|---:|---:|---:|
| ACOM / AutoDisk | empad_g2_16frames | 45 | 52.06% | 67.43% |
| ACOM / DoG-RGM | empad_g2_16frames | 45 | 40.99% | 54.32% |
| ACOM / find_Bragg_disks | empad_g2_16frames | 45 | 45.12% | 61.15% |
| Pyxem / image | empad_g2_16frames | 45 | 6.70% | 8.51% |
| ACOM / AutoDisk | empad_g2_1frame | 45 | 59.76% | 76.60% |
| ACOM / DoG-RGM | empad_g2_1frame | 45 | 48.36% | 63.84% |
| ACOM / find_Bragg_disks | empad_g2_1frame | 45 | 53.61% | 70.39% |
| Pyxem / image | empad_g2_1frame | 45 | 8.22% | 10.71% |
| ACOM / AutoDisk | empad_g2_4frames | 45 | 56.44% | 72.78% |
| ACOM / DoG-RGM | empad_g2_4frames | 45 | 44.32% | 59.20% |
| ACOM / find_Bragg_disks | empad_g2_4frames | 45 | 49.61% | 65.97% |
| Pyxem / image | empad_g2_4frames | 45 | 7.38% | 9.54% |
| ACOM / AutoDisk | empad_g2_64frames | 45 | 47.10% | 61.33% |
| ACOM / DoG-RGM | empad_g2_64frames | 45 | 37.29% | 49.72% |
| ACOM / find_Bragg_disks | empad_g2_64frames | 45 | 40.30% | 56.13% |
| Pyxem / image | empad_g2_64frames | 45 | 6.00% | 7.56% |
| ACOM / AutoDisk | noiseless | 9 | 80.95% | 97.90% |
| ACOM / DoG-RGM | noiseless | 9 | 80.81% | 97.90% |
| ACOM / find_Bragg_disks | noiseless | 9 | 80.92% | 97.86% |
| Pyxem / image | noiseless | 9 | 31.01% | 43.21% |
| ACOM / AutoDisk | poisson_only | 45 | 73.64% | 92.80% |
| ACOM / DoG-RGM | poisson_only | 45 | 73.17% | 93.08% |
| ACOM / find_Bragg_disks | poisson_only | 45 | 69.23% | 88.05% |
| Pyxem / image | poisson_only | 45 | 20.65% | 32.16% |

## Execution and validation status

- ACOM: 706 completed conditions; 5,958 inputs had no saved candidate because the peak list contained too few usable peaks. These are reproducible indexing failures, not suppressed rows.
- Pyxem: 235 completed conditions; 0 missing predictions.
- Every ACOM candidate array was checked for shape `[matched, 5, 3, 3]`, finite values, and agreement between recomputed and saved Top-K metrics.
- The four disjoint Pyxem Clean-C shards were merged only after verifying exactly one owner for every expected condition.

## Artifacts

- Comparison data: `V5_ACOM_PYXEM_TOP5_COMPARISON.json`
- Dose plot: `V5_ACOM_PYXEM_TOP5_BY_DOSE.png`
- Full summaries: `V5_ACOM_TOP5_FULL_SUMMARY.json` and `V5_PYXEM_TOP5_FULL_SUMMARY.json`
- Unified ACOM provenance: `V5_ACOM_TOP5_RUN_MANIFEST.json`
