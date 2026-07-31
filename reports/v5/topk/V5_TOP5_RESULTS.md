# V5 Clean Top-1 / Top-5 results

This report is generated from the saved per-condition candidate files. No failed or inaccurate scientific result is removed.

## Scope

- Dataset: 2,048 orientations per condition; `kmax = 1.5 Å⁻¹`.
- Clean-E: deterministic expectation-intensity diffraction images.
- Clean-C: electron-counted images at 9 independent dose levels. Each dose has a noiseless condition plus Poisson-only and EMPAD-G2 detector noise levels; the counted noise conditions have 5 repeats.
- ACOM: py4DSTEM ACOM using saved peak lists. Clean-E additionally compares oracle, AutoDisk, DoG-RGM, and py4DSTEM peak inputs.
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

| Electrons / pattern | ACOM coverage | ACOM Top-1 Acc@2° | ACOM Top-5 Acc@2° | Pyxem coverage | Pyxem Top-1 Acc@2° | Pyxem Top-5 Acc@2° |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | 91.33% | 3.97% | 7.30% | 100.00% | 0.21% | 0.68% |
| 300 | 99.51% | 11.67% | 18.93% | 100.00% | 0.57% | 1.44% |
| 1,000 | 100.00% | 26.71% | 42.48% | 100.00% | 1.73% | 3.93% |
| 3,000 | 100.00% | 50.11% | 73.13% | 100.00% | 4.96% | 8.81% |
| 10,000 | 100.00% | 67.11% | 90.18% | 100.00% | 10.75% | 15.68% |
| 30,000 | 100.00% | 73.11% | 94.07% | 100.00% | 14.25% | 19.33% |
| 100,000 | 100.00% | 75.54% | 95.34% | 100.00% | 16.55% | 22.01% |
| 300,000 | 100.00% | 76.55% | 96.23% | 100.00% | 18.49% | 24.27% |
| 1,000,000 | 100.00% | 79.41% | 97.38% | 100.00% | 20.61% | 27.13% |

## Clean-C: independent noise-model groups

These rows aggregate all 9 dose levels. Dose and noise remain separate experimental variables in the stored condition table.

| Noise model | Conditions | ACOM Top-1 Acc@2° | ACOM Top-5 Acc@2° | Pyxem Top-1 Acc@2° | Pyxem Top-5 Acc@2° |
|---|---:|---:|---:|---:|---:|
| empad_g2_16frames | 45 | 45.12% | 61.15% | 6.70% | 8.51% |
| empad_g2_1frame | 45 | 53.61% | 70.39% | 8.22% | 10.71% |
| empad_g2_4frames | 45 | 49.61% | 65.97% | 7.38% | 9.54% |
| empad_g2_64frames | 45 | 40.30% | 56.13% | 6.00% | 7.56% |
| noiseless | 9 | 80.92% | 97.86% | 31.01% | 43.21% |
| poisson_only | 45 | 69.23% | 88.05% | 20.65% | 32.16% |

## Execution and validation status

- ACOM: 238 completed conditions; 4,695 inputs had no saved candidate because the peak list contained too few usable peaks. These are reproducible indexing failures, not suppressed rows.
- Pyxem: 235 completed conditions; 0 missing predictions.
- Every ACOM candidate array was checked for shape `[matched, 5, 3, 3]`, finite values, and agreement between recomputed and saved Top-K metrics.
- The four disjoint Pyxem Clean-C shards were merged only after verifying exactly one owner for every expected condition.

## Artifacts

- Comparison data: `V5_ACOM_PYXEM_TOP5_COMPARISON.json`
- Dose plot: `V5_ACOM_PYXEM_TOP5_BY_DOSE.png`
- Full summaries: `V5_ACOM_TOP5_FULL_SUMMARY.json` and `V5_PYXEM_TOP5_FULL_SUMMARY.json`
- Unified ACOM provenance: `V5_ACOM_TOP5_RUN_MANIFEST.json`
