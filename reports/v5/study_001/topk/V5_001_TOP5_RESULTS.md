# V5 independent [001] study Top-1 / Top-5 results

This report is generated from the saved per-condition candidate files. No failed or inaccurate scientific result is removed.

## Scope

- Dataset: 512 orientations per condition; `kmax = 1.5 Å⁻¹`.
- Clean-E: deterministic expectation-intensity diffraction images.
- Clean-C: electron-counted images at 9 independent dose levels. Each dose has a noiseless condition plus Poisson-only and EMPAD-G2 detector noise levels; the counted noise conditions have 5 repeats.
- ACOM: py4DSTEM ACOM using saved peak lists. Clean-E compares oracle, AutoDisk, DoG-RGM, and find_Bragg_disks inputs; Clean-C runs all three automatic disk detectors independently.
- Pyxem: accelerated template matching from the diffraction image.
- Error: minimum misorientation over proper crystal point-group symmetry and the detector-plane Friedel branch.
- Top-K: the best symmetry/Friedel-equivalent error among the first K saved candidates. Accuracy uses all input samples as the denominator; an indexing failure therefore counts as incorrect.

## Clean-E

| Method / input | Coverage | Top-1 Acc@2° | Top-5 Acc@2° | Top-1 median | Top-1 P95 |
|---|---:|---:|---:|---:|---:|
| ACOM / oracle | 100.00% | 2.73% | 41.41% | 71.715° | 73.058° |
| ACOM / autodisk | 100.00% | 3.12% | 39.26% | 71.647° | 72.814° |
| ACOM / dog_rgm | 100.00% | 4.30% | 35.74% | 71.672° | 73.058° |
| ACOM / py4dstem | 100.00% | 6.25% | 40.62% | 71.672° | 73.058° |
| Pyxem / expectation image | 100.00% | 18.36% | 68.55% | 29.001° | 71.095° |

## Clean-C: counted conditions grouped by electron dose

| Method / input | Electrons / pattern | Coverage | Top-1 Acc@2° | Top-5 Acc@2° | Top-1 median | Top-1 P95 |
|---|---:|---:|---:|---:|---:|---:|
| ACOM / AutoDisk | 100 | 99.21% | 0.44% | 1.54% | 88.023° | 93.077° |
| ACOM / DoG-RGM | 100 | 99.58% | 0.49% | 1.48% | 89.120° | 93.177° |
| ACOM / find_Bragg_disks | 100 | 92.68% | 0.66% | 1.39% | 89.769° | 93.323° |
| Pyxem / image | 100 | 100.00% | 0.35% | 0.74% | 89.570° | 92.943° |
| ACOM / AutoDisk | 300 | 99.99% | 0.67% | 4.26% | 72.849° | 92.444° |
| ACOM / DoG-RGM | 300 | 100.00% | 0.91% | 3.81% | 88.217° | 93.171° |
| ACOM / find_Bragg_disks | 300 | 99.81% | 1.16% | 5.08% | 76.019° | 93.106° |
| Pyxem / image | 300 | 100.00% | 0.83% | 1.88% | 89.657° | 92.973° |
| ACOM / AutoDisk | 1,000 | 100.00% | 0.89% | 6.38% | 72.008° | 89.919° |
| ACOM / DoG-RGM | 1,000 | 100.00% | 1.54% | 7.10% | 73.884° | 92.817° |
| ACOM / find_Bragg_disks | 1,000 | 99.99% | 1.17% | 9.19% | 72.053° | 91.974° |
| Pyxem / image | 1,000 | 100.00% | 2.63% | 5.28% | 89.952° | 93.019° |
| ACOM / AutoDisk | 3,000 | 100.00% | 1.69% | 7.59% | 71.910° | 74.087° |
| ACOM / DoG-RGM | 3,000 | 100.00% | 1.71% | 10.65% | 72.017° | 91.394° |
| ACOM / find_Bragg_disks | 3,000 | 100.00% | 1.45% | 11.54% | 71.911° | 75.627° |
| Pyxem / image | 3,000 | 100.00% | 4.07% | 9.46% | 89.892° | 92.922° |
| ACOM / AutoDisk | 10,000 | 100.00% | 2.28% | 11.17% | 71.874° | 73.122° |
| ACOM / DoG-RGM | 10,000 | 100.00% | 1.29% | 12.55% | 71.902° | 74.145° |
| ACOM / find_Bragg_disks | 10,000 | 100.00% | 1.56% | 14.53% | 71.849° | 73.270° |
| Pyxem / image | 10,000 | 100.00% | 5.50% | 13.80% | 73.881° | 92.066° |
| ACOM / AutoDisk | 30,000 | 100.00% | 2.21% | 16.49% | 71.796° | 73.014° |
| ACOM / DoG-RGM | 30,000 | 100.00% | 1.29% | 15.10% | 71.849° | 73.270° |
| ACOM / find_Bragg_disks | 30,000 | 100.00% | 1.16% | 17.17% | 71.780° | 73.021° |
| Pyxem / image | 30,000 | 100.00% | 6.42% | 15.38% | 70.154° | 83.816° |
| ACOM / AutoDisk | 100,000 | 100.00% | 2.86% | 23.94% | 71.764° | 72.886° |
| ACOM / DoG-RGM | 100,000 | 100.00% | 2.20% | 21.73% | 71.768° | 72.939° |
| ACOM / find_Bragg_disks | 100,000 | 100.00% | 1.25% | 22.62% | 71.759° | 72.939° |
| Pyxem / image | 100,000 | 100.00% | 6.95% | 16.91% | 69.926° | 78.199° |
| ACOM / AutoDisk | 300,000 | 100.00% | 3.34% | 30.31% | 71.757° | 73.058° |
| ACOM / DoG-RGM | 300,000 | 100.00% | 3.85% | 26.92% | 71.764° | 73.058° |
| ACOM / find_Bragg_disks | 300,000 | 100.00% | 1.16% | 32.62% | 71.757° | 73.058° |
| Pyxem / image | 300,000 | 100.00% | 7.11% | 17.65% | 70.579° | 75.342° |
| ACOM / AutoDisk | 1,000,000 | 100.00% | 4.44% | 34.95% | 71.735° | 73.058° |
| ACOM / DoG-RGM | 1,000,000 | 100.00% | 4.41% | 27.25% | 71.735° | 73.058° |
| ACOM / find_Bragg_disks | 1,000,000 | 100.00% | 3.02% | 41.68% | 71.735° | 73.058° |
| Pyxem / image | 1,000,000 | 100.00% | 7.11% | 18.03% | 70.663° | 74.283° |

## Clean-C: independent noise-model groups

These rows aggregate all 9 dose levels. Dose and noise remain separate experimental variables in the stored condition table.

| Method / input | Noise model | Conditions | Top-1 Acc@2° | Top-5 Acc@2° |
|---|---|---:|---:|---:|
| ACOM / AutoDisk | empad_g2_16frames | 45 | 1.43% | 13.68% |
| ACOM / DoG-RGM | empad_g2_16frames | 45 | 1.67% | 12.69% |
| ACOM / find_Bragg_disks | empad_g2_16frames | 45 | 0.72% | 15.82% |
| Pyxem / image | empad_g2_16frames | 45 | 3.29% | 3.81% |
| ACOM / AutoDisk | empad_g2_1frame | 45 | 2.18% | 15.81% |
| ACOM / DoG-RGM | empad_g2_1frame | 45 | 1.90% | 15.51% |
| ACOM / find_Bragg_disks | empad_g2_1frame | 45 | 1.31% | 18.45% |
| Pyxem / image | empad_g2_1frame | 45 | 3.42% | 4.04% |
| ACOM / AutoDisk | empad_g2_4frames | 45 | 1.80% | 15.26% |
| ACOM / DoG-RGM | empad_g2_4frames | 45 | 1.74% | 13.92% |
| ACOM / find_Bragg_disks | empad_g2_4frames | 45 | 1.02% | 17.48% |
| Pyxem / image | empad_g2_4frames | 45 | 3.44% | 3.98% |
| ACOM / AutoDisk | empad_g2_64frames | 45 | 1.18% | 12.16% |
| ACOM / DoG-RGM | empad_g2_64frames | 45 | 1.45% | 10.30% |
| ACOM / find_Bragg_disks | empad_g2_64frames | 45 | 0.82% | 14.43% |
| Pyxem / image | empad_g2_64frames | 45 | 3.27% | 3.76% |
| ACOM / AutoDisk | noiseless | 9 | 3.12% | 39.26% |
| ACOM / DoG-RGM | noiseless | 9 | 4.30% | 35.74% |
| ACOM / find_Bragg_disks | noiseless | 9 | 6.25% | 40.62% |
| Pyxem / image | noiseless | 9 | 18.34% | 68.47% |
| ACOM / AutoDisk | poisson_only | 45 | 3.86% | 18.99% |
| ACOM / DoG-RGM | poisson_only | 45 | 3.08% | 17.91% |
| ACOM / find_Bragg_disks | poisson_only | 45 | 3.12% | 20.39% |
| Pyxem / image | poisson_only | 45 | 9.34% | 39.48% |

## Execution and validation status

- ACOM: 706 completed conditions; 1,118 inputs had no saved candidate because the peak list contained too few usable peaks. These are reproducible indexing failures, not suppressed rows.
- Pyxem: 235 completed conditions; 0 missing predictions.
- Every ACOM candidate array was checked for shape `[matched, 5, 3, 3]`, finite values, and agreement between recomputed and saved Top-K metrics.
- The four disjoint Pyxem Clean-C shards were merged only after verifying exactly one owner for every expected condition.

## Artifacts

- Comparison data: `V5_001_ACOM_PYXEM_TOP5_COMPARISON.json`
- Dose plot: `V5_001_ACOM_PYXEM_TOP5_BY_DOSE.png`
- Full ACOM summary and provenance use the artifact prefix `V5_001`.
