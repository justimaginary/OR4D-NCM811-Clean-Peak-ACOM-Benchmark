# V4 — diffraction image to ACOM

V4 replaces the direct peak input with a 2D Clean-E expectation image or a
Clean-C counted image, detects disks using AutoDisk or
`find_Bragg_disks`, and passes the recovered peak list to the same ACOM
evaluation.

## Full-run result snapshot

Clean-E peak recovery is 100% precision and recall for both detectors.
Position RMSE is `0.410 px` for AutoDisk and `0.011 px` for
`find_Bragg_disks`. End-to-end Acc@2° is `90.29%` for both image paths,
compared with `90.19%` for the physical-oracle peak path.

For Clean-C, increasing dose materially improves peak recovery. At
`10⁴ / 10⁵ / 10⁶ e⁻`, `find_Bragg_disks` recall is
`80.05% / 99.38% / 100%`; AutoDisk recall is
`64.36% / 95.66% / 99.95%`. The corresponding mean end-to-end Acc@2° is:

| Detector | 10⁴ e⁻ | 10⁵ e⁻ | 10⁶ e⁻ |
| --- | ---: | ---: | ---: |
| AutoDisk | 88.58% | 90.10% | 90.25% |
| `find_Bragg_disks` | 89.55% | 90.10% | 90.19% |

All values above come directly from the retained full-run JSON files.

- `CLEAN_IMAGE_ACOM_VISUALIZATION.html`: self-contained image-pipeline page.
- `clean_*_evaluation.json`: disk-recovery and end-to-end summaries.
- `acom_clean_details_*.json`: full expectation/oracle sample results.
- `clean_*_comparison.json`: detector and dose comparisons.
- `runs/`: local-only full per-dose/per-repeat details and plan audits.

Smoke-only files were removed. The retained files are completed full-run
results or their reproducibility metadata.
