# V5 — dose, noise and Top-5

- `ACOM_CLEAN_V5_VISUALIZATION.html`: main self-contained visualization.
- `topk/`: tracked ACOM/Pyxem Top-1…Top-5 aggregate results, comparison,
  static plot and ACOM run manifest.
- `pipeline/`: tracked dataset-generation, counting, noise and disk-detection
  manifests.
- `run_records/`: tracked compact execution record for the 468 added Clean-C
  AutoDisk/DoG-RGM conditions.
- `study_001/`: separate [001] diagnostic dataset summaries.
- `full_results/`: ignored optional local staging area for large per-sample
  candidate/detail artifacts.
- `MANIFEST.json`: immutable construction-time inventory of the large V5 input
  datasets and intermediate traces.

Clean-E and Clean-C are both full-dataset tracks. Electron dose and detector
noise are independent axes. The main page reports indexing coverage,
Acc@1°/2°/5°, median and P95 error, plus Top-1 through Top-5 success.

## Completed full runs

- Clean-E ACOM: oracle, AutoDisk, DoG-RGM and `find_Bragg_disks`, 2,048
  samples per method.
- Clean-C ACOM: AutoDisk, DoG-RGM and `find_Bragg_disks`, 234 conditions per
  detector (`9 doses × 6 noise levels × repeats`, with the saved condition
  layout defined by the V5 manifests), 702 conditions total.
- Clean-E/C Pyxem accelerated template matching: 1 and 234 conditions.
- ACOM summary condition count: 706 (`4 Clean-E + 702 Clean-C`); Pyxem: 235
  (`1 Clean-E + 234 Clean-C`).
- All Top-1…Top-5 candidates are evaluated with the same crystal-symmetry and
  Friedel-equivalent error definition. Missing ACOM candidates remain failures.

The compact summaries, plots, HTML and run records are retained in this local
repository. Large raw per-sample detail/candidate files remain reproducibly
stored on the server under:

```text
/mnt/data/xietianhong/or4d-clean-v5/results/acom_top5/
/mnt/data/xietianhong/or4d-clean-v5/results/acom_top5_candidates/
```

The local entry points are:

```text
topk/V5_TOP5_RESULTS.md
topk/V5_ACOM_TOP5_FULL_SUMMARY.json
topk/V5_ACOM_TOP5_RUN_MANIFEST.json
topk/V5_ACOM_PYXEM_TOP5_COMPARISON.json
run_records/v5_clean_c_autodisk_dog_full.json
ACOM_CLEAN_V5_VISUALIZATION.html
```
