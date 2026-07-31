# Benchmark result index

The report tree is grouped by benchmark generation. The HTML files are
self-contained snapshots and can be opened directly without a Python server.

| Version | Input and purpose | Main page |
| --- | --- | --- |
| v3 | Direct `(qx, qy, intensity)` peak input; ACOM coordinate baseline | [V3 coordinate visualization](v3/ACOM_COORDINATE_VISUALIZATION.html) |
| v4 | 2D Clean-E/C diffraction images, disk detection, then ACOM | [V4 image-pipeline visualization](v4/CLEAN_IMAGE_ACOM_VISUALIZATION.html) |
| v5 | 2048 Clean samples; independent dose/noise ladders; ACOM and Pyxem Top-1…Top-5 | [V5 Top-5 visualization](v5/ACOM_CLEAN_V5_VISUALIZATION.html) |

- `common/` contains dataset/version validation shared by older runs.
- `legacy/` preserves the historical v1 baseline.
- `v4/runs/` contains reproducible per-dose/per-repeat detail files. It is
  intentionally local-only because the full set is large.
- `v5/full_results/` contains downloaded full-resolution result artifacts. It
  is intentionally ignored; tracked summaries and the manifest remain under
  `v5/topk/`, `v5/pipeline/`, and `v5/study_001/`.
- Smoke-only and obsolete Top-1-only outputs are not retained.
