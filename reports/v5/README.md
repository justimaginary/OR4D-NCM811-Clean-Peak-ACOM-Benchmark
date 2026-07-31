# V5 — dose, noise and Top-5

- `ACOM_CLEAN_V5_VISUALIZATION.html`: main self-contained visualization.
- `topk/`: tracked ACOM/Pyxem Top-1…Top-5 aggregate results and comparison.
- `pipeline/`: tracked generation, noise and detector manifests.
- `study_001/`: separate [001] diagnostic dataset summaries.
- `full_results/`: ignored local copy of full candidate/detail artifacts.
- `MANIFEST.json`: artifact inventory and provenance.

Clean-E and Clean-C are both full-dataset tracks. Electron dose and detector
noise are independent axes. The main page reports indexing coverage,
Acc@1°/2°/5°, median and P95 error, plus Top-1 through Top-5 success.
