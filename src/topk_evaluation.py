from __future__ import annotations

import numpy as np


def summarize_topk_errors(
    errors_deg: np.ndarray,
    *,
    total_input_samples: int,
) -> list[dict[str, float | int]]:
    """Summarize score-ordered candidates without GT-based re-ranking.

    ``errors_deg`` contains only successfully indexed samples and has shape
    ``[indexed_sample, rank]``. Missing/failed inputs remain in the accuracy
    denominator through ``total_input_samples``.
    """
    errors = np.asarray(errors_deg, dtype=np.float64)
    if errors.ndim != 2 or errors.shape[1] == 0:
        raise ValueError(
            f"errors_deg must have shape [sample,rank>0], got {errors.shape}"
        )
    if total_input_samples < errors.shape[0] or total_input_samples <= 0:
        raise ValueError("invalid total_input_samples")
    rows: list[dict[str, float | int]] = []
    for k in range(1, errors.shape[1] + 1):
        prefix = np.where(np.isfinite(errors[:, :k]), errors[:, :k], np.inf)
        best = np.min(prefix, axis=1)
        valid = best[np.isfinite(best)]
        row: dict[str, float | int] = {
            "k": k,
            "num_input_samples": int(total_input_samples),
            "num_indexed_samples": int(errors.shape[0]),
            "num_valid_predictions": int(valid.size),
            "prediction_coverage": float(valid.size / total_input_samples),
        }
        if valid.size:
            row.update(
                {
                    "median_misorientation_deg_indexed": float(
                        np.median(valid)
                    ),
                    "p95_misorientation_deg_indexed": float(
                        np.percentile(valid, 95)
                    ),
                    "accuracy_all_inputs_within_1deg": float(
                        np.sum(valid <= 1.0) / total_input_samples
                    ),
                    "accuracy_all_inputs_within_2deg": float(
                        np.sum(valid <= 2.0) / total_input_samples
                    ),
                    "accuracy_all_inputs_within_5deg": float(
                        np.sum(valid <= 5.0) / total_input_samples
                    ),
                }
            )
        else:
            row.update(
                {
                    "median_misorientation_deg_indexed": float("nan"),
                    "p95_misorientation_deg_indexed": float("nan"),
                    "accuracy_all_inputs_within_1deg": 0.0,
                    "accuracy_all_inputs_within_2deg": 0.0,
                    "accuracy_all_inputs_within_5deg": 0.0,
                }
            )
        rows.append(row)
    return rows

