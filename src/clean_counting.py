from __future__ import annotations

import numpy as np


def multinomial_count_image(
    expectation: np.ndarray,
    num_electrons: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw a fixed-total ideal-counting detector image."""
    probability = np.asarray(expectation, dtype=np.float64)
    if probability.ndim != 2:
        raise ValueError("expectation must be a 2D image")
    if not np.all(np.isfinite(probability)) or np.any(probability < 0.0):
        raise ValueError("expectation must contain finite non-negative values")
    if num_electrons <= 0:
        raise ValueError("num_electrons must be positive")
    total = float(probability.sum())
    if total <= 0.0:
        raise ValueError("expectation has zero total intensity")
    probability = (probability / total).ravel()
    probability /= probability.sum()
    counts = rng.multinomial(int(num_electrons), probability)
    return counts.reshape(expectation.shape).astype(np.uint32)
