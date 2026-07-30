from __future__ import annotations

import hashlib

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


def poisson_count_image(
    expectation: np.ndarray,
    expected_electrons: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw ideal primary-electron counts with Poisson total fluctuation."""
    probability = np.asarray(expectation, dtype=np.float64)
    if probability.ndim != 2:
        raise ValueError("expectation must be a 2D image")
    if not np.all(np.isfinite(probability)) or np.any(probability < 0.0):
        raise ValueError("expectation must contain finite non-negative values")
    if expected_electrons <= 0:
        raise ValueError("expected_electrons must be positive")
    total = float(probability.sum())
    if total <= 0.0:
        raise ValueError("expectation has zero total intensity")
    lam = probability / total * int(expected_electrons)
    counts = rng.poisson(lam)
    if np.any(counts > np.iinfo(np.uint32).max):
        raise OverflowError("Poisson count exceeds uint32 range")
    return counts.astype(np.uint32)


def deterministic_count_seed(
    seed_base: int,
    sample_id: str,
    dose_electrons: int,
    repeat: int,
) -> int:
    """Derive a stable seed independent of ordering, batching and workers."""
    payload = (
        f"or4d-clean-v5|{int(seed_base)}|{sample_id}|"
        f"{int(dose_electrons)}|{int(repeat)}"
    ).encode("utf-8")
    return int.from_bytes(
        hashlib.blake2b(payload, digest_size=8).digest(), "little"
    )
