from __future__ import annotations

import re
from collections.abc import Iterable

import numpy as np

from topk_evaluation import summarize_topk_errors


_CLEAN_C_STEM = re.compile(
    r"^dose(?P<dose>\d+)_noise_"
    r"(?P<noise>noiseless|poisson_only|empad_g2_(?P<frames>\d+)frames?)"
    r"(?:_repeat(?P<repeat>\d+))?_"
    r"(?P<detector>py4dstem|autodisk|dog_rgm)$"
)


def parse_clean_c_condition_stem(stem: str) -> dict[str, int | str | None]:
    match = _CLEAN_C_STEM.fullmatch(stem)
    if match is None:
        raise ValueError(f"unrecognized Clean-C condition stem: {stem}")
    values = match.groupdict()
    noise = str(values["noise"])
    repeat_text = values["repeat"]
    if noise == "noiseless" and repeat_text is not None:
        raise ValueError("noiseless conditions must not have a repeat")
    if noise != "noiseless" and repeat_text is None:
        raise ValueError("counted conditions must have a repeat")
    return {
        "track": "Clean-C",
        "dose_electrons": int(values["dose"]),
        "noise": noise,
        "repeat": None if repeat_text is None else int(repeat_text),
        "frames": (
            None if values["frames"] is None else int(values["frames"])
        ),
        "detector": str(values["detector"]),
    }


def aggregate_topk_error_blocks(
    blocks: Iterable[tuple[np.ndarray, int]],
) -> list[dict[str, float | int]]:
    arrays: list[np.ndarray] = []
    total_inputs = 0
    rank_count: int | None = None
    for errors, total_input_samples in blocks:
        values = np.asarray(errors, dtype=float)
        if values.ndim != 2:
            raise ValueError(f"expected [sample,rank] errors, got {values.shape}")
        if rank_count is None:
            rank_count = values.shape[1]
        elif values.shape[1] != rank_count:
            raise ValueError("Top-K rank counts differ between blocks")
        if total_input_samples < values.shape[0]:
            raise ValueError("total inputs cannot be below indexed rows")
        arrays.append(values)
        total_inputs += int(total_input_samples)
    if not arrays:
        raise ValueError("cannot aggregate an empty block collection")
    return summarize_topk_errors(
        np.concatenate(arrays, axis=0),
        total_input_samples=total_inputs,
    )


def aggregate_group_keys(
    label: dict[str, int | str | None],
) -> list[tuple[str, str]]:
    track = str(label["track"])
    keys = [("track", track)]
    if track == "Clean-E":
        keys.append(("clean_e_input", str(label["input"])))
        return keys
    if track != "Clean-C":
        return keys
    dose = int(label["dose_electrons"])
    noise = str(label["noise"])
    detector = str(label["detector"])
    keys.extend(
        [
            ("detector", detector),
            ("dose_all_detector", f"{detector}|{dose}"),
            ("dose_noise_detector", f"{detector}|{dose}|{noise}"),
            ("noise_all_detector", f"{detector}|{noise}"),
        ]
    )
    if noise != "noiseless":
        keys.append(("dose_counted_detector", f"{detector}|{dose}"))
    return keys


def group_label(group_by: str, key: str) -> dict[str, int | str]:
    if group_by == "track":
        return {"group_by": group_by, "track": key}
    if group_by == "clean_e_input":
        return {"group_by": group_by, "track": "Clean-E", "input": key}
    if group_by == "detector":
        return {
            "group_by": group_by,
            "track": "Clean-C",
            "detector": key,
        }
    if group_by in {"dose_all_detector", "dose_counted_detector"}:
        detector, dose = key.split("|", 1)
        return {
            "group_by": group_by,
            "track": "Clean-C",
            "detector": detector,
            "dose_electrons": int(dose),
        }
    if group_by == "dose_noise_detector":
        detector, dose, noise = key.split("|", 2)
        return {
            "group_by": group_by,
            "track": "Clean-C",
            "detector": detector,
            "dose_electrons": int(dose),
            "noise": noise,
        }
    if group_by == "noise_all_detector":
        detector, noise = key.split("|", 1)
        return {
            "group_by": group_by,
            "track": "Clean-C",
            "detector": detector,
            "noise": noise,
        }
    if group_by in {"dose_all", "dose_counted"}:
        return {
            "group_by": group_by,
            "track": "Clean-C",
            "dose_electrons": int(key),
        }
    if group_by == "dose_noise":
        dose, noise = key.split("|", 1)
        return {
            "group_by": group_by,
            "track": "Clean-C",
            "dose_electrons": int(dose),
            "noise": noise,
        }
    if group_by == "noise_all":
        return {"group_by": group_by, "track": "Clean-C", "noise": key}
    raise ValueError(f"unknown aggregate group {group_by}")
