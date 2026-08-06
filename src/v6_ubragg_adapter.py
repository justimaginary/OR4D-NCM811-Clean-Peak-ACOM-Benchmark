from __future__ import annotations

import hashlib
import importlib
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class UBraggXResult:
    qx_Ainv: np.ndarray
    qy_Ainv: np.ndarray
    intensity: np.ndarray
    score: np.ndarray
    objectness: np.ndarray
    quality: np.ndarray
    row_px: np.ndarray
    col_px: np.ndarray
    cov_xx_Ainv2: np.ndarray
    cov_xy_Ainv2: np.ndarray
    cov_yy_Ainv2: np.ndarray


def sha256_file(path: Path, block_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_bytes), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_ubragg_x_artifacts(config: dict[str, Any]) -> dict[str, Any]:
    settings = config["v6"]["ubragg_x"]
    repository = Path(settings["repository_path"]).resolve()
    checkpoint = Path(settings["checkpoint_path"]).resolve()
    if not repository.is_dir():
        raise FileNotFoundError(f"UBragg repository is missing: {repository}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"UBragg-X checkpoint is missing: {checkpoint}")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    expected_commit = str(settings["repository_commit"])
    if commit != expected_commit:
        raise RuntimeError(
            f"UBragg repository HEAD is {commit}, expected {expected_commit}"
        )
    size = checkpoint.stat().st_size
    expected_size = int(settings["checkpoint_size_bytes"])
    if size != expected_size:
        raise RuntimeError(
            f"UBragg-X checkpoint size is {size}, expected {expected_size}"
        )
    digest = sha256_file(checkpoint)
    expected_digest = str(settings["checkpoint_sha256"])
    if digest != expected_digest:
        raise RuntimeError(
            f"UBragg-X checkpoint SHA-256 is {digest}, expected {expected_digest}"
        )
    return {
        "repository_path": str(repository),
        "repository_commit": commit,
        "checkpoint_path": str(checkpoint),
        "checkpoint_size_bytes": size,
        "checkpoint_sha256": digest,
    }


class UBraggXInference:
    def __init__(self, config: dict[str, Any], *, device: str = "cuda:0") -> None:
        artifacts = verify_ubragg_x_artifacts(config)
        repository = Path(artifacts["repository_path"])
        source_path = str(repository / "src")
        if source_path not in sys.path:
            sys.path.insert(0, source_path)
        torch = importlib.import_module("torch")
        xmodel = importlib.import_module("ubragg.xmodel")
        geometry = importlib.import_module("ubragg.geometry")
        checkpoint_path = Path(artifacts["checkpoint_path"])
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if "config" not in checkpoint or "model" not in checkpoint:
            raise ValueError("UBragg-X checkpoint lacks config/model fields")
        model_config = dict(checkpoint["config"])
        self.torch = torch
        self.pixel_to_q = geometry.pixel_to_q
        self.covariance_pixel_to_q = geometry.covariance_pixel_to_q
        self.model = xmodel.build_ubragg_x(model_config).to(device)
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()
        self.device = torch.device(device)
        settings = config["v6"]["ubragg_x"]
        if bool(settings["resize_before_inference"]):
            raise ValueError(
                "V6 UBragg-X inference is configured for native detector pixels; "
                "resize_before_inference must remain false"
            )
        self.input_shape = tuple(
            int(value) for value in settings["v6_inference_image_shape_px"]
        )
        self.score_threshold = float(settings["score_threshold"])
        self.artifacts = artifacts
        self.model_config = model_config

    def infer_batch(
        self,
        images: np.ndarray,
        expected_total_electrons: np.ndarray,
        read_noise_sigma_e_per_pixel: np.ndarray,
        vacuum_probe: np.ndarray,
        valid_mask: np.ndarray,
        qx_axis_Ainv: np.ndarray,
        qy_axis_Ainv: np.ndarray,
    ) -> list[UBraggXResult]:
        torch = self.torch
        values = np.asarray(images, dtype=np.float32)
        if values.ndim != 3:
            raise ValueError("UBragg-X images must have shape [N,H,W]")
        if values.shape[1:] != self.input_shape:
            raise ValueError(
                f"UBragg-X V6 input shape is {values.shape[1:]}, "
                f"expected native detector shape {self.input_shape}"
            )
        dose = np.asarray(expected_total_electrons, dtype=np.float32)
        sigma = np.asarray(read_noise_sigma_e_per_pixel, dtype=np.float32)
        if dose.shape != (len(values),) or sigma.shape != (len(values),):
            raise ValueError("UBragg-X dose/sigma arrays must match batch length")
        if np.any(dose <= 0.0):
            raise ValueError("UBragg-X total electron dose must be positive")
        probe = np.asarray(vacuum_probe, dtype=np.float32)
        valid = np.asarray(valid_mask, dtype=np.float32)
        if probe.shape != values.shape[1:] or valid.shape != values.shape[1:]:
            raise ValueError("UBragg-X probe/valid mask must match image shape")
        batch = {
            "image": torch.from_numpy(values).unsqueeze(1).to(self.device),
            "dose": torch.from_numpy(dose).to(self.device),
            "read_noise_sigma": torch.from_numpy(sigma).to(self.device),
            "probe": torch.from_numpy(
                np.broadcast_to(probe, values.shape).copy()
            ).unsqueeze(1).to(self.device),
            "valid_mask": torch.from_numpy(
                np.broadcast_to(valid, values.shape).copy()
            ).unsqueeze(1).to(self.device),
        }
        qx_axis = np.asarray(qx_axis_Ainv, dtype=np.float32)
        qy_axis = np.asarray(qy_axis_Ainv, dtype=np.float32)
        results: list[UBraggXResult] = []
        with torch.inference_mode():
            output = self.model(batch)
            for index in range(len(values)):
                score = output["candidate_score"][index].detach().cpu().numpy()
                valid_candidates = (
                    output["candidate_valid"][index].detach().cpu().numpy()
                )
                keep = valid_candidates & (score >= self.score_threshold)
                rows = (
                    output["candidate_row"][index].detach().cpu().numpy()[keep]
                )
                cols = (
                    output["candidate_col"][index].detach().cpu().numpy()[keep]
                )
                covariance = (
                    output["candidate_covariance"][index]
                    .detach()
                    .cpu()
                    .numpy()[keep]
                )
                qx, qy = self.pixel_to_q(
                    rows, cols, qx_axis, qy_axis
                )
                covariance_q = self.covariance_pixel_to_q(
                    covariance, rows, cols, qx_axis, qy_axis
                )
                results.append(
                    UBraggXResult(
                        qx_Ainv=np.asarray(qx, dtype=np.float32),
                        qy_Ainv=np.asarray(qy, dtype=np.float32),
                        intensity=(
                            output["candidate_intensity"][index]
                            .detach()
                            .cpu()
                            .numpy()[keep]
                            .astype(np.float32)
                        ),
                        score=score[keep].astype(np.float32),
                        objectness=(
                            output["proposal_score"][index]
                            .detach()
                            .cpu()
                            .numpy()[keep]
                            .astype(np.float32)
                        ),
                        quality=(
                            output["candidate_quality"][index]
                            .detach()
                            .cpu()
                            .numpy()[keep]
                            .astype(np.float32)
                        ),
                        row_px=np.asarray(rows, dtype=np.float32),
                        col_px=np.asarray(cols, dtype=np.float32),
                        cov_xx_Ainv2=np.asarray(
                            covariance_q[:, 0, 0], dtype=np.float32
                        ),
                        cov_xy_Ainv2=np.asarray(
                            covariance_q[:, 0, 1], dtype=np.float32
                        ),
                        cov_yy_Ainv2=np.asarray(
                            covariance_q[:, 1, 1], dtype=np.float32
                        ),
                    )
                )
        return results
