#!/usr/bin/env python3
"""Generate the reproducible V5 Clean inputs outside the Git checkout.

The repository tracks the source CIF, configuration, code, compact manifests,
and result summaries.  This command writes generated orientations, images,
count images, noise manifests, and reflection traces below ``--data-root``;
that directory is deliberately external to the checkout on a shared machine.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "benchmark_v5.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="External directory for generated V5 data and intermediates.",
    )
    parser.add_argument("--study", choices=("main", "001", "all"), default="all")
    parser.add_argument("--backend", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate existing outputs; otherwise complete outputs are reused.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def verify_gpu() -> str:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible.isdigit():
        raise RuntimeError(
            "--backend cuda requires CUDA_VISIBLE_DEVICES to contain exactly "
            "one physical GPU index"
        )
    rows = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    status = {}
    for line in rows.splitlines():
        fields = [part.strip() for part in line.split(",")]
        if len(fields) == 3:
            status[fields[0]] = (int(fields[1]), int(fields[2]))
    if visible not in status:
        raise RuntimeError(f"GPU {visible} is absent from nvidia-smi output")
    memory_mib, utilization = status[visible]
    if memory_mib > 100 or utilization > 5:
        raise RuntimeError(
            f"GPU {visible} is not empty: {memory_mib} MiB, {utilization}%"
        )
    return visible


def output_state(paths: list[Path], overwrite: bool) -> str:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        if len(existing) == len(paths):
            return "reuse"
        raise RuntimeError(
            "Partial V5 output set exists; remove it explicitly or pass "
            f"--overwrite: {[str(path) for path in existing]}"
        )
    return "run"


def run(
    command: list[str],
    env: dict[str, str],
    records: list[dict],
    outputs: list[Path],
    overwrite: bool,
) -> None:
    state = output_state(outputs, overwrite)
    record = {"command": command, "outputs": [str(path) for path in outputs]}
    if state == "reuse":
        record["status"] = "reused"
        records.append(record)
        print("reuse", " ".join(command), flush=True)
        return
    started = time.perf_counter()
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)
    record["status"] = "generated"
    record["seconds"] = time.perf_counter() - started
    records.append(record)


def prepare_study(
    *,
    study: str,
    data_root: Path,
    env: dict[str, str],
    records: list[dict],
    overwrite: bool,
) -> None:
    if study == "main":
        count = 2048
        role = "headline_core"
        orientation_name = "clean_v5_orientations.jsonl"
        prefix = "clean_v5_first_born"
        report_prefix = "first_born"
    else:
        count = 512
        role = "study_001"
        orientation_name = "clean_v5_001_orientations.jsonl"
        prefix = "clean_v5_001_first_born"
        report_prefix = "clean_v5_001_first_born"

    manifests = data_root / "manifests"
    datasets = data_root / "datasets"
    intermediates = data_root / "intermediates"
    reports = data_root / "reports"
    for directory in (manifests, datasets, intermediates, reports):
        directory.mkdir(parents=True, exist_ok=True)
    orientation = manifests / orientation_name
    expectation = datasets / f"{prefix}_expectation_{count}.h5"
    counted = datasets / f"{prefix}_counted_{count}.h5"
    dose_noiseless = datasets / f"{prefix}_dose_noiseless_{count}.h5"
    oracle = intermediates / f"{prefix}_oracle_{count}.h5"
    trace = intermediates / f"{prefix}_trace_{count}.h5"
    noise_manifest = manifests / f"{prefix.replace('_first_born', '')}_instrument_noise_{count}.h5"

    if study == "main":
        run(
            [
                sys.executable,
                str(ROOT / "scripts/01_generate_orientations.py"),
                "--output",
                str(orientation),
            ],
            env,
            records,
            [orientation],
            overwrite,
        )
    else:
        run(
            [
                sys.executable,
                str(ROOT / "scripts/01b_generate_001_study.py"),
                "--output",
                str(orientation),
                "--report-output",
                str(reports / "clean_v5_001_manifest_summary.json"),
            ],
            env,
            records,
            [orientation, reports / "clean_v5_001_manifest_summary.json"],
            overwrite,
        )

    run(
        [
            sys.executable,
            str(ROOT / "scripts/02b_generate_clean_images.py"),
            "--orientation-file",
            str(orientation),
            "--role",
            role,
            "--forward-model",
            "coherent_first_born",
            "--compute-backend",
            env["OR4D_V5_BACKEND"],
            "--output",
            str(expectation),
            "--oracle-output",
            str(oracle),
            "--raw-output",
            str(trace),
            "--report-output",
            str(reports / f"{report_prefix}_generation_{count}.json"),
        ],
        env,
        records,
        [expectation, oracle, trace, reports / f"{report_prefix}_generation_{count}.json"],
        overwrite,
    )
    run(
        [
            sys.executable,
            str(ROOT / "scripts/02c_generate_clean_counted_images.py"),
            "--expectation-file",
            str(expectation),
            "--output",
            str(counted),
            "--report-output",
            str(reports / f"{report_prefix}_counted_generation_{count}.json"),
        ],
        env,
        records,
        [counted, reports / f"{report_prefix}_counted_generation_{count}.json"],
        overwrite,
    )
    run(
        [
            sys.executable,
            str(ROOT / "scripts/02d_generate_clean_dose_expectations.py"),
            "--expectation-file",
            str(expectation),
            "--output",
            str(dose_noiseless),
            "--report-output",
            str(reports / f"{report_prefix}_dose_noiseless_generation_{count}.json"),
        ],
        env,
        records,
        [dose_noiseless, reports / f"{report_prefix}_dose_noiseless_generation_{count}.json"],
        overwrite,
    )
    run(
        [
            sys.executable,
            str(ROOT / "scripts/02e_generate_clean_noise_manifest.py"),
            "--count-file",
            str(counted),
            "--output",
            str(noise_manifest),
            "--report-output",
            str(reports / f"{report_prefix}_instrument_noise_manifest_{count}.json"),
        ],
        env,
        records,
        [noise_manifest, reports / f"{report_prefix}_instrument_noise_manifest_{count}.json"],
        overwrite,
    )


def main() -> None:
    args = parse_args()
    if not 1 <= args.cpu_threads <= 16:
        raise ValueError("--cpu-threads must be between 1 and 16")
    data_root = args.data_root.resolve()
    data_root.mkdir(parents=True, exist_ok=True)
    physical_gpu = verify_gpu() if args.backend == "cuda" else ""
    env = os.environ.copy()
    env["OR4D_CONFIG"] = str(CONFIG)
    env["OR4D_V5_BACKEND"] = args.backend
    env["PYTHONPATH"] = str(ROOT / "src")
    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        env[variable] = str(args.cpu_threads)
    records: list[dict] = []
    studies = ("main", "001") if args.study == "all" else (args.study,)
    for study in studies:
        prepare_study(
            study=study,
            data_root=data_root,
            env=env,
            records=records,
            overwrite=args.overwrite,
        )
    manifest = {
        "schema": "or4d-v5-input-preparation-v1",
        "git_commit": git_commit(),
        "config": str(CONFIG.relative_to(ROOT)),
        "data_root": str(data_root),
        "study": args.study,
        "backend": args.backend,
        "physical_gpu_index": physical_gpu,
        "cpu_threads": args.cpu_threads,
        "python": platform.python_version(),
        "commands": records,
    }
    for record in records:
        record["output_sha256"] = {
            str(path): sha256_file(Path(path))
            for path in record["outputs"]
            if Path(path).is_file()
        }
    manifest_path = data_root / "run_records" / "v5_input_preparation.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"V5 input manifest: {manifest_path}")


if __name__ == "__main__":
    main()
