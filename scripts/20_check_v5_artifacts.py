#!/usr/bin/env python3
"""Check the already-generated V5 server artifact set.

This command is intentionally read-only.  V5 inputs were generated and
evaluated on the server; a clone reuses that data root instead of regenerating
another benchmark.  ``reports/v5/MANIFEST.json`` is the local copy of the
server artifact manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "reports/v5/MANIFEST.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--study", choices=("main", "001", "all"), default="all")
    parser.add_argument(
        "--verify-hash",
        action="store_true",
        help="Read every listed artifact and compare its recorded SHA-256.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def selected(artifact_id: str, study: str) -> bool:
    if study == "all":
        return True
    is_001 = artifact_id.startswith("study001_")
    return is_001 if study == "001" else not is_001


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    manifest_path = args.manifest.resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts = [
        artifact
        for artifact in payload.get("artifacts", [])
        if selected(str(artifact["artifact_id"]), args.study)
    ]
    if not artifacts:
        raise ValueError(f"manifest has no artifacts for study={args.study}")

    missing: list[str] = []
    mismatched: list[dict] = []
    for artifact in artifacts:
        path = data_root / str(artifact["server_relative_path"])
        if not path.is_file():
            missing.append(str(path))
            continue
        expected_size = artifact.get("size_bytes")
        if expected_size is not None and path.stat().st_size != int(expected_size):
            mismatched.append(
                {
                    "artifact_id": artifact["artifact_id"],
                    "field": "size_bytes",
                    "expected": int(expected_size),
                    "actual": path.stat().st_size,
                }
            )
        if args.verify_hash and artifact.get("sha256"):
            actual_hash = sha256_file(path)
            if actual_hash != str(artifact["sha256"]):
                mismatched.append(
                    {
                        "artifact_id": artifact["artifact_id"],
                        "field": "sha256",
                        "expected": artifact["sha256"],
                        "actual": actual_hash,
                    }
                )

    result = {
        "manifest": str(manifest_path),
        "data_root": str(data_root),
        "study": args.study,
        "verify_hash": bool(args.verify_hash),
        "artifact_count": len(artifacts),
        "missing": missing,
        "mismatched": mismatched,
        "status": "ok" if not missing and not mismatched else "failed",
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if missing or mismatched:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
