from __future__ import annotations

import getpass
import os
import subprocess
from pathlib import Path
from typing import Any, Iterable


def _resolved(path: Path | str) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def _is_within(path: Path, roots: Iterable[Path]) -> bool:
    for root in roots:
        if path == root or root in path.parents:
            return True
    return False


def enforce_server_write_scope(
    path: Path | str,
    config: dict[str, Any],
    *,
    operation: str = "write",
) -> Path:
    """Reject server-side writes outside the configured user-owned roots.

    The policy is activated automatically for configured server usernames and
    can also be forced in smoke tests with ``OR4D_ENFORCE_SERVER_PATHS=1``.
    It deliberately validates only write targets; input files may remain in
    system-managed read-only locations such as Conda environments.
    """
    target = _resolved(path)
    runtime = config["v6"]["runtime"]
    guarded_users = {str(value) for value in runtime["enforce_write_scope_for_users"]}
    forced = os.environ.get("OR4D_ENFORCE_SERVER_PATHS", "0") == "1"
    if getpass.getuser() not in guarded_users and not forced:
        return target

    roots = [_resolved(value) for value in runtime["allowed_write_roots"]]
    if not roots:
        raise RuntimeError("V6 server write policy has no allowed roots")
    if not _is_within(target, roots):
        rendered = ", ".join(str(root) for root in roots)
        raise PermissionError(
            f"Refusing to {operation} outside V6 server write roots: "
            f"target={target}; allowed={rendered}"
        )
    return target


def enforce_server_write_scopes(
    paths: Iterable[Path | str],
    config: dict[str, Any],
    *,
    operation: str = "write",
) -> list[Path]:
    return [
        enforce_server_write_scope(path, config, operation=operation)
        for path in paths
    ]


def require_empty_bound_gpu(config: dict[str, Any]) -> str:
    """Validate the one physical GPU exposed to a CUDA worker before use."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible.isdigit():
        raise RuntimeError(
            "CUDA V6 runs require CUDA_VISIBLE_DEVICES to expose one physical GPU"
        )
    rows = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    status: dict[str, tuple[str, int, int]] = {}
    for row in rows.splitlines():
        fields = [value.strip() for value in row.split(",")]
        if len(fields) == 4:
            status[fields[0]] = (fields[1], int(fields[2]), int(fields[3]))
    if visible not in status:
        raise RuntimeError(f"Physical GPU {visible} is absent from nvidia-smi")
    gpu_uuid, memory, utilization = status[visible]
    process_rows = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    active_pids = []
    for row in process_rows.splitlines():
        fields = [value.strip() for value in row.split(",")]
        if len(fields) == 2 and fields[0] == gpu_uuid:
            active_pids.append(fields[1])
    runtime = config["v6"]["runtime"]
    if (
        active_pids
        or memory > int(runtime["empty_gpu_max_memory_MiB"])
        or utilization > int(runtime["empty_gpu_max_utilization_percent"])
    ):
        raise RuntimeError(
            f"Physical GPU {visible} is not empty: {memory} MiB, {utilization}%, "
            f"compute PIDs={active_pids}"
        )
    return visible
