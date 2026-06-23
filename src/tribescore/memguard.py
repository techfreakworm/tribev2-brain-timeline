"""System-memory governor for local MPS runs — guards on `vm_stat`, NOT RSS.

MPS/Metal WIRED memory is invisible to process RSS, so psutil-based guards are
blind to it and would let the OS OOM (it crashes near ~125 GB on a 128 GB Mac).
This reads `vm_stat` (active + wired + compressed pages) for true SYSTEM memory
use and gates the local MPS video encode under a safe ceiling. macOS-only; the
callers gate on MPS so this never runs on the CUDA Space.
"""

from __future__ import annotations

import subprocess

GATE_GB: float = 105.0          # operational ceiling (OS crashes ~125 GB)
SAFETY_GB: float = 10.0         # reserved for the ~25 s post-kill wired reclaim
PER_CLIP_FLOOR_GB: float = 25.0  # a single bounded clip's expected need
ABORT_GB: float = 90.0          # mid-run hard abort threshold (system used)


def _parse_vm_stat(text: str) -> float:
    """System memory in use (GB) = (active + wired + compressed) pages × page size."""
    page = 16384
    active = wired = compressed = 0
    for line in text.splitlines():
        if "page size of" in line:
            page = int(line.split()[-2])
        elif "Pages active" in line:
            active = int(line.split()[-1].rstrip("."))
        elif "Pages wired down" in line:
            wired = int(line.split()[-1].rstrip("."))
        elif "occupied by compressor" in line:
            compressed = int(line.split()[-1].rstrip("."))
    return (active + wired + compressed) * page / 1e9


def system_used_gb() -> float:
    return _parse_vm_stat(subprocess.check_output(["vm_stat"]).decode())


def headroom_gb(gate_gb: float = GATE_GB, safety_gb: float = SAFETY_GB) -> float:
    return gate_gb - system_used_gb() - safety_gb


def require_headroom(min_gb: float = PER_CLIP_FLOOR_GB) -> None:
    h = headroom_gb()
    if h < min_gb:
        raise MemoryError(
            f"Insufficient memory headroom to start: {h:.1f} GB < {min_gb} GB "
            f"(system using {system_used_gb():.1f} GB). Close other apps and retry."
        )


def check_or_abort(abort_gb: float = ABORT_GB) -> None:
    used = system_used_gb()
    if used > abort_gb:
        raise MemoryError(
            f"System memory {used:.1f} GB exceeded the {abort_gb} GB safety ceiling "
            f"mid-run; aborting before the OS OOMs."
        )
