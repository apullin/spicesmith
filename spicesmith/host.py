"""Validation of caller-selected affinity; no host-specific CPU reservations."""
from __future__ import annotations

import os

from .config import CpuSet


def check_cpus(cpus: str | None) -> None:
    if cpus is None:
        return
    requested = CpuSet.parse(cpus)
    if (not requested or len(set(requested)) != len(requested)
            or not set(requested) <= os.sched_getaffinity(0)):
        raise ValueError(f'invalid or unavailable CPU set: {cpus}')
