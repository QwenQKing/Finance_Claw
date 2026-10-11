from __future__ import annotations

import ctypes
import os
import sys
from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class MachineProfile:
    cpu_count: int
    available_memory_mb: Optional[int]

def _windows_available_memory_mb() -> Optional[int]:
    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    stat = MEMORYSTATUSEX()
    stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    ok = ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
    if not ok:
        return None
    return int(stat.ullAvailPhys // (1024 * 1024))

def _linux_available_memory_mb() -> Optional[int]:
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                if not line.startswith("MemAvailable:"):
                    continue
                parts = line.split()
                if len(parts) < 2:
                    break
                return int(parts[1]) // 1024
    except Exception:
        return None
    return None

def detect_machine_profile() -> MachineProfile:
    cpu_count = max(1, int(os.cpu_count() or 1))
    available_memory_mb: Optional[int] = None

    try:
        if sys.platform.startswith("win"):
            available_memory_mb = _windows_available_memory_mb()
        elif sys.platform.startswith("linux"):
            available_memory_mb = _linux_available_memory_mb()
    except Exception:
        available_memory_mb = None

    return MachineProfile(cpu_count=cpu_count, available_memory_mb=available_memory_mb)

def clamp_workers(value: int, *, upper_bound: Optional[int] = None) -> int:
    value = max(1, int(value))
    if upper_bound is not None:
        value = min(value, max(1, int(upper_bound)))
    return value

def recommended_workers(task: str, profile: Optional[MachineProfile] = None) -> int:
    profile = profile or detect_machine_profile()
    cpu = profile.cpu_count
    mem = profile.available_memory_mb

    def mem_at_least(limit_mb: int) -> bool:
        return mem is None or mem >= limit_mb

    if task in {"runner_samples", "bench_samples"}:
        if cpu >= 24 and mem_at_least(16000):
            return 8
        if cpu >= 12 and mem_at_least(8000):
            return 6
        if cpu >= 8:
            return 4
        if cpu >= 4:
            return 2
        return 1

    if task == "kg_samples":
        if cpu >= 24 and mem_at_least(16000):
            return 6
        if cpu >= 12 and mem_at_least(8000):
            return 4
        if cpu >= 8:
            return 3
        if cpu >= 4:
            return 2
        return 1

    if task == "kg_datasets":
        if cpu >= 24 and mem_at_least(16000):
            return 2
        if cpu >= 12 and mem_at_least(12000):
            return 2
        return 1

    if task == "kg_llm_chunks":
        return 1

    return 1

def resolve_worker_count(
    *,
    explicit: Optional[int] = None,
    env_var: Optional[str] = None,
    task: str,
    upper_bound: Optional[int] = None,
) -> int:
    raw = None
    if explicit is not None:
        raw = explicit
    elif env_var:
        env_value = os.getenv(env_var, "").strip()
        if env_value:
            raw = int(env_value)

    if raw is None:
        raw = recommended_workers(task)

    return clamp_workers(int(raw), upper_bound=upper_bound)
