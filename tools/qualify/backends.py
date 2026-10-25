"""Backend profiles: where the python, the engine, the locks and the GPU counters live. One entry point per backend, same stages.
MLX: add a profile with its own server/lock when the Ultra arrives (stages and report are backend-neutral)."""
from __future__ import annotations

import os
from pathlib import Path

H = Path.home() / "hermes-work"

PROFILES = {
    "cuda": dict(
        python=str(H / "rtx1/venv/bin/python"), engine=str(H / "rtx1/engine"),
        locks=[str(H / "rtx1/gpu3090.lock")], env={"EXL3_CUDA_GRAPH": "1"},
        gpu_used_mib="nvidia-smi", host_peak_note="3090 24 GiB",
    ),
    "rocm": dict(  # not exercised yet (iGPU busy); same stages, locks as in mu_common.md (one holder, this order)
        python=str(H / "kyojin-rb/.venv/bin/python"), engine=str(H / "kyojin-rb"),
        locks=[str(Path.home() / "strix-bench/gpu.lock"), str(H / "strixfix/kigpu.lock")], env={},
        gpu_used_mib="/sys/class/drm/card1/device/mem_info_gtt_used", host_peak_note="Strix unified memory",
    ),
}


def profile(name: str, repo: str | None = None) -> dict:
    """repo = the engine checkout with the built extension (PYTHONPATH and cwd of every server, as the lanes run them)."""
    if name not in PROFILES:
        raise SystemExit(f"unknown backend {name!r}: {sorted(PROFILES)}")
    p = dict(PROFILES[name])
    p["env"] = {**p["env"], "PYTHONPATH": repo or p["engine"]}
    return p


def mem_available_gib() -> float:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 2**20
    return 0.0


def gpu_free_mib(backend: str) -> float | None:
    """Free device memory in MiB, None when it cannot be read (used by the max-ctx upper bound)."""
    import subprocess
    try:
        if backend == "cuda":
            r = subprocess.run(["nvidia-smi", "--query-gpu=memory.total,memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20)
            tot, used = (float(x) for x in r.stdout.strip().splitlines()[0].split(","))
            return tot - used
        # rocm: GTT is the unified-memory pool, invisible to `free`
        d = Path("/sys/class/drm/card1/device")
        return (int((d / "mem_info_gtt_total").read_text()) - int((d / "mem_info_gtt_used").read_text())) / 2**20
    except Exception:
        return None
