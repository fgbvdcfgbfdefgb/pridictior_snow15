"""Runtime hardware discovery and conservative auto-sizing profiles."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib.util
import os
import subprocess
from typing import Any


@dataclass(frozen=True)
class ModelProfile:
    name: str
    channels: int
    hidden: int
    depth: int
    batch_size: int
    analyzer_windows: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["analyzer_windows"] = list(self.analyzer_windows)
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ModelProfile":
        return cls(
            name=str(value["name"]),
            channels=int(value["channels"]),
            hidden=int(value["hidden"]),
            depth=int(value["depth"]),
            batch_size=int(value["batch_size"]),
            analyzer_windows=tuple(int(v) for v in value["analyzer_windows"]),
        )


def detect_hardware() -> dict[str, Any]:
    """Report visible CPU, RAM, and CUDA devices without requiring PyTorch."""
    result: dict[str, Any] = {
        "cpu_logical": os.cpu_count() or 1,
        "ram_total_gb": None,
        "ram_available_gb": None,
        "cuda_available": False,
        "cuda_device_count": 0,
        "gpus": [],
        "torch_installed": importlib.util.find_spec("torch") is not None,
    }
    try:
        import psutil

        mem = psutil.virtual_memory()
        result["ram_total_gb"] = round(mem.total / (1024**3), 2)
        result["ram_available_gb"] = round(mem.available / (1024**3), 2)
    except Exception as exc:  # pragma: no cover - depends on host
        result["ram_probe_error"] = str(exc)

    if result["torch_installed"]:
        try:
            import torch

            result["cuda_available"] = bool(torch.cuda.is_available())
            result["cuda_device_count"] = int(torch.cuda.device_count())
            if result["cuda_available"]:
                for index in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(index)
                    try:
                        free_bytes, total_bytes = torch.cuda.mem_get_info(index)
                    except TypeError:  # older PyTorch versions
                        free_bytes, total_bytes = torch.cuda.mem_get_info()
                    result["gpus"].append(
                        {
                            "index": index,
                            "name": props.name,
                            "total_memory_gb": round(total_bytes / (1024**3), 2),
                            "free_memory_gb": round(free_bytes / (1024**3), 2),
                        }
                    )
        except Exception as exc:  # pragma: no cover - depends on host/driver
            result["cuda_probe_error"] = str(exc)

    # Still report physical/driver-visible NVIDIA devices before PyTorch is
    # installed; training itself still requires a CUDA-enabled PyTorch build.
    if not result["gpus"]:
        try:
            probe = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,memory.total,memory.free",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
            for line in probe.stdout.splitlines():
                parts = [part.strip() for part in line.split(",", 3)]
                if len(parts) != 4:
                    continue
                index, name, total_mb, free_mb = parts
                result["gpus"].append(
                    {
                        "index": int(index),
                        "name": name,
                        "total_memory_gb": round(float(total_mb) / 1024.0, 2),
                        "free_memory_gb": round(float(free_mb) / 1024.0, 2),
                        "detected_by": "nvidia-smi",
                    }
                )
            if result["gpus"]:
                result["cuda_device_count"] = max(result["cuda_device_count"], len(result["gpus"]))
                result["nvidia_smi_visible"] = True
        except (FileNotFoundError, subprocess.SubprocessError, ValueError):
            result["nvidia_smi_visible"] = False
    return result


def choose_model_profile(hardware: dict[str, Any]) -> ModelProfile:
    """Scale predictor capacity and CPU analyser windows to discovered resources.

    The thresholds are intentionally conservative. They are starting points, not a
    promise that a particular batch fits alongside other workloads on the machine.
    """
    cores = int(hardware.get("cpu_logical") or 1)
    ram = float(hardware.get("ram_available_gb") or 0.0)
    gpus = hardware.get("gpus") or []
    max_vram = max((float(g.get("free_memory_gb", 0.0)) for g in gpus), default=0.0)

    if max_vram >= 40 or (cores >= 48 and ram >= 96):
        return ModelProfile(
            name="large",
            channels=128,
            hidden=384,
            depth=7,
            batch_size=16,
            analyzer_windows=(5, 10, 30, 60, 300, 900, 3600, 21600, 43200),
        )
    if max_vram >= 12 or (cores >= 16 and ram >= 32):
        return ModelProfile(
            name="balanced",
            channels=64,
            hidden=192,
            depth=5,
            batch_size=8,
            analyzer_windows=(10, 60, 300, 900, 3600, 21600),
        )
    return ModelProfile(
        name="compact",
        channels=32,
        hidden=96,
        depth=4,
        batch_size=4,
        analyzer_windows=(60, 300, 3600),
    )
