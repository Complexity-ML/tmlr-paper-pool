"""Device and precision helpers for local training scripts."""

from __future__ import annotations

from contextlib import nullcontext

import torch


def setup_mps(unlimited_watermark: bool = True, cpu_fallback: bool = True, seed: int | None = None) -> torch.device:
    if seed is not None:
        torch.manual_seed(seed)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autocast_dtype(device: torch.device) -> torch.dtype | None:
    if device.type in {"cuda", "mps"}:
        return torch.bfloat16
    return None


def autocast(device: torch.device, dtype: torch.dtype | None = None, enabled: bool = True):
    if not enabled or dtype is None or device.type == "cpu":
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def empty_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()
