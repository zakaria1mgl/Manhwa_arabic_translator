from __future__ import annotations

import logging
import os
import threading
from typing import Any, Callable, Dict, Optional

log = logging.getLogger(__name__)

_lock = threading.Lock()
_cache: Dict[str, Any] = {}


def detect_device() -> str:
    """Return 'cuda' or 'cpu' (also exposes 'mps' for completeness)."""
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def cuda_available() -> bool:
    return detect_device() == "cuda"


def vram_mb(device: Optional[str] = None) -> int:
    """Free VRAM in MB for the selected device (0 on CPU)."""
    device = device or detect_device()
    if device != "cuda":
        return 0
    try:
        import torch
        free, total = torch.cuda.mem_get_info()
        return int(free / (1024 * 1024))
    except Exception:
        return 0


def low_vram_env() -> bool:
    return os.environ.get("MANHWA_LOW_VRAM", "").lower() in ("1", "true", "yes")


def vram_safe_min(mb: int, low_vram: bool = False) -> bool:
    """Return True when the environment is VRAM-safe to load a model of `mb` MB."""
    if not cuda_available():
        return True  # CPU has no VRAM pressure
    free = vram_mb()
    if low_vram:
        return free > max(mb, 512)
    return free > mb + 512


def cached(key: str, builder: Callable[[], Any], cache: bool = True) -> Any:
    """Thread-safe lazy singleton cache."""
    if not cache:
        return builder()
    if key not in _cache:
        with _lock:
            if key not in _cache:
                _cache[key] = builder()
    return _cache[key]


def invalidate(key: Optional[str] = None) -> None:
    """Drop cached model(s); used by the low-VRAM manager."""
    with _lock:
        if key is None:
            _cache.clear()
        else:
            _cache.pop(key, None)


def model_size_hint(mb: int) -> None:
    """Book-keeping helper; called when a model registers its approx footprint."""
    log.info("model footprint hint: %d MB", mb)
