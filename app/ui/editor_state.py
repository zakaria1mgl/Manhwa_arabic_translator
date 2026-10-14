"""Interactive editor state: per-page region fixes without full re-runs."""
from __future__ import annotations

import os
import threading
from typing import Dict, List, Optional

from ..utils.types import PageResult, TextRegion


class EditorState:
    """Holds the last processed page + manual overrides so a single bubble
    can be corrected and re-rendered without re-running the pipeline."""

    def __init__(self):
        self._lock = threading.Lock()
        self.page: Optional[PageResult] = None
        self.base_image_path: str = ""
        self.cleaned_path: str = ""
        self.overrides: Dict[int, Dict[str, str]] = {}

    def set_page(self, page: PageResult, image_path: str,
                 cleaned_path: str = "") -> None:
        with self._lock:
            self.page = page
            self.base_image_path = image_path
            self.cleaned_path = cleaned_path or getattr(page, "cleaned_path", "")
            self.overrides = {}

    def render_base(self) -> str:
        """Best available base for re-rendering: the inpainted (English-free)
        page when it exists, otherwise the original."""
        if self.cleaned_path and os.path.isfile(self.cleaned_path):
            return self.cleaned_path
        return self.base_image_path

    def apply_override(self, region_id: int, *, arabic: Optional[str] = None,
                       font_size: Optional[int] = None,
                       font_name: Optional[str] = None) -> None:
        with self._lock:
            ov = self.overrides.setdefault(region_id, {})
            if arabic is not None:
                ov["arabic"] = arabic
            if font_size is not None:
                ov["font_size"] = str(font_size)
            if font_name is not None:
                ov["font_name"] = font_name

    def get_override(self, region_id: int) -> Dict[str, str]:
        with self._lock:
            return dict(self.overrides.get(region_id, {}))

    def clear(self) -> None:
        with self._lock:
            self.overrides = {}
