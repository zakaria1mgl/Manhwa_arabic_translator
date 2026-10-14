from __future__ import annotations

import logging
import os
from typing import Callable, List, Optional

from ..config import AppConfig
from ..pipeline import Pipeline, ProgressFn
from ..utils.types import PageResult

log = logging.getLogger(__name__)


def list_images(folder: str, exts: Optional[List[str]] = None) -> List[str]:
    exts = exts or [".png", ".jpg", ".jpeg", ".webp"]
    files = []
    for name in sorted(os.listdir(folder)):
        if os.path.splitext(name)[1].lower() in exts:
            files.append(os.path.join(folder, name))
    return files


def natural_key(name: str):
    import re
    return [int(t) if t.isdigit() else t.lower()
            for t in re.split(r"(\d+)", os.path.basename(name))]


def translate_chapter(pipeline: Pipeline, image_paths: List[str],
                      progress: Optional[ProgressFn] = None,
                      page_numbers: Optional[List[int]] = None) -> List[PageResult]:
    """Batch-translate a chapter, maintaining context between pages."""
    results: List[PageResult] = []
    paths = sorted(image_paths, key=natural_key)
    total = max(1, len(paths))
    for i, path in enumerate(paths):
        if progress:
            progress(f"page {i + 1}/{total}: {os.path.basename(path)}",
                     i / total)
        page_no = page_numbers[i] if page_numbers else i + 1
        try:
            res = pipeline.translate_image(path, page_number=page_no)
        except Exception as e:  # pragma: no cover
            log.exception("page failed: %s", path)
            res = PageResult(image_path=path, error=str(e))
        results.append(res)
        if res.error == "cancelled":
            break
    if progress:
        progress("chapter done", 1.0)
    return results
