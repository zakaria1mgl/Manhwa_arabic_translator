from __future__ import annotations

from typing import List

import numpy as np

from ..utils.types import PageResult


def draw_debug_overlay(image_bgr: np.ndarray, page: PageResult) -> np.ndarray:
    """Draw detection boxes, ids, confidence and type tags for debugging."""
    import cv2
    out = image_bgr.copy()
    colors = {
        "dialogue": (80, 180, 80),
        "thought": (200, 140, 60),
        "narration": (60, 120, 220),
        "caption": (60, 120, 220),
        "sfx": (60, 60, 230),
        "sign": (200, 80, 200),
        "ui": (160, 160, 60),
        "unknown": (128, 128, 128),
    }
    for r in page.regions:
        color = colors.get(r.region_type.value, (128, 128, 128))
        pts = r.quad.polygon.reshape(-1, 1, 2).astype(np.int32)
        cv2.polylines(out, [pts], True, color, 2)
        if r.bubble_polygon is not None:
            cv2.polylines(out, [r.bubble_polygon.reshape(-1, 1, 2).astype(np.int32)],
                          True, (0, 200, 255), 1)
        x, y = r.bbox[0], max(0, r.bbox[1] - 6)
        label = f"#{r.id} {r.region_type.value} {r.ocr_conf:.2f}"
        cv2.putText(out, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    color, 1, cv2.LINE_AA)
    return out
