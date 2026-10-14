from __future__ import annotations

import logging
from typing import List

import numpy as np

from ..models.manager import cached, detect_device
from .base import BaseTextDetector

log = logging.getLogger(__name__)


class EasyOCRDetector(BaseTextDetector):
    """CRAFT-based detector from EasyOCR.

    EasyOCR's Reader.detect() returns, per batch image:
      horizontal_list: [[x_min, x_max, y_min, y_max], ...]   axis-aligned boxes
      free_list:       [[x1, x2, x3, x4, y1, y2, y3, y4], ...] skewed/rotated quads

    Both are converted here into 4x2 float quads (one per text line/block).
    """

    name = "easyocr"

    def __init__(self, *, link_threshold: float = 0.4, text_threshold: float = 0.6,
                 low_vram: bool = False, max_area_ratio: float = 0.20,
                 min_size_px: int = 6):
        self._link = link_threshold
        self._text = text_threshold
        self._low_vram = low_vram
        self._max_area_ratio = max_area_ratio
        self._min_size = min_size_px

    def _reader(self):
        import easyocr

        def build():
            gpu = detect_device() == "cuda"
            # detection-only reader keeps the recognizer unloaded for this backend
            return easyocr.Reader(lang_list=["en"], gpu=gpu, detector=True,
                                  recognizer=False, verbose=False)

        return cached("easyocr_detector", build)

    def detect(self, image: np.ndarray) -> List[np.ndarray]:
        reader = self._reader()
        rgb = image[..., ::-1]  # BGR -> RGB
        h, w = rgb.shape[:2]
        try:
            horizontal_list, free_list = reader.detect(
                rgb,
                text_threshold=self._text,
                link_threshold=self._link,
                width_ths=0.7,
                height_ths=0.5,
                ycenter_ths=0.5,
                add_margin=0.15,
            )
        except Exception as e:
            log.warning("detection failed: %s", e)
            return []

        quads: List[np.ndarray] = []

        def _sane(pts: np.ndarray) -> bool:
            """Reject degenerate or page-spanning 'crazy diagonal' detections.

            CRAFT can glue distant text blocks into one giant rotated quad that
            crosses panels (seen in the detection overlay as long diagonals).
            A single text line/bubble never covers a large share of the page.
            """
            xs, ys = pts[:, 0], pts[:, 1]
            wq, hq = xs.max() - xs.min(), ys.max() - ys.min()
            if wq < self._min_size or hq < self._min_size:
                return False
            if wq * hq > self._max_area_ratio * w * h:
                return False
            # extremely long AND tall quads are merged panels, not one text line
            if wq > 0.9 * w and hq > 0.15 * h:
                return False
            if hq > 0.9 * h and wq > 0.15 * w:
                return False
            return True

        # axis-aligned boxes: [x_min, x_max, y_min, y_max] (+ optional padding)
        for item in (horizontal_list[0] if horizontal_list else []):
            try:
                vals = [float(v) for v in list(item)[:4]]
            except Exception:
                continue
            if len(vals) < 4:
                continue
            x_min, x_max, y_min, y_max = vals
            if x_max <= x_min or y_max <= y_min:
                continue
            pts = np.array([[x_min, y_min], [x_max, y_min],
                            [x_max, y_max], [x_min, y_max]], dtype=np.float32)
            if _sane(pts):
                quads.append(_clip(pts, w, h))

        # free-form quads: [x1, x2, x3, x4, y1, y2, y3, y4] (flat or nested)
        for item in (free_list[0] if free_list else []):
            try:
                arr = np.array(item, dtype=np.float32).flatten()
            except Exception:
                continue
            if arr.size < 8:
                continue
            pts = np.stack([arr[:4], arr[4:8]], axis=1)
            if (pts[:, 0].max() - pts[:, 0].min()) < 2 or \
               (pts[:, 1].max() - pts[:, 1].min()) < 2:
                continue
            if _sane(pts):
                quads.append(_clip(pts, w, h))

        return _dedupe(quads)


def _clip(pts: np.ndarray, w: int, h: int) -> np.ndarray:
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
    return pts.astype(np.float32)


def _dedupe(quads: List[np.ndarray], iou_thr: float = 0.6) -> List[np.ndarray]:
    """Keep the larger quad when two detections overlap heavily."""
    out: List[np.ndarray] = []
    for q in quads:
        a = _bbox(q)
        keep = True
        for i, o in enumerate(out):
            if _iou(a, _bbox(o)) > iou_thr:
                if _area(a) > _area(_bbox(o)):
                    out[i] = q
                keep = False
                break
        if keep:
            out.append(q)
    return out


def _bbox(q: np.ndarray):
    x0, y0 = q.min(axis=0)
    x1, y1 = q.max(axis=0)
    return (float(x0), float(y0), float(x1), float(y1))


def _area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _iou(a, b) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter == 0:
        return 0.0
    union = _area(a) + _area(b) - inter
    return inter / max(1e-6, union)
