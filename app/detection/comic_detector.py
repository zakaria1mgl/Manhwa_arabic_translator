from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

from ..models.manager import cached, detect_device
from .base import BaseTextDetector

log = logging.getLogger(__name__)


class ComicTextDetector(BaseTextDetector):
    """Adapter for the Comic Text Detector (dmMaze/comic-text-detector).

    This is an OPTIONAL backend: the project is not on PyPI, so install it
    manually (clone the repo into `vendor/comic-text-detector` or add it to
    PYTHONPATH). It detects comic text blocks AND speech-bubble masks far
    better than CRAFT on stylized comic art. When the import fails, callers
    must fall back to the EasyOCR backend.
    """

    name = "comic"

    def __init__(self, weights_dir: str = "vendor/comic-text-detector/data"):
        self.weights_dir = weights_dir
        self._available: Optional[bool] = None

    # ------------------------------------------------------------------
    def _model(self):
        def build():
            import os
            import sys
            ctd_path = os.path.abspath(
                os.path.join(os.path.dirname(self.weights_dir), ".."))
            if ctd_path not in sys.path:
                sys.path.insert(0, ctd_path)
            from ctd.inferencer import TextDetector  # type: ignore
            return TextDetector(model_path=self.weights_dir,
                                device="cuda" if detect_device() == "cuda"
                                else "cpu")

        return cached("comic_text_detector", build)

    def available(self) -> bool:
        if self._available is None:
            try:
                self._model()
                self._available = True
            except Exception as e:
                log.info("comic-text-detector unavailable: %s", e)
                self._available = False
        return self._available

    # ------------------------------------------------------------------
    def detect(self, image: np.ndarray) -> List[np.ndarray]:
        """Return text-block quads (4x2 float). Bubble masks are not used by
        the current pipeline (it has its own bubble snapping), but blocks map
        directly onto quads."""
        model = self._model()
        rgb = image[..., ::-1]
        # ctd.inferencer.TextDetector.__call__ returns
        # (mask, mask_refined, blk_list) in current releases; handle both.
        result = model(rgb)
        blk_list = result[2] if isinstance(result, tuple) else result
        quads: List[np.ndarray] = []
        for blk in blk_list:
            lines = getattr(blk, "lines", None)
            if lines:
                for ln in lines:
                    pts = np.asarray(ln, dtype=np.float32).reshape(-1, 2)
                    if len(pts) >= 4:
                        quads.append(pts[:4])
            else:
                try:
                    x0, y0, x1, y1 = [float(v) for v in blk.xyxy[:4]]
                except Exception:
                    continue
                quads.append(np.array([[x0, y0], [x1, y0],
                                       [x1, y1], [x0, y1]], dtype=np.float32))
        return quads
