from __future__ import annotations

import logging
from typing import List

import numpy as np

from ..config import AppConfig
from ..utils.types import PageResult, TextRegion
from .backends import CompositeInpainter
from .masking import build_text_mask, uniform_bubble_fill

log = logging.getLogger(__name__)


class TextRemovalService:
    """Erase original English text and reconstruct the background."""

    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.inpainter = CompositeInpainter(cfg)

    def remove_text(self, image: np.ndarray, regions: List[TextRegion]) -> np.ndarray:
        if not regions:
            return image
        dilate = int(self.cfg.get("inpainting", "mask_dilate_px", default=4))
        tol = int(self.cfg.get("inpainting", "uniform_tolerance", default=18))
        img = image
        if self.cfg.get("inpainting", "color_fill_uniform_bubbles", default=True):
            img = uniform_bubble_fill(image, regions, tolerance=tol)
        mask = build_text_mask(img, regions, dilate_px=dilate,
                               uniform_tolerance=tol)
        if mask.max() == 0:
            return img
        return self.inpainter.inpaint(img, mask)
