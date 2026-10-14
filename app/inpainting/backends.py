from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from ..config import AppConfig
from ..models.manager import cached, detect_device, vram_safe_min, invalidate

log = logging.getLogger(__name__)

LAMA_MIN_FREE_MB = 3500


class InpaintBackend(ABC):
    name = "base"

    @abstractmethod
    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray: ...

    def close(self) -> None:  # pragma: no cover
        pass


class LamaBackend(InpaintBackend):
    """Big-LaMa via simple-lama-inpainting (best quality)."""

    name = "lama"

    def __init__(self, low_vram: bool = False):
        self.low_vram = low_vram

    def _model(self):
        from simple_lama_inpainting import SimpleLama  # type: ignore

        def build():
            return SimpleLama()

        return cached("lama_model", build)

    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        from PIL import Image
        model = self._model()
        img_pil = Image.fromarray(image[..., ::-1])  # BGR -> RGB
        mask_pil = Image.fromarray(mask)
        result = model(img_pil, mask_pil)  # returns PIL RGB
        arr = np.asarray(result.convert("RGB"))
        return arr[..., ::-1]  # back to BGR


class OpenCVBackend(InpaintBackend):
    """Telea inpainting; CPU-friendly fallback for non-uniform backgrounds."""

    name = "opencv"

    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        import cv2
        # first pass: Telea for structure
        out = cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        # second pass: for large uniform holes, Telea leaves smudges.
        # replace big connected mask components with local background colour
        # sampled from the hole's border + mild noise, which looks much
        # cleaner inside speech bubbles.
        num, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
        for i in range(1, num):
            x, y, bw, bh, area = stats[i]
            if area < 400:
                continue  # small holes: Telea is fine
            comp = (labels == i)
            # sample border ring around the component
            x0, y0 = max(0, x - 6), max(0, y - 6)
            x1, y1 = min(image.shape[1], x + bw + 6), min(image.shape[0], y + bh + 6)
            ring_mask = np.zeros((y1 - y0, x1 - x0), dtype=bool)
            ring_mask[:, :3] = True; ring_mask[:, -3:] = True
            ring_mask[:3, :] = True; ring_mask[-3:, :] = True
            comp_crop = comp[y0:y1, x0:x1]
            ring_mask &= ~comp_crop
            ring_px = out[y0:y1, x0:x1][ring_mask]
            if len(ring_px) < 10:
                continue
            med = np.median(ring_px, axis=0)
            # flat fill only when the ring itself is uniform (bubble interior)
            spread = float(np.abs(ring_px.astype(np.float32) - med).sum(axis=1).std())
            if spread < 40:
                noise = np.random.default_rng(0).normal(0, 2.5, size=(y1 - y0, x1 - x0, 1))
                fill = np.clip(med[None, None, :] + noise, 0, 255).astype(np.uint8)
                out[y0:y1, x0:x1] = np.where(comp_crop[..., None], fill,
                                             out[y0:y1, x0:x1])
        return out


class CompositeInpainter:
    """Chooses the best available backend, with a VRAM-aware LaMa guard."""

    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.low_vram = bool(cfg.get("device", "low_vram_mode", default=False))
        self._lama: Optional[LamaBackend] = None
        self._cv = OpenCVBackend()

    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        backend = self.cfg.get("inpainting", "backend", default="auto")
        if self._use_lama(backend):
            try:
                if self._lama is None:
                    self._lama = LamaBackend(low_vram=self.low_vram)
                return self._lama.inpaint(image, mask)
            except Exception as e:
                log.warning("LaMa failed (%s); falling back to OpenCV", e)
        return self._cv.inpaint(image, mask)

    def _use_lama(self, setting: str) -> bool:
        if setting == "opencv":
            return False
        if setting == "lama":
            return True
        # auto: prefer LaMa when it is installed and VRAM allows
        try:
            import simple_lama_inpainting  # noqa: F401
        except Exception:
            return False
        if detect_device() == "cuda" and not vram_safe_min(LAMA_MIN_FREE_MB, self.low_vram):
            log.info("low VRAM: skipping LaMa, using OpenCV inpainting")
            return False
        return True

    def release_models(self) -> None:
        invalidate("lama_model")
        self._lama = None
