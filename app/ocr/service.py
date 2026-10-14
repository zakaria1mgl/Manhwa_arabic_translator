from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

from ..config import AppConfig
from ..utils.types import Quad, RegionType, TextRegion
from .easyocr_engine import EasyOCREngine
from .preprocessing import preprocess_crop
from .reading_order import (classify_regions, detect_panels, estimate_bubbles,
                            merge_bubble_lines, sort_reading_order)

log = logging.getLogger(__name__)


class OCRService:
    """Detect + recognize + order + classify text regions for one page.

    Recognition happens exactly once per detected box (detect_and_read in the
    engine). A preprocessed re-read is attempted ONLY for low-confidence
    lines, and is accepted only if it is both non-empty and more confident.
    """

    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self._engine = EasyOCREngine()
        # sticky backend: once vision proves itself broken (exception or an
        # empty page) on this run, stop paying for a doomed API call per page
        self._vision_broken = False

    def detect_and_read(self, image: np.ndarray) -> List[TextRegion]:
        backend = self.cfg.get("ocr", "backend", default="auto")
        # explicit "vision" always tries vision (the user asked for it);
        # "auto" gives up on vision after the first hard failure so a broken
        # LLM config doesn't add a doomed API call to every page.
        use_vision = backend == "vision" or (backend == "auto"
                                             and not self._vision_broken)
        if use_vision:
            regions = self._try_vision(image)
            if regions is not None:
                return self._postprocess(image, regions)
        return self._classical_detect_and_read(image)

    # ------------------------------------------------------------------
    def _try_vision(self, image: np.ndarray) -> Optional[List[TextRegion]]:
        """Vision-LLM OCR (item: one multimodal call does detection, reading,
        speaker attribution and SFX classification). Returns None when the
        backend isn't configured/fails so the caller falls back to EasyOCR."""
        from .vision_llm import VisionOCR
        vo = VisionOCR(self.cfg)
        if not vo.ready:
            return None
        try:
            regions = vo.detect_and_read(image)
        except Exception as e:
            self._vision_broken = True
            log.warning("vision OCR failed (%s); falling back to EasyOCR "
                        "for the rest of this run", e)
            return None
        if not regions:
            # A manhwa page with zero text is possible but rare; one empty
            # reply alone shouldn't disable the backend permanently.
            log.warning("vision OCR returned no regions; falling back to "
                        "EasyOCR for this page")
            return None
        # color sampling still needs the original page. Vision bboxes are
        # whole-bubble blocks: the tight-crop border median would pick the
        # bubble *outline* as bg_color and flood-fill it as text colour on
        # dark-outlined bubbles. Sample the interior instead: background =
        # modal bright cluster, text = cluster far from it.
        for r in regions:
            self._sample_colors_interior(r, image)
        return regions

    def _sample_colors_interior(self, region: TextRegion, image: np.ndarray,
                                pad: int = 4) -> None:
        """Interior-cluster colour sampling for vision block bboxes."""
        try:
            crop = self._crop(image, region.quad, pad=pad)
            if crop is None or crop.ndim != 3 or crop.shape[2] != 3:
                return
            h, w = crop.shape[:2]
            if h < 8 or w < 8:
                self._sample_colors(region, image)
                return
            inner = crop[pad + 1:h - pad - 1, pad + 1:w - pad - 1]
            px = inner.reshape(-1, 3).astype(np.float32)
            lum = px.mean(axis=1)
            # bubble interiors are usually light; fall back to dark-mode
            # (white-on-black narration) when the interior is mostly dark
            face_mask = lum >= 120 if float(np.median(lum)) >= 100 else lum <= 135
            face_px = px[face_mask]
            if len(face_px) < max(6, 0.05 * len(px)):
                face_px = px
            region.bg_color = tuple(int(v) for v in np.median(face_px, axis=0))
            diff = np.abs(px - np.array(region.bg_color)).sum(axis=1)
            ink = px[diff > 150]
            if len(ink) >= 4:
                region.text_color = tuple(int(v) for v in np.median(ink, axis=0))
            else:
                # no clear ink cluster inside the block box — reuse the
                # proven tight-crop heuristic on the same region
                self._sample_colors(region, image)
        except Exception as e:  # pragma: no cover
            log.debug("vision color sampling failed: %s", e)

    def _postprocess(self, image: np.ndarray,
                     regions: List[TextRegion]) -> List[TextRegion]:
        panels = None
        if bool(self.cfg.get("detection", "panel_segmentation", default=True)):
            try:
                panels = detect_panels(image) or None
            except Exception as e:  # pragma: no cover
                log.debug("panel segmentation failed: %s", e)
        regions = sort_reading_order(
            regions,
            panel_order=self.cfg.get("batch", "panel_order", default="ltr"),
            row_overlap_ratio=self.cfg.get("batch", "row_overlap_ratio",
                                           default=0.5),
            panels=panels,
        )
        estimate_bubbles(regions, image.shape, image=image)
        classify_regions(regions, image.shape)
        merge_bubble_lines(regions)
        return regions

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    def _comic_detect_and_read(self, image: np.ndarray):
        """Optional Comic-Text-Detector backend: better block detection on
        comic art. Returns None when the optional dependency is missing."""
        from ..detection.comic_detector import ComicTextDetector
        det = ComicTextDetector(
            weights_dir=self.cfg.get("detection", "comic_weights_dir",
                                     default="vendor/comic-text-detector/data"))
        if not det.available():
            log.warning("comic backend requested but unavailable; "
                        "falling back to easyocr")
            return None
        quads = det.detect(image)
        results = self._engine.read(image, quads)
        raw = []
        for r in results:
            pts = np.asarray(r["quad"], dtype=np.float32).reshape(-1, 2)
            raw.append({"quad": pts, "english": r["english"],
                        "raw_english": r["english"],
                        "confidence": r["confidence"]})
        return raw

    def _classical_detect_and_read(self, image: np.ndarray) -> List[TextRegion]:
        d = self.cfg
        use_comic = d.get("detection", "backend", default="easyocr") == "comic"
        raw = None
        if use_comic:
            raw = self._comic_detect_and_read(image)
        if raw is None:
            raw = self._engine.detect_and_read(
                image,
                text_threshold=float(d.get("detection", "text_threshold", default=0.6)),
                link_threshold=float(d.get("detection", "link_threshold", default=0.4)),
                width_ths=float(d.get("detection", "width_ths", default=0.7)),
                height_ths=float(d.get("detection", "height_ths", default=0.5)),
                ycenter_ths=float(d.get("detection", "ycenter_ths", default=0.5)),
                add_margin=float(d.get("detection", "add_margin", default=0.05)),
                tile_height_px=int(d.get("detection", "tile_height_px",
                                         default=2200)),
            )
        max_area_ratio = float(d.get("detection", "max_quad_area_ratio",
                                     default=0.20))
        min_h = float(d.get("detection", "min_text_height", default=10))
        min_conf = float(d.get("ocr", "min_confidence", default=0.35))
        page_area = float(image.shape[0] * image.shape[1])

        regions: List[TextRegion] = []
        for item in raw:
            quad = Quad(points=[(float(p[0]), float(p[1]))
                                for p in item["quad"]])
            if not self._sane_quad(quad, image.shape, max_area_ratio):
                continue
            if quad.height < min_h:
                continue
            eng, conf = item["english"], item["confidence"]
            orient = quad.orientation
            # one enhanced retry for weak reads only
            if conf and conf < min_conf + 0.15 and eng:
                crop = self._crop(image, quad)
                if crop is not None:
                    retry = self._read_enhanced(crop, quad.height)
                    if (retry["english"] and
                            retry["confidence"] > conf + 0.1):
                        eng, conf = retry["english"], retry["confidence"]
            elif not eng:
                crop = self._crop(image, quad)
                if crop is not None:
                    retry = self._read_enhanced(crop, quad.height)
                    if retry["english"]:
                        eng, conf = retry["english"], retry["confidence"]
            # vertical text (signs, SFX columns): EasyOCR reads horizontal
            # crops of vertical lettering as garbage — retry on a 90° rotation
            if (not eng or conf < min_conf) and quad.height > 1.8 * quad.width:
                crop = self._crop(image, quad)
                if crop is not None:
                    import cv2
                    for rot in (cv2.ROTATE_90_CLOCKWISE,
                                cv2.ROTATE_90_COUNTERCLOCKWISE):
                        rc = cv2.rotate(crop, rot)
                        retry = self._engine.read_crop(rc)
                        if retry["english"] and retry["confidence"] > conf:
                            eng, conf = retry["english"], retry["confidence"]
                            orient = 90.0 if rot == cv2.ROTATE_90_CLOCKWISE \
                                else -90.0
                            break
            if not eng or conf < min_conf:
                # drop unreadable / too-uncertain regions entirely instead of
                # shipping hallucinated text to the translator
                continue

            region = TextRegion(
                id=0,
                quad=quad,
                english=eng,
                ocr_conf=conf,
                orientation=orient,
            )
            if conf < 0.5:
                region.qa_flags.append("low_ocr_confidence")
            if eng != item["raw_english"]:
                region.qa_flags.append("ocr_sanitized")
            self._sample_colors(region, image)
            regions.append(region)

        return self._postprocess(image, regions)

    # ------------------------------------------------------------------
    @staticmethod
    def _sane_quad(quad: Quad, image_shape, max_area_ratio: float) -> bool:
        """Reject degenerate or page-spanning 'crazy diagonal' detections."""
        h, w = image_shape[:2]
        x0, y0, x1, y1 = quad.bbox
        wq, hq = x1 - x0, y1 - y0
        if wq < 6 or hq < 6:
            return False
        if wq * hq > max_area_ratio * w * h:
            return False
        if wq > 0.9 * w and hq > 0.15 * h:
            return False
        if hq > 0.9 * h and wq > 0.15 * w:
            return False
        return True

    @staticmethod
    def _crop(image: np.ndarray, quad: Quad, pad: int = 6) -> Optional[np.ndarray]:
        from ..utils.images import crop_quad
        crop = crop_quad(image, quad.polygon, pad=pad)
        return crop if crop.size else None

    def _read_enhanced(self, crop: np.ndarray, h: float) -> dict:
        target = self.cfg.get("ocr", "upscale_target_height", default=48)
        upscale = self.cfg.get("ocr", "upscale_small_text", default=True)
        if upscale and h < target:
            crop = preprocess_crop(crop, target_height=int(target), enhance=True)
        elif self.cfg.get("ocr", "preprocess", default=True):
            crop = preprocess_crop(crop, target_height=0, enhance=True)
        return self._engine.read_crop(crop)

    def _sample_colors(self, region: TextRegion, image: np.ndarray) -> None:
        """Estimate bubble interior background colour + text colour.

        Uses median of border pixels for the background (robust when text
        fills much of the crop) and the darkest/brightest outlier cluster
        for the text colour.
        """
        try:
            crop = self._crop(image, region.quad, pad=2)
            if crop is None or crop.ndim != 3 or crop.shape[2] != 3:
                return
            h, w = crop.shape[:2]
            if h < 3 or w < 3:
                return
            border = np.concatenate([
                crop[0, :].reshape(-1, 3), crop[-1, :].reshape(-1, 3),
                crop[:, 0].reshape(-1, 3), crop[:, -1].reshape(-1, 3),
            ]).astype(np.float32)
            region.bg_color = tuple(int(v) for v in np.median(border, axis=0))
            diff = np.abs(crop.reshape(-1, 3).astype(np.float32)
                          - np.array(region.bg_color)).sum(axis=1)
            text_px = crop.reshape(-1, 3)[diff > 150]
            if len(text_px) > 0:
                region.text_color = tuple(int(v) for v in np.median(text_px, axis=0))
            else:
                region.text_color = (20, 20, 20)
        except Exception as e:  # pragma: no cover
            log.debug("color sampling failed: %s", e)
