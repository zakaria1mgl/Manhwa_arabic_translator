from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

import numpy as np

from ..models.manager import cached, detect_device
from .base import OCREngineBase

log = logging.getLogger(__name__)

# Reasonable characters for English comic text. Anything outside this set is
# almost always an OCR hallucination on stylized art (e.g. reading decoration
# as Thai/CJK/dingbats) and poisons the translation stage.
_ALLOWED_RE = re.compile(
    r"[^A-Za-z0-9\s\.\,\!\?\'\"\-\:\;\(\)\[\]\{\}…&%\$#\*\+/\\~@<>=\|^_`]")


def sanitize_text(text: str) -> str:
    """Strip hallucinated non-Latin glyphs from an English OCR read."""
    if not text:
        return ""
    cleaned = _ALLOWED_RE.sub("", text)
    return " ".join(cleaned.split())


class EasyOCREngine(OCREngineBase):
    """EasyOCR reader used the way it was designed: one ``detect`` pass over
    the page followed by one ``recognize`` pass on those boxes.

    This replaces the old flow which (1) ran a detector-only Reader, (2) ran
    ``readtext`` *again* on every crop (re-detecting inside the crop and
    frequently returning nothing), and (3) sometimes read a preprocessed copy
    too — three OCR passes per text line with crops so tight the internal
    detector failed.
    """

    name = "easyocr"

    def reader(self):
        import easyocr

        def build():
            gpu = detect_device() == "cuda"
            return easyocr.Reader(lang_list=["en"], gpu=gpu, verbose=False)

        return cached("easyocr_reader", build)

    # ------------------------------------------------------------------
    def detect_and_read(self, image_bgr: np.ndarray, *,
                        text_threshold: float = 0.6,
                        link_threshold: float = 0.4,
                        width_ths: float = 0.7,
                        height_ths: float = 0.5,
                        ycenter_ths: float = 0.5,
                        add_margin: float = 0.05,
                        tile_height_px: int = 2200,
                        tile_overlap_px: int = 150) -> List[Dict[str, Any]]:
        """Detect text on the page and recognize each box in one flow.

        Long webtoon strips are processed in overlapping horizontal tiles —
        CRAFT internally downscales huge canvases, which used to make text at
        the far ends of a tall strip simply vanish. Results from overlapping
        seams are de-duplicated by IoU.

        Returns a list of {"quad": 4x2 float array, "english", "confidence"}.
        """
        h = image_bgr.shape[0]
        tiles = _tile_ranges(h, tile_height_px, tile_overlap_px)
        results: List[Dict[str, Any]] = []
        for y0, y1 in tiles:
            view = image_bgr[y0:y1]
            for item in self._detect_read_once(
                    view,
                    text_threshold=text_threshold, link_threshold=link_threshold,
                    width_ths=width_ths, height_ths=height_ths,
                    ycenter_ths=ycenter_ths, add_margin=add_margin):
                item["quad"][:, 1] += y0  # shift tile-local quads to page coords
                results.append(item)
        if len(tiles) > 1:
            results = _dedupe_results(results)
        return results

    def _detect_read_once(self, image_bgr: np.ndarray, *,
                          text_threshold: float,
                          link_threshold: float,
                          width_ths: float,
                          height_ths: float,
                          ycenter_ths: float,
                          add_margin: float) -> List[Dict[str, Any]]:
        reader = self.reader()
        rgb = image_bgr[..., ::-1]  # BGR -> RGB
        try:
            horizontal_list, free_list = reader.detect(
                rgb,
                text_threshold=text_threshold,
                link_threshold=link_threshold,
                width_ths=width_ths,
                height_ths=height_ths,
                ycenter_ths=ycenter_ths,
                add_margin=add_margin,
            )
        except Exception as e:
            log.warning("detection failed: %s", e)
            return []

        horiz = horizontal_list[0] if horizontal_list else []
        free = free_list[0] if free_list else []
        if not horiz and not free:
            return []

        try:
            results = reader.recognize(rgb, horiz, free, detail=1,
                                       paragraph=False)
        except Exception as e:
            log.warning("recognition failed: %s", e)
            return []

        out: List[Dict[str, Any]] = []
        for box, text, conf in results:
            quad = _box_to_quad(box, rgb.shape[1], rgb.shape[0])
            if quad is None:
                continue
            clean = sanitize_text(str(text))
            if not clean:
                continue
            out.append({
                "quad": quad,
                "english": clean,
                "raw_english": str(text).strip(),
                "confidence": float(conf),
            })
        return out

    # ------------------------------------------------------------------
    def read_crop(self, crop_bgr: np.ndarray) -> Dict[str, Any]:
        """Recognize text in an already-cropped region (single attempt).

        Used only as a low-confidence retry on enhanced/inverted crops —
        normal recognition happens in :meth:`detect_and_read`.
        Falls back to perspective-warping when the crop is not axis-aligned.
        """
        best: Dict[str, Any] = {"english": "", "confidence": 0.0}
        for img in _crop_variants(crop_bgr):
            try:
                raw = self.reader().readtext(img, detail=1, paragraph=False)
            except Exception as e:  # pragma: no cover
                log.debug("readtext failed: %s", e)
                continue
            if not raw:
                continue
            text = sanitize_text(" ".join(str(item[1]) for item in raw))
            conf = float(np.mean([float(item[2]) for item in raw]))
            if conf > best["confidence"] and text:
                best = {"english": text, "confidence": conf}
        return best

    # legacy interface -----------------------------------------------------
    def read(self, image_bgr: np.ndarray, quads: Optional[List[np.ndarray]] = None):
        """Recognize text inside each quad (kept for external callers/tests)."""
        reader = self.reader()
        results = []
        for q in quads or []:
            pts = np.asarray(q, dtype=np.float32).reshape(-1, 2)
            try:
                crop = warp_quad(image_bgr, pts)
            except Exception:
                x0, y0 = pts[:, 0].min(), pts[:, 1].min()
                x1, y1 = pts[:, 0].max(), pts[:, 1].max()
                crop = image_bgr[int(y0):int(y1), int(x0):int(x1)]
            if crop is None or crop.size == 0:
                results.append({"english": "", "confidence": 0.0, "quad": pts})
                continue
            r = self.read_crop(crop)
            r["quad"] = pts
            results.append(r)
        return results


def _crop_variants(crop_bgr: np.ndarray):
    """Yield the crop, and an inverted copy when the text looks light-on-dark
    (white lettering on dark art is unreadable for EasyOCR unless inverted)."""
    yield crop_bgr
    try:
        import cv2
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        hist = np.histogram(gray, bins=8, range=(0, 255))[0]
        dark_bg = hist[-3:].sum() <= hist[:3].sum()  # mostly dark background
        if dark_bg:
            yield cv2.bitwise_not(crop_bgr)
    except Exception:  # pragma: no cover
        return


def _box_to_quad(box, w: int, h: int) -> Optional[np.ndarray]:
    """Normalize EasyOCR box formats to a 4x2 float quad, clipped to the image.

    Horizontal boxes are [x_min, x_max, y_min, y_max]; free boxes are 4 points.
    """
    arr = np.asarray(box, dtype=np.float32).flatten()
    if arr.size == 4:
        x_min, x_max, y_min, y_max = arr
        if x_max <= x_min or y_max <= y_min:
            return None
        pts = np.array([[x_min, y_min], [x_max, y_min],
                        [x_max, y_max], [x_min, y_max]], dtype=np.float32)
    elif arr.size >= 8:
        pts = np.stack([arr[:4], arr[4:8]], axis=1).astype(np.float32)
    else:
        return None
    if (pts[:, 0].max() - pts[:, 0].min()) < 2 or \
       (pts[:, 1].max() - pts[:, 1].min()) < 2:
        return None
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
    return pts


def _tile_ranges(h: int, tile_h: int, overlap: int) -> List[tuple]:
    """[y0, y1) ranges splitting a page of height h into overlapping tiles."""
    if h <= tile_h or tile_h <= 0:
        return [(0, h)]
    ranges = []
    y = 0
    step = tile_h - overlap
    while y < h:
        y1 = min(h, y + tile_h)
        ranges.append((max(0, y1 - tile_h), y1))
        if y1 == h:
            break
        y += step
    # de-duplicate in case the last window equals the previous
    out = []
    for r in ranges:
        if not out or r != out[-1]:
            out.append(r)
    return out


def _dedupe_results(results: List[Dict[str, Any]], iou_thr: float = 0.5
                    ) -> List[Dict[str, Any]]:
    """Drop duplicate detections from overlapping tile seams; keep the
    higher-confidence duplicate."""
    out: List[Dict[str, Any]] = []
    for item in results:
        q = item["quad"]
        x0, y0 = q[:, 0].min(), q[:, 1].min()
        x1, y1 = q[:, 0].max(), q[:, 1].max()
        dup_i = -1
        for i, o in enumerate(out):
            b = o["quad"]
            bx0, by0 = b[:, 0].min(), b[:, 1].min()
            bx1, by1 = b[:, 0].max(), b[:, 1].max()
            iw = max(0.0, min(x1, bx1) - max(x0, bx0))
            ih = max(0.0, min(y1, by1) - max(y0, by0))
            inter = iw * ih
            if inter == 0:
                continue
            union = (x1 - x0) * (y1 - y0) + (bx1 - bx0) * (by1 - by0) - inter
            if inter / max(1e-6, union) > iou_thr:
                dup_i = i
                break
        if dup_i < 0:
            out.append(item)
        elif item["confidence"] > out[dup_i]["confidence"]:
            out[dup_i] = item
    return out


def order_quad_points(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as top-left, top-right, bottom-right, bottom-left.

    ``cv2.boxPoints`` emits corners in a rotating order that depends on the
    rect angle; warping without reordering produced mirrored / 180°-rotated
    crops. This canonicalizes the order by angle around the centroid.
    """
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    if len(pts) != 4:
        return pts
    c = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    pts = pts[np.argsort(ang)]            # CCW starting at left-ish
    # find top-left: smallest x+y
    tl = int(np.argmin(pts.sum(axis=1)))
    ordered = np.roll(pts, -tl, axis=0)
    # ensure clockwise tl,tr,br,bl (screen coords: tr has larger x than bl)
    if ordered[1][0] < ordered[3][0]:
        ordered = ordered[[0, 3, 2, 1]]
    return ordered


def warp_quad(image_bgr: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Perspective-warp a (possibly rotated/skewed) quad into an upright crop."""
    import cv2
    src = order_quad_points(pts)
    w = int(max(np.linalg.norm(src[0] - src[1]), np.linalg.norm(src[3] - src[2])))
    h = int(max(np.linalg.norm(src[1] - src[2]), np.linalg.norm(src[0] - src[3])))
    w, h = max(8, w), max(8, h)
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]],
                   dtype=np.float32)
    M = cv2.getPerspectiveTransform(src.astype(np.float32), dst)
    return cv2.warpPerspective(image_bgr, M, (w, h))


# backwards compatibility
_warp_quad = warp_quad
