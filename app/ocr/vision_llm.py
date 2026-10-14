from __future__ import annotations

import base64
import io
import json
import logging
from typing import Any, Dict, List, Optional

import numpy as np

from ..config import AppConfig
from ..translation.llm_client import extract_json
from ..utils.types import Quad, RegionType, TextRegion

log = logging.getLogger(__name__)

_MAX_SIDE = 1568  # vision-token friendly; bboxes are scaled back to full res

_PROMPT = """You are reading a manhwa/webtoon page image ({W}x{H} pixels).
Find EVERY piece of English text on the page that should be translated:
speech bubbles, thought bubbles, narration boxes, captions. Also include
sound effects (SFX) ONLY when they are plain readable words (e.g. "BOOM").

Do NOT include: lettering fused into logos or artwork, tiny background signs,
watermarks, page numbers, or any text you cannot read with certainty.

Return ONLY valid JSON:
{"regions": [
  {"bbox": [x0, y0, x1, y1],          # integer pixels, top-left/bottom-right
   "text": "exact English text",       # join multi-line bubbles with spaces
   "type": "dialogue|thought|narration|caption|sfx|sign",
   "confidence": 0.0,                  # your certainty this region+text is right
   "speaker": "character name or empty string",
   "gender": "male|female|unknown"}
]}

Rules:
- The image is exactly {W} wide and {H} tall. bbox coordinates MUST be
  absolute pixel positions within THAT image (top-left origin). Never use
  normalized 0-1000 or 0-999 values, even if you were trained that way.
- One entry per bubble / text block; bbox must tightly enclose the WHOLE
  text block (all lines of one bubble) — not the whole page, not half of it.
- Read the text EXACTLY as written, including punctuation and casing.
- SFX = stylized onomatopoeia drawn into the art (BOOM, WHOOSH...).
- speaker = the character speaking this bubble, if identifiable from the page,
  else "". gender = the SPEAKER's gender.
- confidence >= 0.85 only for text you can transcribe verbatim.
- If there is no text, return {"regions": []}.
"""


class _VisionClient:
    """Tiny OpenAI-compatible vision chat helper (image_url + JSON reply)."""

    def __init__(self, cfg: AppConfig):
        self._app_cfg = cfg

    @property
    def ready(self) -> bool:
        llm = self._app_cfg.llm
        return bool(llm.api_key and llm.base_url and llm.model)

    def analyze(self, image_bgr: np.ndarray) -> Dict[str, Any]:
        llm = self._app_cfg.llm
        import requests
        from PIL import Image

        h, w = image_bgr.shape[:2]
        scale = min(1.0, _MAX_SIDE / max(h, w))
        if scale < 1.0:
            import cv2
            small = cv2.resize(image_bgr, (int(w * scale), int(h * scale)),
                               interpolation=cv2.INTER_AREA)
        else:
            small = image_bgr
        buf = io.BytesIO()
        Image.fromarray(small[..., ::-1]).save(buf, format="JPEG", quality=88)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        body = {
            "model": llm.model,
            "temperature": 0.1,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text",
                     "text": _PROMPT.format(W=small.shape[1],
                                            H=small.shape[0])},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ],
            }],
        }
        url = llm.base_url.rstrip("/") + "/chat/completions"
        resp = requests.post(url, json=body, timeout=llm.timeout, headers={
            "Authorization": f"Bearer {llm.api_key}",
            "Content-Type": "application/json",
        })
        resp.raise_for_status()
        payload = resp.json()
        choice = payload["choices"][0]["message"]
        reply = choice.get("content")
        # Some OpenAI-compatible gateways (Gemini via OpenRouter etc.) put the
        # answer in `reasoning` or split it across content parts when the
        # model hits max_tokens with no finish_reason handling.
        if not reply:
            reply = choice.get("reasoning") or ""
        if isinstance(reply, list):  # multimodal-style content parts
            reply = "".join(p.get("text", "") for p in reply
                            if isinstance(p, dict))
        data = extract_json(reply)
        if not isinstance(data, dict):
            raise ValueError(f"vision reply is not a JSON object "
                             f"(finish_reason={payload['choices'][0].get('finish_reason')}, "
                             f"reply head={reply[:120]!r})")
        data["_scale"] = scale
        return data


class VisionOCR:
    """Vision-LLM page reader: one multimodal call replaces the classical
    detect -> recognize -> speaker-guess chain, fixing OCR accuracy, speaker
    attribution and SFX classification in a single shot.

    Falls back cleanly: callers should catch exceptions and use the classical
    engine when no LLM is configured or the call fails.
    """

    name = "vision"

    def __init__(self, cfg: AppConfig):
        self.client = _VisionClient(cfg)

    @property
    def ready(self) -> bool:
        return self.client.ready

    def detect_and_read(self, image_bgr: np.ndarray) -> List[TextRegion]:
        data = self.client.analyze(image_bgr)
        scale = float(data.get("_scale", 1.0))
        inv = 1.0 / max(1e-6, scale)
        h, w = image_bgr.shape[:2]
        sh, sw = int(round(w * scale)), int(round(h * scale))
        raw_items = list(data.get("regions") or data.get("items")
                         or data.get("texts") or [])
        # user-tunable floor (ocr.min_confidence); models that omit the
        # confidence field keep the historical default of 0.9.
        min_conf = float(self._app_cfg.get("ocr", "min_confidence",
                                           default=0.35))

        # --- coordinate-space guard -------------------------------------
        # Several vision models (Qwen-VL and friends) return bboxes in a
        # normalized 0..1000 space even when the prompt asks for pixels.
        # Interpreting those as raw pixels makes every box explode to the
        # page corners, which destroys detection, masking and rendering.
        # Detect the coordinate space from the batch of boxes and rescale.
        sx, sy = _infer_coord_scales(raw_items, sw, sh)

        regions: List[TextRegion] = []
        rejected = 0
        for item in raw_items:
            try:
                x0, y0, x1, y1 = [float(v) for v in item["bbox"][:4]]
            except Exception:
                rejected += 1
                continue
            # normalized -> sent-image pixels (per axis when the model used
            # a non-square grid, e.g. Qwen-VL on portrait webtoon pages)
            x0, x1 = x0 * sx, x1 * sx
            y0, y1 = y0 * sy, y1 * sy
            # back from the downscaled "sent" image to full resolution
            x0, x1 = x0 * inv, x1 * inv
            y0, y1 = y0 * inv, y1 * inv
            x0, x1 = sorted((max(0.0, x0), min(float(w), x1)))
            y0, y1 = sorted((max(0.0, y0), min(float(h), y1)))
            if x1 - x0 < 6 or y1 - y0 < 6:
                rejected += 1
                continue
            # LLM bboxes are eyeballed; reject degenerate/page-spanning boxes
            # exactly like the classical path does, otherwise one "crazy
            # diagonal" region masks half the page during removal.
            quad = Quad(points=[(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
            from .service import OCRService  # local import: no cycle at runtime
            if not OCRService._sane_quad(quad, image_bgr.shape, 0.20):
                log.debug("vision: rejected implausible bbox %s",
                          (x0, y0, x1, y1))
                rejected += 1
                continue
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            try:
                conf = float(item.get("confidence", 0.9))
            except (TypeError, ValueError):
                conf = 0.9
            conf = min(1.0, max(0.0, conf))
            if conf < min_conf:
                log.debug("vision: dropped low-confidence region %r (%.2f)",
                          text[:40], conf)
                rejected += 1
                continue
            rtype = _parse_type(item.get("type"))
            gender = str(item.get("gender") or "").strip().lower() or None
            if gender not in ("male", "female", "unknown"):
                gender = None
            region = TextRegion(
                id=0,
                quad=quad,
                english=text,
                ocr_conf=conf,
                region_type=rtype,
                speaker=str(item.get("speaker") or "") or None,
                speaker_gender=gender,
                orientation=_reported_orientation(item, quad),
            )
            region.qa_flags.append("vision_ocr")
            if conf < 0.5:
                region.qa_flags.append("low_ocr_confidence")
            regions.append(region)
        # duplicate suppression: models sometimes emit the same bubble twice
        # (or nested boxes). Keep the higher-confidence copy so the translator
        # does not produce two competing Arabic strings for one bubble.
        kept: List[TextRegion] = []
        for r in sorted(regions, key=lambda x: -x.ocr_conf):
            dup_of = None
            for k in kept:
                if _iou(r.quad.bbox, k.quad.bbox) >= 0.85:
                    dup_of = k
                    break
            if dup_of is not None:
                log.debug("vision: dropped duplicate %r (kept %r)",
                          r.english[:30], dup_of.english[:30])
                continue
            kept.append(r)
        return kept


def _infer_coord_scales(raw_items: List[dict], sent_w: int,
                        sent_h: int) -> tuple:
    """Return ``(scale_x, scale_y)`` mapping the model's bbox numbers onto the
    image we actually sent (``sent_w`` x ``sent_h`` pixels).

    Most vision models return absolute pixel coordinates of the image they
    were given. Qwen-VL-style models instead return values normalized to a
    0..1000 grid regardless of the real resolution — and some return 0..1
    floats. If we treat those as pixels, every box explodes toward the page
    corners and the whole pipeline (masking, inpainting, rendering) is
    destroyed.

    Heuristic, evaluated over the whole batch so genuine pixel coords are
    never shrunk:
      - global max coord <= ~1.5 and min >= 0 -> 0..1 normalized floats
      - boxes essentially all inside the sent image
        as-is                                  -> pixels, (1.0, 1.0)
      - otherwise                              -> some 0..1000 grid; we pick
        the variant (per-axis free grid vs aspect-preserving square grid)
        that puts more boxes inside the image.

    For the 0..1000 case we first check the FREE-GRID hypothesis (x mapped by
    sent_w/1000, y by sent_h/1000). If that still leaves most boxes outside
    the image, the model used an ASPECT-PRESERVING square grid (the classic
    Qwen-VL behaviour on portrait pages: both axes scaled by min(w,h)/1000)
    and we switch to that. Applying the wrong variant shifts every bubble by
    hundreds of pixels, so the fit is verified against actual containment
    rather than guessed from a single signature.
    """
    boxes: List[tuple] = []
    for item in raw_items:
        try:
            x0, y0, x1, y1 = [float(v) for v in item["bbox"][:4]]
        except Exception:
            continue
        boxes.append((min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)))

    if not boxes:
        return 1.0, 1.0

    vmax = max(max(b[2], b[3]) for b in boxes)
    vmin = min(min(b[0], b[1]) for b in boxes)

    # 0..1 normalized floats: on a real page nothing reaches even ~1.5 px;
    # require the sent image to be clearly larger to avoid false positives.
    if vmax <= 1.5 and vmin >= 0.0 and max(sent_w, sent_h) > 8:
        log.warning("vision: bboxes look normalized to 0..1; rescaling to "
                    "%dx%d", sent_w, sent_h)
        return float(sent_w), float(sent_h)

    def contained(sx: float, sy: float) -> int:
        c = 0
        for x0, y0, x1, y1 in boxes:
            cx0, cy0, cx1, cy1 = x0 * sx, y0 * sy, x1 * sx, y1 * sy
            if -2 <= cx0 and cx1 <= sent_w + 2 and -2 <= cy0 and cy1 <= sent_h + 2:
                c += 1
        return c

    # Candidate coordinate spaces. Pixel space must contain (essentially) all
    # boxes — with a small tolerance for eyeballed LLM overhang — otherwise
    # the numbers are not pixels of the sent image at all.
    tol = 0.02
    n_min = max(1, int(len(boxes) * (1.0 - tol)))
    n_pixel = contained(1.0, 1.0)
    pixel_ok = n_pixel >= n_min

    # A normalized grid is only plausible if some coordinate actually reaches
    # into the top of the 0..1000 range; well-behaved pixel coords on huge
    # pages must never be mistaken for a grid.
    looks_gridish = vmax >= 600 or vmax > max(sent_w, sent_h)

    free_s = (sent_w / 1000.0, sent_h / 1000.0)
    sq = min(sent_w, sent_h) / 1000.0
    square_s = (sq, sq)
    n_free, n_square = contained(*free_s), contained(*square_s)

    # Tie-breaker between "messy pixels" and a grid hypothesis that also
    # contains everything: genuine pixel batches hug the image edges (bubbles
    # spread across the whole page, so max coords sit close to W and H),
    # whereas rescaled 0..1000 grid coords cluster in the middle of the range.
    def pixel_extent() -> float:
        fx = max((b[2] for b in boxes), default=0) / float(sent_w)
        fy = max((b[3] for b in boxes), default=0) / float(sent_h)
        # average axis utilisation; > ~0.85 means coords really reach the
        # far edges of THIS image and are therefore almost certainly pixels.
        return (fx + fy) / 2.0

    all_ints = all(float(v).is_integer() for b in boxes for v in b)
    edge_hugging = pixel_extent() > 0.85

    if pixel_ok:
        # Pixel space contains the batch. A grid hypothesis is strictly more
        # likely when it explains every box while raw pixels leave at least
        # one box outside the image (eyeballed grids overshoot constantly;
        # genuine LLM pixel coords rarely do across a whole page) — unless
        # the values clearly track this image's own dimensions.
        ambiguous = (looks_gridish and max(n_free, n_square) == len(boxes)
                     and n_pixel < len(boxes))
        if ambiguous and not edge_hugging and all_ints:
            pass  # fall through to the grid branch below
        else:
            return 1.0, 1.0

    # Normalized 0..1000 grid: pick whichever variant actually fits the image.
    best_n = max(n_free, n_square)
    grid_ok = looks_gridish and best_n >= n_min
    if grid_ok:
        if n_square > n_free:
            log.warning("vision: bboxes fit an aspect-preserving 0..1000 "
                        "square grid (%d/%d inside %dx%d); rescaling by %.4f",
                        n_square, len(boxes), sent_w, sent_h, sq)
            return square_s
        log.warning("vision: bboxes fit a per-axis 0..1000 grid (%d/%d inside "
                    "%dx%d); rescaling", n_free, len(boxes), sent_w, sent_h)
        return free_s
    if pixel_ok:
        return 1.0, 1.0
    if n_free == 0 and n_square == 0:
        # Nothing fits any hypothesis; last resort is the per-axis grid since
        # coordinates get clamped to the page afterwards anyway.
        log.warning("vision: bbox values don't fit the %dx%d sent image; "
                    "falling back to per-axis 0..1000 rescale", sent_w, sent_h)
        return free_s
    log.debug("vision: bboxes partially outside the %dx%d sent image but no "
              "normalized grid explains them fully; keeping pixel space",
              sent_w, sent_h)
    return 1.0, 1.0


def _iou(b1, b2) -> float:
    """Intersection-over-union of two (x0, y0, x1, y1) bbox tuples."""
    ix = max(0.0, min(b1[2], b2[2]) - max(b1[0], b2[0]))
    iy = max(0.0, min(b1[3], b2[3]) - max(b1[1], b2[1]))
    inter = ix * iy
    a1 = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
    a2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def _reported_orientation(item: dict, quad: Quad) -> float:
    """Honour an explicit `rotated`/`orientation` hint when the model gives
    one; otherwise infer vertical lettering from geometry (tall + narrow =
    vertical SFX/sign column) so downstream mask/crop logic rotates correctly.
    """
    raw = item.get("orientation")
    if isinstance(raw, (int, float)) and abs(float(raw)) > 5:
        return float(raw)
    if item.get("rotated") and quad.height > 1.8 * max(1.0, quad.width):
        return 90.0
    if quad.height > 2.2 * max(1.0, quad.width):
        # purely geometric guess for vertical text blocks
        return 90.0
    return 0.0


def _parse_type(value: Any) -> RegionType:
    try:
        return RegionType(str(value).strip().lower())
    except Exception:
        return RegionType.UNKNOWN
