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

_PROMPT = """You are reading a manhwa/webtoon page image.
Find EVERY piece of English text on the page: speech bubbles, narration
boxes, captions, signs, sound effects (SFX), floating text.

Return ONLY valid JSON:
{"regions": [
  {"bbox": [x0, y0, x1, y1],          # integer pixels, top-left/bottom-right
   "text": "exact English text",       # join multi-line bubbles with spaces
   "type": "dialogue|thought|narration|caption|sfx|sign",
   "speaker": "character name or empty string",
   "gender": "male|female|unknown"}
]}

Rules:
- bbox must tightly enclose the WHOLE text block (all lines of one bubble).
- Read the text EXACTLY as written, including punctuation and casing.
- SFX = stylized onomatopoeia drawn into the art (BOOM, WHOOSH...).
- speaker = the character speaking this bubble, if identifiable from the page,
  else "". gender = the SPEAKER's gender.
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
                    {"type": "text", "text": _PROMPT},
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
        reply = resp.json()["choices"][0]["message"]["content"]
        data = extract_json(reply)
        if not isinstance(data, dict):
            raise ValueError("vision reply is not a JSON object")
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
        regions: List[TextRegion] = []
        for item in data.get("regions", []):
            try:
                x0, y0, x1, y1 = [float(v) * inv for v in item["bbox"][:4]]
            except Exception:
                continue
            x0, x1 = sorted((max(0.0, x0), min(float(w), x1)))
            y0, y1 = sorted((max(0.0, y0), min(float(h), y1)))
            if x1 - x0 < 6 or y1 - y0 < 6:
                continue
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            rtype = _parse_type(item.get("type"))
            region = TextRegion(
                id=0,
                quad=Quad(points=[(x0, y0), (x1, y0), (x1, y1), (x0, y1)]),
                english=text,
                ocr_conf=0.9,  # vision read; no per-word confidence exists
                region_type=rtype,
                speaker=str(item.get("speaker") or "") or None,
                speaker_gender=str(item.get("gender") or "") or None,
            )
            region.qa_flags.append("vision_ocr")
            regions.append(region)
        return regions


def _parse_type(value: Any) -> RegionType:
    try:
        return RegionType(str(value).strip().lower())
    except Exception:
        return RegionType.UNKNOWN
