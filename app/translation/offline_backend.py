from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from ..config import AppConfig
from ..models.manager import cached
from ..utils.types import TextRegion

log = logging.getLogger(__name__)


class OfflineTranslator:
    """Helsinki-NLP MarianMT en->ar fallback (no API key needed)."""

    def __init__(self, model_name: str = "Helsinki-NLP/opus-mt-en-ar"):
        self.model_name = model_name

    def _pipeline(self):
        from transformers import MarianMTModel, MarianTokenizer  # type: ignore

        def build():
            device = _device()
            tok = MarianTokenizer.from_pretrained(self.model_name)
            model = MarianMTModel.from_pretrained(self.model_name)
            model.to(device)
            model.eval()
            return (tok, model)

        return cached("marian_en_ar", build)

    def translate(self, texts: List[str]) -> List[str]:
        if not texts:
            return []
        tok, model = self._pipeline()
        import torch
        device = next(model.parameters()).device
        out: List[str] = []
        batch = 16
        for i in range(0, len(texts), batch):
            chunk = [t if t.strip() else "." for t in texts[i:i + batch]]
            enc = tok(chunk, return_tensors="pt", padding=True,
                      truncation=True, max_length=256).to(device)
            with torch.no_grad():
                gen = model.generate(**enc, num_beams=4, max_length=256)
            out.extend(tok.batch_decode(gen, skip_special_tokens=True))
        return out


def _device() -> str:
    from ..models.manager import detect_device
    return detect_device()
