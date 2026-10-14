from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

import yaml

DEFAULT_CONFIG: Dict[str, Any] = {
    "device": {
        "prefer_gpu": True,           # auto-fallback to CPU
        "low_vram_mode": False,
    },
    "detection": {
        "backend": "easyocr",          # easyocr | external (comic-text-detector adapter)
        "min_text_height": 10,         # px, ignore smaller
        "link_threshold": 0.4,
        "text_threshold": 0.6,
        "max_quad_area_ratio": 0.20,   # drop 'crazy diagonal' cross-panel detections
        # EasyOCR box-merging knobs (previously hardcoded)
        "width_ths": 0.7,
        "height_ths": 0.5,
        "ycenter_ths": 0.5,
        "add_margin": 0.05,            # was 0.15: large margins bleed into neighbors
        "panel_segmentation": True,    # gutter-based panel-aware reading order
        "tile_height_px": 2200,        # detection tiling for tall webtoon strips
        "comic_weights_dir": "vendor/comic-text-detector/data",  # opt. backend
    },
    "ocr": {
        "backend": "auto",            # auto | easyocr | vision (auto: vision when an LLM key is set)
        "min_confidence": 0.35,
        "upscale_small_text": True,
        "upscale_target_height": 48,   # px per text line before recognition
        "preprocess": True,            # contrast/sharpen for low-contrast crops
    },
    "translation": {
        "backend": "auto",             # auto | llm | offline
        "llm": {
            "base_url": "",            # OpenAI-compatible endpoint, e.g. Gemini
            "api_key": "",
            "model": "",
            "temperature": 0.3,
            "max_retries": 3,
            "timeout": 120,
        },
        "offline_model": "Helsinki-NLP/opus-mt-en-ar",
        "review_pass": True,           # critical review pass
        "fit_pass": True,              # length-constrained rewrite pass
        "sfx_dictionary": True,
    },
    "glossary": {
        "path": "glossary.json",
        "auto_learn": True,            # persist new terms chosen by the LLM
    },
    "inpainting": {
        "backend": "auto",             # auto | lama | opencv
        "mask_dilate_px": 4,
        "color_fill_uniform_bubbles": True,   # fast path for plain bubble interiors
        "uniform_tolerance": 18,
    },
    "typography": {
        "font_download_dir": "fonts",
        "fonts": {
            "dialogue": "Noto Naskh Arabic",
            "narration": "Noto Naskh Arabic",
            "sfx": "Noto Sans Arabic",
            "classic": "Amiri",
        },
        "auto_style_match": True,      # serif heuristic -> Amiri/Naskh
        "min_font_size": 10,
        "max_font_size": 96,
        "padding_ratio": 0.06,         # fraction of region height kept as padding
        "line_spacing": 1.25,
        "text_color_dialogue": [26, 26, 26],
        "stroke_ratio": 0.045,         # outline width / font size
    },
    "quality_control": {
        "min_page_score": 0.75,
        "auto_retry": True,
        "max_retries": 2,
        "residual_text_check": True,   # re-OCR rendered page (GPU recommended)
    },
    "batch": {
        "panel_order": "ltr",          # manhwa is LTR; set rtl for manga scans
        "row_overlap_ratio": 0.5,
        "supported_exts": [".png", ".jpg", ".jpeg", ".webp"],
    },
    "output": {
        "dir": "output",
        "save_debug_json": True,
        "save_debug_overlay": True,
        "save_cleaned": True,      # inpainted base for the editor / retries
    },
}


@dataclass
class LLMConfig:
    base_url: str = ""
    api_key: str = ""
    model: str = ""
    temperature: float = 0.3
    max_retries: int = 3
    timeout: int = 120


@dataclass
class AppConfig:
    raw: Dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_CONFIG))

    # convenience accessors -------------------------------------------------
    def get(self, *keys: str, default: Any = None) -> Any:
        node: Any = self.raw
        for k in keys:
            if not isinstance(node, dict) or k not in node:
                return default
            node = node[k]
        return node

    def set(self, value: Any, *keys: str) -> None:
        node = self.raw
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = value

    @property
    def llm(self) -> LLMConfig:
        c = self.get("translation", "llm", default={}) or {}
        return LLMConfig(
            base_url=os.environ.get("MANHWA_LLM_BASE_URL", c.get("base_url", "")),
            api_key=os.environ.get("MANHWA_LLM_API_KEY", c.get("api_key", "")),
            model=os.environ.get("MANHWA_LLM_MODEL", c.get("model", "")),
            temperature=float(c.get("temperature", 0.3)),
            max_retries=int(c.get("max_retries", 3)),
            timeout=int(c.get("timeout", 120)),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    # loading ----------------------------------------------------------------
    @classmethod
    def load(cls, path: Optional[str] = None) -> "AppConfig":
        raw = _deep_copy(DEFAULT_CONFIG)
        if path and os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                user_cfg = yaml.safe_load(f) or {}
            _deep_merge(raw, user_cfg)
        cfg = cls(raw=raw)
        cfg._apply_env()
        return cfg

    def _apply_env(self) -> None:
        if os.environ.get("MANHWA_LOW_VRAM", "").lower() in ("1", "true", "yes"):
            self.set(True, "device", "low_vram_mode")
        if os.environ.get("MANHWA_PANEL_ORDER"):
            self.set(os.environ["MANHWA_PANEL_ORDER"], "batch", "panel_order")
        if os.environ.get("MANHWA_INPAINT_BACKEND"):
            self.set(os.environ["MANHWA_INPAINT_BACKEND"], "inpainting", "backend")
        if os.environ.get("MANHWA_OCR_BACKEND"):
            self.set(os.environ["MANHWA_OCR_BACKEND"], "ocr", "backend")
        # Convenience: forcing the translation LLM without also setting
        # MANHWA_OCR_BACKEND implies the user wants vision OCR (that is the
        # only place the LLM affects detection/reading). Without this, an
        # env-var-only setup silently keeps the old EasyOCR path and none of
        # the vision-LLM fixes ever run. "auto" behaves identically when a
        # key is present, so only upgrade the default "auto" value.
        elif os.environ.get("MANHWA_LLM_API_KEY") or os.environ.get(
                "MANHWA_LLM_MODEL"):
            if self.get("ocr", "backend", default="auto") == "auto":
                self.set("vision", "ocr", "backend")

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.raw, f, allow_unicode=True, sort_keys=False)


def _deep_copy(d: Dict[str, Any]) -> Dict[str, Any]:
    return {k: _deep_copy(v) if isinstance(v, dict) else (list(v) if isinstance(v, list) else v)
            for k, v in d.items()}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> None:
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)
