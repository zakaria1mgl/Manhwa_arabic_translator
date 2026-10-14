from __future__ import annotations

import logging
import os
import urllib.request
from typing import Dict, List, Optional, Tuple

from PIL import ImageFont

log = logging.getLogger(__name__)

# Google Fonts sources. IMPORTANT: PIL does not perform OpenType shaping;
# we shape via arabic_reshaper into presentation-form codepoints (U+FExx) and
# draw those directly. The font therefore MUST contain those glyphs.
# Cairo (variable) covers only ~89/144 presentation forms -> renders tofu.
# Noto Naskh Arabic / Amiri cover 140+/144 -> use them as defaults.
FONT_SOURCES: Dict[str, str] = {
    "Noto Naskh Arabic": "https://github.com/google/fonts/raw/main/ofl/notonaskharabic/NotoNaskhArabic%5Bwght%5D.ttf",
    "Amiri": "https://github.com/google/fonts/raw/main/ofl/amiri/Amiri-Bold.ttf",
    "Noto Sans Arabic": "https://github.com/google/fonts/raw/main/ofl/notosansarabic/NotoSansArabic%5Bwdth%2Cwght%5D.ttf",
    "Cairo": "https://github.com/google/fonts/raw/main/ofl/cairo/Cairo%5Bslnt%2Cwght%5D.ttf",
}

# Fallback raw static-bold URLs (jsDelivr fontsource packages, reliable and
# pinned to real static weights).
FONT_BOLD_FALLBACK: Dict[str, str] = {
    "Cairo": "https://cdn.jsdelivr.net/fontsource/fonts/cairo@latest/arabic-700-normal.woff2",
    "Noto Sans Arabic": "https://cdn.jsdelivr.net/fontsource/fonts/noto-sans-arabic@latest/arabic-700-normal.woff2",
}

_font_cache: Dict[Tuple[str, int], ImageFont.FreeTypeFont] = {}


def ensure_fonts(download_dir: str, names: Optional[List[str]] = None) -> Dict[str, str]:
    """Download any missing fonts; returns {name: path}."""
    os.makedirs(download_dir, exist_ok=True)
    wanted = names or list(FONT_SOURCES)
    paths = {}
    for name in wanted:
        url = FONT_SOURCES.get(name)
        if not url:
            continue
        path = os.path.join(download_dir, _safe_name(name) + ".ttf")
        if not os.path.isfile(path):
            try:
                log.info("downloading font %s", name)
                urllib.request.urlretrieve(url, path)
            except Exception as e:
                log.warning("font download failed for %s: %s", name, e)
                continue
        paths[name] = path
    return paths


def _safe_name(name: str) -> str:
    return name.replace(" ", "_")


def _set_variation(font: ImageFont.FreeTypeFont, weight: int = 700) -> ImageFont.FreeTypeFont:
    """Force a bold instance on variable fonts (PIL set_variation_by_axes)."""
    try:
        axes = font.get_variation_axes()
        names = [a["name"] if isinstance(a, dict) else a.name for a in axes]
        vals = []
        for nm in names:
            key = (nm.decode() if isinstance(nm, bytes) else str(nm)).lower()
            if "weight" in key or key == "wght":
                vals.append(weight)
            elif "slant" in key or "slnt" in key:
                vals.append(0)
            elif "width" in key or "wdth" in key:
                vals.append(100)
            else:
                vals.append(None)
        # fill any remaining unknown axes with their default
        defaults = [a["default"] if isinstance(a, dict) else a.default for a in axes]
        vals = [defaults[i] if vals[i] is None else vals[i] for i in range(len(axes))]
        font.set_variation_by_axes(vals)
    except Exception:
        pass  # static font: nothing to do
    return font


def load_font(name: str, size: int, download_dir: str = "fonts",
              bold: bool = True) -> ImageFont.FreeTypeFont:
    key = (name, size)
    if key in _font_cache:
        return _font_cache[key]
    paths = ensure_fonts(download_dir, [name])
    path = paths.get(name)
    if path is None:
        for cand in ("Noto Naskh Arabic", "Amiri", "Noto Sans Arabic"):
            p = os.path.join(download_dir, _safe_name(cand) + ".ttf")
            if os.path.isfile(p):
                path = p
                break
    if path is None:
        raise FileNotFoundError(f"no Arabic font available for {name!r}")
    font = ImageFont.truetype(path, size)
    if bold:
        font = _set_variation(font, 700)
    _font_cache[key] = font
    return font


def has_glyphs(font: ImageFont.FreeTypeFont, text: str) -> bool:
    """True when the font can render every character of the text.

    Rendering uses arabic_reshaper presentation forms (U+FExx), so the check
    must run against the *shaped* string too — a font covering base Arabic
    codepoints but missing presentation forms renders tofu (the old Cairo
    bug).
    """
    try:
        from fontTools.ttLib import TTFont
        tt = TTFont(font.path, fontNumber=0, lazy=True)
        cm = tt.getBestCmap()
        try:
            from .shaping import shape_arabic
            texts = (text, shape_arabic(text))
        except Exception:
            texts = (text,)
        for t in texts:
            for ch in t:
                if ch.isspace():
                    continue
                if ord(ch) not in cm:
                    return False
        return True
    except Exception:
        return True


def font_coverage(font_path: str) -> float:
    """Fraction of Arabic presentation forms (U+FE70..U+FEFF) covered."""
    try:
        from fontTools.ttLib import TTFont
        tt = TTFont(font_path, fontNumber=0, lazy=True)
        cm = tt.getBestCmap()
        covered = sum(1 for cp in range(0xFE70, 0xFF00) if cp in cm)
        return covered / 144.0
    except Exception:
        return 1.0
