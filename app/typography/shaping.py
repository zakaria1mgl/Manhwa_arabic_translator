from __future__ import annotations

import logging
import re
from typing import List, Optional, Tuple

log = logging.getLogger(__name__)

_reshaper = None
_bidi = None

try:  # optional at import time; required at render time
    import arabic_reshaper as _ar  # type: ignore
    from bidi.algorithm import get_display as _get_display  # type: ignore
    _reshaper = _ar.ArabicReshaper(
        configuration={
            "delete_harakat": False,
            "support_ligatures": True,
        }
    )
    _bidi = _get_display
except Exception:  # pragma: no cover
    log.warning("arabic_reshaper/python-bidi not installed; shaping may be wrong")


def shape_arabic(text: str) -> str:
    """Return display-ready text: reshaped + bidi-reordered."""
    if not text:
        return text
    if _reshaper is None or _bidi is None:
        return text
    return _bidi(_reshaper.reshape(text))


def has_raqm() -> bool:
    """True when PIL was built with libraqm (native complex text)."""
    from PIL import features
    return bool(features.check("raqm"))


def normalize_arabic(text: str) -> str:
    """Unicode normalization tuned for Arabic typesetting."""
    import unicodedata
    text = unicodedata.normalize("NFC", text)
    # Localize punctuation only when adjacent to Arabic letters — a blanket
    # replace corrupted numbers ("1,000") and glossary-preserved Latin names.
    ar = r"\u0600-\u06FF"
    text = re.sub(rf"(?<=[{ar}])\?|\?(?=[{ar}])", "؟", text)
    text = re.sub(rf"(?<=[{ar}]);|;(?=[{ar}])", "؛", text)
    text = re.sub(rf"(?<=[{ar}]),|,(?=[{ar}])", "،", text)
    # collapse whitespace
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def wrap_arabic(text: str, font, max_width: int,
                width_cache: Optional[dict] = None) -> List[str]:
    """Greedy word wrap that never splits a word. Returns logical-order lines
    (shaping happens at draw time). `width_cache` optionally memoizes shaped
    word widths across calls (the font-size binary search re-wraps a lot)."""
    words = text.split()
    if not words:
        return [""]

    def wwidth(s: str) -> int:
        if width_cache is not None and s in width_cache:
            return width_cache[s]
        v = _text_width(s, font)
        if width_cache is not None:
            width_cache[s] = v
        return v

    space_w = _text_width(" ", font)
    lines: List[str] = []
    current = words[0]
    cur_w = wwidth(current)
    for w in words[1:]:
        w_w = wwidth(w)
        if cur_w + space_w + w_w <= max_width:
            current = current + " " + w
            cur_w += space_w + w_w
        else:
            lines.append(current)
            current = w
            cur_w = w_w
    lines.append(current)
    return lines


def _text_width(text: str, font) -> int:
    shaped = shape_arabic(text)
    try:
        return font.getlength(shaped)
    except AttributeError:  # pragma: no cover
        bbox = font.getbbox(shaped)
        return bbox[2] - bbox[0]
