from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Tuple

from ..utils.types import RegionType, TextRegion

log = logging.getLogger(__name__)

# Base SFX suggestions; the LLM adapts them to scene context. Keys are matched
# on the *stem* of the English SFX (vowels/length ignored).
SFX_DICTIONARY: Dict[str, str] = {
    "bam": "بـــااام!",
    "boom": "بووووم!",
    "bang": "بــانغ!",
    "pow": "بـــاو!",
    "thud": "طَـخ!",
    "thump": "طُـــم!",
    "whoosh": "فــــش!",
    "swish": "ســـويش!",
    "wham": "وآآم!",
    "crash": "تحطّـــم!",
    "smash": "تكسّـــر!",
    "crack": "كرَـــك!",
    "snap": "طقّـــة!",
    "clang": "صليـــنغ!",
    "clash": "كلاشـــ!",
    "click": "كليـــك!",
    "creak": "صريـــر!",
    "squeak": "صريـــر!",
    "splash": "سبلاشـ!",
    "drip": "قطـــر!",
    "gulp": "بلـــع!",
    "gasp": "لهـــث!",
    "sigh": "تنهيـــدة!",
    "roar": "زئيـــر!",
    "growl": "هديـــر!",
    "scream": "صرخـــة!",
    "shriek": "صرخـــة!",
    "grunt": "تذمّـــر!",
    "sob": "شهيـــق!",
    "knock": "طرقعـــة!",
    "punch": "ضربـــة!",
    "slash": "سلاشـــ!",
    "stab": "طعنـــة!",
    "slice": "شقّـــ!",
    "step": "خطـــوة!",
    "stomp": "دوســـة!",
    "rumble": "هديـــر!",
    "explosion": "انفجـــار!",
    "boom2": "دومـــ!",
    "heart": "دقّـــات!",
    "heartbeat": "دوم.. دوم..",
    "silence": "...",
    "wind": "هديـــر!",
    "footsteps": "تك.. تك..",
    "door": "صرصـــرة!",
    "phone": "رنينـــ!",
    "flash": "فلاشـــ!",
}


def sfx_stem(word: str) -> str:
    w = re.sub(r"[^a-z]", "", word.lower())
    if len(w) >= 3:
        w = re.sub(r"(.)\1{1,}", r"\1", w)  # collapse repeats
    # strip common suffixes
    for suf in ("ing", "ed", "s"):
        if len(w) > 4 and w.endswith(suf):
            w = w[: -len(suf)]
            break
    return w


def lookup_sfx(text: str) -> Optional[str]:
    """Return a dictionary suggestion for an SFX string, if any."""
    words = re.findall(r"[A-Za-z]+", text)
    if not words:
        return None
    stem = sfx_stem(words[0])
    if stem in SFX_DICTIONARY:
        return SFX_DICTIONARY[stem]
    # try full phrase
    key = re.sub(r"[^a-z]", "", text.lower())
    return SFX_DICTIONARY.get(key)
