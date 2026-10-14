from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional

import numpy as np

from ..config import AppConfig
from ..utils.types import PageResult, RegionType, TextRegion

log = logging.getLogger(__name__)

_ARABIC_RE = re.compile(r"[\u0600-\u06FF]")
_LATIN_RE = re.compile(r"[A-Za-z]{2,}")


def region_qa(region: TextRegion, cfg: AppConfig) -> float:
    """Score one region 0..1 and append flags."""
    # Merged satellites (multi-line bubbles folded into one translation unit
    # by merge_bubble_lines) intentionally carry empty english/arabic — they
    # are NOT failures. Scoring them normally produced guaranteed false
    # `empty_ocr` + `missing_translation` flags on every multi-line bubble,
    # which dragged the page score down and triggered pointless LLM retries.
    # Carry over any layout flags that were recorded before the merge, then
    # mark the region as passed; the primary carries the real QA.
    if region.merged_into is not None:
        carried = [f for f in region.qa_flags
                   if f in ("text_overflow_risk", "layout_failed")]
        region.qa_flags = list(dict.fromkeys(carried))
        region.qa_score = 1.0
        return region.qa_score

    flags: List[str] = []
    score = 1.0

    if region.ocr_conf < 0.5:
        flags.append("low_ocr_confidence")
        score -= 0.25
    if not region.english.strip():
        flags.append("empty_ocr")
        score -= 0.3
    if not region.arabic.strip():
        flags.append("missing_translation")
        score -= 0.5
    else:
        if not _ARABIC_RE.search(region.arabic):
            flags.append("no_arabic_glyphs")
            score -= 0.4
        if region.region_type != RegionType.SFX and _LATIN_RE.search(region.arabic):
            # latin words surviving inside an Arabic dialogue line
            flags.append("latin_left_in_arabic")
            score -= 0.15
        if region.review_issues:
            score -= 0.05 * min(3, len(region.review_issues))
    if "text_overflow_risk" in region.qa_flags:
        score -= 0.15
    if "layout_failed" in region.qa_flags:
        score -= 0.3
    if len(region.arabic) > 4.5 * max(1, len(region.english)) and region.english:
        flags.append("translation_much_longer")
        score -= 0.1

    region.qa_flags = flags + [f for f in region.qa_flags
                               if f in ("text_overflow_risk", "layout_failed")]
    region.qa_score = max(0.0, min(1.0, score))
    return region.qa_score


def page_qa(page: PageResult, cfg: AppConfig) -> float:
    if not page.regions:
        page.page_qa_score = 1.0
        return 1.0
    # Score on *translation units*, not raw regions: merged satellites always
    # score 1.0 after region_qa, so averaging them in dilutes a broken
    # bubble (e.g. 1 failing primary + 3 satellites = 0.75 avg, hiding the
    # failure). Primaries carry the real quality; the worst-region term is
    # taken over primaries only for the same reason.
    primaries = [r for r in page.regions if r.merged_into is None]
    scored = primaries or page.regions  # all-satellite edge case fallback
    scores = [r.qa_score for r in scored]
    avg = float(np.mean(scores))
    worst = min(scores)
    # page score blends average with the worst region (a broken bubble hurts)
    page.page_qa_score = round(max(0.0, min(1.0, 0.7 * avg + 0.3 * worst)), 3)
    return page.page_qa_score


def visual_render_check(original_bgr: np.ndarray, cleaned_bgr: np.ndarray,
                        rendered_bgr: np.ndarray, regions: List[TextRegion],
                        *, diff_thr: int = 30) -> None:
    """Pixel-level QA that heuristic flags cannot see (item: true visual QA).

    For each primary region:
      - removal_missing: cleaned still looks like the original -> the English
        text was never erased (ghost text under the Arabic).
      - render_missing: rendered looks identical to cleaned -> the Arabic
        text was never drawn (layout failed silently / empty render).

    Flags are appended to the regions; they also feed `region_qa` penalties.
    """
    h, w = rendered_bgr.shape[:2]
    for r in regions:
        if r.merged_into is not None:
            continue  # satellites share the primary's pixels
        x0, y0, x1, y1 = r.merged_span or r.bbox
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, x1), min(h, y1)
        if x1 - x0 < 6 or y1 - y0 < 6:
            continue
        orig = original_bgr[y0:y1, x0:x1].astype(np.int16)
        clean = cleaned_bgr[y0:y1, x0:x1].astype(np.int16)
        rend = rendered_bgr[y0:y1, x0:x1].astype(np.int16)
        n = max(1, orig.shape[0] * orig.shape[1])
        # fraction of pixels the removal stage actually changed
        removal = float((np.abs(orig - clean).sum(axis=2) > diff_thr).sum()) / n
        # fraction of pixels the renderer actually drew on
        painted = float((np.abs(clean - rend).sum(axis=2) > diff_thr).sum()) / n
        if r.english.strip() and removal < 0.01:
            r.qa_flags.append("removal_missing")
        if r.arabic.strip() and painted < 0.005:
            r.qa_flags.append("render_missing")


def residual_english_check(rendered_bgr: np.ndarray, regions: List[TextRegion],
                           ocr_engine) -> List[str]:
    """Re-OCR the rendered page around each region; report leftover English."""
    leftovers: List[str] = []
    for r in regions:
        if r.merged_into is not None:
            continue  # satellite boxes live inside the primary's span; only
                      # the primary needs a residual-English re-check
        x0, y0, x1, y1 = r.bbox
        h, w = rendered_bgr.shape[:2]
        pad = 4
        crop = rendered_bgr[max(0, y0 - pad):min(h, y1 + pad),
                            max(0, x0 - pad):min(w, x1 + pad)]
        if crop.size == 0:
            continue
        res = ocr_engine.read_crop(crop)
        text = res.get("english", "")
        if _LATIN_RE.search(text) and res.get("confidence", 0) > 0.5:
            leftovers.append(f"region {r.id}: {text!r}")
    return leftovers


def build_quality_report(page: PageResult) -> Dict:
    flagged = [r for r in page.regions if r.merged_into is None
               and (r.qa_flags or r.qa_score < 0.7)]
    return {
        "page": page.image_path,
        "score": page.page_qa_score,
        # report translation units, not raw regions — satellites are noise
        "regions_total": len([r for r in page.regions
                              if r.merged_into is None]),
        "regions_flagged": [
            {"id": r.id, "flags": r.qa_flags, "score": round(r.qa_score, 2),
             "english": r.english, "arabic": r.arabic}
            for r in flagged
        ],
        "passed": page.page_qa_score >= 0.75,
    }
