"""Regression test for the merged-satellite QA bug (fixed in qa.py).

Before the fix, `merge_bubble_lines` blanked satellite regions' english and
arabic, and `region_qa` scored those blanks as guaranteed failures
(`empty_ocr` + `missing_translation`). Consequences:

  * every multi-line bubble produced false flags on its satellites,
  * `missing_translation` on a satellite is in the pipeline's retryable set,
    so pages with perfectly good translations burned 1-2 extra full LLM
    translation passes,
  * page scores were diluted by always-1.0 satellites once fixed naively,
    so `page_qa` now averages over translation units (primaries) only.

Runs without torch/easyocr/gradio; needs numpy (+ cv2 for nothing here).
"""
from __future__ import annotations

import sys
sys.path.insert(0, ".")

from app.config import AppConfig
from app.ocr.reading_order import merge_bubble_lines
from app.quality_control.qa import build_quality_report, page_qa, region_qa
from app.utils.types import PageResult, Quad, RegionType, TextRegion

RETRYABLE = ("missing_translation", "no_arabic_glyphs",
             "latin_left_in_arabic", "translation_unavailable")


def line(i: int, y: int, text: str) -> TextRegion:
    return TextRegion(
        id=i,
        quad=Quad([(60, y), (240, y), (240, y + 18), (60, y + 18)]),
        english=text, ocr_conf=0.9,
        region_type=RegionType.DIALOGUE, bubble_cluster=[0, 1, 2])


def main() -> None:
    cfg = AppConfig({})

    # --- realistic multi-line bubble + one standalone bubble --------------
    regions = [line(0, 100, "Hello there my"),
               line(1, 122, "friend, how are"),
               line(2, 144, "you today?")]
    solo = TextRegion(id=3, quad=Quad([(60, 300), (200, 300),
                                       (200, 320), (60, 320)]),
                      english="Fine.", ocr_conf=0.95,
                      region_type=RegionType.DIALOGUE)
    regions.append(solo)

    merge_bubble_lines(regions)
    primaries = [r for r in regions if r.merged_into is None]
    sats = [r for r in regions if r.merged_into is not None]
    assert len(primaries) == 2 and len(sats) == 2, \
        "merge should leave 1 cluster primary + solo"
    cluster_primary = next(p for p in primaries if p.english.startswith("Hello"))
    assert cluster_primary.english == "Hello there my friend, how are you today?"

    # simulate a GOOD translation pass
    cluster_primary.arabic = "مرحبًا يا صديقي، كيف حالك اليوم؟"
    solo.arabic = "بخير."

    page = PageResult(regions=regions)
    for r in regions:
        region_qa(r, cfg)

    # 1. satellites must NOT carry failure flags any more
    for s in sats:
        assert s.qa_flags == [], f"satellite {s.id} flagged: {s.qa_flags}"
        assert s.qa_score == 1.0

    # 2. no retryable flag anywhere -> pipeline would not waste an LLM retry
    retryable = [f for r in regions for f in r.qa_flags if f in RETRYABLE]
    assert not retryable, f"false-retry flags present: {retryable}"

    # 3. healthy page passes the min-score gate
    score = page_qa(page, cfg)
    assert score >= 0.75, f"healthy page scored {score}"

    # 4. a genuinely broken primary still fails the gate (no dilution):
    #    satellites scoring 1.0 must not rescue the page average.
    cluster_primary.arabic = ""
    region_qa(cluster_primary, cfg)
    assert "missing_translation" in cluster_primary.qa_flags
    bad = page_qa(page, cfg)
    assert bad < 0.75, f"broken bubble diluted away: page={bad}"

    # 5. report counts translation units, flags only the real failure
    rep = build_quality_report(page)
    assert rep["regions_total"] == 2, rep["regions_total"]
    ids = [x["id"] for x in rep["regions_flagged"]]
    assert ids == [cluster_primary.id], ids

    print("QA_MERGE_TEST_PASSED")


if __name__ == "__main__":
    main()
