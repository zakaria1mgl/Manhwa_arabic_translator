from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from ..config import AppConfig
from ..context.chapter_context import CharacterProfile, ChapterContext
from ..glossary.glossary import Glossary
from ..typography.shaping import normalize_arabic
from ..utils.types import RegionType, TextRegion
from .llm_client import LLMClient, extract_json
from .sfx import lookup_sfx

log = logging.getLogger(__name__)


class PageTranslator:
    """Multi-pass, context-aware EN->AR translator using an LLM backend.

    Passes:
      1. page-level context + speaker analysis (single JSON call)
      2. translation per page (single JSON call, all regions at once)
      3. critical review (single JSON call)
      4. fit optimization (shortens long lines when enabled)
    """

    def __init__(self, cfg: AppConfig, glossary: Glossary, context: ChapterContext):
        self.cfg = cfg
        self.glossary = glossary
        self.context = context
        self.llm = LLMClient(cfg)

    @property
    def backend_name(self) -> str:
        return "llm" if self.llm.ready else "offline"

    # ------------------------------------------------------------------
    def translate_page(self, page_number: int, regions: List[TextRegion],
                       page_summary_hint: str = "") -> None:
        """Fill region.arabic + region.speaker for every region (in place)."""
        dialogues = [r for r in regions if r.english.strip()]
        if not dialogues:
            return
        if not self.llm.ready:
            self._translate_offline(dialogues)
            return

        context_block = self._context_block()
        # PASS 1: context analysis (speakers + page summary + character updates)
        analysis = self._pass1_context(page_number, dialogues, context_block,
                                      page_summary_hint)
        self._apply_analysis(analysis, page_number, dialogues)

        # PASS 2: translation (now with speaker/gender info attached)
        translation = self._pass2_translate(page_number, dialogues,
                                            self._context_block())
        self._apply_translation(translation, dialogues)

        # automatic retry for regions the LLM failed to fill
        missing = [r for r in dialogues if not r.arabic.strip()]
        if missing and self.llm.ready:
            log.info("retrying %d untranslated regions", len(missing))
            retry = self._pass2_translate(page_number, missing,
                                          self._context_block())
            self._apply_translation(retry, missing)

        # PASS 3: critical review
        if self.cfg.get("translation", "review_pass", default=True):
            review = self._pass3_review(page_number, dialogues, self._context_block())
            self._apply_review(review, dialogues)

        # PASS 4: fit optimization (shorten lines that overflow their bubble)
        if self.cfg.get("translation", "fit_pass", default=True) and self.llm.ready:
            self._pass4_fit(page_number, dialogues)

        # SFX: keep the scene-aware LLM result; dictionary is the fallback
        for r in dialogues:
            if r.region_type == RegionType.SFX and not r.arabic:
                sug = lookup_sfx(r.english)
                if sug:
                    r.arabic = sug
                else:
                    r.arabic = self._default_sfx(r.english)
                    r.qa_flags.append("sfx_passthrough")
            r.arabic = normalize_arabic(r.arabic) if r.arabic else r.arabic

    # ------------------------------------------------------------------
    def _context_block(self) -> str:
        parts = []
        g = self.glossary.as_prompt_block()
        if g:
            parts.append(g)
        c = self.context.character_block()
        if c:
            parts.append(c)
        d = self.context.recent_dialogue_block()
        if d:
            parts.append(d)
        if self.context.story_summary:
            parts.append("Story so far:\n" + self.context.story_summary)
        return "\n\n".join(parts)

    # PASS 1 -----------------------------------------------------------------
    def _pass1_context(self, page_number: int, dialogues: List[TextRegion],
                       context_block: str, hint: str) -> Dict[str, Any]:
        items = "\n".join(
            f"[{r.id}] type={r.region_type.value} text={json.dumps(r.english, ensure_ascii=False)}"
            for r in dialogues)
        sys = SYSTEM_BASE + CONTEXT_ANALYSIS_INSTRUCTION
        user = (
            f"Page {page_number}. Analyze this manhwa page.\n"
            f"{context_block}\n\n"
            f"Page text lines:\n{items}\n"
            + (f"\nPrevious summary: {hint}\n" if hint else "")
            + "\nReturn JSON."
        )
        reply = self.llm.chat([
            {"role": "system", "content": sys},
            {"role": "user", "content": user},
        ], json_mode=True, temperature=0.2)
        try:
            return extract_json(reply)
        except Exception as e:
            log.warning("pass1 JSON parse failed: %s", e)
            return {}

    def _apply_analysis(self, analysis: Dict[str, Any], page_number: int,
                        dialogues: Optional[List[TextRegion]] = None) -> None:
        summary = analysis.get("page_summary")
        if summary:
            self.context.story_summary = str(summary)
        learn_terms = bool(self.cfg.get("glossary", "auto_learn", default=True))
        for char in analysis.get("characters", []):
            try:
                terms = dict(char.get("preferred_terms", {}) or {})
                if terms and learn_terms:
                    for en, ar in terms.items():
                        if str(en).strip() and str(ar).strip():
                            self.glossary.set(str(en), str(ar))
                profile = CharacterProfile(
                    name=str(char.get("name", "")),
                    gender=str(char.get("gender", "unknown")),
                    personality=str(char.get("personality", "")),
                    speaking_style=str(char.get("speaking_style", "")),
                    formality=str(char.get("formality", "")),
                    preferred_terms=terms,
                    first_seen_page=page_number,
                    last_seen_page=page_number,
                )
                if profile.name:
                    self.context.upsert_character(profile)
            except Exception as e:  # pragma: no cover
                log.debug("character parse: %s", e)
        # speaker assignment -> BOTH the memory and the regions themselves,
        # so pass 2/3, the dialogue history and the UI editor all see it
        speakers = analysis.get("speakers", {})
        if not isinstance(speakers, dict):
            return
        by_id = {r.id: r for r in (dialogues or [])}
        for rid, info in speakers.items():
            if not isinstance(info, dict):
                continue
            name = info.get("name")
            gender = info.get("gender")
            if name:
                self.context.upsert_character(CharacterProfile(
                    name=str(name), gender=str(gender or "unknown"),
                    first_seen_page=page_number, last_seen_page=page_number))
            try:
                region = by_id.get(int(rid))
            except (TypeError, ValueError):
                continue
            if region is not None:
                if name:
                    region.speaker = str(name)
                if gender:
                    region.speaker_gender = str(gender)

    # PASS 2 -----------------------------------------------------------------
    def _pass2_translate(self, page_number: int, dialogues: List[TextRegion],
                         context_block: str) -> Dict[str, Any]:
        items = "\n".join(
            f"[{r.id}] ({r.region_type.value}"
            + (f", speaker={r.speaker or 'unknown'} / "
               f"{r.speaker_gender or 'unknown'} gender" if r.speaker else "")
            + f") {json.dumps(r.english, ensure_ascii=False)}"
            for r in dialogues)
        sys = SYSTEM_BASE + TRANSLATION_INSTRUCTION
        user = (
            f"Translate the following manhwa page {page_number} into natural "
            f"Modern Standard Arabic (Fusha).\n{context_block}\n\n{items}\n"
            "\nReturn JSON with a \"translations\" object keyed by region id "
            "(keys as strings)."
        )
        reply = self.llm.chat([
            {"role": "system", "content": sys},
            {"role": "user", "content": user},
        ], json_mode=True, temperature=0.3)
        try:
            data = extract_json(reply)
        except Exception as e:
            log.warning("pass2 JSON parse failed: %s", e)
            return {}
        return data if isinstance(data, dict) else {"translations": {}}

    def _apply_translation(self, data: Dict[str, Any],
                           dialogues: List[TextRegion]) -> None:
        translations = data.get("translations", data)
        if not isinstance(translations, dict):
            return
        for r in dialogues:
            ar = translations.get(str(r.id)) or translations.get(r.id)
            if isinstance(ar, dict):
                ar = ar.get("arabic") or ar.get("text")
            if ar:
                r.arabic = normalize_arabic(str(ar).strip())
                r.translation_backend = "llm"

    # PASS 3 -----------------------------------------------------------------
    def _pass3_review(self, page_number: int, dialogues: List[TextRegion],
                      context_block: str) -> Dict[str, Any]:
        items = "\n".join(
            f"[{r.id}] EN={json.dumps(r.english, ensure_ascii=False)} AR={json.dumps(r.arabic, ensure_ascii=False)}"
            for r in dialogues if r.arabic)
        sys = SYSTEM_BASE + REVIEW_INSTRUCTION
        user = f"Critically review these translations for page {page_number}.\n{context_block}\n\n{items}\nReturn JSON."
        reply = self.llm.chat([
            {"role": "system", "content": sys},
            {"role": "user", "content": user},
        ], json_mode=True, temperature=0.1)
        try:
            return extract_json(reply)
        except Exception as e:
            log.warning("pass3 JSON parse failed: %s", e)
            return {}

    def _apply_review(self, review: Dict[str, Any], dialogues: List[TextRegion]) -> None:
        fixes = review.get("fixes", {})
        if isinstance(fixes, dict):
            for r in dialogues:
                fix = fixes.get(str(r.id)) or fixes.get(r.id)
                if isinstance(fix, dict):
                    new_ar = fix.get("arabic") or fix.get("text")
                    issues = fix.get("issues", [])
                    if new_ar and isinstance(new_ar, str):
                        r.arabic = normalize_arabic(str(new_ar).strip())
                    r.review_issues = list(issues) if isinstance(issues, list) else []
        # record issues for regions without fixes
        issues_all = review.get("issues", {})
        if isinstance(issues_all, dict):
            for r in dialogues:
                iss = issues_all.get(str(r.id)) or issues_all.get(r.id)
                if isinstance(iss, list):
                    r.review_issues = list(iss)

    # PASS 4 -----------------------------------------------------------------
    def _pass4_fit(self, page_number: int, dialogues: List[TextRegion]) -> None:
        """Ask the LLM to compress translations that cannot fit their bubble.

        Budget estimation is purely geometric: usable bubble box (merged span
        or bubble hull) / approximate Arabic glyph cell. Regions well under
        their budget are left untouched.
        """
        over: List[TextRegion] = []
        budgets: Dict[int, int] = {}
        for r in dialogues:
            if not r.arabic or r.region_type == RegionType.SFX:
                continue
            budget = _fit_budget_chars(r)
            budgets[r.id] = budget
            if len(r.arabic) > budget:
                over.append(r)
        if not over:
            return
        log.info("fit pass: %d regions exceed their bubble budget", len(over))
        items = "\n".join(
            f"[{r.id}] limit=~{budgets[r.id]} chars (currently {len(r.arabic)}): "
            f"{json.dumps(r.arabic, ensure_ascii=False)}"
            for r in over)
        reply = self.llm.chat([
            {"role": "system", "content": SYSTEM_BASE + FIT_INSTRUCTION},
            {"role": "user", "content":
                f"Page {page_number}. Shorten these Arabic translations so each "
                f"fits its speech bubble (never change meaning or tone):\n{items}\n"
                "Return JSON."},
        ], json_mode=True, temperature=0.2)
        try:
            data = extract_json(reply)
        except Exception as e:
            log.warning("pass4 JSON parse failed: %s", e)
            return
        fixed = data.get("shortened", data) if isinstance(data, dict) else {}
        if not isinstance(fixed, dict):
            return
        by_id = {r.id: r for r in over}
        for rid, new_ar in fixed.items():
            if isinstance(new_ar, dict):
                new_ar = new_ar.get("arabic") or new_ar.get("text")
            try:
                r = by_id.get(int(rid))
            except (TypeError, ValueError):
                continue
            if r is not None and isinstance(new_ar, str) and new_ar.strip():
                candidate = normalize_arabic(new_ar.strip())
                # only accept genuinely shorter rewrites
                if len(candidate) < len(r.arabic):
                    r.arabic = candidate

    # offline fallback ---------------------------------------------------------
    def _translate_offline(self, dialogues: List[TextRegion]) -> None:
        from .offline_backend import OfflineTranslator
        try:
            ot = OfflineTranslator(self.cfg.get(
                "translation", "offline_model",
                default="Helsinki-NLP/opus-mt-en-ar"))
            texts = [r.english for r in dialogues]
            out = ot.translate(texts)
            for r, ar in zip(dialogues, out):
                r.arabic = normalize_arabic(str(ar).strip())
                r.translation_backend = "offline"
                if r.region_type == RegionType.SFX:
                    sug = lookup_sfx(r.english)
                    if sug:
                        r.arabic = sug
        except Exception as e:
            log.warning("offline translation unavailable: %s", e)
            for r in dialogues:
                r.qa_flags.append("translation_unavailable")
        else:
            # offline backend is a LAST RESORT with visibly worse quality;
            # surface it instead of silently shipping stiff translations
            for r in dialogues:
                if "offline_translation_low_quality" not in r.qa_flags:
                    r.qa_flags.append("offline_translation_low_quality")

    def _default_sfx(self, text: str) -> str:
        # Fallback when neither the LLM nor the dictionary produced an Arabic
        # SFX: keep the original art text (blanking a big SFX would leave the
        # artwork damaged), but flag it so QA surfaces the passthrough.
        return text


def _fit_budget_chars(region: TextRegion) -> int:
    """Rough upper bound on Arabic characters that fit the region's bubble.

    Geometry only: usable box = merged span or bubble quad bbox, minus
    padding; line height ~= 1.25 x the OCR quad height; an Arabic glyph is
    about 0.35 x line height wide on average.
    """
    if region.merged_span:
        x0, y0, x1, y1 = region.merged_span
    else:
        x0, y0, x1, y1 = region.bbox
    bw, bh = max(24, x1 - x0), max(24, y1 - y0)
    line_h = max(14.0, region.quad.height * 1.25)
    n_lines = max(1, int((bh * 0.92) // line_h))
    chars_per_line = max(6, int(bw / (0.35 * line_h)))
    return max(14, n_lines * chars_per_line)


SYSTEM_BASE = (
    "You are an expert manhwa/webtoon localizer translating English into "
    "professional Modern Standard Arabic (Fusha). You understand RTL Arabic "
    "typography, webtoon conventions, character voice, and onomatopoeia. "
    "Meaning and naturalness beat literal translation. Preserve emotion, "
    "tone, personality, humor, threats and sarcasm. NEVER translate character "
    "or place names arbitrarily — keep proper nouns as-is unless a glossary "
    "says otherwise. ALWAYS use Arabic punctuation (؟ ، ؛ …) in Arabic text. "
    "Arabic verbs, pronouns, adjectives and address forms MUST match the "
    "speaker's gender AND the addressee's gender — get ة/ت agreement right. "
    "Dialogue must be SHORT and punchy like real webtoon speech — long "
    "wordy sentences do not fit bubbles. Respond ONLY with valid JSON."
)

CONTEXT_ANALYSIS_INSTRUCTION = (
    "\nTask: page-level context analysis. Output JSON:\n"
    '{"page_summary": "...", '
    '"characters": [{"name","gender","personality","speaking_style","formality",'
    '"preferred_terms":{}}], '
    '"speakers": {"<region_id>": {"name","gender"}}}'
)

TRANSLATION_INSTRUCTION = (
    "\nTask: translate. Use the glossary and character voice blocks exactly. "
    "When a speaker is given for a region, translate in THAT character's "
    "voice and use verb forms matching their gender. "
    "Arabic grows visually longer than English — keep every translation "
    "AT MOST roughly the length of the English text, rephrase freely to fit. "
    "Output JSON: {\"translations\": {\"<region_id>\": \"arabic text\"}}. "
    "For SFX (type=sfx) give a short stylized Arabic onomatopoeia, not a sentence. "
    "For narration keep it literary but natural."
)

FIT_INSTRUCTION = (
    "\nTask: bubble-fit rewrite. Output JSON: {\"shortened\": {\"<region_id>\": "
    '"shorter arabic"}}. Compress each line below its character limit: drop '
    "filler words, use shorter synonyms, merge clauses — but NEVER change the "
    "meaning, register, emotion or the character's voice, and keep any "
    "glossary terms and proper nouns byte-identical."
)

REVIEW_INSTRUCTION = (
    "\nTask: critical review. Output JSON: {\"fixes\": {\"<region_id>\": "
    '{"arabic":"...", "issues":["..."]}}, "issues": {"<region_id>":["..."]}}. '
    "Rewrite any line that is unnatural, wrong register, inconsistent with a "
    "character (including gender agreement), or ignores the glossary. "
    "Leave correct lines out of fixes."
)

# NOTE: FIT_INSTRUCTION used to be defined twice (the second definition
# silently shadowed the first). Keep only the one above.
