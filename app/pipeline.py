from __future__ import annotations

import logging
import os
import threading
import time
from typing import Callable, List, Optional

import numpy as np

from .config import AppConfig, setup_logging
from .context.chapter_context import ChapterContext
from .glossary.glossary import Glossary
from .inpainting.service import TextRemovalService
from .ocr.service import OCRService
from .quality_control.qa import (page_qa, region_qa, residual_english_check,
                                 visual_render_check, build_quality_report)
from .rendering.renderer import ArabicRenderer
from .translation.service import PageTranslator
from .utils.images import load_image, save_image
from .utils.types import CancelledError, PageResult, TextRegion
from .utils.debug import draw_debug_overlay

log = logging.getLogger(__name__)

ProgressFn = Callable[[str, float], None]


class Pipeline:
    """Full EN->AR manhwa page pipeline with cancellation and progress."""

    def __init__(self, cfg: AppConfig, *, chapter_id: str = "default",
                 workdir: str = "."):
        self.cfg = cfg
        self.workdir = workdir
        self.glossary = Glossary(
            os.path.join(workdir, cfg.get("glossary", "path", default="glossary.json")),
            auto_learn=bool(cfg.get("glossary", "auto_learn", default=True)))
        self.context = ChapterContext(chapter_id, workdir)
        self.ocr = OCRService(cfg)
        self.removal = TextRemovalService(cfg)
        self.translator = PageTranslator(cfg, self.glossary, self.context)
        self.renderer = ArabicRenderer(cfg)
        self._cancel = threading.Event()

    # ------------------------------------------------------------------
    def cancel(self) -> None:
        self._cancel.set()

    def _check_cancel(self) -> None:
        if self._cancel.is_set():
            raise CancelledError("cancelled by user")

    def reset_context(self) -> None:
        self.context.reset()
        self.context.save()

    # ------------------------------------------------------------------
    def translate_image(self, image_path: str, page_number: int = 1,
                        progress: Optional[ProgressFn] = None) -> PageResult:
        """Process one page end-to-end."""
        self._cancel.clear()
        self._last_cleaned_image = None  # don't leak the previous page's base

        def report(msg: str, frac: float) -> None:
            if progress:
                progress(msg, frac)

        result = PageResult(image_path=image_path)
        try:
            report("Loading image", 0.02)
            img = load_image(image_path)
            result.width, result.height = img.shape[1], img.shape[0]

            self._check_cancel()
            report("Detecting and reading text", 0.12)
            regions = self.ocr.detect_and_read(img)
            result.regions = regions
            log.info("page %d: %d text regions", page_number, len(regions))

            # removal is independent of the translation, so do it once and
            # let the translate -> typeset -> QA pass retry cheaply on low scores
            report("Removing original text", 0.3)
            cleaned = self.removal.remove_text(img, regions)
            # keep the inpainted (English-free) base so the Editor can
            # re-render single bubbles without re-running the pipeline
            self._last_cleaned_image = cleaned

            auto_retry = bool(self.cfg.get("quality_control", "auto_retry", default=True))
            max_tries = int(self.cfg.get("quality_control", "max_retries", default=2))
            min_score = float(self.cfg.get("quality_control", "min_page_score", default=0.75))
            attempts = 1 + (max_tries if (auto_retry and self.translator.backend_name == "llm") else 0)

            rendered = img
            for attempt in range(attempts):
                self._check_cancel()
                report("Translating with context"
                       + (f" (retry {attempt})" if attempt else ""), 0.4)
                self.translator.translate_page(page_number, regions)

                self._check_cancel()
                report("Typesetting Arabic", 0.8)
                for r in regions:
                    r.layout = None  # translations changed -> refit
                rendered = self.renderer.render_page(cleaned, regions)

                # QA ------------------------------------------------------------
                report("Quality check", 0.9)
                for r in regions:
                    region_qa(r, self.cfg)
                result.page_summary = self.context.story_summary
                page_qa(result, self.cfg)
                if result.page_qa_score >= min_score or attempt == attempts - 1:
                    break
                # Retrying only makes sense when the LLM can actually move the
                # score: untranslated or untranslated-looking regions. Flags
                # like low_ocr_confidence/empty_ocr/residual_english do not
                # change with re-translation, so a score driven by them used
                # to trigger identical, pointless retries (3+ LLM calls).
                retryable = any(
                    f in ("missing_translation", "no_arabic_glyphs",
                          "latin_left_in_arabic", "translation_unavailable")
                    for r in regions for f in r.qa_flags)
                if not retryable:
                    log.info("page score %.2f < %.2f but no translation-fixable "
                             "flags; not retrying",
                             result.page_qa_score, min_score)
                    break
                log.info("page score %.2f < %.2f -> retrying translation",
                         result.page_qa_score, min_score)

            if (self.cfg.get("quality_control", "residual_text_check", default=True)
                    and not self._cancel.is_set()):
                # true pixel-level QA: catches English ghosts the OCR-based
                # check misses, and Arabic text that silently failed to render
                try:
                    visual_render_check(img, cleaned, rendered, regions)
                except Exception as e:  # pragma: no cover
                    log.debug("visual QA failed: %s", e)
                try:
                    leftovers = residual_english_check(rendered, regions,
                                                       self.ocr._engine)
                    # note: leftovers are strings like "region 3: 'ABC'" —
                    # match by id (zipping pairs them with the wrong regions)
                    for lo in leftovers:
                        for r in regions:
                            if lo.startswith(f"region {r.id}:"):
                                r.qa_flags.append("residual_english")
                                break
                except Exception as e:  # pragma: no cover
                    log.debug("residual check failed: %s", e)

            self.context.remember_page(page_number, result)
            self.context.save()

            # save ------------------------------------------------------------
            out_dir = os.path.join(self.workdir,
                                   self.cfg.get("output", "dir", default="output"))
            base = os.path.splitext(os.path.basename(image_path))[0]
            out_path = os.path.join(out_dir, f"{base}_ar.png")
            report("Saving result", 0.97)
            save_image(rendered, out_path)
            result.output_path = out_path

            if self.cfg.get("output", "save_cleaned", default=True):
                cleaned_path = os.path.join(out_dir, f"{base}_cleaned.png")
                cleaned_img = self._last_cleaned_image
                save_image(cleaned_img if cleaned_img is not None else img,
                           cleaned_path)
                result.cleaned_path = cleaned_path

            if self.cfg.get("output", "save_debug_json", default=True):
                import json
                dbg_path = os.path.join(out_dir, f"{base}_debug.json")
                with open(dbg_path, "w", encoding="utf-8") as f:
                    json.dump(result.to_dict(), f, ensure_ascii=False, indent=1)
            if self.cfg.get("output", "save_debug_overlay", default=True):
                from .utils.debug import draw_debug_overlay
                ov = draw_debug_overlay(img, result)
                result.debug_overlay_path = os.path.join(
                    out_dir, f"{base}_overlay.png")
                save_image(ov, result.debug_overlay_path)
            report("Done", 1.0)
        except CancelledError:
            result.error = "cancelled"
        except Exception as e:
            log.exception("pipeline failed for %s", image_path)
            result.error = str(e)
        return result
