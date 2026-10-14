from __future__ import annotations

import json
import os
import threading
import traceback
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import numpy as np

from ..config import AppConfig
from ..pipeline import Pipeline
from ..utils.types import PageResult
from .editor_state import EditorState

CSS = """
#title { text-align: center; font-size: 1.6rem; margin: 0.2rem 0; }
.gr-button-primary { background: #1a73e8 !important; }
.status-ok { color: #188038; }
.status-bad { color: #c5221f; }
"""


def _pil_from_path(path: str):
    from PIL import Image
    if not path or not os.path.isfile(path):
        return None
    return Image.open(path)


def _pil_from_bgr(bgr: np.ndarray):
    from PIL import Image
    return Image.fromarray(bgr[..., ::-1])


class GradioApp:
    def __init__(self, cfg: AppConfig, workdir: str = "."):
        self.cfg = cfg
        self.workdir = workdir
        self.pipeline: Optional[Pipeline] = None
        self.editor = EditorState()
        self._cancel_flag = threading.Event()
        self.chapter_id = "default"
        self._page_counters: Dict[str, int] = {}

    def _get_pipeline(self, chapter_id: str) -> Pipeline:
        if self.pipeline is None or self.pipeline.context.chapter_id != chapter_id:
            self.pipeline = Pipeline(self.cfg, chapter_id=chapter_id,
                                     workdir=self.workdir)
        return self.pipeline

    # -- single page --------------------------------------------------------
    def translate_single(self, image, chapter_id, progress=gr.Progress()) -> Tuple:
        if image is None:
            return None, None, None, "⚠️ Upload an image first.", ""
        import tempfile
        from ..utils.images import save_image
        tmp = tempfile.mktemp(suffix=".png")
        if isinstance(image, str):
            import shutil
            shutil.copy(image, tmp)
        else:
            save_image(np.asarray(image)[..., ::-1], tmp)  # RGB->BGR

        pipe = self._get_pipeline(chapter_id or "default")
        self._cancel_flag.clear()
        # track page numbers per chapter so sequential single-page uploads
        # build up the chapter context instead of all being "page 1"
        cid = chapter_id or "default"
        page_no = self._page_counters.get(cid, 0) + 1
        self._page_counters[cid] = page_no
        try:
            result = pipe.translate_image(tmp, page_number=page_no)
        except Exception as e:  # pragma: no cover
            result = PageResult(image_path=tmp, error=str(e))
        self.editor.set_page(result, tmp, cleaned_path=result.cleaned_path)

        original = _pil_from_path(tmp)
        arabic = _pil_from_path(result.output_path)
        overlay = _pil_from_path(result.debug_overlay_path)
        status = self._status_html(result)
        debug_json = json.dumps(result.to_dict(), ensure_ascii=False, indent=1)
        return original, overlay, arabic, status, debug_json

    # -- batch --------------------------------------------------------------
    def translate_batch(self, folder, chapter_id, progress=gr.Progress()) -> Tuple:
        from ..batch.chapter import list_images, translate_chapter
        if not folder or not os.path.isdir(folder):
            return "⚠️ Select a folder.", "", ""
        paths = list_images(folder)
        if not paths:
            return "⚠️ No images found.", "", ""
        pipe = self._get_pipeline(chapter_id or "default")
        self._cancel_flag.clear()

        def cb(msg, frac):
            progress(frac, desc=msg)
            if self._cancel_flag.is_set():
                pipe.cancel()

        results = translate_chapter(pipe, paths, progress=cb)
        html = self._batch_report(results)
        debug = json.dumps([r.to_dict() for r in results],
                           ensure_ascii=False, indent=1)
        out_dir = os.path.join(self.workdir,
                               self.cfg.get("output", "dir", default="output"))
        return html, debug, out_dir

    def cancel(self) -> str:
        self._cancel_flag.set()
        if self.pipeline:
            self.pipeline.cancel()
        return "⏹ Cancel requested (stops between stages)."

    # -- glossary -----------------------------------------------------------
    def glossary_rows(self) -> List[List[str]]:
        pipe = self.pipeline
        if pipe is None:
            pipe = self._get_pipeline("default")
        return pipe.glossary.to_rows()

    def save_glossary(self, rows) -> str:
        pipe = self.pipeline or self._get_pipeline("default")
        pipe.glossary.replace_all(rows)
        return "✅ Glossary saved."

    # -- editor / single-bubble edit ----------------------------------------
    def region_table(self) -> List[List[Any]]:
        page = self.editor.page
        if page is None:
            return []
        rows = []
        for r in page.regions:
            ov = self.editor.get_override(r.id)
            rows.append([
                r.id, r.region_type.value, r.english,
                ov.get("arabic", r.arabic),
                ov.get("font_size", str(r.layout.font_size if r.layout else "")),
                round(r.qa_score, 2),
                ", ".join(r.qa_flags),
            ])
        return rows

    def apply_region_edit(self, region_id, arabic, font_size) -> Tuple:
        page = self.editor.page
        if page is None or not page.regions:
            return None, "⚠️ No page loaded."
        try:
            rid = int(region_id)
        except (TypeError, ValueError):
            return None, "⚠️ Invalid region id."
        if not any(r.id == rid for r in page.regions):
            return None, f"⚠️ Region {rid} not found."
        if arabic:
            self.editor.apply_override(rid, arabic=arabic)
        if font_size:
            self.editor.apply_override(rid, font_size=int(font_size))
        # re-render from the inpainted base is expensive; simple path:
        # mark override so the next render uses it, and re-run render of base.
        new_pil = self._rerender_page()
        return new_pil, f"✅ Applied override to region {rid}."

    def _rerender_page(self):
        """Re-render the current page from the INPAINTED base (English-free),
        applying editor text/size overrides with a real refit."""
        page = self.editor.page
        if page is None:
            return None
        from ..utils.images import load_image, save_image
        from ..rendering.renderer import ArabicRenderer
        from ..typography.fitting import AutoFitter
        from ..typography.shaping import normalize_arabic
        base = self.editor.render_base()
        img = load_image(base)
        renderer = ArabicRenderer(self.cfg)
        fitter = AutoFitter(self.cfg)
        # apply overrides; force a refit so text actually fits after edits
        for r in page.regions:
            ov = self.editor.get_override(r.id)
            if "arabic" in ov:
                r.arabic = normalize_arabic(ov["arabic"])
                r.layout = None
            if "font_size" in ov:
                if r.layout is None:
                    r.layout = fitter.fit(r)
                if ov["font_size"]:
                    refit = fitter.refit_at_size(r, int(ov["font_size"]))
                    if refit is not None:
                        r.layout = refit
        rendered = renderer.render_page(img, page.regions)
        # persist so the exported result matches what the user sees
        if page.output_path:
            try:
                save_image(rendered, page.output_path)
            except Exception:
                pass
        return _pil_from_bgr(rendered)

    # -- report helpers -----------------------------------------------------
    def _status_html(self, result: PageResult) -> str:
        if result.error:
            return f'<span class="status-bad">❌ {result.error}</span>'
        score = result.page_qa_score
        color = "status-ok" if score >= 0.75 else "status-bad"
        html = (f'<span class="{color}">✅ {len(result.regions)} regions · '
                f'page score {score:.2f}</span>')
        if any("offline_translation_low_quality" in r.qa_flags
               for r in result.regions):
            html += ('<br><span class="status-bad">⚠️ No LLM configured — '
                     'used the offline translation backend, which has much '
                     'lower quality. Set an LLM key in ⚙️ Settings for '
                     'good results.</span>')
        return html

    def _batch_report(self, results: List[PageResult]) -> str:
        ok = sum(1 for r in results if not r.error)
        failed = [r for r in results if r.error]
        rows = "".join(
            f"<tr><td>{os.path.basename(r.image_path)}</td>"
            f"<td>{'❌ ' + r.error if r.error else '✅ ' + f'{r.page_qa_score:.2f}'}</td></tr>"
            for r in results)
        return (f"<h3>Batch complete: {ok}/{len(results)} pages</h3>"
                f"<table><tr><th>Page</th><th>Result</th></tr>{rows}</table>")

    # -- build --------------------------------------------------------------
    def build(self) -> gr.Blocks:
        with gr.Blocks(title="Manhwa Arabic Translator", css=CSS) as demo:
            gr.HTML("<div id='title'>📖 Manhwa/Webtoon English → Arabic Translator</div>")
            chapter = gr.Textbox(value="default", label="Chapter / Session ID",
                                 info="Same ID keeps characters & terminology consistent.")
            with gr.Tabs():
                # ---------- Translate ----------
                with gr.Tab("🖼 Translate"):
                    with gr.Row():
                        with gr.Column():
                            in_img = gr.Image(type="filepath", label="Upload manhwa page",
                                              height=560)
                            btn = gr.Button("Translate", variant="primary")
                            cancel_btn = gr.Button("Cancel")
                        with gr.Column():
                            gr.Markdown("### Detection / Debug")
                            overlay = gr.Image(type="pil", label="Detection overlay", height=260)
                            gr.Markdown("### Arabic Result")
                            arabic = gr.Image(type="pil", label="Translated page", height=320)
                    status = gr.HTML()
                    debug_json = gr.Code(label="Debug JSON (structured output)",
                                         language="json", lines=14)
                    btn.click(self.translate_single,
                              inputs=[in_img, chapter],
                              outputs=[in_img, overlay, arabic, status, debug_json])
                    cancel_btn.click(self.cancel, outputs=[status])

                # ---------- Batch ----------
                with gr.Tab("📚 Batch / Chapter"):
                    folder = gr.Textbox(label="Folder path containing page images")
                    batch_btn = gr.Button("Translate Chapter", variant="primary")
                    batch_report = gr.HTML()
                    batch_debug = gr.Code(label="Batch debug JSON", language="json", lines=10)
                    out_dir = gr.Textbox(label="Output folder")
                    batch_btn.click(self.translate_batch,
                                    inputs=[folder, chapter],
                                    outputs=[batch_report, batch_debug, out_dir])

                # ---------- Glossary ----------
                with gr.Tab("📖 Glossary"):
                    gr.Markdown("English → Arabic terminology (names, places, skills, SFX).")
                    glossary = gr.Dataframe(headers=["English", "Arabic"],
                                            datatype=["str", "str"],
                                            value=self.glossary_rows,
                                            interactive=True, row_count=(10, "dynamic"))
                    save_gloss = gr.Button("Save Glossary")
                    gloss_status = gr.HTML()
                    save_gloss.click(self.save_glossary, inputs=[glossary],
                                     outputs=[gloss_status])

                # ---------- Editor ----------
                with gr.Tab("✏️ Editor"):
                    gr.Markdown("Fix one bubble without re-running the page.")
                    regions = gr.Dataframe(
                        headers=["ID", "Type", "English", "Arabic", "Font size",
                                 "Score", "Flags"],
                        value=self.region_table, interactive=False,
                        row_count=(6, "dynamic"))
                    with gr.Row():
                        rid = gr.Number(label="Region ID", precision=0)
                        new_ar = gr.Textbox(label="New Arabic text")
                        new_size = gr.Number(label="Font size (optional)", precision=0)
                    apply_btn = gr.Button("Apply & Re-render")
                    edit_out = gr.Image(type="pil", label="Re-rendered page", height=400)
                    edit_status = gr.HTML()
                    apply_btn.click(self.apply_region_edit,
                                    inputs=[rid, new_ar, new_size],
                                    outputs=[edit_out, edit_status])

                # ---------- Settings ----------
                with gr.Tab("⚙️ Settings"):
                    llm_base = gr.Textbox(value=self.cfg.llm.base_url,
                                          label="LLM base URL (OpenAI-compatible)")
                    llm_key = gr.Textbox(value=self.cfg.llm.api_key, type="password",
                                         label="API key")
                    llm_model = gr.Textbox(value=self.cfg.llm.model, label="Model name")
                    low_vram = gr.Checkbox(value=bool(self.cfg.get("device", "low_vram_mode")),
                                           label="Low VRAM mode")
                    save_cfg = gr.Button("Apply (this session)")
                    cfg_status = gr.HTML()

                    def _apply_cfg(base, key, model, vram):
                        self.cfg.set(base, "translation", "llm", "base_url")
                        self.cfg.set(key, "translation", "llm", "api_key")
                        self.cfg.set(model, "translation", "llm", "model")
                        self.cfg.set(bool(vram), "device", "low_vram_mode")
                        # recreate the pipeline so device/model changes take
                        # effect (LLM settings are read lazily either way)
                        self.pipeline = None
                        return "✅ Settings applied."

                    save_cfg.click(_apply_cfg,
                                   inputs=[llm_base, llm_key, llm_model, low_vram],
                                   outputs=[cfg_status])

        return demo
