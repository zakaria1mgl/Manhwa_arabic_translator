# 📖 Manhwa / Webtoon English → Arabic Image Translator

Professional-grade localization pipeline for manhwa/webtoon pages:
**English image in → professionally localized Arabic image out**, with the
artwork untouched except for the replaced text.

```
Image → Preprocessing → Text Detection → OCR → Reading Order
      → Region Classification → Page-Level Context Analysis
      → Context-Aware EN→AR Translation (multi-pass LLM)
      → Translation Review → Text Removal → Inpainting
      → Arabic RTL Typesetting → Auto Text Fitting → QA → Final Render
```

## Features

- **Detection/OCR (3 backends)**:
  - `vision` (default when an LLM key is set): one multimodal LLM call does
    detection + reading + speaker/gender + SFX classification — by far the
    most accurate on stylized comic art.
  - `easyocr`: CRAFT detection + single-pass recognition (no crop
    re-detection), hallucination sanitizing, polarity inversion for
    white-on-dark text, vertical-text retry, tiled detection for long
    webtoon strips.
  - `comic`: optional adapter for dmMaze/comic-text-detector (install
    separately into `vendor/`), best classical block detection.
- **Panel-aware reading order**: gutter-based panel segmentation; panels are
  read in order, then bubbles inside each panel (LTR/RTL switchable).
- **True bubble detection**: text-line clusters snap to the actual bright
  *or dark* bubble contour on the page (dark narration boxes included), so
  filling, removal and typesetting respect the real bubble shape.
- **Page-level context**: characters (gender, personality, speaking style,
  formality), story summary, recent dialogue — persisted per chapter so page 8
  sounds like page 1. Speaker + gender are attached to every region, so Arabic
  verb/pronoun agreement is correct.
- **Multi-pass translation**: ① context analysis ② translation ③ critical
  review ④ bubble-fit rewrite (length-constrained when text would overflow) —
  all as structured JSON against an OpenAI-compatible LLM (Gemini, OpenAI,
  local vLLM…). Automatic retry fills any region a pass skipped. Glossary
  auto-learns terms chosen by the LLM. Offline MarianMT fallback without a key.
- **SFX handling**: onomatopoeia dictionary + scene-aware LLM adaptation,
  size/colour/orientation preserved.
- **Text removal**: colour-aware text-pixel masks (no white halos),
  whole-bubble fast-fill using the real contour, Big-LaMa inpainting with
  OpenCV Telea fallback.
- **Arabic typesetting**: arabic_reshaper + python-bidi shaping, punctuation
  normalization (؟ ، ؛), professional fonts (Amiri / Noto Naskh Arabic —
  auto-downloaded), bubble-aware binary-search font fitting with an
  ellipse-aware inset so text never pokes out of round bubble corners.
- **QA + auto-retry**: per-region and per-page scores, pixel-level checks
  (un-erased English ghosts, silent render failures), residual-English re-OCR
  check, and re-translation retry that only fires when the score can
  actually improve.
- **UI (Gradio)**: original / detection / result views, single page, batch
  chapter, glossary editor, single-bubble editor (re-renders from the
  inpainted base — no leftover English), settings, cancel.
- **Performance**: model caching (load once per session), GPU auto-detect,
  VRAM-aware LaMa guard, low-VRAM mode.

## Quick start (Google Colab, recommended)

1. Open `ManhwaTranslator_Colab.ipynb` in Colab.
2. **Runtime → Change runtime type → GPU (T4)**.
3. Run cells in order. Upload the project zip when asked.
4. (Optional but recommended) set an LLM key in the Settings tab:
   - Gemini: base URL `https://generativelanguage.googleapis.com/v1beta/openai`,
     model `gemini-2.0-flash`, your API key.
5. Use the public Gradio link.

## Local install (Linux / Windows / any NVIDIA GPU)

```bash
cd manhwa_arabic_translator
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m app.main --workdir . --port 7860
# open http://127.0.0.1:7860
```

CPU-only works (slower); omit `simple-lama-inpainting` to use the OpenCV
inpainting fallback.

## Configuration

Copy the defaults from `app/config.py` into `config.yaml` and pass
`--config config.yaml`. Environment variables:

| Variable | Purpose |
|---|---|
| `MANHWA_LLM_BASE_URL` / `MANHWA_LLM_API_KEY` / `MANHWA_LLM_MODEL` | LLM translation backend |
| `MANHWA_LOW_VRAM=1` | low-VRAM mode (skips LaMa when tight) |
| `MANHWA_PANEL_ORDER=rtl` | manga-style right-to-left panel order |
| `MANHWA_INPAINT_BACKEND=opencv` | force OpenCV inpainting |
| `MANHWA_OCR_BACKEND` | `auto` (default) / `easyocr` / `vision` OCR backend |

## Project layout

```
app/
├── ui/            Gradio app + editor state
├── detection/     text detection backends (EasyOCR/CRAFT)
├── ocr/           recognition, preprocessing, reading order, classification
├── context/       chapter memory: characters, story, dialogue history
├── translation/   LLM client, multi-pass service, SFX, offline fallback
├── glossary/      persistent EN→AR terminology
├── inpainting/    masks, LaMa/OpenCV backends
├── typography/    fonts, Arabic shaping, auto-fitting
├── rendering/     final Arabic text drawing
├── quality_control/ region/page scoring, residual-English check
├── batch/         chapter batch processing
├── models/        device/VRAM management + model cache
├── utils/         types, polygons, images, debug overlay
├── pipeline.py    end-to-end orchestration
└── main.py        entry point
```

## Notes

- Same **Chapter ID** in the UI keeps character voice and terminology
  consistent across pages; change it to start a fresh context.
- The **Editor** tab fixes a single bubble (text or font size) without
  re-running the whole page.
- Output images are saved to `output/` with `_ar`, `_debug.json`,
  `_overlay.png` and `_cleaned` (the English-free inpainted base used by the
  editor) companions.
