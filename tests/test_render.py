"""Rendering test: end-to-end typography on a synthetic bubble image.

Draws a fake speech bubble with English text, then runs:
masking -> OpenCV inpaint -> Arabic fit+render, and verifies the output
contains Arabic glyph pixels inside the bubble and no black text remnants.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import cv2

from app.config import AppConfig
from app.utils.types import Quad, RegionType, TextRegion
from app.inpainting.masking import build_text_mask
from app.inpainting.backends import OpenCVBackend
from app.typography.fitting import AutoFitter
from app.rendering.renderer import ArabicRenderer

cfg = AppConfig.load()

# --- build a synthetic page with one white bubble -----------------------
img = np.full((600, 900, 3), 200, dtype=np.uint8)
cv2.circle(img, (450, 300), 180, (255, 255, 255), -1)
cv2.rectangle(img, (380, 420), (420, 470), (255, 255, 255), -1)  # bubble tail
cv2.putText(img, 'What is', (330, 280), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 3)
cv2.putText(img, 'going on?', (300, 330), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 3)

region = TextRegion(
    id=1,
    quad=Quad([(300, 240), (600, 240), (600, 350), (300, 350)]),
    english='What is going on?',
    arabic='ما الذي يجري هنا؟',
    region_type=RegionType.DIALOGUE,
    bg_color=(255, 255, 255),
    text_color=(0, 0, 0),
)

# --- remove text -----------------------------------------------------------
mask = build_text_mask(img, [region], dilate_px=3)
cleaned = OpenCVBackend().inpaint(img, mask)
bubble_roi = cleaned[150:450, 300:600]
dark_ratio_before = (bubble_roi < 60).mean()

# --- typeset Arabic ---------------------------------------------------------
fitter = AutoFitter(cfg)
region.bubble_polygon = np.array([[270, 120], [630, 120], [630, 480], [270, 480]])
layout = fitter.fit(region)
assert layout is not None, 'auto-fit returned no layout'
region.layout = layout

renderer = ArabicRenderer(cfg)
out = renderer.render_page(cleaned, [region])

# --- verify -----------------------------------------------------------------
assert out.shape == img.shape
roi = out[150:450, 300:600]
dark_ratio_after = (roi < 60).mean()
assert dark_ratio_after > dark_ratio_before, 'no Arabic glyphs rendered'
assert layout.max_line_width <= 360, f'text wider than bubble: {layout.max_line_width}'
print(f'render OK: font={layout.font_size}px lines={len(layout.lines)} '
      f'dark_before={dark_ratio_before:.3f} dark_after={dark_ratio_after:.3f}')

# save artifacts for visual inspection
os.makedirs('output_test', exist_ok=True)
cv2.imwrite('output_test/synth_original.png', img)
cv2.imwrite('output_test/synth_cleaned.png', cleaned)
cv2.imwrite('output_test/synth_arabic.png', out)
print('RENDER_TEST_PASSED')
