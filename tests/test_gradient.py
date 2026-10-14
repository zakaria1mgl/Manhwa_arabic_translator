"""Realistic test: gradient bubble interior (like the user's manhwa),
verifies NO white-rectangle patch and bold, bubble-filling Arabic."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np, cv2
from PIL import Image

from app.config import AppConfig
from app.utils.types import Quad, RegionType, TextRegion
from app.ocr.reading_order import sort_reading_order, estimate_bubbles, merge_bubble_lines, classify_regions
from app.inpainting.masking import build_text_mask, uniform_bubble_fill
from app.inpainting.backends import OpenCVBackend
from app.typography.fitting import AutoFitter
from app.rendering.renderer import ArabicRenderer

cfg = AppConfig.load()
rng = np.random.default_rng(3)

# page with soft cloudy background
art = rng.integers(80, 220, size=(900, 700, 3), dtype=np.uint8)
art = cv2.GaussianBlur(art, (51, 51), 20)
img = art.copy()

# bubble 1 with a subtle vertical gradient interior (not flat white!)
cv2.ellipse(img, (350, 180), (260, 150), 0, 0, 360, (255, 255, 255), -1)
grad = np.linspace(248, 232, 300, dtype=np.uint8)[:, None]
mask_ellipse = np.zeros((900, 700), dtype=np.uint8)
cv2.ellipse(mask_ellipse, (350, 180), (258, 148), 0, 0, 360, 255, -1)
region_px = mask_ellipse > 0
ys, xs = np.where(region_px)
# vertical gradient inside the ellipse: brightness depends on y
yy0, yy1 = ys.min(), ys.max()
for c in range(3):
    ch = img[..., c]
    t = (ys - yy0) / max(1, (yy1 - yy0))       # 0 top -> 1 bottom
    vals = (248 - 16 * t - c * 2).astype(np.uint8)
    ch[ys, xs] = vals
# 5 lines of text
lines1 = ["IF THERE'S TROUBLE,", "YOU CAN ENTIRELY USE", "A JADE SLIP TO SEND",
          "A MESSAGE. I'D SUGGEST", "YOU USE THE ABILITY"]
y = 110
for ln in lines1:
    cv2.putText(img, ln, (140, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 2)
    y += 34

# rebuild ellipse mask gradient test image (the loop above used img directly)
regions = []
lid = 0
for j, ln in enumerate(lines1):
    yy = 96 + j * 34
    regions.append(TextRegion(id=lid, quad=Quad([(130, yy), (580, yy), (580, yy + 26), (130, yy + 26)]),
                              english=ln, ocr_conf=0.9, bg_color=(242, 241, 243)))
    lid += 1
regions = sort_reading_order(regions)
estimate_bubbles(regions, img.shape, image=img)
classify_regions(regions, img.shape)
merge_bubble_lines(regions)
primaries = [r for r in regions if r.english and r.merged_into is None]
primaries[0].arabic = 'إذا كان هناك متاعب، يمكنك استخدام انزلاق اليشم لإرسال رسالة. أقترح أن تستخدم القدرة على إحضار قارب صغير.'
primaries[0].region_type = RegionType.DIALOGUE
active = regions  # full region list (primaries + satellites) as in the pipeline

# 1) gradient preserved by fill?
filled = uniform_bubble_fill(img, active)
# sample inside the bubble away from text: top area vs bottom area of quad span.
# (the whole bubble interior may now be painted, so the sampled values must
#  track the ORIGINAL gradient level at that height, not a flat white wall:
#  original top strip ~244, bottom strip ~235)
top = filled[80:100, 300:400].mean()
bot = filled[240:258, 300:400].mean()
print(f'gradient preserved: top={top:.1f} bottom={bot:.1f} (expect top>bot, both ~235-245)')
assert top - bot > 4, 'fill flattened the gradient -> white patch effect'
assert 232 <= top <= 250 and 222 <= bot <= 242, \
    f'fill drifted from original gradient: top={top:.1f} bot={bot:.1f}'

# 2) no white RECTANGLE: check corners of the quad span (outside ellipse but
#    inside span corners) keep artwork, not white
cx0, cy0, cx1, cy1 = primaries[0].merged_span
corner = filled[cy0:cy0 + 8, cx0:cx0 + 8].mean()
print(f'corner (outside ellipse) mean={corner:.1f} (artwork ~120-200, not ~250)')
assert corner < 230, 'fill stamped outside the text area -> rectangle patch'

# 3) inpaint + render Arabic: bold and bubble-filling
mask = build_text_mask(filled, active, dilate_px=3)
inpainted = OpenCVBackend().inpaint(filled, mask)
roi = inpainted[100:250, 140:560]
dark = (cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) < 60).mean()
print(f'text removed: dark={dark:.4f}')
assert dark < 0.02

fitter = AutoFitter(cfg)
layout = fitter.fit(active[0])
assert layout is not None
print(f'font={layout.font_size}px lines={len(layout.lines)} block={layout.max_line_width}x{layout.total_height}')
assert layout.font_size >= 26, f'font too small again: {layout.font_size}'
active[0].layout = layout
renderer = ArabicRenderer(cfg)
out = renderer.render_page(inpainted, active)

os.makedirs('output_test', exist_ok=True)
cv2.imwrite('output_test/grad_original.png', img)
cv2.imwrite('output_test/grad_result.png', out)
print('GRADIENT_TEST_PASSED')
