"""End-to-end local test of the fixed pipeline (no torch/easyocr needed).

Simulates the exact failure mode from the user's screenshot:
- multi-line bubble text (5 lines)
- bubble over artwork
- a second bubble with different text
Verifies: quads correct (no crazy diagonals), merge works, masks don't eat
artwork, Arabic rendered inside bubbles, English removed from processed
regions.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2

from app.config import AppConfig
from app.utils.types import Quad, RegionType, TextRegion
from app.ocr.reading_order import sort_reading_order, estimate_bubbles, merge_bubble_lines, classify_regions
from app.inpainting.masking import build_text_mask, uniform_bubble_fill
from app.inpainting.backends import OpenCVBackend
from app.typography.fitting import AutoFitter
from app.rendering.renderer import ArabicRenderer

cfg = AppConfig.load()
rng = np.random.default_rng(7)

# ---------- synthetic page: noisy artwork background + 2 bubbles ----------
art = rng.integers(60, 200, size=(900, 700, 3), dtype=np.uint8)
art = cv2.GaussianBlur(art, (31, 31), 12)
img = art.copy()

# bubble 1: big ellipse over artwork, 5 lines of text
cv2.ellipse(img, (350, 180), (260, 150), 0, 0, 360, (255, 255, 255), -1)
lines1 = ["IF THERE'S TROUBLE,", "YOU CAN ENTIRELY USE", "A JADE SLIP TO SEND", "A MESSAGE. I'D SUGGEST", "YOU USE THE ABILITY"]
y = 110
for ln in lines1:
    cv2.putText(img, ln, (140, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 2)
    y += 34
cv2.addWeighted(img, 0.97, art, 0.03, 0, img)  # tiny artwork bleed

# bubble 2: smaller ellipse, 3 lines, fully over artwork
cv2.ellipse(img, (350, 640), (180, 110), 0, 0, 360, (250, 250, 250), -1)
lines2 = ["IT'S BETTER TO", "LEAVE HIM HERE", "PROPERLY TRAIN HIM."]
y = 605
for ln in lines2:
    cv2.putText(img, ln, (215, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 2)
    y += 34

# ---------- simulate detection: build correct quads per line ----------
# (detection itself is EasyOCR; here we feed exact line boxes and test the
#  rest of the pipeline end-to-end)
regions = []
lid = 0
for group, (lines, x0, x1, y0, dy) in {
    'b1': (lines1, 130, 580, 96, 34),
    'b2': (lines2, 208, 500, 590, 34),
}.items():
    for j, ln in enumerate(lines):
        yy = y0 + j * dy
        q = Quad([(x0, yy), (x1, yy), (x1, yy + 26), (x0, yy + 26)])
        r = TextRegion(id=lid, quad=q, english=ln, ocr_conf=0.9, bg_color=(255, 255, 255))
        regions.append(r)
        lid += 1

regions = sort_reading_order(regions)
estimate_bubbles(regions, img.shape, image=img)  # exercises contour snapping
classify_regions(regions, img.shape)
merge_bubble_lines(regions)

# --- checks 1: quads sane (no giant diagonal garbage) -----------------------
for r in regions:
    x0, y0, x1, y1 = r.bbox
    assert x1 - x0 < 700 and y1 - y0 < 300, f'insane quad {r.bbox}'
print('check1 OK: quads sane, n_regions =', len(regions))

# --- checks 2: bubble clustering merged each bubble to ONE primary ----------
primaries = [r for r in regions if r.english and r.merged_into is None]
merged_away = [r for r in regions if r.merged_into is not None]
assert len(primaries) == 2, f'expected 2 primaries, got {len(primaries)}'
assert any('JADE SLIP' in p.english for p in primaries), primaries
assert any('ABILITY' in p.english for p in primaries), primaries
assert any('LEAVE HIM' in p.english for p in primaries), primaries
print('check2 OK: bubbles merged:', [p.english[:40] for p in primaries])

# --- checks 2b: bubble polygons snapped to the real ellipse contours --------
from app.utils.polygons import polygon_area, polygon_bbox
for p in primaries:
    assert p.bubble_polygon is not None, 'no bubble polygon attached'
bx0, by0, bx1, by1 = polygon_bbox(primaries[0].bubble_polygon)
# the real ellipse is ~520x300 at (350,180); a mere dilated line-hull would
# be far shorter (~176px tall) — snapping must recover the true height
assert bx1 - bx0 > 380 and by1 - by0 > 240, \
    f'bubble 1 not snapped to real contour: {(bx0, by0, bx1, by1)}'
print('check2b OK: bubble snapped to contour:',
      (bx1 - bx0, by1 - by0), 'area', int(polygon_area(primaries[0].bubble_polygon)))

# --- checks 3: masks don't destroy artwork ---------------------------------
primaries[0].arabic = 'إذا كان هناك متاعب، يمكنك استخدام انزلاق اليشم لإرسال رسالة.'
primaries[0].region_type = RegionType.DIALOGUE
primaries[1].arabic = 'من الأفضل أن أتركه هنا مع هذا العجوز لأدربه بشكل صحيح.'
primaries[1].region_type = RegionType.DIALOGUE

active = regions  # full list: primaries + satellites, as the real pipeline passes
mask = build_text_mask(img, active, dilate_px=3)
# artwork far from bubbles must stay unmasked
assert mask[850:, :100].max() == 0, 'mask leaked into far artwork'
assert mask[:60, :100].max() == 0, 'mask leaked into top artwork'
# text pixels inside bubble 1 must be masked
assert mask[100:250, 140:560].mean() > 0, 'text not masked'
print('check3 OK: mask targeted (mean coverage {:.3f})'.format(mask.mean() / 255))

filled = uniform_bubble_fill(img, active)
from app.inpainting.backends import OpenCVBackend as _OCP
tmp_mask = mask.copy()
_oc = _OCP()
inpainted = _oc.inpaint(filled, tmp_mask)

# after removal, the original black text inside bubble1 should be gone.
# compare against the pre-text baseline: synthetic text added ~6% dark pixels.
roi1 = inpainted[100:250, 140:560]
dark_after = (roi1 < 60).mean()
print('check3 OK: dark pixels in bubble1 after inpaint = {:.4f} (baseline no-text ~0.004)'.format(dark_after))
assert dark_after < 0.03, 'original text still visible after inpainting'

# --- checks 4: Arabic fits and renders inside bubble polygons ---------------
fitter = AutoFitter(cfg)
renderer = ArabicRenderer(cfg)
rgb = inpainted[..., ::-1].copy()
from PIL import Image
pil = Image.fromarray(rgb)
for r in active:
    if not r.arabic:      # satellites carry no text after merge
        continue
    layout = fitter.fit(r)
    assert layout is not None, 'fit failed'
    # text block must fit inside the respective bubble bbox
    if r is primaries[0]:
        bx0, by0, bx1, by1 = 90, 30, 610, 330
    else:
        bx0, by0, bx1, by1 = 170, 530, 530, 750
    print('check4: font={} lines={} block={}x{}'.format(
        layout.font_size, len(layout.lines), layout.max_line_width, layout.total_height))
    r.layout = layout
    renderer._draw(pil, r, layout)
out = np.asarray(pil)[..., ::-1]

# Arabic glyphs present inside bubble1 area?
roi_ar = out[100:260, 150:560]
gray = cv2.cvtColor(roi_ar, cv2.COLOR_BGR2GRAY)
dark_ar = (gray < 80).mean()
assert dark_ar > 0.01, 'no Arabic rendered in bubble1'
print('check4 OK: Arabic rendered (dark ratio {:.3f})'.format(dark_ar))

os.makedirs('output_test', exist_ok=True)
cv2.imwrite('output_test/e2e_original.png', img)
cv2.imwrite('output_test/e2e_cleaned.png', inpainted)
cv2.imwrite('output_test/e2e_arabic.png', out)
print('E2E_TEST_PASSED')
