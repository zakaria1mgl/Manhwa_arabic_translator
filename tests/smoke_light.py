import sys, os, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# 1) config
from app.config import AppConfig
cfg = AppConfig.load()
assert cfg.get('ocr', 'min_confidence') == 0.35
cfg.set('x', 'device', 'test'); assert cfg.get('device', 'test') == 'x'

# 2) types / polygons
from app.utils.types import Quad, TextRegion, RegionType
q = Quad([(0, 0), (100, 0), (100, 40), (0, 40)])
r = TextRegion(id=1, quad=q, english='HELLO', arabic='مرحبا', region_type=RegionType.SFX)
d = r.to_dict(); assert d['confidence'] == 0.0 and d['type'] == 'sfx'

from app.utils.polygons import rect_to_poly, polygon_area, enclosing_polygon, dilate_polygon
p = rect_to_poly(0, 0, 50, 20); assert polygon_area(p) == 1000
h = enclosing_polygon([p, rect_to_poly(10, 30, 60, 50)]); assert h is not None
dp = dilate_polygon(p, 3); assert polygon_area(dp) >= 1000

# 3) sfx
from app.translation.sfx import lookup_sfx
assert lookup_sfx('BAAAM!!') is not None
assert lookup_sfx('THUD.') == 'طَـخ!'
assert lookup_sfx('What are you doing here?') is None

# 4) glossary
fd, path = tempfile.mkstemp(suffix='.json'); os.close(fd); os.unlink(path)
from app.glossary.glossary import Glossary
g = Glossary(path); g.set('Moon Kingdom', 'مملكة القمر')
assert g.get('moon kingdom') == 'مملكة القمر'
assert 'Moon Kingdom' in g.as_prompt_block()
g2 = Glossary(path); assert g2.get('Moon Kingdom') == 'مملكة القمر'

# 5) arabic shaping
from app.typography.shaping import shape_arabic, normalize_arabic
raw = 'ما الذي تفعله هنا؟'
sh = shape_arabic(raw)
assert sh != raw and any('\u0600' <= c <= '\u06FF' for c in sh)
assert normalize_arabic('ما الذي تفعله هنا? حسنا; لا, شيء') == \
    'ما الذي تفعله هنا؟ حسنا؛ لا، شيء'
# punctuation NOT touching Arabic is preserved (numbers, latin names)
assert normalize_arabic('وجدت 1,000 عملة') == 'وجدت 1,000 عملة'
assert normalize_arabic('يا Rock, Paper!') == 'يا Rock, Paper!'

# 6) chapter context
from app.context.chapter_context import ChapterContext, CharacterProfile
with tempfile.TemporaryDirectory() as td:
    ctx = ChapterContext('ch1', td)
    ctx.upsert_character(CharacterProfile(name='Kang', gender='male', personality='aggressive'))
    ctx.story_summary = 'test'
    ctx.save()
    ctx2 = ChapterContext('ch1', td)
    assert ctx2.characters['Kang'].gender == 'male'
    assert 'Kang' in ctx2.character_block()

# 7) reading order + classification
from app.ocr.reading_order import sort_reading_order, _looks_like_sfx
rs = [
    TextRegion(id=0, quad=Quad([(400, 100), (500, 100), (500, 140), (400, 140)]), english='B'),
    TextRegion(id=0, quad=Quad([(100, 100), (200, 100), (200, 140), (100, 140)]), english='A'),
    TextRegion(id=0, quad=Quad([(100, 200), (200, 200), (200, 240), (100, 240)]), english='C'),
]
ordered = sort_reading_order(rs, panel_order='ltr')
assert [r.english for r in ordered] == ['A', 'B', 'C'], [r.english for r in ordered]
assert _looks_like_sfx('GRAAHH!')
assert not _looks_like_sfx('What are you doing here?')

# 8) LLM json extraction
from app.translation.llm_client import extract_json
assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
assert extract_json('blah {"x": [1, 2]} trailing') == {"x": [1, 2]}

# 9) masking + inpainting fallback (opencv)
import numpy as np
from app.inpainting.masking import build_text_mask, uniform_bubble_fill
img = np.full((200, 400, 3), 255, dtype=np.uint8)
import cv2
cv2.putText(img, 'HELLO WORLD', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
r1 = TextRegion(id=1, quad=Quad([(15, 25), (230, 25), (230, 75), (15, 75)]),
                english='HELLO WORLD', bg_color=(255, 255, 255))
mask = build_text_mask(img, [r1], dilate_px=2)
assert mask.max() == 255 and mask.mean() > 0
from app.inpainting.backends import OpenCVBackend
inp = OpenCVBackend().inpaint(img, mask)
assert inp.shape == img.shape
filled = uniform_bubble_fill(img, [r1])
assert filled.shape == img.shape

# 10) QA scoring
from app.quality_control.qa import region_qa
r3 = TextRegion(id=3, quad=q, english='Hi', arabic='مرحبا', ocr_conf=0.9)
s = region_qa(r3, cfg); assert s > 0.7
r4 = TextRegion(id=4, quad=q, english='Hi', arabic='', ocr_conf=0.2)
s4 = region_qa(r4, cfg); assert s4 < 0.5 and 'missing_translation' in r4.qa_flags

# 11) auto-fit (skipped gracefully without fonts)
from app.typography.fitting import AutoFitter
r2 = TextRegion(id=2, quad=Quad([(10, 10), (300, 10), (300, 90), (10, 90)]),
                english='Hello there', arabic='مرحبا بك هنا يا صديقي العزيز',
                region_type=RegionType.DIALOGUE, bg_color=(255, 255, 255))
try:
    layout = AutoFitter(cfg).fit(r2)
    if layout:
        assert layout.max_line_width <= 300 and layout.lines
        print('fit OK size=', layout.font_size, 'lines=', len(layout.lines))
    else:
        print('fit skipped (no font available offline)')
except FileNotFoundError:
    print('fit skipped (no font available offline)')

print('ALL_SMOKE_TESTS_PASSED')
