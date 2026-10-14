from __future__ import annotations

import logging
import math
from typing import List, Optional

import numpy as np
from PIL import Image, ImageDraw

from ..config import AppConfig
from ..utils.types import PageResult, TextRegion
from ..typography.fitting import AutoFitter, TextLayout
from ..typography.fonts import load_font
from ..typography.shaping import shape_arabic

log = logging.getLogger(__name__)


class ArabicRenderer:
    """Draw shaped Arabic text layouts onto the inpainted page."""

    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.fitter = AutoFitter(cfg)

    def render_page(self, image_bgr: np.ndarray,
                    regions: List[TextRegion]) -> np.ndarray:
        rgb = image_bgr[..., ::-1].copy()
        pil = Image.fromarray(rgb)
        for region in regions:
            if not region.arabic:
                continue
            layout = region.layout or self.fitter.fit(region)
            region.layout = layout
            if layout is None:
                region.qa_flags.append("layout_failed")
                continue
            self._draw(pil, region, layout)
        return np.asarray(pil)[..., ::-1]  # back to BGR

    # ------------------------------------------------------------------
    def _draw(self, pil: Image.Image, region: TextRegion, layout: TextLayout) -> None:
        font = load_font(layout.font_name, layout.font_size,
                         self.cfg.get("typography", "font_download_dir", default="fonts"))
        if layout.angle:
            self._draw_rotated(pil, region, layout, font)
            return
        draw = ImageDraw.Draw(pil)
        anchor = layout.anchor_xy
        y = anchor[1]
        for line in layout.lines:
            shaped = shape_arabic(line)
            lw = int(_len(font, shaped))
            x = anchor[0] + (layout.max_line_width - lw) // 2  # centered
            # RTL: PIL draws the display-ordered string; centering keeps RTL sane
            draw.text((x, y), shaped, font=font, fill=layout.color,
                      stroke_width=layout.stroke_width,
                      stroke_fill=layout.stroke_color or layout.color)
            y += layout.line_height

    def _draw_rotated(self, pil: Image.Image, region: TextRegion,
                      layout: TextLayout, font) -> None:
        # render text block to a transparent tile, rotate, paste centered
        pad = 8 + layout.stroke_width
        w = layout.max_line_width + 2 * pad
        h = layout.total_height + 2 * pad
        tile = Image.new("RGBA", (max(1, w), max(1, h)), (0, 0, 0, 0))
        d = ImageDraw.Draw(tile)
        y = pad
        for line in layout.lines:
            shaped = shape_arabic(line)
            lw = int(_len(font, shaped))
            x = pad + (layout.max_line_width - lw) // 2
            d.text((x, y), shaped, font=font, fill=(*layout.color, 255),
                   stroke_width=layout.stroke_width,
                   stroke_fill=(*(layout.stroke_color or layout.color), 255))
            y += layout.line_height
        tile = tile.rotate(-layout.angle, expand=True, resample=Image.BICUBIC)
        cx = layout.anchor_xy[0] + layout.max_line_width // 2
        cy = layout.anchor_xy[1] + layout.total_height // 2
        pil.paste(tile, (int(cx - tile.width / 2), int(cy - tile.height / 2)), tile)


def _len(font, shaped: str) -> float:
    try:
        return font.getlength(shaped)
    except AttributeError:  # pragma: no cover
        b = font.getbbox(shaped)
        return b[2] - b[0]
