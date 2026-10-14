from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from ..config import AppConfig
from ..utils.polygons import polygon_area
from ..utils.types import RegionType, TextRegion
from .fonts import load_font
from .shaping import shape_arabic, wrap_arabic

log = logging.getLogger(__name__)


@dataclass
class LineLayout:
    text: str
    y_top: int
    width: int


@dataclass
class TextLayout:
    font_name: str
    font_size: int
    lines: List[str]
    line_height: int
    total_height: int
    max_line_width: int
    color: Tuple[int, int, int] = (20, 20, 20)
    stroke_color: Optional[Tuple[int, int, int]] = None
    stroke_width: int = 0
    anchor_xy: Tuple[int, int] = (0, 0)   # top-left of the text block
    angle: float = 0.0                    # rotation for SFX / rotated text
    # fitting constraints (kept so the editor can refit at a given size)
    max_w: int = 0
    max_h: int = 0
    line_spacing: float = 1.25

    def to_dict(self):
        return {
            "font": self.font_name, "size": self.font_size,
            "lines": self.lines, "color": list(self.color),
            "anchor": list(self.anchor_xy), "angle": self.angle,
        }


class AutoFitter:
    """Bubble-aware automatic Arabic text fitting.

    Given the region's placement polygon, find the largest font size whose
    wrapped text fits strictly inside with padding, using binary search.
    """

    def __init__(self, cfg: AppConfig):
        self.cfg = cfg

    def fit(self, region: TextRegion) -> Optional[TextLayout]:
        if not region.arabic:
            return None
        poly = region.placement_polygon
        # a merged multi-line cluster: constrain to the union span of its lines
        span = getattr(region, 'merged_span', None)
        if span and region.bubble_polygon is None:
            x0, y0, x1, y1 = span
            poly = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.int32)
        x0, y0, x1, y1 = _poly_bbox(poly)
        w, h = x1 - x0, y1 - y0
        if w < 8 or h < 8:
            return None
        font_name = self._font_for(region)
        pad = max(2, int(h * self.cfg.get("typography", "padding_ratio", default=0.06)))
        max_w, max_h = w - 2 * pad, h - 2 * pad
        if max_w < 10 or max_h < 10:
            pad = 2
            max_w, max_h = max(10, w - 2 * pad), max(10, h - 2 * pad)
        # round/oval bubbles: the inscribed rectangle is much smaller than the
        # bounding box — shrink so text never pokes out of the bubble corners
        area_ratio = polygon_area(poly) / float(max(1, w * h))
        if region.bubble_polygon is not None and area_ratio < 0.85:
            max_w, max_h = int(max_w * 0.80), int(max_h * 0.80)
        # vertical text is typeset horizontally then rotated 90°: the fitting
        # box axes must be swapped or the size search optimizes the wrong one
        rotate_out = region.region_type in (RegionType.SFX, RegionType.SIGN) \
            and abs(region.orientation) > 4
        if rotate_out and abs(region.orientation) > 45:
            max_w, max_h = max_h, max_w

        min_s = int(self.cfg.get("typography", "min_font_size", default=10))
        max_s = int(self.cfg.get("typography", "max_font_size", default=96))
        # NOTE: no artificial per-word cap — the binary search below already
        # finds the largest size whose wrapped text fits (max_w, max_h).
        # (The old 'bubble height / word count' heuristic crushed text.)

        spacing = float(self.cfg.get("typography", "line_spacing", default=1.25))
        best: Optional[TextLayout] = None
        lo, hi = min_s, max_s
        while lo <= hi:
            mid = (lo + hi) // 2
            layout = self._try_size(region, font_name, mid, max_w, max_h, spacing)
            if layout is not None:
                best = layout
                lo = mid + 1
            else:
                hi = mid - 1
        if best is None:
            # squeeze: smallest allowed size, allow overflow flags
            best = self._try_size(region, font_name, min_s, max_w, max_h, spacing,
                                  force=True)
            if best is not None:
                region.qa_flags.append("text_overflow_risk")
        # center inside the polygon bbox
        if best is not None:
            cx = (x0 + x1) // 2
            cy = (y0 + y1) // 2
            best.anchor_xy = (cx - best.max_line_width // 2,
                              cy - best.total_height // 2)
            best.angle = region.orientation if rotate_out else 0.0
        return best

    # ------------------------------------------------------------------
    def _try_size(self, region: TextRegion, font_name: str, size: int,
                  max_w: int, max_h: int, spacing: float,
                  force: bool = False) -> Optional[TextLayout]:
        try:
            font = load_font(font_name, size,
                             self.cfg.get("typography", "font_download_dir", default="fonts"))
        except FileNotFoundError:
            return None
        # glyph-coverage guard: if this font can't render the shaped text
        # (e.g. Cairo misses many presentation forms -> tofu), fall back to
        # Noto Naskh Arabic which has near-complete coverage.
        from .fonts import has_glyphs
        if not has_glyphs(font, region.arabic):
            try:
                font = load_font("Noto Naskh Arabic", size,
                                 self.cfg.get("typography", "font_download_dir", default="fonts"))
                font_name = "Noto Naskh Arabic"
            except FileNotFoundError:
                return None
        wrapped = wrap_arabic(region.arabic, font, max_w, width_cache={})
        if not wrapped:
            return None
        line_h = int(size * spacing)
        total_h = line_h * len(wrapped)
        line_w = max(int(_w(font, ln)) for ln in wrapped)
        ok = line_w <= max_w and total_h <= max_h
        if not ok and not force:
            return None
        stroke = 0
        if region.region_type == RegionType.SFX:
            stroke = max(1, int(size * self.cfg.get("typography", "stroke_ratio", default=0.045)))
        color = self._text_color(region)
        return TextLayout(
            font_name=font_name, font_size=size, lines=wrapped,
            line_height=line_h, total_height=total_h,
            max_line_width=line_w, color=color,
            stroke_color=region.stroke_color, stroke_width=stroke,
            max_w=max_w, max_h=max_h, line_spacing=spacing,
        )

    def refit_at_size(self, region: TextRegion, size: int) -> Optional[TextLayout]:
        """Re-wrap the region's arabic at an explicit font size, reusing the
        constraints stored by fit(). Used by the editor's live override."""
        prev = region.layout
        if prev is None or prev.max_w <= 0 or not region.arabic:
            return None
        layout = self._try_size(region, prev.font_name, int(size),
                                prev.max_w, prev.max_h, prev.line_spacing,
                                force=True)
        if layout is not None:
            # keep the same anchor/box; recenter vertically around old center
            layout.anchor_xy = (prev.anchor_xy[0] + (prev.max_line_width - layout.max_line_width) // 2,
                                prev.anchor_xy[1] + (prev.total_height - layout.total_height) // 2)
            layout.angle = prev.angle
        return layout

    def _font_for(self, region: TextRegion) -> str:
        fonts = self.cfg.get("typography", "fonts", default={}) or {}
        if region.region_type == RegionType.SFX:
            return fonts.get("sfx", "Noto Sans Arabic")
        if region.region_type in (RegionType.NARRATION, RegionType.CAPTION):
            return fonts.get("narration", "Noto Naskh Arabic")
        return fonts.get("dialogue", "Noto Naskh Arabic")

    def _text_color(self, region: TextRegion) -> Tuple[int, int, int]:
        if region.region_type in (RegionType.SFX, RegionType.SIGN):
            # SFX usually keeps its original colour
            return region.text_color or (20, 20, 20)
        cfgc = self.cfg.get("typography", "text_color_dialogue", default=[26, 26, 26])
        return tuple(cfgc) if cfgc else (26, 26, 26)


def _w(font, line: str) -> float:
    shaped = shape_arabic(line)
    try:
        return font.getlength(shaped)
    except AttributeError:
        b = font.getbbox(shaped)
        return b[2] - b[0]


def _poly_bbox(poly: np.ndarray) -> Tuple[int, int, int, int]:
    pts = np.asarray(poly).reshape(-1, 2)
    return (int(pts[:, 0].min()), int(pts[:, 1].min()),
            int(pts[:, 0].max()), int(pts[:, 1].max()))
