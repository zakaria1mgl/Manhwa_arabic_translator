from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

Point = Tuple[float, float]


class RegionType(str, Enum):
    DIALOGUE = "dialogue"
    THOUGHT = "thought"
    NARRATION = "narration"
    CAPTION = "caption"
    SFX = "sfx"
    SIGN = "sign"
    UI = "ui"
    UNKNOWN = "unknown"


@dataclass
class Quad:
    """Axis-degenerate quadrilateral around a text line (tl, tr, br, bl)."""
    points: List[Point]

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))

    @property
    def center(self) -> Point:
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        return sum(xs) / len(xs), sum(ys) / len(ys)

    @property
    def width(self) -> float:
        x0, _, x1, _ = self.bbox
        return x1 - x0

    @property
    def height(self) -> float:
        _, y0, _, y1 = self.bbox
        return y1 - y0

    @property
    def polygon(self) -> np.ndarray:
        return np.array(self.points, dtype=np.int32)

    @property
    def orientation(self) -> float:
        """Rotation angle in degrees (0 = horizontal)."""
        (x0, y0), (x1, y1) = self.points[0], self.points[1]
        import math
        return math.degrees(math.atan2(y1 - y0, x1 - x0))

    def scaled(self, sx: float, sy: Optional[float] = None) -> "Quad":
        sy = sy or sx
        return Quad([(p[0] * sx, p[1] * sy) for p in self.points])

    def to_list(self) -> List[List[float]]:
        return [[float(x), float(y)] for x, y in self.points]


@dataclass
class TextRegion:
    id: int
    quad: Quad
    region_type: RegionType = RegionType.UNKNOWN

    # OCR
    english: str = ""
    ocr_conf: float = 0.0
    orientation: float = 0.0

    # context
    speaker: Optional[str] = None
    speaker_gender: Optional[str] = None
    bubble_cluster: List[int] = field(default_factory=list)
    merged_into: Optional[int] = None
    merged_span: Optional[tuple] = None  # (x0,y0,x1,y1) span of a merged cluster

    # geometry of the containing bubble (if any)
    bubble_polygon: Optional[np.ndarray] = None

    # colours sampled from the original
    bg_color: Optional[Tuple[int, int, int]] = None
    text_color: Optional[Tuple[int, int, int]] = None
    stroke_color: Optional[Tuple[int, int, int]] = None

    # translation
    arabic: str = ""
    translation_backend: str = ""
    review_issues: List[str] = field(default_factory=list)
    sfx_semantics: str = ""

    # rendering (filled by typography.fitting)
    layout: Optional[Any] = None

    # QA
    qa_flags: List[str] = field(default_factory=list)
    qa_score: float = 0.0

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        return self.quad.bbox

    @property
    def placement_polygon(self) -> np.ndarray:
        """Polygon that constrains the Arabic text placement."""
        if self.bubble_polygon is not None and len(self.bubble_polygon) >= 3:
            return self.bubble_polygon
        return self.quad.polygon

    def to_dict(self) -> Dict[str, Any]:
        from .polygons import polygon_to_list
        return {
            "id": self.id,
            "type": self.region_type.value,
            "bbox": list(self.quad.bbox),
            "polygon": self.quad.to_list(),
            "bubble": polygon_to_list(self.bubble_polygon),
            "english": self.english,
            "arabic": self.arabic,
            "confidence": round(self.ocr_conf, 3),
            "speaker": self.speaker,
            "orientation": round(self.orientation, 1),
            "qa_flags": self.qa_flags,
            "qa_score": round(self.qa_score, 3),
            "review_issues": self.review_issues,
        }


@dataclass
class PageResult:
    image_path: str = ""
    width: int = 0
    height: int = 0
    regions: List[TextRegion] = field(default_factory=list)
    page_summary: str = ""
    page_qa_score: float = 0.0
    output_path: str = ""
    cleaned_path: str = ""      # inpainted base (for the single-bubble editor)
    debug_overlay_path: str = ""
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "image": self.image_path,
            "width": self.width,
            "height": self.height,
            "page_qa_score": round(self.page_qa_score, 3),
            "summary": self.page_summary,
            "regions": [r.to_dict() for r in self.regions],
        }


class CancelledError(RuntimeError):
    """Raised when the user requests cancellation between pipeline stages."""
