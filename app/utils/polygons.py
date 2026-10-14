from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

import numpy as np

PolyLike = np.ndarray


def polygon_area(poly: PolyLike) -> float:
    if poly is None or len(poly) < 3:
        return 0.0
    x = poly[:, 0].astype(float)
    y = poly[:, 1].astype(float)
    return float(0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def polygon_to_list(poly: Optional[PolyLike]) -> Optional[List[List[float]]]:
    if poly is None:
        return None
    return [[float(p[0]), float(p[1])] for p in np.asarray(poly).reshape(-1, 2)]


def polygon_bbox(poly: PolyLike) -> Tuple[int, int, int, int]:
    pts = np.asarray(poly).reshape(-1, 2)
    return (int(pts[:, 0].min()), int(pts[:, 1].min()),
            int(pts[:, 0].max()), int(pts[:, 1].max()))


def point_in_polygon(pt: Tuple[float, float], poly: PolyLike) -> bool:
    from cv2 import pointPolygonTest
    return pointPolygonTest(np.asarray(poly, dtype=np.int32).reshape(-1, 1, 2),
                            (float(pt[0]), float(pt[1])), False) >= 0


def polygon_centroid(poly: PolyLike) -> Tuple[float, float]:
    pts = np.asarray(poly).reshape(-1, 2).astype(float)
    if len(pts) == 0:
        return (0.0, 0.0)
    return (float(pts[:, 0].mean()), float(pts[:, 1].mean()))


def dilate_polygon(poly: PolyLike, px: int) -> PolyLike:
    """Approximate polygon dilation via convex hull of offset vertices."""
    import cv2
    pts = np.asarray(poly, dtype=np.int32).reshape(-1, 1, 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    # cheap approach: draw + dilate on a padded canvas, then hull
    x, y, x2, y2 = polygon_bbox(pts.reshape(-1, 2))
    pad = px + 8
    w, h = x2 - x + 1 + 2 * pad, y2 - y + 1 + 2 * pad
    canvas = np.zeros((h, w), dtype=np.uint8)
    shifted = pts - np.array([[[x - pad, y - pad]]])
    cv2.fillPoly(canvas, [shifted.reshape(-1, 1, 2)], 255)
    canvas = cv2.dilate(canvas, kernel)
    contours, _ = cv2.findContours(canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.asarray(poly, dtype=np.int32).reshape(-1, 2)
    hull = cv2.convexHull(contours[0]).reshape(-1, 2)
    return hull + np.array([x - pad, y - pad])


def rect_to_poly(x0: int, y0: int, x1: int, y1: int) -> PolyLike:
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.int32)


def rects_overlap(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> bool:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return not (ax1 <= bx0 or bx1 <= ax0 or ay1 <= by0 or by1 <= ay0)


def overlap_ratio(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    if not rects_overlap(a, b):
        return 0.0
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix = min(ax1, bx1) - max(ax0, bx0)
    iy = min(ay1, by1) - max(ay0, by0)
    inter = ix * iy
    area_b = max(1, (bx1 - bx0) * (by1 - by0))
    return inter / area_b


def enclosing_polygon(quads: Iterable[PolyLike]) -> Optional[PolyLike]:
    """Convex hull around a set of text polygons -> approximate bubble outline."""
    import cv2
    all_pts = []
    for q in quads:
        all_pts.extend(np.asarray(q).reshape(-1, 2).tolist())
    if len(all_pts) < 3:
        return None
    pts = np.array(all_pts, dtype=np.int32)
    hull = cv2.convexHull(pts).reshape(-1, 2)
    return hull
