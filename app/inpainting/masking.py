from __future__ import annotations

import logging
from typing import List

import numpy as np

from ..config import AppConfig
from ..utils.types import TextRegion

log = logging.getLogger(__name__)


def _odd(n: int) -> int:
    n = max(3, int(n))
    return n if n % 2 == 1 else n + 1


def build_text_mask(image: np.ndarray, regions: List[TextRegion], *,
                    dilate_px: int = 4,
                    uniform_tolerance: int = 18) -> np.ndarray:
    """Mask the actual text strokes inside each region's quad.

    Uses a local-median background estimate: text strokes are sparse and get
    removed by the median filter, so |pixel - local_bg| flags exactly the
    letters — while bubble borders, panel lines and soft shading survive the
    filter and are NOT masked. This avoids the white-patch and clipped-border
    artifacts of naive colour-difference masks.
    """
    import cv2
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    for r in regions:
        x0, y0, x1, y1 = r.bbox
        exp = 6
        x0, y0 = max(0, x0 - exp), max(0, y0 - exp)
        x1, y1 = min(w, x1 + exp), min(h, y1 + exp)
        if x1 <= x0 or y1 <= y0:
            continue
        roi = image[y0:y1, x0:x1]
        local_mask = np.zeros(roi.shape[:2], dtype=np.uint8)
        shifted = r.quad.polygon.reshape(-1, 1, 2).astype(np.int32) - \
            np.array([[[x0, y0]]], dtype=np.int32)
        cv2.fillPoly(local_mask, [shifted], 255)
        quad_area = float((x1 - x0) * (y1 - y0))

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # kernel sized to the text: bigger for SFX/large text
        k = _odd(max(21, int(0.6 * (r.bbox[3] - r.bbox[1]))))
        bg_local = cv2.medianBlur(gray, k)
        diff = cv2.absdiff(gray, bg_local)
        text_px = (diff > 55) & (local_mask > 0)
        local_mask = text_px.astype(np.uint8) * 255

        # drop huge connected components: those are bubble borders or shading
        # edges caught inside the quad, not letters
        local_mask = _drop_large_components(local_mask, quad_area)

        coverage = local_mask.sum() / max(1.0, 255.0 * quad_area)
        if coverage < 0.02:
            # colour approach found nothing: fall back to the whole quad, but
            # only when the region sits on a plain interior (ring check)
            if _is_uniform(roi, r, uniform_tolerance):
                cv2.fillPoly(local_mask, [shifted], 255)

        dil = dilate_px if _is_uniform(roi, r, uniform_tolerance) else max(1, dilate_px - 2)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dil + 1, 2 * dil + 1))
        local_mask = cv2.dilate(local_mask, kernel)
        mask[y0:y1, x0:x1] = cv2.bitwise_or(mask[y0:y1, x0:x1], local_mask)
    return mask


def uniform_bubble_fill(image: np.ndarray, regions: List[TextRegion],
                        tolerance: int = 18) -> np.ndarray:
    """Fast path for plain bubble interiors: repaint the real bubble contour
    (when contour snapping found one and its interior ring is uniform) or,
    failing that, the text-line hull — always slightly inset, filled with the
    LOCAL MEDIAN background. Never a bounding rectangle, whose corners cross
    the bubble outline and stamp white patches on the artwork; the local
    median drops the sparse text strokes while keeping gradients/shading."""
    import cv2
    from ..utils.polygons import enclosing_polygon
    out = image.copy()
    h, w = image.shape[:2]
    # group by bubble cluster so multi-line fills cover all lines jointly.
    # NOTE: after merge_bubble_lines the cluster has ONE primary (merged_into
    # is None) plus satellites (merged_into set) — group by the cluster key
    # and collect every member of that cluster, not just primaries.
    by_id = {r.id: r for r in regions}
    groups: dict = {}
    for r in regions:
        if r.bg_color is None or r.merged_into is not None:
            continue
        cluster = list(r.bubble_cluster) if r.bubble_cluster else [r.id]
        members = [by_id[i] for i in cluster if i in by_id]
        members = [m for m in members if m.bg_color is not None] or [r]
        key = tuple(sorted(cluster))
        groups[key] = members
    for members in groups.values():
        # skip if any member sits on non-uniform ground
        probes_ok = True
        for m in members:
            mx0, my0, mx1, my1 = m.bbox
            mx0, my0 = max(0, mx0 - 8), max(0, my0 - 8)
            mx1, my1 = min(w, mx1 + 8), min(h, my1 + 8)
            if mx1 <= mx0 or my1 <= my0:
                probes_ok = False
                break
            if not _is_uniform(out[my0:my1, mx0:mx1], m, tolerance):
                probes_ok = False
                break
        if not probes_ok:
            continue
        # prefer the real (snapped) bubble contour: repaint the WHOLE interior
        # in one pass. Guard: a thin ring just inside the contour must be
        # uniform and match the bubble background — otherwise it's not really
        # a flat bubble (artwork crosses it) and we fall back to the text hull.
        bubble = members[0].bubble_polygon
        bgs = np.array([m.bg_color for m in members if m.bg_color is not None])
        target = None
        if bubble is not None and len(bgs):
            poly = np.asarray(bubble).reshape(-1, 2)
            if len(poly) >= 3 and _interior_ring_uniform(
                    out, poly, tuple(int(v) for v in np.median(bgs, axis=0)),
                    tolerance):
                target = poly.astype(np.int32).reshape(-1, 1, 2)
        if target is None:
            hull = enclosing_polygon([m.quad.polygon for m in members])
            if hull is None:
                continue
            target = cv2.convexHull(np.asarray(hull, dtype=np.int32).reshape(-1, 1, 2))
        pts = target.reshape(-1, 2).astype(np.float32)
        c = pts.mean(axis=0)
        # inset so we never touch the bubble outline itself
        inset = 4
        pts = c + (pts - c) * (1.0 - inset / max(8.0, float(np.linalg.norm(pts - c, axis=1).mean())))
        hull_i = pts.astype(np.int32).reshape(-1, 1, 2)
        x0, y0, x1, y1 = _hull_bbox(hull_i, w, h, pad=8)
        roi = out[y0:y1, x0:x1]
        if roi.size == 0:
            continue
        # local median background: keeps gradients/shading, drops thin strokes
        k = _odd(max(21, int(0.5 * (y1 - y0))))
        local_bg_bgr = cv2.medianBlur(roi, k)
        poly_mask = np.zeros(roi.shape[:2], dtype=np.uint8)
        shifted = hull_i - np.array([[[x0, y0]]], dtype=np.int32)
        cv2.fillPoly(poly_mask, [shifted], 255)
        poly_mask = cv2.erode(poly_mask, np.ones((3, 3), np.uint8))
        out[y0:y1, x0:x1] = np.where(poly_mask[..., None] > 0, local_bg_bgr, roi)
    return out


def _interior_ring_uniform(image: np.ndarray, poly: np.ndarray,
                           bg_bgr, tolerance: int, ring_px: int = 6) -> bool:
    """True when a thin ring just inside `poly` is uniform and close to bg.

    Confirms the polygon really is a flat bubble interior before repainting
    the whole thing (the text lines inside are removed by the local median
    during the fill, so they must not block the decision).
    """
    import cv2
    from ..utils.polygons import polygon_bbox
    h, w = image.shape[:2]
    x0, y0, x1, y1 = polygon_bbox(poly)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    if x1 - x0 < 12 or y1 - y0 < 12:
        return False
    roi = image[y0:y1, x0:x1]
    shifted = np.asarray(poly, dtype=np.int32).reshape(-1, 1, 2) - \
        np.array([[[x0, y0]]], dtype=np.int32)
    m = np.zeros(roi.shape[:2], dtype=np.uint8)
    cv2.fillPoly(m, [shifted], 255)
    k = 2 * ring_px + 1
    inner = cv2.erode(m, np.ones((k, k), np.uint8))
    ring = cv2.subtract(m, inner)
    if cv2.countNonZero(ring) < 50:
        return False
    px = roi[ring > 0].astype(np.float32)
    diff = np.abs(px - np.array(bg_bgr, dtype=np.float32)).sum(axis=1)
    return float(np.median(diff)) <= tolerance * 3


def _hull_bbox(hull: np.ndarray, w: int, h: int, pad: int = 8):
    pts = hull.reshape(-1, 2)
    x0 = max(0, int(pts[:, 0].min()) - pad)
    y0 = max(0, int(pts[:, 1].min()) - pad)
    x1 = min(w, int(pts[:, 0].max()) + pad)
    y1 = min(h, int(pts[:, 1].max()) + pad)
    return x0, y0, x1, y1


def _ring_diff(roi: np.ndarray, r: TextRegion) -> np.ndarray:
    if r.bg_color is None:
        return np.full(4, 1e9, dtype=np.float32)  # unknown bg -> treat as non-uniform
    bg = np.array(r.bg_color, dtype=np.int16)
    diff = np.abs(roi.astype(np.int16) - bg).sum(axis=2)
    return np.concatenate([diff[0, :], diff[-1, :], diff[:, 0], diff[:, -1]])


def _is_uniform(roi: np.ndarray, r: TextRegion, tolerance: int) -> bool:
    """True when the region sits on a plain bubble-like interior."""
    if r.bg_color is None:
        return False
    return float(np.median(_ring_diff(roi, r))) <= tolerance * 3


def _drop_large_components(mask: np.ndarray, quad_area: float,
                           limit: float = 0.45) -> np.ndarray:
    """Remove mask components whose bbox covers a large share of the quad —
    those are bubble borders / shading, not individual letters."""
    import cv2
    m = (mask > 0).astype(np.uint8)
    if m.sum() == 0:
        return mask
    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    out = mask.copy()
    for i in range(1, num):
        x, y, bw, bh, area = stats[i]
        if bw * bh > limit * quad_area:
            out[labels == i] = 0
    return out
