from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..utils.polygons import enclosing_polygon, polygon_area, polygon_centroid
from ..utils.types import Quad, RegionType, TextRegion

log = logging.getLogger(__name__)


@dataclass
class RegionRow:
    y: float
    x: float
    regions: List[TextRegion] = field(default_factory=list)


def detect_panels(image: np.ndarray,
                  min_area_ratio: float = 0.06) -> List[Tuple[int, int, int, int]]:
    """Cheap panel segmentation via gutter analysis.

    Manhwa pages are panels separated by (usually white) gutters. We find
    "content" pixels (anything differing from the border background colour),
    close them into solid blobs, and take the big enclosing contours' bboxes
    as panels. Returns [] when no sensible segmentation is found — callers
    fall back to plain row ordering.
    """
    import cv2
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    border = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
    bg = float(np.median(border))
    fg = (np.abs(gray.astype(np.int16) - int(bg)) > 28).astype(np.uint8) * 255
    k = max(9, int(min(h, w) / 60) | 1)  # ~1.5% of short side, odd
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    closed = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel, iterations=3)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    panels = []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        if bw * bh < min_area_ratio * w * h:
            continue
        panels.append((x, y, x + bw, y + bh))
    # drop panels almost fully covered by another panel (nested dupes)
    panels.sort(key=lambda p: (p[2] - p[0]) * (p[3] - p[1]))
    out: List[Tuple[int, int, int, int]] = []
    for p in panels:
        dup = False
        for q in out:
            ix0, iy0 = max(p[0], q[0]), max(p[1], q[1])
            ix1, iy1 = min(p[2], q[2]), min(p[3], q[3])
            inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
            if inter > 0.85 * (p[2] - p[0]) * (p[3] - p[1]):
                dup = True
                break
        if not dup:
            out.append(p)
    return out if len(out) >= 2 else []


def _order_rows(regions: List[TextRegion], rev: bool,
                row_overlap_ratio: float) -> List[TextRegion]:
    """Group regions into visual rows; order rows top->bottom, regions
    left->right (or reversed) inside each row."""
    if not regions:
        return []
    items = sorted(regions, key=lambda r: r.bbox[1])
    rows: List[RegionRow] = []
    for r in items:
        x0, y0, x1, y1 = r.bbox
        h = max(1, y1 - y0)
        placed = False
        for row in rows:
            ry0 = min(o.bbox[1] for o in row.regions)
            ry1 = max(o.bbox[3] for o in row.regions)
            inter = min(y1, ry1) - max(y0, ry0)
            if inter > row_overlap_ratio * h:
                row.regions.append(r)
                placed = True
                break
        if not placed:
            rows.append(RegionRow(y=y0, x=x0, regions=[r]))
    ordered: List[TextRegion] = []
    for row in sorted(rows, key=lambda rw: rw.y):
        ordered.extend(sorted(row.regions, key=lambda r: r.bbox[0],
                              reverse=rev))
    return ordered


def sort_reading_order(regions: List[TextRegion], panel_order: str = "ltr",
                       row_overlap_ratio: float = 0.5,
                       panels: Optional[List[Tuple[int, int, int, int]]] = None
                       ) -> List[TextRegion]:
    """Order regions for reading.

    With panels: assign each region to the smallest panel containing its
    centre, order panels in reading sequence (row-grouped), and order the
    regions inside each panel. Panel-less leftovers keep row ordering.
    Without panels this is the legacy row grouping.
    """
    if not regions:
        return []
    rev = panel_order == "rtl"
    if panels:
        by_panel: Dict[int, List[TextRegion]] = {i: [] for i in range(len(panels))}
        leftovers: List[TextRegion] = []
        for r in regions:
            cx = (r.bbox[0] + r.bbox[2]) / 2
            cy = (r.bbox[1] + r.bbox[3]) / 2
            best_i, best_area = -1, float("inf")
            for i, (x0, y0, x1, y1) in enumerate(panels):
                if x0 <= cx <= x1 and y0 <= cy <= y1:
                    area = (x1 - x0) * (y1 - y0)
                    if area < best_area:
                        best_i, best_area = i, area
            (by_panel[best_i] if best_i >= 0 else leftovers).append(r)
        # panel reading order via the same row grouping used for regions
        pseudo = [TextRegion(id=i + 1, quad=Quad(
            points=[(p[0], p[1]), (p[2], p[1]), (p[2], p[3]), (p[0], p[3])]))
            for i, p in enumerate(panels)]
        panel_seq = [p.id - 1 for p in _order_rows(pseudo, rev,
                                                   row_overlap_ratio)]
        ordered: List[TextRegion] = []
        for i in panel_seq:
            ordered.extend(_order_rows(by_panel[i], rev, row_overlap_ratio))
        ordered.extend(_order_rows(leftovers, rev, row_overlap_ratio))
    else:
        ordered = _order_rows(regions, rev, row_overlap_ratio)
    for i, r in enumerate(ordered):
        r.id = i + 1
    return ordered


def estimate_bubbles(regions: List[TextRegion], image_shape: Tuple[int, int],
                     image: Optional[np.ndarray] = None) -> None:
    """Attach a bubble polygon to each region.

    Strategy: cluster nearby text regions whose quads are close and aligned;
    then grow a convex hull around them padded by a typical bubble margin.
    When the page image is available, snap each cluster to the *actual*
    bright bubble contour enclosing the text — the dilated hull is only the
    fallback. Region type is refined afterwards.
    """
    h, w = image_shape[:2]
    unassigned = list(range(len(regions)))
    clusters: List[List[int]] = []
    while unassigned:
        seed = unassigned.pop(0)
        cluster = [seed]
        changed = True
        while changed:
            changed = False
            for idx in list(unassigned):
                if _near_cluster(regions, cluster, idx):
                    cluster.append(idx)
                    unassigned.remove(idx)
                    changed = True
        clusters.append(cluster)
    for cluster in clusters:
        quads = [regions[i].quad.polygon for i in cluster]
        hull = enclosing_polygon(quads)
        if hull is None:
            continue
        # pad hull outward to approximate the bubble outline
        margin = _cluster_margin(regions, cluster)
        from ..utils.polygons import dilate_polygon
        bubble = dilate_polygon(hull, margin)
        for i in cluster:
            regions[i].bubble_polygon = bubble
            regions[i].bubble_cluster = list(cluster)
    if image is not None:
        _snap_to_bubble_contours(regions, image)


def _snap_to_bubble_contours(regions: List[TextRegion], image: np.ndarray,
                             white_thr: int = 232,
                             dark_thr: int = 65) -> None:
    """Replace dilated-hull 'bubbles' with the real bubble outline.

    Speech bubbles / narration boxes are bright, solid shapes — except dark
    narration boxes and thought panels, which are the opposite. So we run two
    passes: a near-white threshold pass, then a near-black pass for groups
    that are still unsnapped AND sit on dark sampled backgrounds.
    """
    import cv2
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    med_h = float(np.median([r.quad.height for r in regions])) if regions else 20.0
    k = max(9, int(0.8 * med_h) | 1)  # odd, roughly a text-line height

    # group regions by bubble cluster so all lines share the snapped contour
    groups: Dict[Tuple[int, ...], List[TextRegion]] = {}
    for r in regions:
        key = tuple(sorted(r.bubble_cluster)) if r.bubble_cluster else (r.id,)
        groups.setdefault(key, []).append(r)

    # pass 1: bright (near-white) bubbles
    bright = cv2.threshold(gray, white_thr, 255, cv2.THRESH_BINARY)[1]
    snapped = _close_and_snap(groups, image.shape, bright, k)

    # pass 2: dark caption boxes / black bubbles, only for groups that are
    # still unsnapped AND sit on a dark sampled background
    dark_groups = {}
    for key, members in groups.items():
        if key in snapped:
            continue
        bg = members[0].bg_color
        if bg is not None and float(np.mean(bg)) > 150:
            continue  # bright bg but unsnapped: not a dark box either
        dark_groups[key] = members
    if dark_groups:
        dark = cv2.threshold(gray, dark_thr, 255, cv2.THRESH_BINARY_INV)[1]
        _close_and_snap(dark_groups, image.shape, dark, k)


def _hull_of(members: List[TextRegion]) -> Optional[np.ndarray]:
    return enclosing_polygon([m.quad.polygon for m in members])


def _close_and_snap(groups: Dict[Tuple[int, ...], List[TextRegion]],
                    image_shape: Tuple[int, int], mask: "np.ndarray",
                    k: int) -> set:
    """Close text-stroke holes in `mask`, find solid contours, and snap each
    group to the smallest contour that contains its text lines.
    Returns the set of group keys that were snapped."""
    import cv2
    h, w = image_shape[:2]
    snapped_keys: set = set()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return snapped_keys
    cand = []
    for c in contours:
        area = float(cv2.contourArea(c))
        if area < 400 or area > 0.7 * h * w:
            continue
        cand.append((area, c))
    cand.sort(key=lambda t: t[0])  # try smallest areas first
    page_area = float(h * w)

    for members in groups.values():
        hull = _hull_of(members)
        if hull is None:
            continue
        hull_area = max(1.0, polygon_area(hull))
        cx, cy = polygon_centroid(hull)
        corners = [(float(p[0]), float(p[1]))
                   for m in members for p in m.quad.polygon.reshape(-1, 2)]
        best = None
        for area, c in cand:
            if area < 1.05 * hull_area or area > 30 * hull_area:
                continue
            if area > 0.55 * page_area:
                continue
            cpts = c.reshape(-1, 2)
            # contour must contain the centroid and (almost) all text corners
            if cv2.pointPolygonTest(cpts.astype(np.int32).reshape(-1, 1, 2),
                                    (cx, cy), False) < 0:
                continue
            inside = sum(
                1 for p in corners
                if cv2.pointPolygonTest(cpts.astype(np.int32).reshape(-1, 1, 2),
                                        p, False) >= -2)
            if inside < max(1, int(0.75 * len(corners))):
                continue
            best = c
            break  # smallest containing contour wins
        if best is None:
            continue
        peri = cv2.arcLength(best, True)
        approx = cv2.approxPolyDP(best, 0.01 * peri, True).reshape(-1, 2)
        if len(approx) < 3:
            continue
        # inset ~3px toward the centroid so text stays off the outline
        c0 = approx.mean(axis=0)
        d = np.linalg.norm(approx - c0, axis=1).mean()
        inset = min(0.06, 3.0 / max(8.0, float(d)))
        approx = (c0 + (approx - c0) * (1.0 - inset)).astype(np.int32)
        snapped = approx.reshape(-1, 2)
        log.debug("cluster %d members -> snapped bubble (area x%.1f)",
                  len(members), polygon_area(snapped) / hull_area)
        for m in members:
            m.bubble_polygon = snapped
        key = tuple(sorted(members[0].bubble_cluster)) \
            if members[0].bubble_cluster else (members[0].id,)
        snapped_keys.add(key)
    return snapped_keys


def merge_bubble_lines(regions: List[TextRegion]) -> None:
    """Merge multi-line text inside one bubble into a single translation unit.

    The region with the largest quad keeps the translated text; the others
    become satellites (empty arabic) so the renderer places the whole bubble's
    text once, centered in the bubble — and the translator sees one coherent
    sentence instead of fragmented lines.
    """
    if not regions:
        return
    by_cluster: dict = {}
    for r in regions:
        if r.bubble_cluster:
            key = tuple(sorted(r.bubble_cluster))
            by_cluster.setdefault(key, []).append(r)
    for members in by_cluster.values():
        if len(members) <= 1:
            continue
        # order lines top->bottom inside the bubble, join with spaces
        members_sorted = sorted(members, key=lambda m: m.bbox[1])
        primary = max(members_sorted, key=lambda m: m.quad.width * m.quad.height)
        joined = " ".join(m.english for m in members_sorted if m.english)
        # never let a merge ERASE content: if the primary had no text but a
        # satellite did, the joined text still survives on the primary
        primary.english = joined
        primary.ocr_conf = min(m.ocr_conf for m in members_sorted) or primary.ocr_conf
        # union bbox of all lines -> mask/fitting must cover the whole block
        xs0 = min(m.bbox[0] for m in members_sorted)
        ys0 = min(m.bbox[1] for m in members_sorted)
        xs1 = max(m.bbox[2] for m in members_sorted)
        ys1 = max(m.bbox[3] for m in members_sorted)
        primary.merged_span = (int(xs0), int(ys0), int(xs1), int(ys1))
        # CRITICAL: expand the primary's quad to the union span. Downstream
        # fit-budget and mask code read quad.height — leaving it at a single
        # line makes the LLM shorten translations to fit one line's height.
        primary.quad = Quad(points=[(xs0, ys0), (xs1, ys0),
                                    (xs1, ys1), (xs0, ys1)])
        for m in members_sorted:
            if m is not primary:
                m.english = ""
                m.arabic = ""
                m.merged_into = primary.id


def _near_cluster(regions: List[TextRegion], cluster: List[int], idx: int,
                  dist_factor: float = 1.6) -> bool:
    """True when region `idx` belongs to the same text block as `cluster`.

    Guards against transitive gluing (a floating SFX bridging two speakers'
    bubbles): require comparable line heights, real vertical proximity, and a
    modest horizontal gap relative to the *line height* — not the line width,
    which let long lines reach halfway across the page.
    """
    r = regions[idx]
    rh = max(1.0, r.quad.height)
    for ci in cluster:
        c = regions[ci]
        ch = max(1.0, c.quad.height)
        ch0 = min(rh, ch)
        # lines in one bubble have similar heights (within ~2.2x)
        if max(rh, ch) > 2.2 * ch0:
            continue
        y_gap = max(0, max(r.bbox[1], c.bbox[1]) - min(r.bbox[3], c.bbox[3]))
        x_gap = max(0, max(r.bbox[0], c.bbox[0]) - min(r.bbox[2], c.bbox[2]))
        if y_gap < dist_factor * ch0 and x_gap < 5.0 * ch0:
            return True
    return False


def _cluster_margin(regions: List[TextRegion], cluster: List[int]) -> int:
    heights = [max(1.0, regions[i].quad.height) for i in cluster]
    return max(6, int(0.9 * np.mean(heights)))


def classify_regions(regions: List[TextRegion], image_shape: Tuple[int, int]) -> None:
    """Heuristic region classification (refined by the context LLM later)."""
    h, w = image_shape[:2]
    for r in regions:
        if r.region_type not in (RegionType.UNKNOWN,):
            continue
        qh, qw = r.quad.height, max(1.0, r.quad.width)
        area_ratio = polygon_area(r.quad.polygon) / float(h * w)
        words = r.english.split()
        n_words = len(words)
        # big stylized text -> SFX
        if qh > 0.055 * h and n_words <= 3:
            r.region_type = RegionType.SFX
            continue
        if _looks_like_sfx(r.english):
            r.region_type = RegionType.SFX
            continue
        # narrow tall -> vertical sign
        if qh > 2.0 * qw and n_words <= 6:
            r.region_type = RegionType.SIGN
            continue
        # corner / edge -> narration or caption
        x0, y0, x1, y1 = r.bbox
        near_top = y0 < 0.12 * h
        near_left = x0 < 0.10 * w
        near_right = x1 > 0.90 * w
        if (near_top and (near_left or near_right)) or (near_left and y0 < 0.35 * h):
            r.region_type = RegionType.NARRATION
            continue
        if near_top and area_ratio > 0.02 and n_words > 3:
            r.region_type = RegionType.CAPTION
            continue
        # bubble-like hull exists -> dialogue
        if r.bubble_polygon is not None:
            r.region_type = RegionType.DIALOGUE
            continue
        r.region_type = RegionType.DIALOGUE


_SFX_CHARS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!?*")


def _looks_like_sfx(text: str) -> bool:
    """Heuristic SFX detector.

    NOTE: the old version matched all-caps words starting with common
    consonant blends (TH, ST, PL, CR...) — which classified ordinary
    dialogue like "THE", "STOP", "PLEASE" as onomatopoeia. SFX detection is
    now based on visual/style signals (all-caps, very short, punctuation or
    repeated letters) only; semantic classification is left to the LLM.
    """
    t = text.strip()
    if not t or len(t) > 24:
        return False
    letters = [c for c in t if c.isalpha()]
    if not letters:
        return False
    if not (t.upper() == t and len(t.split()) <= 2):
        return False
    # all-caps short strings with repeated letters or bangs -> stylized SFX
    rep = sum(1 for a, b in zip(t, t[1:]) if a == b and a.isalpha())
    if rep >= 1 or any(ch in "!?*" for ch in t):
        return True
    # single short all-caps token that is not a common English word
    if len(t.split()) == 1 and 2 <= len(t) <= 8 and not _is_common_word(t):
        return True
    return False


_COMMON_WORDS = {
    "THE", "A", "AN", "I", "YOU", "HE", "SHE", "IT", "WE", "THEY", "IS",
    "ARE", "WAS", "BE", "DO", "DID", "GO", "NO", "YES", "OK", "OKAY", "HI",
    "OH", "AH", "HEY", "STOP", "WAIT", "WHAT", "WHY", "HOW", "WHO", "WHEN",
    "WHERE", "THIS", "THAT", "THERE", "HERE", "COME", "LOOK", "LET", "NOW",
    "BUT", "AND", "FOR", "NOT", "ALL", "CAN", "WILL", "JUST", "DON'T",
}


def _is_common_word(word: str) -> bool:
    return word.strip("!?.,'\"").upper() in _COMMON_WORDS
