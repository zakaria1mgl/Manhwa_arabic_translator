from __future__ import annotations

import os
import tempfile
from typing import Any, Dict, Optional, Tuple

import numpy as np
from PIL import Image, ImageOps


def load_image(path: str) -> np.ndarray:
    """Load image as BGR uint8 (cv2 convention), keeping EXIF orientation."""
    import cv2
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        # fallback for unusual formats (e.g. some WEBP)
        pil = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        img = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
    return img


def save_image(img: np.ndarray, path: str) -> None:
    import cv2
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    ok = cv2.imwrite(path, img)
    if not ok:
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        pil.save(path)


def to_rgb(img: np.ndarray) -> np.ndarray:
    import cv2
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def to_bgr(img: np.ndarray) -> np.ndarray:
    import cv2
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def crop_quad(img: np.ndarray, quad, pad: int = 0) -> np.ndarray:
    """Return the axis-aligned bounding crop of a quadrilateral region."""
    pts = np.asarray(quad).reshape(-1, 2).astype(int)
    x0, y0 = pts[:, 0].min(), pts[:, 1].min()
    x1, y1 = pts[:, 0].max(), pts[:, 1].max()
    h, w = img.shape[:2]
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(w, x1 + pad)
    y1 = min(h, y1 + pad)
    return img[y0:y1, x0:x1]


def dominant_color(mask_or_crop: np.ndarray, n: int = 1) -> Tuple[int, int, int]:
    """Dominant colour of a crop (optionally restricted by a boolean mask)."""
    data = np.asarray(mask_or_crop)
    if data.ndim == 2:
        # already a mask -> caller should pass colour image; keep simple
        return (255, 255, 255)
    px = data.reshape(-1, 3).astype(np.float32)
    import cv2
    if len(px) == 0:
        return (255, 255, 255)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(px, n, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    centers = centers.astype(int)
    counts = np.bincount(labels.flatten(), minlength=n)
    return tuple(int(v) for v in centers[int(np.argmax(counts))])


def pil_to_temp(img: np.ndarray, suffix: str = ".png") -> str:
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    save_image(img, path)
    return path


def np_to_pil(img: np.ndarray) -> Image.Image:
    return Image.fromarray(to_rgb(img))


def ensure_8bit(img: np.ndarray) -> np.ndarray:
    if img.dtype != np.uint8:
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8) if img.dtype == np.float32 else img.astype(np.uint8)
    return img


def resize_longest(img: np.ndarray, max_px: int) -> Tuple[np.ndarray, float]:
    """Downscale (never upscale) so the longest side is <= max_px. Returns scale."""
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest <= max_px:
        return img, 1.0
    scale = max_px / longest
    import cv2
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA), scale
