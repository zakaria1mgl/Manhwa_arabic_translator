from __future__ import annotations

import logging
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


def preprocess_crop(crop_bgr: np.ndarray, *, target_height: int = 48,
                    enhance: bool = True) -> np.ndarray:
    """Upscale small text and improve contrast before OCR.

    Never destructive: works on a copy, output is BGR.
    """
    import cv2
    img = crop_bgr.copy()
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return img
    if h < target_height:
        scale = target_height / h
        img = cv2.resize(img, (max(1, int(w * scale)), target_height),
                         interpolation=cv2.INTER_CUBIC)
    if enhance:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # denoise lightly, then adaptive contrast
        gray = cv2.bilateralFilter(gray, 5, 40, 40)
        mean, std = float(gray.mean()), float(gray.std())
        if std < 45:  # low contrast
            # tile grid scaled to the crop: a fixed 8x8 grid on a ~48px-tall
            # crop yields ~6px tiles, which amplifies paper/screentone
            # texture into fake glyph edges
            ts = max(2, min(8, min(img.shape[:2]) // 24))
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(ts, ts))
            gray = clahe.apply(gray)
            img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        else:
            # sharpen modestly
            blur = cv2.GaussianBlur(img, (0, 0), 2.0)
            img = cv2.addWeighted(img, 1.4, blur, -0.4, 0)
    return img


def estimate_text_polarity(crop_bgr: np.ndarray) -> bool:
    """True when text is darker than background (typical bubbles)."""
    import cv2
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    hist = np.histogram(gray, bins=8, range=(0, 255))[0]
    return hist[:3].sum() > hist[-3:].sum()


def binarize_for_mask(crop_bgr: np.ndarray, dark_text: bool = True) -> np.ndarray:
    """Binary text mask for a crop: 255 where text pixels are."""
    import cv2
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 5, 40, 40)
    if dark_text:
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.medianBlur(mask, 3)
    return mask
