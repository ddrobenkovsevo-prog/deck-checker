"""Region-of-interest extraction from a normalized card image.

After preprocessing we have a 250×350 BGR card oriented upright. The rank and suit
glyphs live in the top-left corner (and mirrored in the bottom-right). We crop
the top-left corner, then split it vertically into rank (upper) and suit (lower).

All coordinates are in the canonical 250×350 space defined by preprocessing.NORMALIZED_SIZE.
"""

from __future__ import annotations

from typing import Literal

import cv2
import numpy as np

# Corner box covers the rank + suit pip area in the upper-left of a poker card.
# Tuned to be generous: we'd rather include a few extra background pixels than
# clip a tall rank like "10" or a wide suit pip.
CORNER_WIDTH = 50
CORNER_HEIGHT = 130

# Where rank ends and suit begins, as a fraction of CORNER_HEIGHT.
RANK_SUIT_SPLIT = 0.55

Corner = Literal["top-left", "bottom-right"]


def extract_corner(
    card_image: np.ndarray,
    corner: Corner = "top-left",
) -> np.ndarray:
    """Crop the rank+suit corner of a normalized card.

    `top-left` returns the upper-left corner as-is.
    `bottom-right` returns the lower-right corner rotated 180° so it reads upright.
    """
    if card_image.shape[0] < CORNER_HEIGHT or card_image.shape[1] < CORNER_WIDTH:
        raise ValueError(
            f"Card image too small ({card_image.shape}); "
            f"need at least {CORNER_HEIGHT}×{CORNER_WIDTH}"
        )

    if corner == "top-left":
        return card_image[:CORNER_HEIGHT, :CORNER_WIDTH].copy()

    if corner == "bottom-right":
        h, w = card_image.shape[:2]
        crop = card_image[h - CORNER_HEIGHT :, w - CORNER_WIDTH :]
        return cv2.rotate(crop, cv2.ROTATE_180)

    raise ValueError(f"Unknown corner: {corner!r}")


def split_rank_suit(
    corner_roi: np.ndarray,
    split_ratio: float = RANK_SUIT_SPLIT,
) -> tuple[np.ndarray, np.ndarray]:
    """Split a corner crop into rank (top) and suit (bottom) regions."""
    h = corner_roi.shape[0]
    split_y = int(h * split_ratio)
    rank = corner_roi[:split_y].copy()
    suit = corner_roi[split_y:].copy()
    return rank, suit


def to_grayscale_clean(roi: np.ndarray) -> np.ndarray:
    """Convert an ROI to a clean binary grayscale image suitable for template matching.

    Steps:
        BGR → grayscale → Gaussian blur → Otsu inverse threshold.

    Otsu picks the threshold automatically per ROI, which is robust against
    minor illumination drift. INV gives glyph=white on background=black, matching
    how learned templates are stored.
    """
    if roi.ndim == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU,
    )
    return binary


def tightly_crop(binary_roi: np.ndarray, padding: int = 2) -> np.ndarray:
    """Crop a binary ROI to the bounding box of its non-zero pixels.

    Useful for tightening rank/suit glyphs before template matching so we compare
    glyph-to-glyph instead of including different amounts of margin.
    """
    coords = cv2.findNonZero(binary_roi)
    if coords is None:
        return binary_roi
    x, y, w, h = cv2.boundingRect(coords)
    h_img, w_img = binary_roi.shape[:2]
    x0 = max(0, x - padding)
    y0 = max(0, y - padding)
    x1 = min(w_img, x + w + padding)
    y1 = min(h_img, y + h + padding)
    return binary_roi[y0:y1, x0:x1]
