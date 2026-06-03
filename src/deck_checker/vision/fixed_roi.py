"""
vision/fixed_roi.py — Fixed-position ROI extraction.

Unlike the contour-based pipeline (preprocessing.py + roi.py), this module
assumes the card always appears in the SAME position in the camera frame,
held by the card-running machine. No contour detection or perspective
correction needed — just crop the known rank/suit zones.

This is faster and far more robust when the mechanical setup guarantees
a fixed card position.

Geometry (as fraction of full frame, calibrated 2026-06):
  Card is upside-down (180° rotation).
  Rank + suit corner is in the upper-right of the frame.
  Within that corner: suit on the LEFT, rank on the RIGHT (horizontal layout).

Coordinates are configurable via FixedRoiConfig so they can be re-calibrated
without code changes.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class FixedRoiConfig:
    """
    Fractional coordinates of the rank/suit corner zone within the frame.

    All values are fractions [0.0–1.0] of frame width/height.
    The corner zone is split into rank and suit sub-regions.
    """
    # Overall corner zone
    corner_x1: float = 0.66
    corner_x2: float = 0.79
    corner_y1: float = 0.195
    corner_y2: float = 0.265

    # Card orientation in frame (machine holds cards upside-down)
    rotate_180: bool = True

    # Within the (rotated) corner, split point between rank and suit.
    # After 180° rotation: rank ends up on the LEFT, suit on the RIGHT.
    # split_x = fraction of corner width where rank|suit boundary sits.
    rank_suit_split_x: float = 0.5

    # Which side is the rank after rotation: "left" or "right"
    rank_side: str = "left"

    # Output sizes for matching
    rank_w: int = 60
    rank_h: int = 80
    suit_w: int = 60
    suit_h: int = 60


def extract_corner(frame: np.ndarray, cfg: FixedRoiConfig) -> np.ndarray:
    """
    Crop the fixed rank/suit corner zone from a full camera frame.

    Returns the corner as a BGR (or grayscale) image, rotated upright
    if cfg.rotate_180 is set.
    """
    h, w = frame.shape[:2]
    x1 = int(w * cfg.corner_x1)
    x2 = int(w * cfg.corner_x2)
    y1 = int(h * cfg.corner_y1)
    y2 = int(h * cfg.corner_y2)
    corner = frame[y1:y2, x1:x2].copy()
    if cfg.rotate_180:
        corner = cv2.rotate(corner, cv2.ROTATE_180)
    return corner


def split_rank_suit(
    corner: np.ndarray, cfg: FixedRoiConfig
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split the corner crop into rank and suit sub-images (horizontal layout).

    Returns (rank_img, suit_img) as grayscale, resized to config dimensions.
    """
    if corner.ndim == 3:
        gray = cv2.cvtColor(corner, cv2.COLOR_BGR2GRAY)
    else:
        gray = corner

    h, w = gray.shape[:2]
    split = int(w * cfg.rank_suit_split_x)

    left  = gray[:, :split]
    right = gray[:, split:]

    if cfg.rank_side == "left":
        rank_raw, suit_raw = left, right
    else:
        rank_raw, suit_raw = right, left

    rank_img = cv2.resize(rank_raw, (cfg.rank_w, cfg.rank_h),
                          interpolation=cv2.INTER_AREA)
    suit_img = cv2.resize(suit_raw, (cfg.suit_w, cfg.suit_h),
                          interpolation=cv2.INTER_AREA)
    return rank_img, suit_img


def binarise(roi: np.ndarray) -> np.ndarray:
    """
    Otsu threshold + tight crop to isolate the symbol.

    Dark symbol on light card → THRESH_BINARY_INV makes symbol white.
    """
    # Normalise illumination first (the card centre is often over-exposed)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    roi = clahe.apply(roi)

    _, binary = cv2.threshold(
        roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    coords = cv2.findNonZero(binary)
    if coords is None:
        return binary
    x, y, bw, bh = cv2.boundingRect(coords)
    pad = 2
    x = max(0, x - pad)
    y = max(0, y - pad)
    bw = min(binary.shape[1] - x, bw + 2 * pad)
    bh = min(binary.shape[0] - y, bh + 2 * pad)
    return binary[y:y + bh, x:x + bw]


def extract_rank_suit(
    frame: np.ndarray, cfg: FixedRoiConfig | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Full fixed-ROI pipeline: crop corner → split → binarise.

    Parameters
    ----------
    frame: Full BGR camera frame.
    cfg:   FixedRoiConfig (uses defaults if None).

    Returns
    -------
    (rank_binary, suit_binary) — ready for template matching.
    """
    if cfg is None:
        cfg = FixedRoiConfig()
    corner = extract_corner(frame, cfg)
    rank_img, suit_img = split_rank_suit(corner, cfg)
    return binarise(rank_img), binarise(suit_img)
