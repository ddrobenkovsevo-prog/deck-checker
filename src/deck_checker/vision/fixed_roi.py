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
    # Overall corner zone (calibrated 2026-06 on real machine)
    corner_x1: float = 0.655
    corner_x2: float = 0.79
    corner_y1: float = 0.198
    corner_y2: float = 0.242

    # Card orientation: machine holds cards so "9" reads upright already
    rotate_180: bool = False

    # Horizontal split: rank on the right, suit on the left
    rank_suit_split_x: float = 0.52

    # Which side is the rank: "left" or "right"
    rank_side: str = "right"

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


# ─────────────────────────────────────────────────────────────────────────────
# Adaptive extraction — handles ±1cm card drift
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AdaptiveRoiConfig:
    """
    Wide search zone within which rank+suit symbols are located dynamically.
    Robust to horizontal card drift (the card-running machine lets cards
    wander ~1cm left/right).
    """
    # Wide search zone (fractions of frame)
    # Widened on all sides to tolerate card drift both horizontally (~1cm)
    # and vertically.
    zone_x1: float = 0.58
    zone_x2: float = 0.82
    zone_y1: float = 0.13
    zone_y2: float = 0.32

    # Auto-exposure target mean brightness (0-255)
    target_mean: float = 120.0

    # Blob area filter (fraction of zone area)
    min_blob_frac: float = 0.01
    max_blob_frac: float = 0.30

    # Output sizes
    rank_w: int = 60
    rank_h: int = 80
    suit_w: int = 60
    suit_h: int = 60


def _find_symbol_blobs(binary: np.ndarray, cfg: AdaptiveRoiConfig):
    """Return list of (x, y, w, h, area) for symbol-sized blobs."""
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    zh, zw = binary.shape
    zone_area = zh * zw
    blobs = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        frac = area / zone_area
        if cfg.min_blob_frac < frac < cfg.max_blob_frac:
            blobs.append((int(x), int(y), int(bw), int(bh), int(area)))
    return blobs


def extract_rank_suit_adaptive(
    frame: np.ndarray, cfg: AdaptiveRoiConfig | None = None
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Adaptive rank/suit extraction that tolerates card drift.

    Strategy:
      1. Crop the wide search zone.
      2. CLAHE + Otsu to get a binary symbol mask.
      3. Find symbol-sized blobs.
      4. The rank + suit corner sits in the UPPER part of the zone:
         take the two blobs with the smallest y (topmost).
         Of those two, the rightmost is the rank, the other is the suit.
      5. Crop each blob tightly and return binarised symbols.

    Returns (rank_binary, suit_binary), or (None, None) if not found.
    """
    if cfg is None:
        cfg = AdaptiveRoiConfig()

    h, w = frame.shape[:2]
    x1 = int(w * cfg.zone_x1)
    x2 = int(w * cfg.zone_x2)
    y1 = int(h * cfg.zone_y1)
    y2 = int(h * cfg.zone_y2)
    zone = frame[y1:y2, x1:x2]

    if zone.ndim == 3:
        gray = cv2.cvtColor(zone, cv2.COLOR_BGR2GRAY)
    else:
        gray = zone

    # ── Build a mask of the white CARD area ─────────────────────────────────
    # The card is bright (>~100); background/edges are darker.
    # Find the largest bright region = the card, and only look for symbols
    # inside it. This rejects the dark background and the card's edge.
    _, card_mask = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY)
    # Clean up the mask
    mk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    card_mask = cv2.morphologyEx(card_mask, cv2.MORPH_CLOSE, mk)
    card_mask = cv2.morphologyEx(card_mask, cv2.MORPH_OPEN, mk)

    # Keep only the largest white component (the card body)
    n_m, lbl_m, st_m, _ = cv2.connectedComponentsWithStats(card_mask, connectivity=8)
    if n_m > 1:
        largest = 1 + int(np.argmax(st_m[1:, cv2.CC_STAT_AREA]))
        card_mask = np.where(lbl_m == largest, 255, 0).astype(np.uint8)
    # Erode inward so the card's edge isn't included as a "symbol"
    card_mask = cv2.erode(card_mask, mk, iterations=1)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    norm = clahe.apply(gray)
    _, binary = cv2.threshold(
        norm, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    # Restrict symbols to inside the card body
    binary = cv2.bitwise_and(binary, card_mask)

    blobs = _find_symbol_blobs(binary, cfg)
    if len(blobs) < 2:
        return None, None

    # Suit-anchored detection.
    #
    # The SUIT symbol (a compact club/heart/etc.) is the most reliably
    # detected blob: it is small, solid, and roughly square. The RANK sits
    # immediately to its RIGHT at a similar vertical level.
    #
    # 1. Score blobs for "suit-likeness": compact (near-square), small-ish,
    #    in the upper area, NOT touching the zone's right edge (that's the
    #    card edge under an angle).
    # 2. Once the suit is chosen, the rank = nearest blob to the right at
    #    similar Y.
    zh, zw = binary.shape
    zone_area = zh * zw

    def suit_score(b):
        x, y, bw, bh, area = b
        if bw == 0 or bh == 0:
            return -1e9
        aspect = min(bw, bh) / max(bw, bh)      # 1.0 = square (club-like)
        frac = area / zone_area
        cy = (y + bh / 2) / zh
        # Compact + square + upper + medium-small
        score = aspect * 2.0
        score -= cy * 0.5                       # prefer upper
        score -= abs(frac - 0.04) * 5.0         # ideal symbol size ~4% of zone
        # Penalise blobs that span most of the zone height (edges/pips)
        if bh > 0.7 * zh:
            score -= 5.0
        return score

    suit_blob = max(blobs, key=suit_score)
    sx, sy, sbw, sbh, _ = suit_blob
    suit_cx = sx + sbw / 2
    suit_cy = sy + sbh / 2

    # Rank = blob to the RIGHT of suit, similar Y, closest
    rank_blob = None
    best_d = 1e9
    for b in blobs:
        if b is suit_blob:
            continue
        x, y, bw, bh, area = b
        cx = x + bw / 2
        cy = y + bh / 2
        if cx <= suit_cx:           # must be to the right
            continue
        if abs(cy - suit_cy) > 0.5 * zh:   # roughly same row
            continue
        d = abs(cx - suit_cx) + abs(cy - suit_cy)
        if d < best_d:
            best_d = d
            rank_blob = b

    if rank_blob is None:
        return None, None

    def _crop(blob, out_w, out_h):
        x, y, bw, bh, _ = blob
        pad = 3
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1c = min(binary.shape[1], x + bw + pad)
        y1c = min(binary.shape[0], y + bh + pad)
        crop = binary[y0:y1c, x0:x1c]
        return cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_AREA)

    rank_img = _crop(rank_blob, cfg.rank_w, cfg.rank_h)
    suit_img = _crop(suit_blob, cfg.suit_w, cfg.suit_h)
    return rank_img, suit_img
