"""
vision/roi.py -- Fixed-window index-ROI extraction.

Camera is positioned close-up on the card rank+suit corner.
No perspective correction needed -- blob detection finds the symbols
within a calibrated pixel window.

Geometry calibrated 2026-06 on IMX296 (1456x1088 frame, cam0):
    WX0, WX1, WY0, WY1 = 50, 260, 230, 950
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

WX0: int = 50
WX1: int = 260
WY0: int = 230
WY1: int = 950

BLOB_MIN: int = 1_200
BLOB_MAX: int = 30_000
BLOB_DIM_MAX: int = 320
SYM_FRAC: float = 0.45
ROI_MARGIN: int = 10


@dataclass
class WindowConfig:
    """Calibration constants for the index window (absolute pixel coordinates)."""
    wx0: int = WX0
    wx1: int = WX1
    wy0: int = WY0
    wy1: int = WY1


def extract_index_roi(
    frame: np.ndarray,
    cfg: WindowConfig | None = None,
) -> tuple[np.ndarray | None, str]:
    """
    Extract the rank+suit index corner from a raw camera frame.

    Returns (roi_bgr, info) or (None, reason).
    """
    if frame is None:
        return None, "no_image"

    if cfg is None:
        cfg = WindowConfig()

    win = frame[cfg.wy0:cfg.wy1, cfg.wx0:cfg.wx1]

    b, g, _ = cv2.split(win)
    gb = np.minimum(g, b).astype(np.int16)

    bright = gb[gb > 40]
    if bright.size < 5_000:
        return None, "no_card_in_window"

    med = float(np.median(bright))
    sym_mask = (gb < med * SYM_FRAC).astype(np.uint8) * 255
    sym_mask = cv2.morphologyEx(
        sym_mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )

    n, _, stats, _ = cv2.connectedComponentsWithStats(sym_mask)
    H, W = sym_mask.shape

    cands = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if not (BLOB_MIN <= area <= BLOB_MAX):
            continue
        if w > BLOB_DIM_MAX or h > BLOB_DIM_MAX:
            continue
        edges = (x <= 2) + (y <= 2) + (x + w >= W - 2) + (y + h >= H - 2)
        if edges >= 2:
            continue
        if area / (w * h) < 0.15:
            continue
        cands.append((int(x), int(y), int(w), int(h)))

    if not cands:
        return None, "no_symbols"

    cands.sort(key=lambda c: c[1])
    seed = cands[0]
    rank_blobs = [seed]
    for c in cands[1:]:
        ov = min(seed[1] + seed[3], c[1] + c[3]) - max(seed[1], c[1])
        if ov > 0.5 * min(seed[3], c[3]):
            rank_blobs.append(c)

    rank_bot = max(c[1] + c[3] for c in rank_blobs)
    rx0 = min(c[0] for c in rank_blobs)
    rx1 = max(c[0] + c[2] for c in rank_blobs)

    selected = list(rank_blobs)
    suit_cands = [
        c for c in cands
        if c not in rank_blobs
        and c[1] >= rank_bot - 20
        and rx0 - 120 < c[0] + c[2] / 2 < rx1 + 120
    ]
    if suit_cands:
        suit_cands.sort(key=lambda c: c[1])
        selected.append(suit_cands[0])

    x0 = min(c[0] for c in selected)
    y0 = min(c[1] for c in selected)
    x1 = max(c[0] + c[2] for c in selected)
    y1 = max(c[1] + c[3] for c in selected)

    m = ROI_MARGIN
    roi = win[
        max(0, y0 - m): min(H, y1 + m),
        max(0, x0 - m): min(W, x1 + m),
    ]

    return roi.copy(), f"blobs={len(selected)}"


def binarise(grey):
    """Otsu binarisation. Kept for backward compatibility."""
    import numpy as np, cv2
    if grey.dtype != np.uint8:
        grey = grey.astype(np.uint8)
    _, binary = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary
