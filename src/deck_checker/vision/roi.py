"""
Region-of-Interest extraction from a canonical 250x350 card image.
"""
from __future__ import annotations

import cv2
import numpy as np

from deck_checker.vision.preprocessing import CARD_H, CARD_W

CORNER_W_FRAC = 0.20
CORNER_H_FRAC = 0.30
RANK_SUIT_SPLIT = 0.55
RANK_H = 60
RANK_W = 45
SUIT_H = 40
SUIT_W = 40


def extract_corner(normalised_gray: np.ndarray, use_bottom: bool = False) -> np.ndarray:
    img = normalised_gray
    if use_bottom:
        img = cv2.rotate(img, cv2.ROTATE_180)
    cw = int(CARD_W * CORNER_W_FRAC)
    ch = int(CARD_H * CORNER_H_FRAC)
    return img[:ch, :cw].copy()


def split_rank_suit(corner: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h = corner.shape[0]
    split_y = int(h * RANK_SUIT_SPLIT)
    rank_raw = corner[:split_y, :]
    suit_raw = corner[split_y:, :]
    rank_roi = cv2.resize(rank_raw, (RANK_W, RANK_H), interpolation=cv2.INTER_AREA)
    suit_roi = cv2.resize(suit_raw, (SUIT_W, SUIT_H), interpolation=cv2.INTER_AREA)
    return rank_roi, suit_roi


def binarise(roi: np.ndarray) -> np.ndarray:
    _, binary = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = cv2.findNonZero(binary)
    if coords is None:
        return binary
    x, y, w, h = cv2.boundingRect(coords)
    pad = 3
    x = max(0, x - pad)
    y = max(0, y - pad)
    w = min(binary.shape[1] - x, w + 2 * pad)
    h = min(binary.shape[0] - y, h + 2 * pad)
    return binary[y:y + h, x:x + w]


def extract_rois(
    normalised_gray: np.ndarray,
    use_bottom: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    corner = extract_corner(normalised_gray, use_bottom=use_bottom)
    rank_roi, suit_roi = split_rank_suit(corner)
    return binarise(rank_roi), binarise(suit_roi)
