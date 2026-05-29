"""
Card image preprocessing pipeline.
"""
from __future__ import annotations

import cv2
import numpy as np

CARD_W, CARD_H = 250, 350


def _order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def find_card_contour(gray: np.ndarray) -> np.ndarray | None:
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    h, w = gray.shape[:2]
    if area < 0.10 * w * h:
        return None
    peri = cv2.arcLength(largest, closed=True)
    approx = cv2.approxPolyDP(largest, 0.02 * peri, closed=True)
    if len(approx) != 4:
        return None
    return _order_points(approx.reshape(4, 2).astype(np.float32))


def perspective_transform(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    dst = np.array(
        [[0, 0], [CARD_W - 1, 0], [CARD_W - 1, CARD_H - 1], [0, CARD_H - 1]],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(corners, dst)
    return cv2.warpPerspective(image, M, (CARD_W, CARD_H))


def normalize_illumination(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def preprocess(bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    corners = find_card_contour(gray)
    if corners is None:
        return None
    warped_bgr = perspective_transform(bgr, corners)
    warped_gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)
    normalised = normalize_illumination(warped_gray)
    return warped_bgr, normalised
