"""
Card image preprocessing pipeline.

Steps:
  1. Find the largest quadrilateral contour (the card).
  2. Apply perspective transform → 250×350 canonical image.
  3. Normalise illumination with CLAHE.
"""
from __future__ import annotations

import cv2
import numpy as np

# Canonical card size (pixels) used throughout the pipeline
CARD_W, CARD_H = 250, 350


def _order_points(pts: np.ndarray) -> np.ndarray:
    """Return points in [top-left, top-right, bottom-right, bottom-left] order."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # top-left
    rect[2] = pts[np.argmax(s)]   # bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect


def find_card_contour(gray: np.ndarray) -> np.ndarray | None:
    """
    Detect the card quadrilateral in a grayscale image.

    Returns a (4, 2) float32 array of ordered corner points,
    or None if no suitable contour is found.

    Uses two strategies and returns the first that finds a valid card:
      1. Canny edge detection (robust for real photos on dark backgrounds)
      2. Otsu threshold (works well for synthetic / high-contrast images)
    """
    h, w = gray.shape[:2]
    min_area = 0.10 * w * h

    def _largest_quad(mask: np.ndarray) -> np.ndarray | None:
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        best = None
        best_area = min_area
        for c in contours:
            area = cv2.contourArea(c)
            if area < best_area:
                continue
            peri = cv2.arcLength(c, closed=True)
            approx = cv2.approxPolyDP(c, 0.03 * peri, closed=True)
            if len(approx) == 4:
                best_area = area
                best = approx
        if best is None:
            return None
        return _order_points(best.reshape(4, 2).astype(np.float32))

    # ── Strategy 1: Canny edges (best for real camera) ──────────────────────
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blurred, 30, 120)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    result = _largest_quad(closed_edges)
    if result is not None:
        return result

    # ── Strategy 2: Otsu threshold (best for synthetic) ─────────────────────
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed_thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel2)
    return _largest_quad(closed_thresh)


def perspective_transform(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """
    Warp the card region to a canonical CARD_W × CARD_H rectangle.

    Parameters
    ----------
    image:   BGR or grayscale source image.
    corners: (4, 2) ordered corner points from find_card_contour().

    Returns
    -------
    Warped image of shape (CARD_H, CARD_W, C).
    """
    dst = np.array(
        [[0, 0], [CARD_W - 1, 0], [CARD_W - 1, CARD_H - 1], [0, CARD_H - 1]],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(corners, dst)
    return cv2.warpPerspective(image, M, (CARD_W, CARD_H))


def normalize_illumination(gray: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE to equalise uneven lighting across the card surface.

    Input/output: single-channel uint8 image.
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def preprocess(bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Full preprocessing pipeline: detect → warp → normalise.

    Parameters
    ----------
    bgr: Raw BGR frame from the camera.

    Returns
    -------
    (warped_bgr, normalised_gray) or None if card not found.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    corners = find_card_contour(gray)
    if corners is None:
        return None

    warped_bgr = perspective_transform(bgr, corners)
    warped_gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)
    normalised = normalize_illumination(warped_gray)
    return warped_bgr, normalised
