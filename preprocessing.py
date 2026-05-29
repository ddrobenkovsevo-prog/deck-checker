"""Image preprocessing: card detection, perspective correction, illumination normalization.

The pipeline is purposely staged so each step has a single, testable responsibility:

    raw frame  ─→  detect contour  ─→  4 corners
    4 corners  ─→  perspective warp ─→  250×350 BGR
    250×350    ─→  CLAHE on L      ─→  illumination-normalized card

Sizes and thresholds are tuned for an InnoMaker IMX296 sensor at ~10 cm from the card
with a 6 mm C-mount lens. Anything that depends on hardware lives in the config module,
not here.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# Canonical card size after perspective correction. Aspect 5:7 matches a poker card.
NORMALIZED_WIDTH = 250
NORMALIZED_HEIGHT = 350
NORMALIZED_SIZE = (NORMALIZED_WIDTH, NORMALIZED_HEIGHT)


@dataclass(frozen=True, slots=True)
class DetectionParams:
    """Tunable parameters for card contour detection."""

    blur_kernel: int = 5
    canny_low: int = 50
    canny_high: int = 150
    # Minimum contour area as a fraction of the full image area.
    min_area_ratio: float = 0.05
    # approxPolyDP epsilon as a fraction of the contour perimeter.
    approx_epsilon_ratio: float = 0.02


def detect_card_contour(
    image: np.ndarray,
    params: DetectionParams | None = None,
) -> np.ndarray | None:
    """Find the four corners of the card in the image.

    Returns a (4, 2) array of (x, y) corner coordinates in image space,
    or None if no card-like quadrilateral was found.
    """
    p = params or DetectionParams()
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    blurred = cv2.GaussianBlur(gray, (p.blur_kernel, p.blur_kernel), 0)
    edges = cv2.Canny(blurred, p.canny_low, p.canny_high)

    # Close small gaps so the card outline becomes a connected contour.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    image_area = float(image.shape[0] * image.shape[1])
    min_area = p.min_area_ratio * image_area

    # Sort by area, descending. Try each candidate until one approximates to a quad.
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        area = cv2.contourArea(contour)
        if area < min_area:
            break  # remaining contours are smaller
        peri = cv2.arcLength(contour, closed=True)
        approx = cv2.approxPolyDP(contour, p.approx_epsilon_ratio * peri, closed=True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return approx.reshape(4, 2).astype(np.float32)

    return None


def order_corners(corners: np.ndarray) -> np.ndarray:
    """Order 4 corners as [top-left, top-right, bottom-right, bottom-left].

    The trick: top-left has the smallest x+y, bottom-right the largest.
    Top-right has the smallest y-x, bottom-left the largest.
    """
    if corners.shape != (4, 2):
        raise ValueError(f"Expected (4, 2) corners, got {corners.shape}")

    rect = np.zeros((4, 2), dtype=np.float32)
    s = corners.sum(axis=1)
    d = np.diff(corners, axis=1).flatten()  # y - x

    rect[0] = corners[np.argmin(s)]  # top-left
    rect[2] = corners[np.argmax(s)]  # bottom-right
    rect[1] = corners[np.argmin(d)]  # top-right
    rect[3] = corners[np.argmax(d)]  # bottom-left
    return rect


def perspective_correct(
    image: np.ndarray,
    corners: np.ndarray,
    size: tuple[int, int] = NORMALIZED_SIZE,
) -> np.ndarray:
    """Warp the card so it fills the output rectangle at the canonical size.

    The card might be wider-than-tall (sideways) or taller-than-wide (upright).
    We orient long-side vertical by inspecting the input corners after ordering.
    """
    width, height = size
    src = order_corners(corners)

    # Detect orientation: distance(top-left → top-right) vs (top-left → bottom-left)
    top_edge = float(np.linalg.norm(src[1] - src[0]))
    left_edge = float(np.linalg.norm(src[3] - src[0]))
    if top_edge > left_edge:
        # Card is sideways in the source. Rotate the dst so the result is upright.
        dst = np.array(
            [
                [width - 1, 0],
                [width - 1, height - 1],
                [0, height - 1],
                [0, 0],
            ],
            dtype=np.float32,
        )
    else:
        dst = np.array(
            [
                [0, 0],
                [width - 1, 0],
                [width - 1, height - 1],
                [0, height - 1],
            ],
            dtype=np.float32,
        )

    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image, matrix, (width, height), flags=cv2.INTER_CUBIC)


def normalize_illumination(card_image: np.ndarray) -> np.ndarray:
    """Apply CLAHE on the L channel of LAB to even out lighting.

    CLAHE (Contrast-Limited Adaptive Histogram Equalization) handles gradients
    across the card better than global histogram equalization. We touch only
    luminance, never colour channels, so red and black inks keep their identity.
    """
    if card_image.ndim != 3:
        raise ValueError("normalize_illumination expects a BGR image")

    lab = cv2.cvtColor(card_image, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_chan = clahe.apply(l_chan)
    merged = cv2.merge([l_chan, a_chan, b_chan])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


@dataclass(slots=True)
class PreprocessResult:
    """Bundle of intermediate artifacts from the full preprocess pipeline."""

    corners: np.ndarray | None
    warped: np.ndarray | None
    normalized: np.ndarray | None

    @property
    def ok(self) -> bool:
        return self.normalized is not None


def preprocess(image: np.ndarray) -> PreprocessResult:
    """Run the full preprocessing pipeline.

    Returns a PreprocessResult; check `.ok` before using `.normalized`.
    Intermediate fields are populated even on failure to aid debugging.
    """
    corners = detect_card_contour(image)
    if corners is None:
        return PreprocessResult(corners=None, warped=None, normalized=None)

    warped = perspective_correct(image, corners)
    normalized = normalize_illumination(warped)
    return PreprocessResult(corners=corners, warped=warped, normalized=normalized)
