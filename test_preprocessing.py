"""Tests for the CV preprocessing pipeline using synthetic card images."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from deck_checker.vision import preprocessing, roi


def make_synthetic_card_scene(
    canvas_size: tuple[int, int] = (800, 600),
    card_corners: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a synthetic image with a white card on a dark background.

    Returns (image, true_corners). Corners are in standard order:
    top-left, top-right, bottom-right, bottom-left.
    """
    h, w = canvas_size
    image = np.full((h, w, 3), 30, dtype=np.uint8)  # dark grey background

    if card_corners is None:
        # Default: a roughly upright card with mild perspective tilt.
        card_corners = np.array(
            [[200, 100], [550, 120], [560, 600], [180, 580]],
            dtype=np.float32,
        )

    cv2.fillPoly(image, [card_corners.astype(np.int32)], (245, 245, 245))

    # Stamp some glyph-like marks in the top-left corner so we have a recognizable
    # pattern to verify ROI extraction works end-to-end.
    cv2.putText(
        image,
        "A",
        (card_corners[0].astype(int) + (12, 50)).tolist(),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.4,
        (0, 0, 0),
        3,
    )

    return image, card_corners


class TestDetectCardContour:
    def test_detects_card_on_clean_background(self) -> None:
        image, true_corners = make_synthetic_card_scene()
        detected = preprocessing.detect_card_contour(image)

        assert detected is not None
        assert detected.shape == (4, 2)

        # Detected corners should land near the true corners (within ~15 px).
        true_set = preprocessing.order_corners(true_corners)
        detected_set = preprocessing.order_corners(detected)
        distances = np.linalg.norm(true_set - detected_set, axis=1)
        assert np.all(distances < 15), f"corner errors: {distances}"

    def test_returns_none_on_empty_scene(self) -> None:
        empty = np.full((400, 400, 3), 30, dtype=np.uint8)
        assert preprocessing.detect_card_contour(empty) is None


class TestOrderCorners:
    def test_orders_arbitrary_input(self) -> None:
        # Pass corners in a deliberately scrambled order.
        scrambled = np.array(
            [[100, 100], [100, 500], [400, 500], [400, 100]],
            dtype=np.float32,
        )
        ordered = preprocessing.order_corners(scrambled)

        # Top-left has min sum, bottom-right has max sum.
        assert tuple(ordered[0]) == (100.0, 100.0)
        assert tuple(ordered[2]) == (400.0, 500.0)
        assert tuple(ordered[1]) == (400.0, 100.0)
        assert tuple(ordered[3]) == (100.0, 500.0)

    def test_rejects_wrong_shape(self) -> None:
        with pytest.raises(ValueError):
            preprocessing.order_corners(np.zeros((3, 2)))


class TestPerspectiveCorrect:
    def test_produces_canonical_size(self) -> None:
        image, corners = make_synthetic_card_scene()
        warped = preprocessing.perspective_correct(image, corners)
        assert warped.shape[:2] == (
            preprocessing.NORMALIZED_HEIGHT,
            preprocessing.NORMALIZED_WIDTH,
        )

    def test_card_fills_output(self) -> None:
        # After warping, the four output corners should map to bright (card) pixels,
        # confirming the card fills the canonical rectangle.
        image, corners = make_synthetic_card_scene()
        warped = preprocessing.perspective_correct(image, corners)
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        # Sample 10 px inside each corner to dodge any boundary anti-aliasing.
        samples = [gray[10, 10], gray[10, -10], gray[-10, 10], gray[-10, -10]]
        assert all(s > 200 for s in samples), f"warped corner samples: {samples}"


class TestNormalizeIllumination:
    def test_preserves_size_and_channels(self) -> None:
        image = np.full((350, 250, 3), 128, dtype=np.uint8)
        normalized = preprocessing.normalize_illumination(image)
        assert normalized.shape == image.shape

    def test_rejects_grayscale_input(self) -> None:
        gray = np.full((350, 250), 128, dtype=np.uint8)
        with pytest.raises(ValueError):
            preprocessing.normalize_illumination(gray)


class TestPreprocessFullPipeline:
    def test_happy_path(self) -> None:
        image, _ = make_synthetic_card_scene()
        result = preprocessing.preprocess(image)
        assert result.ok
        assert result.normalized is not None
        assert result.normalized.shape[:2] == (
            preprocessing.NORMALIZED_HEIGHT,
            preprocessing.NORMALIZED_WIDTH,
        )

    def test_empty_image_returns_not_ok(self) -> None:
        empty = np.full((400, 400, 3), 30, dtype=np.uint8)
        result = preprocessing.preprocess(empty)
        assert not result.ok
        assert result.normalized is None


class TestRoiExtraction:
    def test_top_left_corner_shape(self) -> None:
        card = np.zeros(
            (preprocessing.NORMALIZED_HEIGHT, preprocessing.NORMALIZED_WIDTH, 3),
            dtype=np.uint8,
        )
        corner = roi.extract_corner(card, "top-left")
        assert corner.shape == (roi.CORNER_HEIGHT, roi.CORNER_WIDTH, 3)

    def test_split_rank_suit(self) -> None:
        corner = np.zeros((roi.CORNER_HEIGHT, roi.CORNER_WIDTH, 3), dtype=np.uint8)
        rank, suit = roi.split_rank_suit(corner)
        assert rank.shape[0] + suit.shape[0] == roi.CORNER_HEIGHT
        assert rank.shape[1] == suit.shape[1] == roi.CORNER_WIDTH

    def test_full_chain_on_synthetic_card(self) -> None:
        image, _ = make_synthetic_card_scene()
        result = preprocessing.preprocess(image)
        assert result.ok
        assert result.normalized is not None

        corner = roi.extract_corner(result.normalized, "top-left")
        rank, suit = roi.split_rank_suit(corner)
        rank_bin = roi.to_grayscale_clean(rank)

        # The "A" we drew in the synthetic card should show up as non-zero pixels
        # in the rank binary ROI.
        assert int(rank_bin.sum()) > 0
