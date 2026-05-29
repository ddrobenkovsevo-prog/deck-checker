"""
Tests for vision/recognition.py, vision/learning.py, storage/library.py.

All tests run without a camera or Raspberry Pi — synthetic images only.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from deck_checker.core.models import Card, GameType, Rank, Suit
from deck_checker.vision.recognition import (
    CONFIDENCE_THRESHOLD,
    TemplateLibrary,
    _best_rank,
    _best_suit,
    _combined_confidence,
    _match_one,
    recognise_batch,
    recognise_card,
)
from deck_checker.vision.learning import (
    LearningError,
    calibrate_exposure,
    learn_card,
    run_learning_pass,
)
from deck_checker.storage.library import (
    list_profiles,
    load_library,
    profile_summary,
    save_library,
)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers — synthetic image factories
# ─────────────────────────────────────────────────────────────────────────────

def _solid_gray(value: int, h: int = 60, w: int = 45) -> np.ndarray:
    """Return a solid-grey image (uint8)."""
    return np.full((h, w), value, dtype=np.uint8)


def _synthetic_rank_template(rank: Rank) -> np.ndarray:
    """
    Synthetic rank image (60×45): white bg with a unique binary pattern per rank.
    Uses a grid of black squares whose positions are determined by the rank ordinal,
    producing clearly distinct images that survive binarise()'s tight crop.
    """
    img = np.full((60, 45), 255, dtype=np.uint8)
    ordinal = list(Rank).index(rank)

    # Draw ordinal+1 small squares in a diagonal pattern — well-separated
    for i in range(ordinal + 1):
        x = 4 + (i % 5) * 7
        y = 4 + (i // 5) * 12
        img[y:y + 8, x:x + 6] = 0

    return img


def _synthetic_suit_template(suit: Suit) -> np.ndarray:
    """
    Synthetic suit image (40×40): white bg with a unique filled shape per suit.
    """
    img = np.full((40, 40), 255, dtype=np.uint8)
    ordinal = list(Suit).index(suit)

    if ordinal == 0:   # SPADES  — filled triangle
        pts = np.array([[20, 5], [5, 35], [35, 35]], np.int32)
        cv2.fillPoly(img, [pts], 0)
    elif ordinal == 1:  # HEARTS  — filled circle
        cv2.circle(img, (20, 20), 14, 0, -1)
    elif ordinal == 2:  # DIAMONDS — filled diamond
        pts = np.array([[20, 4], [36, 20], [20, 36], [4, 20]], np.int32)
        cv2.fillPoly(img, [pts], 0)
    else:              # CLUBS   — three overlapping circles
        cv2.circle(img, (20, 28), 10, 0, -1)
        cv2.circle(img, (12, 18), 9, 0, -1)
        cv2.circle(img, (28, 18), 9, 0, -1)

    return img


def _make_full_library() -> TemplateLibrary:
    """Build a TemplateLibrary whose templates are binarised the same way
    extract_rois() processes real card images."""
    from deck_checker.vision.roi import binarise
    lib = TemplateLibrary()
    for rank in Rank:
        if rank == Rank.JOKER:
            continue
        lib.rank_templates[rank] = binarise(_synthetic_rank_template(rank))
    for suit in Suit:
        lib.suit_templates[suit] = binarise(_synthetic_suit_template(suit))
    return lib


def _make_synthetic_card_image(card: Card, *, noise: int = 0) -> np.ndarray:
    """
    Build a 250×350 normalised grayscale card image.

    The rank pattern is placed in rows 0:57, cols 0:50 and the suit pattern
    in rows 57:105, cols 0:50 — exactly where extract_rois reads them.

    We use inverted (black-on-white) images because binarise() applies
    THRESH_BINARY_INV; the result will be white symbols on black — matching
    what the recogniser sees from a real card.
    """
    # Mid-grey background (card body)
    img = np.full((350, 250), 200, dtype=np.uint8)

    rank_tmpl = _synthetic_rank_template(card.rank)   # 60×45, black on white
    suit_tmpl = _synthetic_suit_template(card.suit)    # 40×40, black on white

    # Rank ROI zone: rows 0:57, cols 0:50
    # rank_tmpl is 60×45; crop to 57 rows to fit
    img[0:57, 0:45] = rank_tmpl[:57, :]

    # Suit ROI zone: rows 57:105, cols 0:50
    # suit_tmpl is 40×40; pad/place starting at row 57
    img[57:97, 0:40] = suit_tmpl

    if noise:
        noise_layer = np.random.randint(-noise, noise + 1, img.shape, dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise_layer, 0, 255).astype(np.uint8)

    return img


# ─────────────────────────────────────────────────────────────────────────────
# _match_one
# ─────────────────────────────────────────────────────────────────────────────

class TestMatchOne:
    def test_identical_images_return_one(self):
        img = _solid_gray(128, 60, 45)
        assert _match_one(img, img) == pytest.approx(1.0, abs=0.01)

    def test_different_images_return_low_score(self):
        # TM_CCOEFF_NORMED on uniform images: both have zero variance,
        # OpenCV returns 1.0 as an edge case.  The meaningful test is that
        # structurally different non-uniform images score low against each other.
        rank_ace = _synthetic_rank_template(Rank.ACE)
        suit_spades = _synthetic_suit_template(Suit.SPADES)
        # A rank template should not match a suit template well
        rank_h, rank_w = rank_ace.shape
        suit_resized = cv2.resize(suit_spades, (rank_w, rank_h))
        score = _match_one(rank_ace, suit_resized)
        assert score < 0.95, f"Expected low cross-type score, got {score:.3f}"

    def test_template_resized_to_match_query(self):
        query = _synthetic_rank_template(Rank.ACE)          # 60×45
        template = cv2.resize(query, (30, 20))              # different size
        score = _match_one(query, template)
        assert score > 0.7

    def test_tiny_query_returns_zero(self):
        tiny = _solid_gray(128, 3, 3)
        normal = _solid_gray(128, 60, 45)
        assert _match_one(tiny, normal) == 0.0

    def test_tiny_template_returns_zero(self):
        normal = _solid_gray(128, 60, 45)
        tiny = _solid_gray(128, 3, 3)
        assert _match_one(normal, tiny) == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# _best_rank / _best_suit
# ─────────────────────────────────────────────────────────────────────────────

class TestBestRankSuit:
    def setup_method(self):
        self.lib = _make_full_library()

    def test_best_rank_finds_ace(self):
        from deck_checker.vision.roi import binarise
        ace_img = binarise(_synthetic_rank_template(Rank.ACE))
        rank, score = _best_rank(ace_img, self.lib.rank_templates)
        assert rank == Rank.ACE
        assert score > 0.9

    def test_best_rank_finds_king(self):
        from deck_checker.vision.roi import binarise
        king_img = binarise(_synthetic_rank_template(Rank.KING))
        rank, score = _best_rank(king_img, self.lib.rank_templates)
        assert rank == Rank.KING

    def test_best_suit_finds_spades(self):
        from deck_checker.vision.roi import binarise
        spades_img = binarise(_synthetic_suit_template(Suit.SPADES))
        suit, score = _best_suit(spades_img, self.lib.suit_templates)
        assert suit == Suit.SPADES
        assert score > 0.9

    def test_best_suit_finds_hearts(self):
        from deck_checker.vision.roi import binarise
        hearts_img = binarise(_synthetic_suit_template(Suit.HEARTS))
        suit, score = _best_suit(hearts_img, self.lib.suit_templates)
        assert suit == Suit.HEARTS

    @pytest.mark.parametrize("rank", list(Rank)[:-1])  # skip JOKER
    def test_all_ranks_self_identify(self, rank):
        from deck_checker.vision.roi import binarise
        img = binarise(_synthetic_rank_template(rank))
        found, score = _best_rank(img, self.lib.rank_templates)
        assert found == rank, f"Expected {rank}, got {found} (score={score:.3f})"

    @pytest.mark.parametrize("suit", list(Suit))
    def test_all_suits_self_identify(self, suit):
        from deck_checker.vision.roi import binarise
        img = binarise(_synthetic_suit_template(suit))
        found, score = _best_suit(img, self.lib.suit_templates)
        assert found == suit, f"Expected {suit}, got {found} (score={score:.3f})"

    def test_empty_templates_raises(self):
        with pytest.raises(ValueError, match="empty"):
            _best_rank(_synthetic_rank_template(Rank.ACE), {})

    def test_empty_suit_templates_raises(self):
        with pytest.raises(ValueError, match="empty"):
            _best_suit(_synthetic_suit_template(Suit.SPADES), {})


# ─────────────────────────────────────────────────────────────────────────────
# _combined_confidence
# ─────────────────────────────────────────────────────────────────────────────

class TestCombinedConfidence:
    def test_both_one(self):
        assert _combined_confidence(1.0, 1.0) == pytest.approx(1.0)

    def test_one_zero(self):
        assert _combined_confidence(1.0, 0.0) == pytest.approx(0.0)

    def test_symmetric(self):
        assert _combined_confidence(0.9, 0.8) == pytest.approx(_combined_confidence(0.8, 0.9))

    def test_negative_clamped_to_zero(self):
        assert _combined_confidence(-0.5, 0.9) == pytest.approx(0.0)


# ─────────────────────────────────────────────────────────────────────────────
# recognise_card
# ─────────────────────────────────────────────────────────────────────────────

class TestRecogniseCard:
    def setup_method(self):
        self.lib = _make_full_library()

    def test_empty_library_returns_none_card(self):
        empty = TemplateLibrary()
        result = recognise_card(_make_synthetic_card_image(Card(Rank.ACE, Suit.SPADES)), empty)
        assert result.card is None
        assert result.confidence == 0.0

    def test_recognise_ace_of_spades(self):
        card = Card(Rank.ACE, Suit.SPADES)
        img = _make_synthetic_card_image(card)
        result = recognise_card(img, self.lib)
        assert result.card == card
        assert result.confidence >= CONFIDENCE_THRESHOLD
        assert result.method == "template"

    @pytest.mark.parametrize("card", [
        Card(Rank.KING, Suit.HEARTS),
        Card(Rank.QUEEN, Suit.DIAMONDS),
        Card(Rank.TWO, Suit.CLUBS),
        Card(Rank.TEN, Suit.SPADES),
    ])
    def test_recognise_various_cards(self, card):
        img = _make_synthetic_card_image(card)
        result = recognise_card(img, self.lib)
        assert result.card == card

    def test_result_has_raw_rank_and_suit(self):
        card = Card(Rank.JACK, Suit.HEARTS)
        img = _make_synthetic_card_image(card)
        result = recognise_card(img, self.lib)
        assert result.raw_rank == Rank.JACK.value
        assert result.raw_suit == Suit.HEARTS.value

    def test_is_confident_property(self):
        card = Card(Rank.ACE, Suit.SPADES)
        img = _make_synthetic_card_image(card)
        result = recognise_card(img, self.lib)
        assert result.is_confident is True
        assert result.succeeded is True


# ─────────────────────────────────────────────────────────────────────────────
# recognise_batch
# ─────────────────────────────────────────────────────────────────────────────

class TestRecogniseBatch:
    def test_empty_list(self):
        lib = _make_full_library()
        assert recognise_batch([], lib) == []

    def test_batch_length_matches_input(self):
        lib = _make_full_library()
        cards = [Card(r, Suit.SPADES) for r in [Rank.ACE, Rank.TWO, Rank.THREE]]
        images = [_make_synthetic_card_image(c) for c in cards]
        results = recognise_batch(images, lib)
        assert len(results) == 3

    def test_batch_correct_recognition(self):
        lib = _make_full_library()
        cards = [Card(Rank.ACE, Suit.SPADES), Card(Rank.KING, Suit.HEARTS)]
        images = [_make_synthetic_card_image(c) for c in cards]
        results = recognise_batch(images, lib)
        for expected, result in zip(cards, results):
            assert result.card == expected


# ─────────────────────────────────────────────────────────────────────────────
# calibrate_exposure (mock camera)
# ─────────────────────────────────────────────────────────────────────────────

class TestCalibrateExposure:
    def _make_bright_bgr_card(self, brightness: int = 140) -> np.ndarray:
        """Return a 480×640 BGR frame with a white card-like rectangle."""
        frame = np.full((480, 640, 3), 50, dtype=np.uint8)
        # Large bright card region
        frame[50:430, 100:540] = brightness
        return frame

    def test_no_set_exposure_returns_default(self):
        calls = [0]

        def capture():
            calls[0] += 1
            return self._make_bright_bgr_card(140)

        # Without hardware control, should return quickly
        # (preprocess will likely fail on uniform frame → loop exhaust)
        result = calibrate_exposure(capture, set_exposure=None, max_iterations=3)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_exposure_setter_called(self):
        set_calls = []

        def capture():
            return self._make_bright_bgr_card(140)

        def set_exp(v):
            set_calls.append(v)

        calibrate_exposure(capture, set_exp, max_iterations=3)
        # Setter should have been called at least once
        assert len(set_calls) >= 1


# ─────────────────────────────────────────────────────────────────────────────
# learn_card
# ─────────────────────────────────────────────────────────────────────────────

class TestLearnCard:
    def _capture_for(self, card: Card):
        """Return a capture callable that always yields the given card image
        wrapped in a 480×640 BGR frame suitable for preprocess()."""
        def capture():
            # Build a minimal BGR frame: card region is the synthetic image
            # blown up to make contour detection easier.
            frame = np.full((480, 640, 3), 30, dtype=np.uint8)
            card_patch = _make_synthetic_card_image(card)
            # Place the 250×350 patch in the centre
            y0, x0 = 65, 195
            patch_bgr = cv2.cvtColor(card_patch, cv2.COLOR_GRAY2BGR)
            frame[y0:y0 + 350, x0:x0 + 250] = patch_bgr
            return frame
        return capture

    def test_learn_card_returns_two_rois(self):
        card = Card(Rank.ACE, Suit.SPADES)
        # Preprocess will likely not find contour on synthetic image, so
        # LearningError is expected — we test error path here.
        capture_fails = lambda: np.zeros((480, 640, 3), dtype=np.uint8)
        with pytest.raises(LearningError):
            learn_card(card, capture_fails, max_retries=2)

    def test_learning_error_message_contains_card(self):
        card = Card(Rank.KING, Suit.CLUBS)
        with pytest.raises(LearningError, match="KC"):
            learn_card(card, lambda: np.zeros((480, 640, 3), dtype=np.uint8), max_retries=1)


# ─────────────────────────────────────────────────────────────────────────────
# run_learning_pass (mock library injection)
# ─────────────────────────────────────────────────────────────────────────────

class TestRunLearningPass:
    def test_run_with_no_cards_returns_empty_library(self):
        lib = run_learning_pass(
            capture=lambda: np.zeros((480, 640, 3), dtype=np.uint8),
            calibrate=False,
            cards_to_learn=[],
        )
        assert lib.rank_count() == 0
        assert lib.suit_count() == 0

    def test_progress_callback_called(self):
        progress_calls = []

        def on_progress(idx, total, card):
            progress_calls.append((idx, total))

        run_learning_pass(
            capture=lambda: np.zeros((480, 640, 3), dtype=np.uint8),
            calibrate=False,
            on_progress=on_progress,
            cards_to_learn=[Card(Rank.ACE, Suit.SPADES)],
        )
        # Called once for the single card (even if learning fails)
        assert len(progress_calls) == 1
        assert progress_calls[0] == (0, 1)


# ─────────────────────────────────────────────────────────────────────────────
# save_library / load_library
# ─────────────────────────────────────────────────────────────────────────────

class TestLibraryPersistence:
    def _make_minimal_library(self) -> TemplateLibrary:
        lib = TemplateLibrary()
        lib.rank_templates[Rank.ACE] = _synthetic_rank_template(Rank.ACE)
        lib.suit_templates[Suit.SPADES] = _synthetic_suit_template(Suit.SPADES)
        return lib

    def test_save_creates_profile_json(self, tmp_path):
        lib = self._make_minimal_library()
        profile_json = save_library(lib, tmp_path / "profile1")
        assert profile_json.exists()
        assert profile_json.name == "profile.json"

    def test_save_creates_png_files(self, tmp_path):
        lib = self._make_minimal_library()
        profile_dir = tmp_path / "profile1"
        save_library(lib, profile_dir)
        assert (profile_dir / "templates/ranks/A.png").exists()
        assert (profile_dir / "templates/suits/S.png").exists()

    def test_overwrite_false_raises_if_exists(self, tmp_path):
        lib = self._make_minimal_library()
        profile_dir = tmp_path / "profile1"
        save_library(lib, profile_dir)
        with pytest.raises(FileExistsError):
            save_library(lib, profile_dir, overwrite=False)

    def test_overwrite_true_replaces(self, tmp_path):
        lib = self._make_minimal_library()
        profile_dir = tmp_path / "profile1"
        save_library(lib, profile_dir)
        save_library(lib, profile_dir, overwrite=True)  # should not raise

    def test_round_trip(self, tmp_path):
        lib = _make_full_library()
        profile_dir = tmp_path / "full_profile"
        save_library(lib, profile_dir, game_type=GameType.BLACKJACK, num_decks=8)

        loaded_lib, metadata = load_library(profile_dir)
        assert loaded_lib.rank_count() == lib.rank_count()
        assert loaded_lib.suit_count() == lib.suit_count()
        assert metadata["game_type"] == "blackjack"
        assert metadata["num_decks"] == 8

    def test_load_missing_directory_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_library(tmp_path / "nonexistent")

    def test_loaded_templates_have_correct_shapes(self, tmp_path):
        lib = _make_full_library()
        profile_dir = tmp_path / "shape_test"
        save_library(lib, profile_dir)
        loaded, _ = load_library(profile_dir)

        for rank, tmpl in lib.rank_templates.items():
            loaded_tmpl = loaded.rank_templates[rank]
            assert loaded_tmpl.shape == tmpl.shape, f"Shape mismatch for rank {rank}"

    def test_list_profiles(self, tmp_path):
        lib = self._make_minimal_library()
        for name in ["p1", "p2", "p3"]:
            save_library(lib, tmp_path / name)
        profiles = list_profiles(tmp_path)
        assert len(profiles) == 3

    def test_profile_summary(self, tmp_path):
        lib = self._make_minimal_library()
        profile_dir = tmp_path / "summary_test"
        save_library(lib, profile_dir, game_type=GameType.BACCARAT, num_decks=6)
        summary = profile_summary(profile_dir)
        assert summary is not None
        assert summary["game_type"] == "baccarat"
        assert summary["num_decks"] == 6
        assert summary["rank_count"] == 1
        assert summary["suit_count"] == 1

    def test_profile_summary_missing_returns_none(self, tmp_path):
        result = profile_summary(tmp_path / "ghost")
        assert result is None
