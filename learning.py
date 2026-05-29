"""
Two-pass deck learning.

Pass 1 — Calibration
    Feed the Ace of Spades.  Measure mean brightness of the card ROI.
    Compute a target exposure multiplier so the card fills ~60 % of the
    dynamic range (ideal for template matching).

Pass 2 — Template capture
    For each of the 52 standard cards + 1 joker:
      - Acquire image (real camera or mock callable).
      - Run preprocessing pipeline.
      - Extract rank_roi and suit_roi via the ROI module.
      - Store binarised ROIs in the TemplateLibrary.

The camera interface is an abstract callable:
    capture() -> np.ndarray   (BGR frame)

This design lets us inject a mock during tests and on Windows
without any hardware attached.
"""
from __future__ import annotations

import logging
from typing import Callable, Protocol

import cv2
import numpy as np

from deck_checker.core.models import Card, Rank, Suit
from deck_checker.vision.preprocessing import preprocess
from deck_checker.vision.recognition import TemplateLibrary
from deck_checker.vision.roi import extract_rois

logger = logging.getLogger(__name__)

# Full set of cards to learn (52 + joker stored as AS duplicate for simplicity)
LEARNING_ORDER: list[Card] = [
    Card(rank=r, suit=s)
    for s in Suit
    for r in Rank
    if r != Rank.JOKER
]

# Ace of Spades is the calibration card (first card in a fresh shoe)
CALIBRATION_CARD = Card(rank=Rank.ACE, suit=Suit.SPADES)

# Brightness target for calibration (0-255 mean of normalised ROI)
TARGET_BRIGHTNESS = 140.0


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------

class CaptureCallable(Protocol):
    """Anything that returns a BGR frame when called with no arguments."""
    def __call__(self) -> np.ndarray: ...


class ExposureCallable(Protocol):
    """Set camera exposure (0.0 = darkest, 1.0 = brightest)."""
    def __call__(self, value: float) -> None: ...


# ---------------------------------------------------------------------------
# Pass 1 — calibration
# ---------------------------------------------------------------------------

def calibrate_exposure(
    capture: CaptureCallable,
    set_exposure: ExposureCallable | None = None,
    *,
    max_iterations: int = 10,
    tolerance: float = 10.0,
) -> float:
    """
    Adjust exposure until the Ace of Spades ROI is near TARGET_BRIGHTNESS.

    Parameters
    ----------
    capture:        Callable that returns a BGR frame (live camera or mock).
    set_exposure:   Callable to set normalised exposure [0, 1].  Pass None
                    to skip hardware control (records current brightness only).
    max_iterations: Safety limit on the calibration loop.
    tolerance:      Acceptable distance from TARGET_BRIGHTNESS (intensity units).

    Returns
    -------
    Final exposure multiplier applied (1.0 if set_exposure is None).
    """
    exposure = 0.5  # start mid-range
    if set_exposure is not None:
        set_exposure(exposure)

    for iteration in range(max_iterations):
        frame = capture()
        result = preprocess(frame)
        if result is None:
            logger.warning("Calibration pass %d: card not detected, retrying", iteration)
            continue

        _, normalised = result
        brightness = float(normalised.mean())
        logger.debug(
            "Calibration pass %d: brightness=%.1f, exposure=%.3f",
            iteration, brightness, exposure,
        )

        if abs(brightness - TARGET_BRIGHTNESS) <= tolerance:
            logger.info(
                "Calibration converged at pass %d: brightness=%.1f, exposure=%.3f",
                iteration, brightness, exposure,
            )
            return exposure

        if set_exposure is None:
            # No hardware control — just measure and return
            return exposure

        # Proportional adjustment
        ratio = TARGET_BRIGHTNESS / max(brightness, 1.0)
        exposure = float(np.clip(exposure * ratio, 0.05, 1.0))
        set_exposure(exposure)

    logger.warning(
        "Calibration did not converge after %d iterations; using exposure=%.3f",
        max_iterations, exposure,
    )
    return exposure


# ---------------------------------------------------------------------------
# Pass 2 — template capture
# ---------------------------------------------------------------------------

class LearningError(Exception):
    """Raised when a card cannot be learned after retries."""


def learn_card(
    card: Card,
    capture: CaptureCallable,
    *,
    max_retries: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Capture and extract ROIs for a single card.

    Returns
    -------
    (rank_roi_binary, suit_roi_binary)

    Raises
    ------
    LearningError if card not detected after max_retries attempts.
    """
    for attempt in range(max_retries):
        frame = capture()
        result = preprocess(frame)
        if result is None:
            logger.warning(
                "Card %s: preprocessing failed (attempt %d/%d)",
                card, attempt + 1, max_retries,
            )
            continue

        _, normalised = result
        rank_roi, suit_roi = extract_rois(normalised, use_bottom=False)

        # Sanity: ROI must have non-trivial content
        if rank_roi.sum() == 0 or suit_roi.sum() == 0:
            logger.warning("Card %s: empty ROI (attempt %d/%d)", card, attempt + 1, max_retries)
            continue

        return rank_roi, suit_roi

    raise LearningError(f"Failed to learn card {card} after {max_retries} attempts")


def run_learning_pass(
    capture: CaptureCallable,
    set_exposure: ExposureCallable | None = None,
    on_progress: Callable[[int, int, Card], None] | None = None,
    *,
    cards_to_learn: list[Card] | None = None,
    calibrate: bool = True,
) -> TemplateLibrary:
    """
    Execute the full two-pass learning sequence.

    Parameters
    ----------
    capture:        Frame source (camera or mock).
    set_exposure:   Exposure setter for Pass 1 calibration.
    on_progress:    Optional callback(current, total, card) for UI progress.
    cards_to_learn: Override the default 52-card learning order (for testing).
    calibrate:      If False, skip Pass 1 (useful when exposure is pre-set).

    Returns
    -------
    Populated TemplateLibrary ready for use by the recogniser.
    """
    library = TemplateLibrary()
    cards = cards_to_learn if cards_to_learn is not None else LEARNING_ORDER

    # ── Pass 1: exposure calibration ────────────────────────────────────────
    if calibrate:
        logger.info("Learning Pass 1: calibrating exposure on %s", CALIBRATION_CARD)
        calibrate_exposure(capture, set_exposure)

    # ── Pass 2: capture all cards ────────────────────────────────────────────
    logger.info("Learning Pass 2: capturing %d card templates", len(cards))
    for idx, card in enumerate(cards):
        if on_progress is not None:
            on_progress(idx, len(cards), card)

        try:
            rank_roi, suit_roi = learn_card(card, capture)
        except LearningError as exc:
            logger.error("Skipping card %s: %s", card, exc)
            continue

        # Accumulate into the library
        # For duplicate suits/ranks across cards, we keep the last captured
        # (or could average — template matching is robust to either approach)
        library.rank_templates[card.rank] = rank_roi
        library.suit_templates[card.suit] = suit_roi

        logger.debug("Learned %s (%d/%d)", card, idx + 1, len(cards))

    logger.info(
        "Learning complete: %d rank templates, %d suit templates",
        library.rank_count(), library.suit_count(),
    )
    return library
