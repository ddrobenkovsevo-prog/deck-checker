"""
Two-pass deck learning.
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

LEARNING_ORDER: list[Card] = [
    Card(rank=r, suit=s)
    for s in Suit
    for r in Rank
    if r != Rank.JOKER
]

CALIBRATION_CARD = Card(rank=Rank.ACE, suit=Suit.SPADES)
TARGET_BRIGHTNESS = 140.0


class CaptureCallable(Protocol):
    def __call__(self) -> np.ndarray: ...


class ExposureCallable(Protocol):
    def __call__(self, value: float) -> None: ...


def calibrate_exposure(
    capture: CaptureCallable,
    set_exposure: ExposureCallable | None = None,
    *,
    max_iterations: int = 10,
    tolerance: float = 10.0,
) -> float:
    exposure = 0.5
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
        logger.debug("Calibration pass %d: brightness=%.1f, exposure=%.3f",
                     iteration, brightness, exposure)
        if abs(brightness - TARGET_BRIGHTNESS) <= tolerance:
            logger.info("Calibration converged at pass %d: brightness=%.1f, exposure=%.3f",
                        iteration, brightness, exposure)
            return exposure
        if set_exposure is None:
            return exposure
        ratio = TARGET_BRIGHTNESS / max(brightness, 1.0)
        exposure = float(np.clip(exposure * ratio, 0.05, 1.0))
        set_exposure(exposure)
    logger.warning("Calibration did not converge after %d iterations; using exposure=%.3f",
                   max_iterations, exposure)
    return exposure


class LearningError(Exception):
    pass


def learn_card(
    card: Card,
    capture: CaptureCallable,
    *,
    max_retries: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    for attempt in range(max_retries):
        frame = capture()
        result = preprocess(frame)
        if result is None:
            logger.warning("Card %s: preprocessing failed (attempt %d/%d)",
                           card, attempt + 1, max_retries)
            continue
        _, normalised = result
        rank_roi, suit_roi = extract_rois(normalised, use_bottom=False)
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
    library = TemplateLibrary()
    cards = cards_to_learn if cards_to_learn is not None else LEARNING_ORDER
    if calibrate:
        logger.info("Learning Pass 1: calibrating exposure on %s", CALIBRATION_CARD)
        calibrate_exposure(capture, set_exposure)
    logger.info("Learning Pass 2: capturing %d card templates", len(cards))
    for idx, card in enumerate(cards):
        if on_progress is not None:
            on_progress(idx, len(cards), card)
        try:
            rank_roi, suit_roi = learn_card(card, capture)
        except LearningError as exc:
            logger.error("Skipping card %s: %s", card, exc)
            continue
        library.rank_templates[card.rank] = rank_roi
        library.suit_templates[card.suit] = suit_roi
        logger.debug("Learned %s (%d/%d)", card, idx + 1, len(cards))
    logger.info("Learning complete: %d rank templates, %d suit templates",
                library.rank_count(), library.suit_count())
    return library
