"""
Card recogniser — template matching primary path, YOLO fallback stub.

Algorithm
---------
1. Extract rank ROI and suit ROI from the normalised card image.
2. Resize each stored template to the same size as the query ROI.
3. Match with cv2.matchTemplate(TM_CCOEFF_NORMED).
4. Best match across all stored templates wins (argmax of max scores).
5. Repeat from the bottom-right corner if top-left confidence is low.
6. If best confidence < CONFIDENCE_THRESHOLD → return low-confidence result
   (caller decides whether to retry via YOLO or request manual input).

Template library structure (managed by storage.library):
    {Card -> np.ndarray}  — loaded rank templates (binary, variable size)
    {Suit -> np.ndarray}  — loaded suit templates (binary, variable size)

This module is pure-vision: it receives numpy arrays and returns
RecognitionResult dataclasses.  No I/O here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from deck_checker.core.models import Card, Rank, RecognitionResult, Suit
from deck_checker.vision.roi import extract_rois

# Minimum normalised cross-correlation score to accept a match
CONFIDENCE_THRESHOLD = 0.82

# If top-left corner score < this, also try the bottom-right corner
RETRY_THRESHOLD = 0.65

# Minimum size (px) for resized template — avoid degenerate tiny templates
MIN_TEMPLATE_DIM = 8


@dataclass
class TemplateLibrary:
    """
    In-memory collection of rank and suit templates.

    Populated by storage.library.load_library(); kept separate from
    disk I/O so the recogniser can be tested without a filesystem.
    """
    rank_templates: dict[Rank, np.ndarray] = field(default_factory=dict)
    suit_templates: dict[Suit, np.ndarray] = field(default_factory=dict)

    def is_ready(self) -> bool:
        return bool(self.rank_templates) and bool(self.suit_templates)

    def rank_count(self) -> int:
        return len(self.rank_templates)

    def suit_count(self) -> int:
        return len(self.suit_templates)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _match_one(query: np.ndarray, template: np.ndarray) -> float:
    """
    Compute TM_CCOEFF_NORMED score between query and template.

    The template is resized to match the query dimensions if they differ.
    Returns a score in [-1, 1]; 1.0 is a perfect match.
    """
    qh, qw = query.shape[:2]
    th, tw = template.shape[:2]

    if qh < MIN_TEMPLATE_DIM or qw < MIN_TEMPLATE_DIM:
        return 0.0
    if th < MIN_TEMPLATE_DIM or tw < MIN_TEMPLATE_DIM:
        return 0.0

    if (th, tw) != (qh, qw):
        resized = cv2.resize(template, (qw, qh), interpolation=cv2.INTER_AREA)
    else:
        resized = template

    # matchTemplate with equal-size images returns a 1×1 result
    result = cv2.matchTemplate(
        query.astype(np.float32),
        resized.astype(np.float32),
        cv2.TM_CCOEFF_NORMED,
    )
    return float(result[0, 0])


def _best_rank(
    rank_roi: np.ndarray,
    rank_templates: dict[Rank, np.ndarray],
) -> tuple[Rank, float]:
    """Return (best_rank, score) across all rank templates."""
    best_rank: Optional[Rank] = None
    best_score = -1.0

    for rank, tmpl in rank_templates.items():
        score = _match_one(rank_roi, tmpl)
        if score > best_score:
            best_score = score
            best_rank = rank

    if best_rank is None:
        raise ValueError("rank_templates is empty")
    return best_rank, best_score


def _best_suit(
    suit_roi: np.ndarray,
    suit_templates: dict[Suit, np.ndarray],
) -> tuple[Suit, float]:
    """Return (best_suit, score) across all suit templates."""
    best_suit: Optional[Suit] = None
    best_score = -1.0

    for suit, tmpl in suit_templates.items():
        score = _match_one(suit_roi, tmpl)
        if score > best_score:
            best_score = score
            best_suit = suit

    if best_suit is None:
        raise ValueError("suit_templates is empty")
    return best_suit, best_score


def _combined_confidence(rank_score: float, suit_score: float) -> float:
    """Geometric mean — both rank and suit must be confident."""
    return float(np.sqrt(max(rank_score, 0.0) * max(suit_score, 0.0)))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def recognise_card(
    normalised_gray: np.ndarray,
    library: TemplateLibrary,
    *,
    allow_bottom_retry: bool = True,
) -> RecognitionResult:
    """
    Recognise a single card from its normalised grayscale image.

    Parameters
    ----------
    normalised_gray:    250×350 CLAHE-normalised grayscale card image.
    library:            Populated TemplateLibrary.
    allow_bottom_retry: If True, retry using the bottom-right corner when
                        the top-left confidence is below RETRY_THRESHOLD.

    Returns
    -------
    RecognitionResult with card, confidence, and method="template".
    confidence < CONFIDENCE_THRESHOLD means the match is uncertain.
    """
    if not library.is_ready():
        return RecognitionResult(card=None, confidence=0.0, method="template")

    # ── Pass 1: top-left corner ──────────────────────────────────────────────
    rank_roi, suit_roi = extract_rois(normalised_gray, use_bottom=False)
    rank1, rank_score1 = _best_rank(rank_roi, library.rank_templates)
    suit1, suit_score1 = _best_suit(suit_roi, library.suit_templates)
    conf1 = _combined_confidence(rank_score1, suit_score1)

    if conf1 >= CONFIDENCE_THRESHOLD:
        return RecognitionResult(
            card=Card(rank=rank1, suit=suit1),
            confidence=conf1,
            method="template",
            raw_rank=rank1.value,
            raw_suit=suit1.value,
        )

    # ── Pass 2: bottom-right corner (card may be upside-down) ───────────────
    if allow_bottom_retry:
        rank_roi2, suit_roi2 = extract_rois(normalised_gray, use_bottom=True)
        rank2, rank_score2 = _best_rank(rank_roi2, library.rank_templates)
        suit2, suit_score2 = _best_suit(suit_roi2, library.suit_templates)
        conf2 = _combined_confidence(rank_score2, suit_score2)

        if conf2 > conf1:
            best_card = Card(rank=rank2, suit=suit2)
            best_conf = conf2
            best_rank_val, best_suit_val = rank2.value, suit2.value
        else:
            best_card = Card(rank=rank1, suit=suit1)
            best_conf = conf1
            best_rank_val, best_suit_val = rank1.value, suit1.value
    else:
        best_card = Card(rank=rank1, suit=suit1)
        best_conf = conf1
        best_rank_val, best_suit_val = rank1.value, suit1.value

    return RecognitionResult(
        card=best_card,
        confidence=best_conf,
        method="template",
        raw_rank=best_rank_val,
        raw_suit=best_suit_val,
    )


def recognise_batch(
    frames: list[np.ndarray],
    library: TemplateLibrary,
) -> list[RecognitionResult]:
    """
    Recognise a list of card images in sequence.

    Convenience wrapper around recognise_card for scanning a full shoe.
    """
    return [recognise_card(frame, library) for frame in frames]
