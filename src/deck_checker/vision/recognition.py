"""
vision/recognition.py -- Combined-index template matching.

Algorithm:
1. extract_index_roi(frame) -> BGR index crop (rank + suit together).
2. canon(roi) -> 200x150 min(G,B) greyscale, fitted + padded to canvas.
3. is_red(roi) -> bool -- split into red/black groups (halves search space).
4. NCC (TM_CCOEFF_NORMED) against every template in the colour group -> argmax.
5. confidence = best NCC score.

TemplateLibrary: one canonical 200x150 uint8 image per Card (52 entries).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from deck_checker.core.models import Card, RecognitionResult
from deck_checker.vision.roi import WindowConfig, extract_index_roi

CANON_W: int = 200
CANON_H: int = 150
CONFIDENCE_THRESHOLD: float = 0.55
DUP_NCC: float = 0.93


@dataclass
class TemplateLibrary:
    """Combined rank+suit index templates. One 200x150 canon image per Card."""
    templates: dict = field(default_factory=dict)   # Card -> np.ndarray
    red_flags: dict = field(default_factory=dict)   # Card -> bool
    rank_templates: dict = field(default_factory=dict)  # Rank -> ndarray (compat)
    suit_templates: dict = field(default_factory=dict)  # Suit -> ndarray (compat)

    def is_ready(self) -> bool:
        return bool(self.templates) or bool(self.rank_templates)

    def card_count(self) -> int:
        return len(self.templates)

    def rank_count(self) -> int:
        return len({c.rank for c in self.templates})

    def suit_count(self) -> int:
        return len({c.suit for c in self.templates})


def canon(roi_bgr: np.ndarray) -> np.ndarray:
    """Normalise BGR index ROI to 200x150 uint8 min(G,B) canvas."""
    b, g, _ = cv2.split(roi_bgr)
    gb = np.minimum(g, b)
    h, w = gb.shape
    scale = min(CANON_W / w, CANON_H / h)
    nw = max(1, int(w * scale))
    nh = max(1, int(h * scale))
    resized = cv2.resize(gb, (nw, nh))
    pad_val = int(np.median(resized))
    canvas = np.full((CANON_H, CANON_W), pad_val, dtype=np.uint8)
    y0 = (CANON_H - nh) // 2
    x0 = (CANON_W - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


def is_red(roi_bgr: np.ndarray) -> bool:
    """Return True if the card is red (Hearts/Diamonds)."""
    b, g, r = cv2.split(roi_bgr.astype(np.int16))
    gb = np.minimum(g, b)
    # Use bright pixels as reference so median stays valid when ink fills >50% of ROI
    bright = gb[gb > 40]
    if bright.size < 10:
        return False
    threshold = float(np.median(bright)) * 0.6
    symbols = gb < threshold
    if symbols.sum() < 50:
        return False
    return float(np.mean((r - np.maximum(g, b))[symbols])) > 15


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    return float(cv2.matchTemplate(a, b, cv2.TM_CCOEFF_NORMED)[0, 0])


def recognise_card(
    frame_bgr: np.ndarray,
    library: TemplateLibrary,
    *,
    roi_cfg: WindowConfig | None = None,
) -> RecognitionResult:
    """Recognise a single card from a raw BGR camera frame."""
    if not library.is_ready():
        return RecognitionResult(card=None, confidence=0.0, method="template")

    roi, info = extract_index_roi(frame_bgr, cfg=roi_cfg)
    if roi is None:
        return RecognitionResult(card=None, confidence=0.0, method="template",
                                 raw_rank=info)

    query = canon(roi)
    red = is_red(roi)

    candidates = [
        (card, tmpl)
        for card, tmpl in library.templates.items()
        if library.red_flags.get(card, False) == red
    ]
    if not candidates:
        candidates = list(library.templates.items())

    scores = [(card, _ncc(query, tmpl)) for card, tmpl in candidates]
    scores.sort(key=lambda x: x[1], reverse=True)
    best_card, best_score = scores[0]

    return RecognitionResult(
        card=best_card,
        confidence=best_score,
        method="template",
        raw_rank=best_card.rank.value,
        raw_suit=best_card.suit.value,
    )


def recognise_batch(
    frames: list,
    library: TemplateLibrary,
    *,
    roi_cfg: WindowConfig | None = None,
    dedup_threshold: float = DUP_NCC,
) -> list:
    """Recognise a list of raw frames, skipping duplicate flashes."""
    results = []
    prev_canon = None

    for frame in frames:
        roi, _ = extract_index_roi(frame, cfg=roi_cfg)
        if roi is None:
            continue
        q = canon(roi)
        if prev_canon is not None and _ncc(prev_canon, q) >= dedup_threshold:
            prev_canon = q
            continue
        prev_canon = q
        results.append(recognise_card(frame, library, roi_cfg=roi_cfg))

    return results
