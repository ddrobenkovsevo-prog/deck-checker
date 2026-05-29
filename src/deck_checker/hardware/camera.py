"""
hardware/camera.py — Camera implementations.
"""
from __future__ import annotations

import logging
import time
from typing import Optional

import cv2
import numpy as np

from deck_checker.hardware.base import USE_MOCK, CameraProtocol
from deck_checker.core.models import Card, Rank, Suit

logger = logging.getLogger(__name__)

IMX296_W, IMX296_H = 1456, 1088
MOCK_W, MOCK_H = 640, 480


def _make_mock_frame(card: Optional[Card] = None, brightness: int = 180) -> np.ndarray:
    frame = np.full((MOCK_H, MOCK_W, 3), 40, dtype=np.uint8)
    if card is None:
        return frame
    margin = 30
    cw = min(250, MOCK_W - 2 * margin)
    ch = min(350, MOCK_H - 2 * margin)
    x0 = (MOCK_W - cw) // 2
    y0 = (MOCK_H - ch) // 2
    frame[y0:y0+ch, x0:x0+cw] = brightness
    ordinal = list(Rank).index(card.rank)
    for i in range(ordinal + 1):
        rx = x0 + 8 + (i % 5) * 10
        ry = y0 + 8 + (i // 5) * 16
        if rx + 10 < x0 + cw and ry + 12 < y0 + ch:
            frame[ry:ry+12, rx:rx+8] = 30
    suit_ordinal = list(Suit).index(card.suit)
    cx_s = x0 + 20
    cy_s = y0 + ch - 40 - suit_ordinal * 5
    radius = 8 + suit_ordinal * 3
    cv2.circle(frame, (cx_s, cy_s), radius, (30, 30, 30), -1)
    return frame


class MockCamera:
    def __init__(
        self,
        mode: str = "random",
        card_sequence: Optional[list[Card]] = None,
        static_card: Optional[Card] = None,
        frame_delay_ms: float = 0.0,
        noise: int = 8,
    ) -> None:
        self.mode = mode
        self._sequence = card_sequence or []
        self._static_card = static_card
        self._delay = frame_delay_ms / 1000.0
        self._noise = noise
        self._index = 0
        self._exposure = 0.5
        self._open = False
        self._all_cards = [
            Card(r, s) for r in Rank if r != Rank.JOKER for s in Suit
        ]

    def open(self) -> None:
        self._open = True
        logger.info("MockCamera opened (mode=%s)", self.mode)

    def close(self) -> None:
        self._open = False

    def capture(self) -> np.ndarray:
        if not self._open:
            raise RuntimeError("MockCamera is not open — call open() first")
        if self._delay:
            time.sleep(self._delay)
        card = self._pick_card()
        brightness = int(40 + self._exposure * 200)
        frame = _make_mock_frame(card, brightness=brightness)
        if self._noise:
            layer = np.random.randint(
                -self._noise, self._noise + 1, frame.shape, dtype=np.int16
            )
            frame = np.clip(frame.astype(np.int16) + layer, 0, 255).astype(np.uint8)
        return frame

    def set_exposure(self, value: float) -> None:
        self._exposure = float(np.clip(value, 0.0, 1.0))

    @property
    def resolution(self) -> tuple[int, int]:
        return MOCK_W, MOCK_H

    def __enter__(self) -> "MockCamera":
        self.open()
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def _pick_card(self) -> Optional[Card]:
        if self.mode == "static":
            return self._static_card
        if self.mode == "sequence":
            if not self._sequence:
                return None
            card = self._sequence[self._index % len(self._sequence)]
            self._index += 1
            return card
        return self._all_cards[np.random.randint(0, len(self._all_cards))]

    def set_sequence(self, cards: list[Card]) -> None:
        self._sequence = list(cards)
        self._index = 0

    def reset_sequence(self) -> None:
        self._index = 0


class RealCamera:
    _EXP_MIN_US = 100
    _EXP_MAX_US = 50_000

    def __init__(
        self,
        camera_index: int = 0,
        resolution: tuple[int, int] = (IMX296_W, IMX296_H),
    ) -> None:
        self._index = camera_index
        self._res = resolution
        self._picam = None
        self._exposure_norm = 0.5

    def open(self) -> None:
        try:
            from picamera2 import Picamera2
        except ImportError as exc:
            raise RuntimeError(
                "picamera2 is not installed. "
                "Run: sudo apt install -y python3-picamera2"
            ) from exc
        self._picam = Picamera2(self._index)
        config = self._picam.create_still_configuration(
            main={"size": self._res, "format": "BGR888"},
            controls={"AeEnable": False, "AwbEnable": False},
        )
        self._picam.configure(config)
        self._picam.start()
        time.sleep(0.2)
        self._apply_exposure(self._exposure_norm)
        logger.info("RealCamera[%d] opened at %dx%d", self._index, *self._res)

    def close(self) -> None:
        if self._picam is not None:
            self._picam.stop()
            self._picam.close()
            self._picam = None

    def capture(self) -> np.ndarray:
        if self._picam is None:
            raise RuntimeError("RealCamera is not open")
        return self._picam.capture_array("main")

    def set_exposure(self, value: float) -> None:
        self._exposure_norm = float(np.clip(value, 0.0, 1.0))
        if self._picam is not None:
            self._apply_exposure(self._exposure_norm)

    @property
    def resolution(self) -> tuple[int, int]:
        return self._res

    def __enter__(self) -> "RealCamera":
        self.open()
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def _apply_exposure(self, norm: float) -> None:
        exp_us = int(self._EXP_MIN_US + norm * (self._EXP_MAX_US - self._EXP_MIN_US))
        self._picam.set_controls({
            "ExposureTime": exp_us,
            "AnalogueGain": 1.0 + norm * 7.0,
        })


def make_camera(mock: Optional[bool] = None, **kwargs) -> CameraProtocol:
    use_mock = USE_MOCK if mock is None else mock
    if use_mock:
        logger.info("Using MockCamera")
        return MockCamera(**kwargs)
    else:
        if not USE_MOCK:
            logger.info("Using RealCamera")
            return RealCamera(**kwargs)
        raise RuntimeError(
            "mock=False requested but not running on Raspberry Pi."
        )
