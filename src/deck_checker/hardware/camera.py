"""
hardware/camera.py — Camera implementations.

RealCamera:   InnoMaker IMX296 Color via libcamera / picamera2 (Pi only).
              Supports XTR hardware trigger mode for Arduino strobe sync.
MockCamera:   Generates synthetic card images deterministically.

Factory:
    make_camera(mock=None) -> CameraProtocol
        mock=None  → auto-detect (real on Pi, mock elsewhere)
        mock=True  → always mock
        mock=False → always real (raises if not on Pi)

XTR trigger workflow (RealCamera in trigger mode):
    1. cam.start()  — resets sensor to free-run
    2. i2c.py trigger on  — MUST come after start()
    3. capture() blocks until brightness > threshold (card frame)
    4. Keepalive frames from Arduino (no flash) are < threshold, skipped
"""
from __future__ import annotations

import logging
import subprocess
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from deck_checker.hardware.base import USE_MOCK, CameraProtocol
from deck_checker.core.models import Card, Rank, Suit

logger = logging.getLogger(__name__)

# IMX296 native resolution
IMX296_W, IMX296_H = 1456, 1088

# Default mock frame resolution
MOCK_W, MOCK_H = 640, 480

# Calibrated for IMX296 + Arduino strobe (session 16.06.2026)
DEFAULT_COLOUR_GAINS = (2.73, 2.13)
DEFAULT_ANALOGUE_GAIN = 2.0
# keepalive frames (no flash) are dark < 60; card frames with flash = 110-145
DEFAULT_BRIGHTNESS_THRESHOLD = 60.0

_I2C_SCRIPT_DEFAULT = (
    Path.home()
    / "cam-imx296raw-trigger"
    / "i2c-tools-python-eeprom-strobe-trigger"
    / "i2c.py"
)


# ─────────────────────────────────────────────────────────────────────────────
# Mock camera
# ─────────────────────────────────────────────────────────────────────────────

def _make_mock_frame(card: Optional[Card] = None, brightness: int = 180) -> np.ndarray:
    frame = np.full((MOCK_H, MOCK_W, 3), 40, dtype=np.uint8)
    if card is None:
        return frame
    margin = 30
    cw = min(250, MOCK_W - 2 * margin)
    ch = min(350, MOCK_H - 2 * margin)
    x0 = (MOCK_W - cw) // 2
    y0 = (MOCK_H - ch) // 2
    frame[y0:y0 + ch, x0:x0 + cw] = brightness
    ordinal = list(Rank).index(card.rank)
    for i in range(ordinal + 1):
        rx = x0 + 8 + (i % 5) * 10
        ry = y0 + 8 + (i // 5) * 16
        if rx + 10 < x0 + cw and ry + 12 < y0 + ch:
            frame[ry:ry + 12, rx:rx + 8] = 30
    suit_ordinal = list(Suit).index(card.suit)
    cx_s = x0 + 20
    cy_s = y0 + ch - 40 - suit_ordinal * 5
    radius = 8 + suit_ordinal * 3
    cv2.circle(frame, (cx_s, cy_s), radius, (30, 30, 30), -1)
    return frame


class MockCamera:
    """Synthetic frame source for development / testing."""

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
        logger.info("MockCamera closed")

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


# ─────────────────────────────────────────────────────────────────────────────
# Real camera (Pi only)
# ─────────────────────────────────────────────────────────────────────────────

class RealCamera:
    """
    InnoMaker IMX296 Color camera via picamera2 (Raspberry Pi 5 only).

    Trigger mode (default, use_trigger=True):
        Arduino fires strobe + XTR pulse → camera captures frame.
        capture() blocks until a bright frame arrives (card with flash).
        Keepalive pulses produce dark frames that are silently skipped.

    Free-run mode (use_trigger=False):
        Standard still capture, no external sync.
    """

    def __init__(
        self,
        camera_index: int = 0,
        resolution: tuple[int, int] = (IMX296_W, IMX296_H),
        use_trigger: bool = True,
        trigger_bus: int = 6,
        i2c_script: Optional[Path] = None,
        colour_gains: tuple[float, float] = DEFAULT_COLOUR_GAINS,
        analogue_gain: float = DEFAULT_ANALOGUE_GAIN,
        brightness_threshold: float = DEFAULT_BRIGHTNESS_THRESHOLD,
    ) -> None:
        self._index = camera_index
        self._res = resolution
        self._use_trigger = use_trigger
        self._trigger_bus = trigger_bus
        self._i2c_script = Path(i2c_script) if i2c_script else _I2C_SCRIPT_DEFAULT
        self._colour_gains = colour_gains
        self._analogue_gain = analogue_gain
        self._threshold = brightness_threshold
        self._picam = None

    def open(self) -> None:
        try:
            from picamera2 import Picamera2  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "picamera2 not installed. Run: sudo apt install -y python3-picamera2"
            ) from exc

        self._picam = Picamera2(self._index)

        if self._use_trigger:
            # Video config + huge FrameDurationLimits so camera waits for trigger
            config = self._picam.create_video_configuration(
                main={"size": self._res, "format": "RGB888"},
                controls={
                    "AnalogueGain": self._analogue_gain,
                    "AeEnable": False,
                    "AwbEnable": False,
                    "ColourGains": self._colour_gains,
                    "FrameDurationLimits": (100, 1_000_000_000),
                },
            )
        else:
            config = self._picam.create_still_configuration(
                main={"size": self._res, "format": "BGR888"},
                controls={
                    "AnalogueGain": self._analogue_gain,
                    "AeEnable": False,
                    "AwbEnable": False,
                    "ColourGains": self._colour_gains,
                },
            )

        self._picam.configure(config)
        self._picam.start()
        time.sleep(0.5)

        if self._use_trigger:
            self._enable_trigger()

        logger.info(
            "RealCamera[%d] opened — trigger=%s", self._index, self._use_trigger
        )

    def close(self) -> None:
        if self._picam is not None:
            self._picam.stop()
            self._picam.close()
            self._picam = None
        logger.info("RealCamera[%d] closed", self._index)

    def capture(self) -> np.ndarray:
        """
        Return next card frame (BGR).

        In trigger mode blocks until a bright frame (card + flash) arrives;
        dark keepalive frames are consumed silently.
        In free-run mode returns the next frame immediately.
        """
        if self._picam is None:
            raise RuntimeError("RealCamera is not open")

        if self._use_trigger:
            while True:
                frame = self._picam.capture_array("main")
                brightness = float(np.mean(frame[::8, ::8]))
                if brightness >= self._threshold:
                    logger.debug("Card frame bright=%.1f", brightness)
                    # picamera2 RGB888 output is effectively BGR — no cvtColor needed
                    return frame
                logger.debug("Keepalive skipped bright=%.1f", brightness)
        else:
            return self._picam.capture_array("main")

    def set_exposure(self, value: float) -> None:
        gain = 1.0 + float(np.clip(value, 0.0, 1.0)) * 7.0
        if self._picam is not None:
            self._picam.set_controls({"AnalogueGain": gain})

    @property
    def resolution(self) -> tuple[int, int]:
        return self._res

    def __enter__(self) -> "RealCamera":
        self.open()
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def _enable_trigger(self) -> None:
        """Activate XTR trigger mode via i2c.py (must run AFTER cam.start())."""
        if not self._i2c_script.exists():
            logger.warning("i2c.py not found at %s — trigger NOT enabled", self._i2c_script)
            return
        try:
            result = subprocess.run(
                ["sudo", "python3", str(self._i2c_script),
                 "trigger", "on", "--bus", str(self._trigger_bus)],
                capture_output=True, text=True, timeout=15,
            )
            if result.returncode == 0:
                logger.info("XTR trigger enabled (bus %d)", self._trigger_bus)
            else:
                logger.error("XTR trigger failed: %s", result.stderr.strip())
        except Exception as exc:
            logger.error("XTR trigger error: %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def make_camera(
    mock: Optional[bool] = None,
    **kwargs,
) -> CameraProtocol:
    """
    Return the appropriate camera implementation.

    mock=None  → auto-detect (real on Pi, mock elsewhere)
    mock=True  → always mock
    mock=False → always real
    """
    use_mock = USE_MOCK if mock is None else mock

    if use_mock:
        logger.info("Using MockCamera")
        return MockCamera(**kwargs)

    logger.info("Using RealCamera")
    return RealCamera(**kwargs)
