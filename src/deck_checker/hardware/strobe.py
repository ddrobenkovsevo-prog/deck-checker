"""
hardware/strobe.py — White LED strobe controller.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Optional

from deck_checker.hardware.base import USE_MOCK, StrobeProtocol

logger = logging.getLogger(__name__)

STROBE_PIN = 24


class MockStrobe:
    def __init__(self) -> None:
        self._on       = False
        self.pulse_count = 0
        self.on_count    = 0

    def on(self) -> None:
        self._on = True
        self.on_count += 1

    def off(self) -> None:
        self._on = False

    def pulse(self, duration_ms: float = 2.0) -> None:
        self._on = True
        self.pulse_count += 1
        threading.Timer(duration_ms / 1000.0, self._auto_off).start()

    @property
    def is_on(self) -> bool:
        return self._on

    def reset_counters(self) -> None:
        self.pulse_count = 0
        self.on_count    = 0

    def _auto_off(self) -> None:
        self._on = False


class RealStrobe:
    def __init__(self, pin: int = STROBE_PIN) -> None:
        self._pin  = pin
        self._gpio = None

    def _setup(self) -> None:
        if self._gpio is not None:
            return
        import RPi.GPIO as GPIO
        self._gpio = GPIO
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self._pin, GPIO.OUT, initial=GPIO.LOW)

    def on(self) -> None:
        self._setup()
        self._gpio.output(self._pin, self._gpio.HIGH)

    def off(self) -> None:
        self._setup()
        self._gpio.output(self._pin, self._gpio.LOW)

    def pulse(self, duration_ms: float = 2.0) -> None:
        self._setup()
        self._gpio.output(self._pin, self._gpio.HIGH)
        time.sleep(duration_ms / 1000.0)
        self._gpio.output(self._pin, self._gpio.LOW)

    def cleanup(self) -> None:
        self.off()
        if self._gpio:
            self._gpio.cleanup(self._pin)


def make_strobe(mock: Optional[bool] = None, **kwargs) -> StrobeProtocol:
    use_mock = USE_MOCK if mock is None else mock
    return MockStrobe(**kwargs) if use_mock else RealStrobe(**kwargs)
