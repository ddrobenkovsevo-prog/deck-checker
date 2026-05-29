"""
hardware/base.py — Protocol interfaces for all hardware components.
"""
from __future__ import annotations

import platform
import sys
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np


def _detect_pi() -> bool:
    if platform.system() != "Linux":
        return False
    model_file = Path("/proc/device-tree/model")
    try:
        return "Raspberry Pi" in model_file.read_text(errors="ignore")
    except OSError:
        return False


IS_PI:  bool = _detect_pi()
IS_WIN: bool = sys.platform == "win32"
USE_MOCK: bool = not IS_PI


@runtime_checkable
class CameraProtocol(Protocol):
    def open(self) -> None: ...
    def close(self) -> None: ...
    def capture(self) -> np.ndarray: ...
    def set_exposure(self, value: float) -> None: ...
    @property
    def resolution(self) -> tuple[int, int]: ...
    def __enter__(self) -> "CameraProtocol":
        self.open()
        return self
    def __exit__(self, *_) -> None:
        self.close()


@runtime_checkable
class StrobeProtocol(Protocol):
    def on(self) -> None: ...
    def off(self) -> None: ...
    def pulse(self, duration_ms: float = 2.0) -> None: ...


@runtime_checkable
class TriggerProtocol(Protocol):
    def is_card_present(self) -> bool: ...
    def wait_for_card(self, timeout_s: float = 5.0) -> bool: ...
    def wait_for_clear(self, timeout_s: float = 5.0) -> bool: ...


@runtime_checkable
class MotorProtocol(Protocol):
    def start(self, speed: float = 1.0) -> None: ...
    def stop(self) -> None: ...
    def is_running(self) -> bool: ...


@runtime_checkable
class HallSensorProtocol(Protocol):
    def is_closed(self) -> bool: ...
