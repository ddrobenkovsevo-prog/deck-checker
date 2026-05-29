"""
hardware/gpio.py — GPIO-based hardware: IR trigger, DC motor, hall sensor.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Optional

from deck_checker.hardware.base import (
    USE_MOCK,
    TriggerProtocol,
    MotorProtocol,
    HallSensorProtocol,
)

logger = logging.getLogger(__name__)

IR_TRIGGER_PIN = 17
MOTOR_IN1_PIN  = 27
MOTOR_IN2_PIN  = 22
MOTOR_PWM_PIN  = 18
HALL_PIN       = 23
PWM_FREQ_HZ    = 1000


class MockTrigger:
    def __init__(self, interval_s: float = 0.8) -> None:
        self._interval = interval_s
        self._present  = False
        self._lock     = threading.Lock()
        self._event    = threading.Event()
        self._clear_ev = threading.Event()
        self._clear_ev.set()
        self._thread: Optional[threading.Thread] = None
        self._running  = False

    def is_card_present(self) -> bool:
        with self._lock:
            return self._present

    def wait_for_card(self, timeout_s: float = 5.0) -> bool:
        return self._event.wait(timeout=timeout_s)

    def wait_for_clear(self, timeout_s: float = 5.0) -> bool:
        return self._clear_ev.wait(timeout=timeout_s)

    def start_auto(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop_auto(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def feed_card(self, hold_s: float = 0.15) -> None:
        self._set_present(True)
        threading.Timer(hold_s, lambda: self._set_present(False)).start()

    def _set_present(self, present: bool) -> None:
        with self._lock:
            self._present = present
        if present:
            self._clear_ev.clear()
            self._event.set()
        else:
            self._event.clear()
            self._clear_ev.set()

    def _loop(self) -> None:
        while self._running:
            time.sleep(self._interval)
            if self._running:
                self.feed_card(hold_s=0.12)


class RealTrigger:
    def __init__(self, pin: int = IR_TRIGGER_PIN) -> None:
        self._pin = pin
        self._gpio = None

    def _setup(self) -> None:
        if self._gpio is not None:
            return
        import RPi.GPIO as GPIO
        self._gpio = GPIO
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self._pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)

    def is_card_present(self) -> bool:
        self._setup()
        return self._gpio.input(self._pin) == self._gpio.LOW

    def wait_for_card(self, timeout_s: float = 5.0) -> bool:
        self._setup()
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.is_card_present():
                return True
            time.sleep(0.002)
        return False

    def wait_for_clear(self, timeout_s: float = 5.0) -> bool:
        self._setup()
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if not self.is_card_present():
                return True
            time.sleep(0.002)
        return False

    def cleanup(self) -> None:
        if self._gpio:
            self._gpio.cleanup(self._pin)


class MockMotor:
    def __init__(self) -> None:
        self._running = False
        self._speed   = 0.0

    def start(self, speed: float = 1.0) -> None:
        self._running = True
        self._speed   = float(max(0.0, min(1.0, speed)))
        logger.info("MockMotor started at speed=%.2f", self._speed)

    def stop(self) -> None:
        self._running = False
        self._speed   = 0.0
        logger.info("MockMotor stopped")

    def is_running(self) -> bool:
        return self._running

    @property
    def speed(self) -> float:
        return self._speed


class RealMotor:
    def __init__(
        self,
        in1: int = MOTOR_IN1_PIN,
        in2: int = MOTOR_IN2_PIN,
        pwm_pin: int = MOTOR_PWM_PIN,
        freq: int = PWM_FREQ_HZ,
    ) -> None:
        self._in1  = in1
        self._in2  = in2
        self._pwm_pin = pwm_pin
        self._freq = freq
        self._gpio = None
        self._pwm  = None
        self._running = False

    def _setup(self) -> None:
        if self._gpio is not None:
            return
        import RPi.GPIO as GPIO
        self._gpio = GPIO
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self._in1,     GPIO.OUT)
        GPIO.setup(self._in2,     GPIO.OUT)
        GPIO.setup(self._pwm_pin, GPIO.OUT)
        self._pwm = GPIO.PWM(self._pwm_pin, self._freq)
        self._pwm.start(0)

    def start(self, speed: float = 1.0) -> None:
        self._setup()
        speed = float(max(0.0, min(1.0, speed)))
        self._gpio.output(self._in1, self._gpio.HIGH)
        self._gpio.output(self._in2, self._gpio.LOW)
        self._pwm.ChangeDutyCycle(speed * 100)
        self._running = True

    def stop(self) -> None:
        if self._gpio is None:
            return
        self._pwm.ChangeDutyCycle(0)
        self._gpio.output(self._in1, self._gpio.LOW)
        self._gpio.output(self._in2, self._gpio.LOW)
        self._running = False

    def is_running(self) -> bool:
        return self._running

    def cleanup(self) -> None:
        self.stop()
        if self._pwm:
            self._pwm.stop()
        if self._gpio:
            self._gpio.cleanup([self._in1, self._in2, self._pwm_pin])


class MockHallSensor:
    def __init__(self, initially_closed: bool = True) -> None:
        self._closed = initially_closed

    def is_closed(self) -> bool:
        return self._closed

    def set_closed(self, closed: bool) -> None:
        self._closed = closed


class RealHallSensor:
    def __init__(self, pin: int = HALL_PIN) -> None:
        self._pin  = pin
        self._gpio = None

    def _setup(self) -> None:
        if self._gpio is not None:
            return
        import RPi.GPIO as GPIO
        self._gpio = GPIO
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self._pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)

    def is_closed(self) -> bool:
        self._setup()
        return self._gpio.input(self._pin) == self._gpio.LOW

    def cleanup(self) -> None:
        if self._gpio:
            self._gpio.cleanup(self._pin)


def make_trigger(mock: Optional[bool] = None, **kwargs) -> TriggerProtocol:
    use_mock = USE_MOCK if mock is None else mock
    return MockTrigger(**kwargs) if use_mock else RealTrigger(**kwargs)


def make_motor(mock: Optional[bool] = None, **kwargs) -> MotorProtocol:
    use_mock = USE_MOCK if mock is None else mock
    return MockMotor(**kwargs) if use_mock else RealMotor(**kwargs)


def make_hall_sensor(mock: Optional[bool] = None, **kwargs) -> HallSensorProtocol:
    use_mock = USE_MOCK if mock is None else mock
    return MockHallSensor(**kwargs) if use_mock else RealHallSensor(**kwargs)
