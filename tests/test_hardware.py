"""
Tests for hardware abstraction layer (mock implementations only).

All tests run without any GPIO or camera hardware.
"""
from __future__ import annotations

import time
import threading

import numpy as np
import pytest

from deck_checker.core.models import Card, Rank, Suit
from deck_checker.hardware.base import USE_MOCK, CameraProtocol, TriggerProtocol, MotorProtocol, HallSensorProtocol, StrobeProtocol
from deck_checker.hardware.camera import MockCamera, make_camera, MOCK_W, MOCK_H
from deck_checker.hardware.gpio import MockTrigger, MockMotor, MockHallSensor, make_trigger, make_motor, make_hall_sensor
from deck_checker.hardware.strobe import MockStrobe, make_strobe


# ─────────────────────────────────────────────────────────────────────────────
# Protocol conformance
# ─────────────────────────────────────────────────────────────────────────────

class TestProtocolConformance:
    def test_mock_camera_implements_protocol(self):
        cam = MockCamera()
        assert isinstance(cam, CameraProtocol)

    def test_mock_trigger_implements_protocol(self):
        assert isinstance(MockTrigger(), TriggerProtocol)

    def test_mock_motor_implements_protocol(self):
        assert isinstance(MockMotor(), MotorProtocol)

    def test_mock_hall_implements_protocol(self):
        assert isinstance(MockHallSensor(), HallSensorProtocol)

    def test_mock_strobe_implements_protocol(self):
        assert isinstance(MockStrobe(), StrobeProtocol)


# ─────────────────────────────────────────────────────────────────────────────
# MockCamera
# ─────────────────────────────────────────────────────────────────────────────

class TestMockCamera:
    def test_capture_requires_open(self):
        cam = MockCamera()
        with pytest.raises(RuntimeError, match="not open"):
            cam.capture()

    def test_context_manager_opens_and_closes(self):
        with MockCamera() as cam:
            frame = cam.capture()
        assert frame is not None

    def test_capture_returns_correct_shape(self):
        with MockCamera() as cam:
            frame = cam.capture()
        assert frame.shape == (MOCK_H, MOCK_W, 3)
        assert frame.dtype == np.uint8

    def test_resolution_property(self):
        cam = MockCamera()
        assert cam.resolution == (MOCK_W, MOCK_H)

    def test_set_exposure_clamps_to_range(self):
        cam = MockCamera()
        cam.set_exposure(2.0)
        assert cam._exposure == 1.0
        cam.set_exposure(-1.0)
        assert cam._exposure == 0.0

    def test_exposure_affects_brightness(self):
        with MockCamera(mode="static", static_card=Card(Rank.ACE, Suit.SPADES)) as cam:
            cam.set_exposure(0.1)
            dark = cam.capture().mean()
            cam.set_exposure(0.9)
            bright = cam.capture().mean()
        assert bright > dark

    def test_static_mode_always_same_card(self):
        card = Card(Rank.KING, Suit.HEARTS)
        with MockCamera(mode="static", static_card=card, noise=0) as cam:
            f1 = cam.capture()
            f2 = cam.capture()
        # Same card → same mean brightness (no noise)
        assert abs(float(f1.mean()) - float(f2.mean())) < 1.0

    def test_static_none_returns_dark_frame(self):
        with MockCamera(mode="static", static_card=None, noise=0) as cam:
            frame = cam.capture()
        # Background is 40 grey
        assert frame.mean() < 50

    def test_sequence_mode_cycles(self):
        cards = [Card(Rank.ACE, Suit.SPADES), Card(Rank.KING, Suit.HEARTS)]
        cam = MockCamera(mode="sequence", card_sequence=cards, noise=0)
        cam.open()
        cam.capture()  # ACE
        cam.capture()  # KING
        cam.capture()  # ACE again (wraps)
        assert cam._index == 3
        cam.close()

    def test_set_sequence_resets_index(self):
        cam = MockCamera(mode="sequence")
        cam.open()
        cam._index = 5
        cam.set_sequence([Card(Rank.TWO, Suit.CLUBS)])
        assert cam._index == 0
        cam.close()

    def test_random_mode_returns_valid_cards(self):
        with MockCamera(mode="random") as cam:
            for _ in range(20):
                frame = cam.capture()
                assert frame is not None

    def test_noise_zero_deterministic(self):
        card = Card(Rank.ACE, Suit.SPADES)
        with MockCamera(mode="static", static_card=card, noise=0) as cam:
            f1 = cam.capture()
            f2 = cam.capture()
        assert np.array_equal(f1, f2)

    def test_noise_nonzero_adds_variation(self):
        card = Card(Rank.ACE, Suit.SPADES)
        with MockCamera(mode="static", static_card=card, noise=20) as cam:
            f1 = cam.capture()
            f2 = cam.capture()
        # With noise=20 two frames will almost never be identical
        assert not np.array_equal(f1, f2)


# ─────────────────────────────────────────────────────────────────────────────
# make_camera factory
# ─────────────────────────────────────────────────────────────────────────────

class TestMakeCameraFactory:
    def test_auto_returns_mock_on_non_pi(self):
        cam = make_camera()
        assert isinstance(cam, MockCamera)

    def test_force_mock(self):
        cam = make_camera(mock=True)
        assert isinstance(cam, MockCamera)

    def test_force_real_raises_on_non_pi(self):
        if USE_MOCK:
            with pytest.raises(RuntimeError, match="Raspberry Pi"):
                make_camera(mock=False)

    def test_kwargs_passed_to_mock(self):
        cam = make_camera(mock=True, mode="static",
                          static_card=Card(Rank.ACE, Suit.SPADES))
        assert isinstance(cam, MockCamera)
        assert cam.mode == "static"


# ─────────────────────────────────────────────────────────────────────────────
# MockTrigger
# ─────────────────────────────────────────────────────────────────────────────

class TestMockTrigger:
    def test_initially_clear(self):
        t = MockTrigger()
        assert not t.is_card_present()

    def test_feed_card_sets_present(self):
        t = MockTrigger()
        t.feed_card(hold_s=0.5)
        time.sleep(0.02)
        assert t.is_card_present()

    def test_feed_card_auto_clears(self):
        t = MockTrigger()
        t.feed_card(hold_s=0.05)
        time.sleep(0.15)
        assert not t.is_card_present()

    def test_wait_for_card_returns_true(self):
        t = MockTrigger()
        threading.Timer(0.05, t.feed_card).start()
        result = t.wait_for_card(timeout_s=1.0)
        assert result is True

    def test_wait_for_card_timeout(self):
        t = MockTrigger()
        result = t.wait_for_card(timeout_s=0.1)
        assert result is False

    def test_wait_for_clear_returns_true_when_no_card(self):
        t = MockTrigger()
        result = t.wait_for_clear(timeout_s=0.5)
        assert result is True

    def test_wait_for_clear_after_card(self):
        t = MockTrigger()
        t.feed_card(hold_s=0.08)
        time.sleep(0.02)
        assert t.is_card_present()
        result = t.wait_for_clear(timeout_s=1.0)
        assert result is True
        assert not t.is_card_present()

    def test_auto_feed_delivers_cards(self):
        t = MockTrigger(interval_s=0.05)
        t.start_auto()
        received = []
        for _ in range(3):
            if t.wait_for_card(timeout_s=0.5):
                received.append(True)
                t.wait_for_clear(timeout_s=0.5)
        t.stop_auto()
        assert len(received) >= 2


# ─────────────────────────────────────────────────────────────────────────────
# MockMotor
# ─────────────────────────────────────────────────────────────────────────────

class TestMockMotor:
    def test_initially_stopped(self):
        m = MockMotor()
        assert not m.is_running()

    def test_start_sets_running(self):
        m = MockMotor()
        m.start()
        assert m.is_running()

    def test_stop_clears_running(self):
        m = MockMotor()
        m.start()
        m.stop()
        assert not m.is_running()

    def test_speed_clamped(self):
        m = MockMotor()
        m.start(speed=2.0)
        assert m.speed == 1.0
        m.start(speed=-0.5)
        assert m.speed == 0.0

    def test_speed_zero_on_stop(self):
        m = MockMotor()
        m.start(0.8)
        m.stop()
        assert m.speed == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# MockHallSensor
# ─────────────────────────────────────────────────────────────────────────────

class TestMockHallSensor:
    def test_initially_closed_by_default(self):
        assert MockHallSensor().is_closed()

    def test_initially_open_when_specified(self):
        assert not MockHallSensor(initially_closed=False).is_closed()

    def test_set_closed(self):
        h = MockHallSensor(initially_closed=False)
        h.set_closed(True)
        assert h.is_closed()

    def test_set_open(self):
        h = MockHallSensor(initially_closed=True)
        h.set_closed(False)
        assert not h.is_closed()


# ─────────────────────────────────────────────────────────────────────────────
# MockStrobe
# ─────────────────────────────────────────────────────────────────────────────

class TestMockStrobe:
    def test_initially_off(self):
        assert not MockStrobe().is_on

    def test_on_turns_on(self):
        s = MockStrobe()
        s.on()
        assert s.is_on

    def test_off_turns_off(self):
        s = MockStrobe()
        s.on()
        s.off()
        assert not s.is_on

    def test_on_count_increments(self):
        s = MockStrobe()
        s.on(); s.on(); s.on()
        assert s.on_count == 3

    def test_pulse_increments_count(self):
        s = MockStrobe()
        s.pulse(1.0); s.pulse(1.0)
        assert s.pulse_count == 2

    def test_pulse_turns_on_briefly(self):
        s = MockStrobe()
        s.pulse(duration_ms=50)
        assert s.is_on   # immediately on
        time.sleep(0.1)
        assert not s.is_on  # auto-off after 50 ms

    def test_reset_counters(self):
        s = MockStrobe()
        s.on(); s.pulse(1.0)
        s.reset_counters()
        assert s.pulse_count == 0
        assert s.on_count == 0


# ─────────────────────────────────────────────────────────────────────────────
# GPIO factories
# ─────────────────────────────────────────────────────────────────────────────

class TestGpioFactories:
    def test_make_trigger_returns_mock(self):
        assert isinstance(make_trigger(mock=True), MockTrigger)

    def test_make_motor_returns_mock(self):
        assert isinstance(make_motor(mock=True), MockMotor)

    def test_make_hall_returns_mock(self):
        assert isinstance(make_hall_sensor(mock=True), MockHallSensor)

    def test_make_strobe_returns_mock(self):
        assert isinstance(make_strobe(mock=True), MockStrobe)

    def test_auto_returns_mock_on_non_pi(self):
        assert isinstance(make_trigger(), MockTrigger)
        assert isinstance(make_motor(), MockMotor)
        assert isinstance(make_hall_sensor(), MockHallSensor)
        assert isinstance(make_strobe(), MockStrobe)
