"""
Tests for hardware abstraction layer (mock implementations only).
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


class TestProtocolConformance:
    def test_mock_camera(self): assert isinstance(MockCamera(), CameraProtocol)
    def test_mock_trigger(self): assert isinstance(MockTrigger(), TriggerProtocol)
    def test_mock_motor(self): assert isinstance(MockMotor(), MotorProtocol)
    def test_mock_hall(self): assert isinstance(MockHallSensor(), HallSensorProtocol)
    def test_mock_strobe(self): assert isinstance(MockStrobe(), StrobeProtocol)


class TestMockCamera:
    def test_capture_requires_open(self):
        with pytest.raises(RuntimeError):
            MockCamera().capture()

    def test_context_manager(self):
        with MockCamera() as cam:
            frame = cam.capture()
        assert frame is not None

    def test_capture_shape(self):
        with MockCamera() as cam:
            frame = cam.capture()
        assert frame.shape == (MOCK_H, MOCK_W, 3)
        assert frame.dtype == np.uint8

    def test_resolution(self):
        assert MockCamera().resolution == (MOCK_W, MOCK_H)

    def test_exposure_clamps(self):
        cam = MockCamera()
        cam.set_exposure(2.0); assert cam._exposure == 1.0
        cam.set_exposure(-1.0); assert cam._exposure == 0.0

    def test_exposure_affects_brightness(self):
        card = Card(Rank.ACE, Suit.SPADES)
        with MockCamera(mode="static", static_card=card) as cam:
            cam.set_exposure(0.1); dark = cam.capture().mean()
            cam.set_exposure(0.9); bright = cam.capture().mean()
        assert bright > dark

    def test_static_none_dark(self):
        with MockCamera(mode="static", static_card=None, noise=0) as cam:
            assert cam.capture().mean() < 50

    def test_sequence_cycles(self):
        cards = [Card(Rank.ACE, Suit.SPADES), Card(Rank.KING, Suit.HEARTS)]
        cam = MockCamera(mode="sequence", card_sequence=cards, noise=0)
        cam.open()
        cam.capture(); cam.capture(); cam.capture()
        assert cam._index == 3
        cam.close()

    def test_noise_zero_deterministic(self):
        card = Card(Rank.ACE, Suit.SPADES)
        with MockCamera(mode="static", static_card=card, noise=0) as cam:
            assert np.array_equal(cam.capture(), cam.capture())

    def test_noise_nonzero_varies(self):
        card = Card(Rank.ACE, Suit.SPADES)
        with MockCamera(mode="static", static_card=card, noise=20) as cam:
            assert not np.array_equal(cam.capture(), cam.capture())


class TestMakeCameraFactory:
    def test_auto_returns_correct_type(self):
        from deck_checker.hardware.base import IS_PI
        cam = make_camera()
        if IS_PI:
            from deck_checker.hardware.camera import RealCamera
            assert isinstance(cam, RealCamera)
        else:
            assert isinstance(cam, MockCamera)
    def test_force_mock(self):
        assert isinstance(make_camera(mock=True), MockCamera)
    def test_force_real_raises_on_non_pi(self):
        if USE_MOCK:
            with pytest.raises(RuntimeError):
                make_camera(mock=False)


class TestMockTrigger:
    def test_initially_clear(self):
        assert not MockTrigger().is_card_present()

    def test_feed_sets_present(self):
        t = MockTrigger()
        t.feed_card(hold_s=0.5)
        time.sleep(0.02)
        assert t.is_card_present()

    def test_feed_auto_clears(self):
        t = MockTrigger()
        t.feed_card(hold_s=0.05)
        time.sleep(0.15)
        assert not t.is_card_present()

    def test_wait_for_card(self):
        t = MockTrigger()
        threading.Timer(0.05, t.feed_card).start()
        assert t.wait_for_card(timeout_s=1.0) is True

    def test_wait_timeout(self):
        assert MockTrigger().wait_for_card(timeout_s=0.1) is False

    def test_wait_for_clear(self):
        assert MockTrigger().wait_for_clear(timeout_s=0.5) is True

    def test_auto_feed(self):
        t = MockTrigger(interval_s=0.05)
        t.start_auto()
        received = []
        for _ in range(3):
            if t.wait_for_card(timeout_s=0.5):
                received.append(True)
                t.wait_for_clear(timeout_s=0.5)
        t.stop_auto()
        assert len(received) >= 2


class TestMockMotor:
    def test_initially_stopped(self): assert not MockMotor().is_running()
    def test_start(self):
        m = MockMotor(); m.start(); assert m.is_running()
    def test_stop(self):
        m = MockMotor(); m.start(); m.stop(); assert not m.is_running()
    def test_speed_clamped(self):
        m = MockMotor(); m.start(2.0); assert m.speed == 1.0
    def test_speed_zero_on_stop(self):
        m = MockMotor(); m.start(0.8); m.stop(); assert m.speed == 0.0


class TestMockHallSensor:
    def test_default_closed(self): assert MockHallSensor().is_closed()
    def test_initially_open(self): assert not MockHallSensor(initially_closed=False).is_closed()
    def test_set_closed(self):
        h = MockHallSensor(initially_closed=False); h.set_closed(True); assert h.is_closed()
    def test_set_open(self):
        h = MockHallSensor(); h.set_closed(False); assert not h.is_closed()


class TestMockStrobe:
    def test_initially_off(self): assert not MockStrobe().is_on
    def test_on(self): s = MockStrobe(); s.on(); assert s.is_on
    def test_off(self): s = MockStrobe(); s.on(); s.off(); assert not s.is_on
    def test_on_count(self):
        s = MockStrobe(); s.on(); s.on(); assert s.on_count == 2
    def test_pulse_count(self):
        s = MockStrobe(); s.pulse(1.0); s.pulse(1.0); assert s.pulse_count == 2
    def test_pulse_auto_off(self):
        s = MockStrobe(); s.pulse(50)
        assert s.is_on
        time.sleep(0.1)
        assert not s.is_on
    def test_reset_counters(self):
        s = MockStrobe(); s.on(); s.pulse(1.0); s.reset_counters()
        assert s.pulse_count == 0 and s.on_count == 0


class TestFactories:
    def test_trigger(self): assert isinstance(make_trigger(mock=True), MockTrigger)
    def test_motor(self): assert isinstance(make_motor(mock=True), MockMotor)
    def test_hall(self): assert isinstance(make_hall_sensor(mock=True), MockHallSensor)
    def test_strobe(self): assert isinstance(make_strobe(mock=True), MockStrobe)
    def test_auto_returns_correct_type(self):
        from deck_checker.hardware.base import IS_PI
        if IS_PI:
            from deck_checker.hardware.gpio import RealTrigger, RealMotor, RealHallSensor
            from deck_checker.hardware.strobe import RealStrobe
            assert isinstance(make_trigger(), RealTrigger)
            assert isinstance(make_motor(), RealMotor)
            assert isinstance(make_hall_sensor(), RealHallSensor)
            assert isinstance(make_strobe(), RealStrobe)
        else:
            assert isinstance(make_trigger(), MockTrigger)
            assert isinstance(make_motor(), MockMotor)
            assert isinstance(make_hall_sensor(), MockHallSensor)
            assert isinstance(make_strobe(), MockStrobe)
