"""
Tests for core/state_machine.py — all hardware mocked.
"""
from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from deck_checker.core.models import (
    Card, GameType, Rank, RecognitionResult, ScanState, Suit,
)
from deck_checker.core.state_machine import (
    MAX_RETRIES, ScanStateMachine, ScanContext,
)
from deck_checker.hardware.camera import MockCamera
from deck_checker.hardware.gpio import MockHallSensor, MockMotor, MockTrigger
from deck_checker.hardware.strobe import MockStrobe
from deck_checker.vision.recognition import TemplateLibrary
from deck_checker.vision.roi import binarise


def _rank_tmpl(rank):
    import cv2
    img = np.full((60, 45), 255, dtype=np.uint8)
    ordinal = list(Rank).index(rank)
    for i in range(ordinal + 1):
        x = 4 + (i % 5) * 7
        y = 4 + (i // 5) * 12
        img[y:y+8, x:x+6] = 0
    return img

def _suit_tmpl(suit):
    import cv2
    img = np.full((40, 40), 255, dtype=np.uint8)
    o = list(Suit).index(suit)
    if o == 0:
        cv2.fillPoly(img, [np.array([[20,5],[5,35],[35,35]], np.int32)], 0)
    elif o == 1:
        cv2.circle(img, (20,20), 14, 0, -1)
    elif o == 2:
        cv2.fillPoly(img, [np.array([[20,4],[36,20],[20,36],[4,20]], np.int32)], 0)
    else:
        cv2.circle(img,(20,28),10,0,-1)
        cv2.circle(img,(12,18),9,0,-1)
        cv2.circle(img,(28,18),9,0,-1)
    return img

def make_library():
    lib = TemplateLibrary()
    for r in Rank:
        if r != Rank.JOKER:
            lib.rank_templates[r] = binarise(_rank_tmpl(r))
    for s in Suit:
        lib.suit_templates[s] = binarise(_suit_tmpl(s))
    return lib

def full_shoe(num_decks=1):
    return [Card(r,s) for _ in range(num_decks)
            for s in Suit for r in Rank if r != Rank.JOKER]

def make_machine(cards, hall_closed=True, num_decks=1, trigger_interval=0.005):
    library = make_library()
    camera  = MockCamera(mode="static", static_card=None, noise=0)
    trigger = MockTrigger(interval_s=trigger_interval)
    motor   = MockMotor()
    strobe  = MockStrobe()
    hall    = MockHallSensor(initially_closed=hall_closed)
    camera.open()

    card_iter = iter(cards)
    def recognise_fn(frame):
        try:
            card = next(card_iter)
            return RecognitionResult(card=card, confidence=0.97, method="template",
                                     raw_rank=card.rank.value, raw_suit=card.suit.value)
        except StopIteration:
            return RecognitionResult(card=None, confidence=0.0, method="template")

    machine = ScanStateMachine(
        camera=camera, trigger=trigger, motor=motor,
        strobe=strobe, hall=hall, library=library,
        game_type=GameType.BLACKJACK, num_decks=num_decks,
        recognise_fn=recognise_fn,
    )
    return machine, trigger, motor, strobe, hall


def run_until(machine, target, timeout=30.0, trigger=None):
    if trigger:
        trigger.start_auto()
    machine.start_async()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if machine.state == target:
            machine.stop()
            if trigger: trigger.stop_auto()
            return True
        time.sleep(0.02)
    machine.stop()
    if trigger: trigger.stop_auto()
    return False


class TestScanContext:
    def test_reset(self):
        ctx = ScanContext()
        ctx.retry_count = 2; ctx.error_reason = "oops"
        ctx.reset()
        assert ctx.retry_count == 0 and ctx.error_reason == "" and ctx.report is None

    def test_fresh_report(self):
        ctx = ScanContext(game_type=GameType.BACCARAT, num_decks=6)
        r = ctx.fresh_report()
        assert r.game_type == GameType.BACCARAT and r.num_decks == 6

    def test_fresh_report_replaces(self):
        ctx = ScanContext()
        r1 = ctx.fresh_report()
        r2 = ctx.fresh_report()
        assert r1 is not r2 and ctx.report is r2


class TestStateTransitions:
    def test_initial_state(self):
        m, *_ = make_machine([])
        assert m.state == ScanState.INIT

    def test_hardware_ready(self):
        m, *_ = make_machine([])
        m.hardware_ready()
        time.sleep(0.05)
        assert m.state == ScanState.IDLE

    def test_reset_from_success(self):
        m, *_ = make_machine([])
        m._transition(ScanState.SUCCESS)
        time.sleep(0.05)
        m.reset()
        time.sleep(0.05)
        assert m.state == ScanState.IDLE

    def test_reset_from_manual(self):
        m, *_ = make_machine([])
        m._transition(ScanState.MANUAL_VALIDATION)
        time.sleep(0.05)
        m.reset()
        time.sleep(0.05)
        assert m.state == ScanState.IDLE

    def test_callback_fires(self):
        changes = []
        m, *_ = make_machine([])
        m.on_state_change = lambda o, n, c: changes.append(n)
        m._transition(ScanState.IDLE)
        time.sleep(0.1)
        assert ScanState.IDLE in changes


class TestSuccessPath:
    def test_full_shoe_success(self):
        shoe = full_shoe(1)
        m, trigger, *_ = make_machine(shoe, num_decks=1)
        assert run_until(m, ScanState.SUCCESS, timeout=30.0, trigger=trigger)

    def test_report_valid(self):
        shoe = full_shoe(1)
        m, trigger, *_ = make_machine(shoe, num_decks=1)
        reports = []
        m.on_scan_complete = lambda r: reports.append(r)
        run_until(m, ScanState.SUCCESS, timeout=30.0, trigger=trigger)
        assert reports and reports[0].is_valid

    def test_motor_stopped(self):
        shoe = full_shoe(1)
        m, trigger, motor, *_ = make_machine(shoe, num_decks=1)
        run_until(m, ScanState.SUCCESS, timeout=30.0, trigger=trigger)
        assert not motor.is_running()

    def test_strobe_pulsed(self):
        shoe = full_shoe(1)
        m, trigger, _, strobe, _ = make_machine(shoe, num_decks=1)
        run_until(m, ScanState.SUCCESS, timeout=30.0, trigger=trigger)
        assert strobe.pulse_count == len(shoe)

    def test_card_scanned_callback(self):
        shoe = full_shoe(1)
        m, trigger, *_ = make_machine(shoe, num_decks=1)
        scanned = []
        m.on_card_scanned = lambda idx, r: scanned.append(idx)
        run_until(m, ScanState.SUCCESS, timeout=30.0, trigger=trigger)
        assert len(scanned) == len(shoe)


class TestLidOpen:
    def test_waits_for_lid(self):
        shoe = full_shoe(1)
        m, trigger, *_, hall = make_machine(shoe, hall_closed=False)
        trigger.start_auto()
        m.start_async()
        time.sleep(0.3)
        assert m.state == ScanState.IDLE
        m.stop(); trigger.stop_auto()

    def test_lid_close_unblocks(self):
        shoe = full_shoe(1)
        m, trigger, *_, hall = make_machine(shoe, hall_closed=False)
        trigger.start_auto()
        m.start_async()
        time.sleep(0.15)
        hall.set_closed(True)
        deadline = time.monotonic() + 20.0
        reached = False
        while time.monotonic() < deadline:
            if m.state == ScanState.SUCCESS:
                reached = True; break
            time.sleep(0.05)
        m.stop(); trigger.stop_auto()
        assert reached


class TestErrorAndRetry:
    def test_error_on_short_shoe(self):
        short = full_shoe(1)[:26]
        states = []
        m, trigger, *_ = make_machine(short, num_decks=1)
        m.on_state_change = lambda o, n, c: states.append(n)
        trigger.start_auto()
        m.start_async()
        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline:
            if m.state in (ScanState.ERROR, ScanState.MANUAL_VALIDATION):
                break
            time.sleep(0.05)
        m.stop(); trigger.stop_auto()
        assert ScanState.ERROR in states or ScanState.MANUAL_VALIDATION in states

    def test_max_retries_manual_validation(self):
        m, *_ = make_machine([])
        m._ctx.retry_count = MAX_RETRIES
        m._ctx.error_reason = "forced"
        states = []
        m.on_state_change = lambda o, n, c: states.append(n)
        t = threading.Thread(target=m._handle_error, daemon=True)
        t.start(); t.join(timeout=2.0)
        time.sleep(0.1)
        assert ScanState.MANUAL_VALIDATION in states


class TestManualOverride:
    def test_outside_manual_ignored(self):
        m, *_ = make_machine([])
        m._transition(ScanState.IDLE)
        time.sleep(0.05)
        m.manual_override(0, Card(Rank.ACE, Suit.SPADES))
        assert m.state == ScanState.IDLE

    def test_override_resolves_to_success(self):
        m, *_ = make_machine([], num_decks=1)
        m._transition(ScanState.MANUAL_VALIDATION)
        time.sleep(0.05)
        shoe = full_shoe(1)
        report = m._ctx.fresh_report()
        report.scanned = [
            RecognitionResult(card=c, confidence=0.95, method="template")
            for c in shoe
        ]
        report.scanned[0] = RecognitionResult(card=shoe[0], confidence=0.5, method="template")
        states = []
        m.on_state_change = lambda o, n, c: states.append(n)
        assert 0 in report.low_confidence_indices
        m.manual_override(0, shoe[0])
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            if ScanState.SUCCESS in states or ScanState.ERROR in states:
                break
            time.sleep(0.02)
        assert ScanState.SUCCESS in states
