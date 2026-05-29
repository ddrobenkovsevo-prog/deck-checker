"""
core/state_machine.py — Deck Checker scan state machine.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Optional

from deck_checker.core.models import (
    Card, GameType, RecognitionResult, ScanReport, ScanState,
)
from deck_checker.hardware.base import (
    CameraProtocol, HallSensorProtocol, MotorProtocol,
    StrobeProtocol, TriggerProtocol,
)
from deck_checker.vision.preprocessing import preprocess
from deck_checker.vision.recognition import TemplateLibrary, recognise_card

logger = logging.getLogger(__name__)

MAX_RETRIES     = 2
SCAN_TIMEOUT_S  = 120.0
CARD_TIMEOUT_S  = 5.0

StateChangeCallback  = Callable[[ScanState, ScanState, dict], None]
CardScannedCallback  = Callable[[int, RecognitionResult], None]
ScanCompleteCallback = Callable[[ScanReport], None]
ErrorCallback        = Callable[[str], None]


@dataclass
class ScanContext:
    game_type:    GameType = GameType.BLACKJACK
    num_decks:    int      = 8
    retry_count:  int      = 0
    scan_start:   float    = 0.0
    report:       Optional[ScanReport] = None
    error_reason: str      = ""

    def reset(self) -> None:
        self.retry_count  = 0
        self.scan_start   = 0.0
        self.report       = None
        self.error_reason = ""

    def fresh_report(self) -> ScanReport:
        self.report = ScanReport(
            game_type=self.game_type,
            num_decks=self.num_decks,
        )
        return self.report


class ScanStateMachine:
    def __init__(
        self,
        camera:   CameraProtocol,
        trigger:  TriggerProtocol,
        motor:    MotorProtocol,
        strobe:   StrobeProtocol,
        hall:     HallSensorProtocol,
        library:  TemplateLibrary,
        game_type: GameType = GameType.BLACKJACK,
        num_decks: int = 8,
        recognise_fn=None,
    ) -> None:
        self._camera  = camera
        self._trigger = trigger
        self._motor   = motor
        self._strobe  = strobe
        self._hall    = hall
        self._library = library
        self._recognise_fn = recognise_fn
        self._state   = ScanState.INIT
        self._lock    = threading.Lock()
        self._stop_ev = threading.Event()
        self._ctx     = ScanContext(game_type=game_type, num_decks=num_decks)
        self.on_state_change:  StateChangeCallback  = lambda *_: None
        self.on_card_scanned:  CardScannedCallback  = lambda *_: None
        self.on_scan_complete: ScanCompleteCallback = lambda *_: None
        self.on_error:         ErrorCallback        = lambda *_: None

    @property
    def state(self) -> ScanState:
        with self._lock:
            return self._state

    @property
    def context(self) -> ScanContext:
        return self._ctx

    def start(self) -> None:
        self._stop_ev.clear()
        self._run()

    def start_async(self) -> threading.Thread:
        self._stop_ev.clear()
        t = threading.Thread(target=self._run, daemon=True, name="StateMachine")
        t.start()
        return t

    def stop(self) -> None:
        self._stop_ev.set()

    def hardware_ready(self) -> None:
        if self.state == ScanState.INIT:
            self._transition(ScanState.IDLE)

    def reset(self) -> None:
        with self._lock:
            if self._state in (ScanState.SUCCESS,
                               ScanState.MANUAL_VALIDATION,
                               ScanState.ERROR):
                self._ctx.reset()
                self._do_transition(ScanState.IDLE)

    def manual_override(self, index: int, card: Card) -> None:
        if self.state != ScanState.MANUAL_VALIDATION:
            logger.warning("manual_override called outside MANUAL_VALIDATION, ignored")
            return
        report = self._ctx.report
        if report is None:
            return
        report.manual_overrides[index] = card
        logger.info("Manual override: index=%d card=%s", index, card)
        if not report.low_confidence_indices:
            if report.is_valid:
                self._transition(ScanState.SUCCESS)
            else:
                self._ctx.error_reason = "Shoe invalid after manual correction"
                self._transition(ScanState.ERROR)

    def _run(self) -> None:
        logger.info("StateMachine started in state %s", self._state)
        self._transition(ScanState.IDLE)
        while not self._stop_ev.is_set():
            state = self.state
            try:
                if state == ScanState.IDLE:
                    self._handle_idle()
                elif state == ScanState.DECK_LOADED:
                    self._handle_deck_loaded()
                elif state == ScanState.SCANNING:
                    self._handle_scanning()
                elif state == ScanState.SUCCESS:
                    self._stop_ev.wait(timeout=1.0)
                elif state == ScanState.ERROR:
                    self._handle_error()
                elif state == ScanState.MANUAL_VALIDATION:
                    self._stop_ev.wait(timeout=1.0)
                else:
                    time.sleep(0.05)
            except Exception as exc:
                logger.exception("Unhandled exception in state %s: %s", state, exc)
                self._ctx.error_reason = f"Internal error: {exc}"
                self._transition(ScanState.ERROR)
        logger.info("StateMachine stopped")

    def _handle_idle(self) -> None:
        if not self._hall.is_closed():
            while not self._hall.is_closed() and not self._stop_ev.is_set():
                time.sleep(0.1)
        detected = self._trigger.wait_for_card(timeout_s=CARD_TIMEOUT_S)
        if detected and not self._stop_ev.is_set():
            self._transition(ScanState.DECK_LOADED)

    def _handle_deck_loaded(self) -> None:
        self._motor.start(speed=0.8)
        self._ctx.fresh_report()
        self._ctx.scan_start = time.monotonic()
        self._transition(ScanState.SCANNING)

    def _handle_scanning(self) -> None:
        report   = self._ctx.report
        expected = self._ctx.num_decks * 52
        while not self._stop_ev.is_set():
            elapsed = time.monotonic() - self._ctx.scan_start
            if elapsed > SCAN_TIMEOUT_S:
                self._ctx.error_reason = f"Scan timeout after {elapsed:.0f}s"
                self._motor.stop()
                self._transition(ScanState.ERROR)
                return
            card_arrived = self._trigger.wait_for_card(timeout_s=CARD_TIMEOUT_S)
            if not card_arrived:
                if len(report.scanned) == expected:
                    break
                self._ctx.error_reason = (
                    f"Card stall: {len(report.scanned)}/{expected} scanned"
                )
                self._motor.stop()
                self._transition(ScanState.ERROR)
                return
            self._strobe.pulse(duration_ms=2.0)
            frame  = self._camera.capture()
            result = self._recognise_frame(frame)
            idx = len(report.scanned)
            report.scanned.append(result)
            self.on_card_scanned(idx, result)
            self._trigger.wait_for_clear(timeout_s=1.0)
            if len(report.scanned) >= expected:
                break
        self._motor.stop()
        if report.is_valid:
            self._transition(ScanState.SUCCESS)
            self.on_scan_complete(report)
        elif report.low_confidence_indices:
            self._ctx.error_reason = (
                f"{len(report.low_confidence_indices)} low-confidence card(s)"
            )
            self._transition(ScanState.ERROR)
        else:
            self._ctx.error_reason = (
                f"Missing: {dict(list(report.missing_cards.items())[:3])}"
            )
            self._transition(ScanState.ERROR)

    def _handle_error(self) -> None:
        self.on_error(self._ctx.error_reason)
        if self._ctx.retry_count < MAX_RETRIES:
            self._ctx.retry_count += 1
            self._ctx.fresh_report()
            self._ctx.scan_start = time.monotonic()
            self._motor.start(speed=0.8)
            self._transition(ScanState.SCANNING)
        else:
            self._transition(ScanState.MANUAL_VALIDATION)

    def _recognise_frame(self, frame) -> RecognitionResult:
        if self._recognise_fn is not None:
            return self._recognise_fn(frame)
        result = preprocess(frame)
        if result is None:
            return RecognitionResult(card=None, confidence=0.0, method="template")
        _, normalised = result
        return recognise_card(normalised, self._library)

    def _transition(self, new_state: ScanState) -> None:
        with self._lock:
            self._do_transition(new_state)

    def _do_transition(self, new_state: ScanState) -> None:
        old_state = self._state
        self._state = new_state
        logger.info("State: %s → %s", old_state.name, new_state.name)
        ctx_snapshot = {
            "retry_count":  self._ctx.retry_count,
            "error_reason": self._ctx.error_reason,
            "cards_scanned": len(self._ctx.report.scanned) if self._ctx.report else 0,
        }
        threading.Thread(
            target=self.on_state_change,
            args=(old_state, new_state, ctx_snapshot),
            daemon=True,
        ).start()
