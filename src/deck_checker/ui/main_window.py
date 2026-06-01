"""
ui/main_window.py — Main kiosk window.

Wires the state machine to the UI screens.
Runs fullscreen on the 10.1" touchscreen (1024×600).
"""
from __future__ import annotations

import logging
import time
from typing import Optional

from PyQt6.QtCore import QObject, QThread, Qt, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QMainWindow, QStackedWidget, QWidget

from deck_checker.core.models import (
    Card, GameType, RecognitionResult, ScanReport, ScanState,
)
from deck_checker.core.state_machine import ScanStateMachine
from deck_checker.hardware.base import IS_PI
from deck_checker.hardware.camera import make_camera
from deck_checker.hardware.gpio import make_hall_sensor, make_motor, make_trigger
from deck_checker.hardware.strobe import make_strobe
from deck_checker.vision.recognition import TemplateLibrary
from deck_checker.ui.screens import IdleScreen, ManualScreen, ResultScreen, ScanningScreen
from deck_checker.ui.theme import COLORS, SCREEN_H, SCREEN_W, STYLESHEET

logger = logging.getLogger(__name__)

# Screen indices in QStackedWidget
IDX_IDLE     = 0
IDX_SCANNING = 1
IDX_RESULT   = 2
IDX_MANUAL   = 3


# ─────────────────────────────────────────────────────────────────────────────
# Bridge — thread-safe signals from state machine → UI thread
# ─────────────────────────────────────────────────────────────────────────────

class MachineBridge(QObject):
    """
    Relay state machine callbacks (called from worker thread)
    to Qt signals (received in the UI thread).
    """
    state_changed  = pyqtSignal(object, object, dict)   # old, new, ctx
    card_scanned   = pyqtSignal(int, object)             # index, RecognitionResult
    scan_complete  = pyqtSignal(object)                  # ScanReport
    error_occurred = pyqtSignal(str)                     # reason


# ─────────────────────────────────────────────────────────────────────────────
# Main Window
# ─────────────────────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    """
    Top-level kiosk window.

    On startup, instantiates hardware (real on Pi, mock elsewhere),
    wires state machine callbacks to UI via MachineBridge, and shows
    the first screen.
    """

    def __init__(
        self,
        library: Optional[TemplateLibrary] = None,
        kiosk: bool = True,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._library   = library or TemplateLibrary()
        self._kiosk     = kiosk
        self._machine: Optional[ScanStateMachine] = None
        self._bridge    = MachineBridge()
        self._scan_start: float = 0.0
        self._last_report: Optional[ScanReport] = None

        self._build_ui()
        self._connect_bridge()
        self._init_hardware()

        if kiosk:
            self.showFullScreen()
        else:
            self.resize(SCREEN_W, SCREEN_H)
            self.show()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        self.setWindowTitle("Deck Checker")
        self.setStyleSheet(STYLESHEET)

        self._stack = QStackedWidget()
        self.setCentralWidget(self._stack)

        self._screen_idle     = IdleScreen()
        self._screen_scanning = ScanningScreen()
        self._screen_result   = ResultScreen()
        self._screen_manual   = ManualScreen()

        self._stack.addWidget(self._screen_idle)      # 0
        self._stack.addWidget(self._screen_scanning)  # 1
        self._stack.addWidget(self._screen_result)    # 2
        self._stack.addWidget(self._screen_manual)    # 3

        # Wire screen signals
        self._screen_idle.start_requested.connect(self._on_manual_start)
        self._screen_scanning.abort_requested.connect(self._on_abort)
        self._screen_result.reset_requested.connect(self._on_reset)
        self._screen_result.reprint_requested.connect(self._on_reprint)
        self._screen_manual.override_submitted.connect(self._on_override)
        self._screen_manual.reset_requested.connect(self._on_reset)

        self._show_screen(IDX_IDLE)

    def _connect_bridge(self) -> None:
        self._bridge.state_changed.connect(self._on_state_changed)
        self._bridge.card_scanned.connect(self._on_card_scanned)
        self._bridge.scan_complete.connect(self._on_scan_complete)
        self._bridge.error_occurred.connect(self._on_error)

    def _init_hardware(self) -> None:
        camera  = make_camera()
        trigger = make_trigger()
        motor   = make_motor()
        strobe  = make_strobe()
        hall    = make_hall_sensor()

        camera.open()

        self._machine = ScanStateMachine(
            camera=camera,
            trigger=trigger,
            motor=motor,
            strobe=strobe,
            hall=hall,
            library=self._library,
        )

        # Wire callbacks → bridge signals (thread-safe)
        self._machine.on_state_change = lambda o, n, c: \
            self._bridge.state_changed.emit(o, n, c)
        self._machine.on_card_scanned = lambda i, r: \
            self._bridge.card_scanned.emit(i, r)
        self._machine.on_scan_complete = lambda rep: \
            self._bridge.scan_complete.emit(rep)
        self._machine.on_error = lambda msg: \
            self._bridge.error_occurred.emit(msg)

        # Start machine in background thread
        self._machine.start_async()
        self._machine.hardware_ready()
        logger.info("Hardware initialised, state machine running")

    # ── Screen helpers ────────────────────────────────────────────────────────

    def _show_screen(self, idx: int) -> None:
        self._stack.setCurrentIndex(idx)

    # ── Bridge slot handlers (UI thread) ─────────────────────────────────────

    @pyqtSlot(object, object, dict)
    def _on_state_changed(self, old: ScanState, new: ScanState, ctx: dict) -> None:
        logger.info("UI state: %s → %s", old.name, new.name)

        if new == ScanState.IDLE:
            self._show_screen(IDX_IDLE)
            if self._last_report:
                verdict = "VALID" if self._last_report.is_valid else "INVALID"
                elapsed = time.monotonic() - self._scan_start
                mins, secs = divmod(int(elapsed), 60)
                self._screen_idle.set_last_scan(
                    f"{verdict}  {mins}:{secs:02d}"
                )

        elif new == ScanState.DECK_LOADED:
            pass  # transitional — scanning follows immediately

        elif new == ScanState.SCANNING:
            num_decks = self._machine.context.num_decks
            self._scan_start = time.monotonic()
            self._screen_scanning.start_scan(num_decks * 52)
            self._show_screen(IDX_SCANNING)

        elif new == ScanState.SUCCESS:
            self._screen_scanning.stop_scan()
            if self._last_report:
                elapsed = time.monotonic() - self._scan_start
                self._screen_result.show_result(self._last_report, elapsed)
            self._show_screen(IDX_RESULT)

        elif new == ScanState.ERROR:
            self._screen_scanning.stop_scan()
            # Stay on scanning screen briefly, then show result with error
            if self._machine.context.report:
                elapsed = time.monotonic() - self._scan_start
                self._screen_result.show_result(
                    self._machine.context.report, elapsed
                )
            self._show_screen(IDX_RESULT)

        elif new == ScanState.MANUAL_VALIDATION:
            report = self._machine.context.report
            if report:
                pending = [
                    (i, str(report.scanned[i].card) if report.scanned[i].card else None)
                    for i in report.low_confidence_indices
                ]
                self._screen_manual.set_pending(pending)
            self._show_screen(IDX_MANUAL)

        # Update lid status on idle screen
        if new in (ScanState.IDLE, ScanState.INIT):
            closed = self._machine._hall.is_closed() if self._machine else False
            self._screen_idle.update_lid_status(closed)

    @pyqtSlot(int, object)
    def _on_card_scanned(self, index: int, result: RecognitionResult) -> None:
        self._screen_scanning.on_card_scanned(index, result)

    @pyqtSlot(object)
    def _on_scan_complete(self, report: ScanReport) -> None:
        self._last_report = report
        logger.info("Scan complete: valid=%s", report.is_valid)

    @pyqtSlot(str)
    def _on_error(self, reason: str) -> None:
        logger.warning("Scan error: %s", reason)

    # ── User action handlers ──────────────────────────────────────────────────

    @pyqtSlot(object, int)
    def _on_manual_start(self, game_type: GameType, num_decks: int) -> None:
        """Operator pressed MANUAL START on idle screen."""
        if not self._machine:
            return
        self._machine.context.game_type = game_type
        self._machine.context.num_decks = num_decks

        # On mock (dev/Windows): feed a card to unblock the trigger wait
        from deck_checker.hardware.gpio import MockTrigger
        if isinstance(self._machine._trigger, MockTrigger):
            self._machine._trigger.start_auto()
            self._machine._trigger.feed_card(hold_s=0.1)

    @pyqtSlot()
    def _on_abort(self) -> None:
        if self._machine:
            self._machine.stop()
            self._machine.reset()

    @pyqtSlot()
    def _on_reset(self) -> None:
        if self._machine:
            self._machine.reset()

    @pyqtSlot()
    def _on_reprint(self) -> None:
        logger.info("Reprint requested")
        # Thermal printer integration — Phase 7

    @pyqtSlot(int, object)
    def _on_override(self, index: int, card: Card) -> None:
        if self._machine:
            self._machine.manual_override(index, card)

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def closeEvent(self, event) -> None:
        if self._machine:
            self._machine.stop()
        super().closeEvent(event)
