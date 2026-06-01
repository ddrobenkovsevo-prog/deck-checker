"""
ui/screens.py — Screen widgets for each state of the Deck Checker.

Each screen is a QWidget that occupies the full 1024×600 display.
Screens communicate back to the main window via signals.
"""
from __future__ import annotations

from collections import Counter
from typing import Optional

from PyQt6.QtCore import Qt, QTimer, QDateTime, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtSvgWidgets import QSvgWidget
from PyQt6.QtWidgets import (
    QComboBox, QFrame, QGridLayout, QHBoxLayout, QLabel,
    QListWidget, QListWidgetItem, QPushButton, QProgressBar,
    QSizePolicy, QSpacerItem, QVBoxLayout, QWidget,
)

from deck_checker.core.models import Card, GameType, Rank, RecognitionResult, Suit
from deck_checker.ui.theme import COLORS, FONTS, SCREEN_H, SCREEN_W
from deck_checker.ui.widgets import CardBadge, ResultBanner, RingProgress, StatusDot


def _label(text: str, obj_name: str = "", parent=None) -> QLabel:
    lbl = QLabel(text, parent)
    if obj_name:
        lbl.setObjectName(obj_name)
    return lbl


def _hline() -> QFrame:
    line = QFrame()
    line.setObjectName("divider")
    line.setFrameShape(QFrame.Shape.HLine)
    return line


# ─────────────────────────────────────────────────────────────────────────────
# IDLE screen
# ─────────────────────────────────────────────────────────────────────────────

class IdleScreen(QWidget):
    """Waiting for lid to close and first card."""

    start_requested = pyqtSignal(GameType, int)   # game_type, num_decks

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build()

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(48, 32, 48, 32)
        root.setSpacing(0)

        # ── Header ────────────────────────────────────────────────────────────
        header = QHBoxLayout()
        dot = StatusDot(COLORS["text_dim"], size=14)
        dot.set_pulsing(False)
        self._status_dot = dot
        title = _label("DECK CHECKER", "label_title")
        header.addWidget(dot, 0, Qt.AlignmentFlag.AlignVCenter)
        header.addSpacing(16)
        header.addWidget(title, 0, Qt.AlignmentFlag.AlignVCenter)
        header.addStretch()

        # Right: clock + date + Evolution logo
        right_header = QVBoxLayout()
        right_header.setSpacing(2)
        right_header.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        # Evolution logo SVG
        import os
        logo_path = os.path.join(os.path.dirname(__file__), "evolution_logo_white.svg")
        if os.path.exists(logo_path):
            evo_logo = QSvgWidget(logo_path)
            evo_logo.setFixedSize(160, 31)
            right_header.addWidget(evo_logo, 0, Qt.AlignmentFlag.AlignRight)
        else:
            evo_lbl = _label("EVOLUTION", "label_info")
            evo_lbl.setStyleSheet(
                f"color: {COLORS['amber']}; font-family: {FONTS['mono']};"
                f"font-size: 11px; font-weight: bold; letter-spacing: 4px;"
            )
            evo_lbl.setAlignment(Qt.AlignmentFlag.AlignRight)
            right_header.addWidget(evo_lbl)

        # Clock
        self._clock_lbl = _label("00:00:00", "label_info")
        self._clock_lbl.setStyleSheet(
            f"color: {COLORS['text_primary']}; font-family: {FONTS['mono']};"
            f"font-size: 18px; font-weight: bold;"
        )
        self._clock_lbl.setAlignment(Qt.AlignmentFlag.AlignRight)
        right_header.addWidget(self._clock_lbl)

        # Date
        self._date_lbl = _label("", "label_info")
        self._date_lbl.setStyleSheet(
            f"color: {COLORS['text_secondary']}; font-family: {FONTS['mono']};"
            f"font-size: 11px;"
        )
        self._date_lbl.setAlignment(Qt.AlignmentFlag.AlignRight)
        right_header.addWidget(self._date_lbl)

        header.addLayout(right_header)
        root.addLayout(header)

        # Clock timer
        self._clock_timer = QTimer(self)
        self._clock_timer.timeout.connect(self._update_clock)
        self._clock_timer.start(1000)
        self._update_clock()

        root.addSpacing(8)
        root.addWidget(_hline())
        root.addSpacing(24)

        # ── Centre content ────────────────────────────────────────────────────
        centre = QHBoxLayout()
        centre.setSpacing(32)

        # Left: big status
        left = QVBoxLayout()
        left.setSpacing(12)

        status_lbl = _label("READY", "label_status")
        status_lbl.setStyleSheet(f"color: {COLORS['text_secondary']};")
        self._status_lbl = status_lbl
        left.addWidget(status_lbl)

        instruction = _label(
            "Place cards in tray",
            "label_subtitle",
        )
        instruction.setWordWrap(True)
        left.addWidget(instruction)

        left.addSpacing(24)

        # Game type selector
        game_lbl = _label("GAME TYPE", "label_info")
        game_lbl.setStyleSheet(f"color: {COLORS['text_dim']}; letter-spacing: 2px;")
        left.addWidget(game_lbl)

        self._game_combo = QComboBox()
        self._game_combo.addItem("Blackjack",  GameType.BLACKJACK)
        self._game_combo.addItem("Baccarat",   GameType.BACCARAT)
        self._game_combo.addItem("Poker",      GameType.BLACKJACK)
        self._game_combo.addItem("Always 6",   GameType.BACCARAT)
        self._game_combo.addItem("Always 7",   GameType.BACCARAT)
        self._game_combo.addItem("Always 8",   GameType.BACCARAT)
        self._game_combo.setMinimumWidth(220)
        left.addWidget(self._game_combo)

        left.addSpacing(12)

        decks_lbl = _label("NUMBER OF DECKS", "label_info")
        decks_lbl.setStyleSheet(f"color: {COLORS['text_dim']}; letter-spacing: 2px;")
        left.addWidget(decks_lbl)

        self._decks_combo = QComboBox()
        for n in [1, 2, 4, 6, 8]:
            self._decks_combo.addItem(f"{n} deck{'s' if n > 1 else ''}", n)
        self._decks_combo.setCurrentIndex(4)   # default 8 decks
        self._decks_combo.setMinimumWidth(220)
        left.addWidget(self._decks_combo)

        left.addStretch()
        centre.addLayout(left, 3)

        # Right: large idle graphic
        right = QVBoxLayout()
        right.setAlignment(Qt.AlignmentFlag.AlignCenter)

        idle_ring = RingProgress(total=416)
        idle_ring.set_current(0)
        idle_ring.set_color(COLORS["text_dim"])
        self._idle_ring = idle_ring
        right.addWidget(idle_ring, 0, Qt.AlignmentFlag.AlignCenter)

        last_lbl = _label("LAST SCAN: —", "label_info")
        last_lbl.setStyleSheet(f"color: {COLORS['text_dim']};")
        last_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._last_lbl = last_lbl
        right.addWidget(last_lbl)

        centre.addLayout(right, 2)
        root.addLayout(centre, 1)

        root.addWidget(_hline())
        root.addSpacing(16)

        # ── Footer ────────────────────────────────────────────────────────────
        footer = QHBoxLayout()
        self._lid_lbl = _label("⬜  LID OPEN", "label_info")
        self._lid_lbl.setStyleSheet(f"color: {COLORS['red']};")
        footer.addWidget(self._lid_lbl)
        footer.addStretch()

        btn_manual = QPushButton("START")
        btn_manual.setObjectName("btn_primary")
        btn_manual.setFixedHeight(52)
        btn_manual.clicked.connect(self._on_manual_start)
        footer.addWidget(btn_manual)

        root.addLayout(footer)

    def _update_clock(self) -> None:
        now = QDateTime.currentDateTime()
        self._clock_lbl.setText(now.toString("HH:mm:ss"))
        self._date_lbl.setText(now.toString("ddd, dd MMM yyyy"))

    def update_lid_status(self, closed: bool) -> None:
        if closed:
            self._lid_lbl.setText("🟢  LID CLOSED")
            self._lid_lbl.setStyleSheet(f"color: {COLORS['green']};")
            self._status_dot.set_color(COLORS["green"])
            self._status_dot.set_pulsing(True)
        else:
            self._lid_lbl.setText("⬜  LID OPEN")
            self._lid_lbl.setStyleSheet(f"color: {COLORS['red']};")
            self._status_dot.set_color(COLORS["text_dim"])
            self._status_dot.set_pulsing(False)

    def set_last_scan(self, text: str) -> None:
        self._last_lbl.setText(f"LAST SCAN: {text}")

    def _on_manual_start(self) -> None:
        game_type = self._game_combo.currentData()
        num_decks = self._decks_combo.currentData()
        self.start_requested.emit(game_type, num_decks)


# ─────────────────────────────────────────────────────────────────────────────
# SCANNING screen
# ─────────────────────────────────────────────────────────────────────────────

class ScanningScreen(QWidget):
    """Live scanning progress view."""

    abort_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._total = 416
        self._build()

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(48, 24, 48, 24)
        root.setSpacing(0)

        # ── Header ────────────────────────────────────────────────────────────
        header = QHBoxLayout()
        self._dot = StatusDot(COLORS["amber"], size=16)
        self._dot.set_pulsing(True)
        header.addWidget(self._dot, 0, Qt.AlignmentFlag.AlignVCenter)
        header.addSpacing(12)

        self._status_lbl = _label("SCANNING", "label_status")
        self._status_lbl.setStyleSheet(f"color: {COLORS['amber']};")
        header.addWidget(self._status_lbl, 0, Qt.AlignmentFlag.AlignVCenter)
        header.addStretch()

        self._elapsed_lbl = _label("0:00", "label_subtitle")
        header.addWidget(self._elapsed_lbl)
        root.addLayout(header)

        root.addSpacing(8)
        root.addWidget(_hline())
        root.addSpacing(16)

        # ── Main content ──────────────────────────────────────────────────────
        content = QHBoxLayout()
        content.setSpacing(32)

        # Ring progress
        self._ring = RingProgress(total=416)
        content.addWidget(self._ring, 0, Qt.AlignmentFlag.AlignCenter)

        # Right side: stats + last cards
        right = QVBoxLayout()
        right.setSpacing(12)

        # Stats grid
        stats_frame = QFrame()
        stats_frame.setObjectName("panel")
        stats_layout = QGridLayout(stats_frame)
        stats_layout.setContentsMargins(16, 12, 16, 12)
        stats_layout.setSpacing(8)

        def stat_pair(label: str, row: int) -> QLabel:
            lbl = _label(label, "label_info")
            lbl.setStyleSheet(f"color: {COLORS['text_dim']}; letter-spacing: 1px;")
            val = _label("—", "label_subtitle")
            val.setStyleSheet(f"color: {COLORS['amber']}; font-family: {FONTS['mono']};")
            stats_layout.addWidget(lbl, row, 0)
            stats_layout.addWidget(val, row, 1, Qt.AlignmentFlag.AlignRight)
            return val

        self._val_scanned  = stat_pair("SCANNED",       0)
        self._val_remaining= stat_pair("REMAINING",     1)
        self._val_lowconf  = stat_pair("LOW CONFIDENCE",2)
        self._val_speed    = stat_pair("CARDS / SEC",   3)
        right.addWidget(stats_frame)

        # Last 6 cards
        last_lbl = _label("LAST CARDS", "label_info")
        last_lbl.setStyleSheet(f"color: {COLORS['text_dim']}; letter-spacing: 2px;")
        right.addWidget(last_lbl)

        cards_row = QHBoxLayout()
        cards_row.setSpacing(8)
        self._card_badges = []
        for _ in range(6):
            badge = CardBadge()
            self._card_badges.append(badge)
            cards_row.addWidget(badge)
        cards_row.addStretch()
        right.addLayout(cards_row)

        right.addStretch()
        content.addLayout(right, 1)
        root.addLayout(content, 1)

        root.addSpacing(12)

        # ── Progress bar ──────────────────────────────────────────────────────
        self._progress = QProgressBar()
        self._progress.setRange(0, 416)
        self._progress.setValue(0)
        self._progress.setFormat("%v / %m cards")
        self._progress.setFixedHeight(28)
        root.addWidget(self._progress)

        root.addSpacing(12)
        root.addWidget(_hline())
        root.addSpacing(12)

        # ── Footer ────────────────────────────────────────────────────────────
        footer = QHBoxLayout()
        footer.addStretch()
        btn_abort = QPushButton("ABORT SCAN")
        btn_abort.setObjectName("btn_danger")
        btn_abort.setFixedHeight(48)
        btn_abort.clicked.connect(self.abort_requested.emit)
        footer.addWidget(btn_abort)
        root.addLayout(footer)

        # ── Elapsed timer ─────────────────────────────────────────────────────
        self._start_time = 0
        self._elapsed_timer = QTimer(self)
        self._elapsed_timer.timeout.connect(self._update_elapsed)
        self._scan_count = 0
        self._low_conf   = 0

    def start_scan(self, total: int) -> None:
        import time
        self._total = total
        self._scan_count = 0
        self._low_conf = 0
        self._start_time = time.monotonic()
        self._ring.set_total(total)
        self._ring.set_current(0)
        self._ring.set_color(COLORS["amber"])
        self._progress.setRange(0, total)
        self._progress.setValue(0)
        self._elapsed_timer.start(500)
        for badge in self._card_badges:
            badge.set_card("")
        self._update_stats()

    def on_card_scanned(self, index: int, result: RecognitionResult) -> None:
        self._scan_count = index + 1
        if not result.is_confident:
            self._low_conf += 1

        self._ring.set_current(self._scan_count)
        self._progress.setValue(self._scan_count)

        # Shift last cards
        card_str = str(result.card) if result.card else "??"
        for i in range(len(self._card_badges) - 1):
            self._card_badges[i].setText(self._card_badges[i+1].text())
            self._card_badges[i].setStyleSheet(self._card_badges[i+1].styleSheet())
        self._card_badges[-1].set_card(card_str)
        self._update_stats()

    def stop_scan(self) -> None:
        self._elapsed_timer.stop()
        self._dot.set_pulsing(False)

    def _update_stats(self) -> None:
        remaining = max(0, self._total - self._scan_count)
        self._val_scanned.setText(str(self._scan_count))
        self._val_remaining.setText(str(remaining))
        lc_color = COLORS["red"] if self._low_conf > 0 else COLORS["amber"]
        self._val_lowconf.setText(str(self._low_conf))
        self._val_lowconf.setStyleSheet(
            f"color: {lc_color}; font-family: {FONTS['mono']};"
        )

    def _update_elapsed(self) -> None:
        import time
        elapsed = time.monotonic() - self._start_time
        mins = int(elapsed) // 60
        secs = int(elapsed) % 60
        self._elapsed_lbl.setText(f"{mins}:{secs:02d}")
        if elapsed > 1 and self._scan_count > 0:
            cps = self._scan_count / elapsed
            self._val_speed.setText(f"{cps:.0f}")


# ─────────────────────────────────────────────────────────────────────────────
# RESULT screen (SUCCESS + ERROR)
# ─────────────────────────────────────────────────────────────────────────────

class ResultScreen(QWidget):
    """Displays scan result — SUCCESS or ERROR with details."""

    reset_requested  = pyqtSignal()
    reprint_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build()

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(48, 24, 48, 24)
        root.setSpacing(16)

        # Banner
        self._banner = ResultBanner()
        self._banner.setFixedHeight(80)
        root.addWidget(self._banner)

        # Details row
        details = QHBoxLayout()
        details.setSpacing(24)

        # Left: summary stats
        left_frame = QFrame()
        left_frame.setObjectName("panel")
        left_layout = QGridLayout(left_frame)
        left_layout.setContentsMargins(20, 16, 20, 16)
        left_layout.setSpacing(10)

        def detail_row(label: str, row: int) -> QLabel:
            l = _label(label, "label_info")
            l.setStyleSheet(f"color: {COLORS['text_dim']}; letter-spacing: 1px;")
            v = _label("—", "label_subtitle")
            v.setStyleSheet(f"font-family: {FONTS['mono']}; color: {COLORS['text_primary']};")
            v.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            left_layout.addWidget(l, row, 0)
            left_layout.addWidget(v, row, 1)
            return v

        self._val_total    = detail_row("TOTAL CARDS",    0)
        self._val_correct  = detail_row("RECOGNISED",     1)
        self._val_missing  = detail_row("MISSING",        2)
        self._val_extra    = detail_row("EXTRA",          3)
        self._val_lowconf  = detail_row("LOW CONFIDENCE", 4)
        self._val_time     = detail_row("SCAN TIME",      5)
        self._val_digest   = detail_row("AUDIT DIGEST",   6)
        details.addWidget(left_frame, 2)

        # Right: missing/extra card lists
        right = QVBoxLayout()
        right.setSpacing(8)

        miss_lbl = _label("MISSING CARDS", "label_info")
        miss_lbl.setStyleSheet(f"color: {COLORS['text_dim']}; letter-spacing: 2px;")
        right.addWidget(miss_lbl)
        self._missing_list = QListWidget()
        self._missing_list.setFixedHeight(120)
        right.addWidget(self._missing_list)

        extra_lbl = _label("EXTRA CARDS", "label_info")
        extra_lbl.setStyleSheet(f"color: {COLORS['text_dim']}; letter-spacing: 2px;")
        right.addWidget(extra_lbl)
        self._extra_list = QListWidget()
        self._extra_list.setFixedHeight(120)
        right.addWidget(self._extra_list)

        details.addLayout(right, 1)
        root.addLayout(details, 1)

        root.addWidget(_hline())

        # Footer buttons
        footer = QHBoxLayout()
        footer.setSpacing(16)

        self._btn_reprint = QPushButton("PRINT REPORT")
        self._btn_reprint.setObjectName("btn_primary")
        self._btn_reprint.setFixedHeight(52)
        self._btn_reprint.clicked.connect(self.reprint_requested.emit)
        footer.addWidget(self._btn_reprint)

        footer.addStretch()

        self._btn_reset = QPushButton("NEW SCAN")
        self._btn_reset.setObjectName("btn_success")
        self._btn_reset.setFixedHeight(52)
        self._btn_reset.setMinimumWidth(160)
        self._btn_reset.clicked.connect(self.reset_requested.emit)
        footer.addWidget(self._btn_reset)

        root.addLayout(footer)

    def show_result(self, report, elapsed_s: float = 0.0) -> None:
        from deck_checker.core.models import ScanReport
        total   = len(report.scanned)
        correct = sum(1 for r in report.scanned if r.is_confident)
        missing = report.missing_cards
        extra   = report.extra_cards
        low_conf= len(report.low_confidence_indices)

        if report.is_valid:
            self._banner.show_success("SHOE VALID")
        else:
            self._banner.show_error("SHOE INVALID")

        self._val_total.setText(str(total))
        self._val_correct.setText(str(correct))

        miss_n = sum(missing.values())
        extra_n = sum(extra.values())

        miss_color = COLORS["red"] if miss_n > 0 else COLORS["green"]
        extra_color = COLORS["red"] if extra_n > 0 else COLORS["green"]
        lc_color = COLORS["red"] if low_conf > 0 else COLORS["green"]

        self._val_missing.setText(str(miss_n))
        self._val_missing.setStyleSheet(f"font-family: {FONTS['mono']}; color: {miss_color};")
        self._val_extra.setText(str(extra_n))
        self._val_extra.setStyleSheet(f"font-family: {FONTS['mono']}; color: {extra_color};")
        self._val_lowconf.setText(str(low_conf))
        self._val_lowconf.setStyleSheet(f"font-family: {FONTS['mono']}; color: {lc_color};")

        mins = int(elapsed_s) // 60
        secs = int(elapsed_s) % 60
        self._val_time.setText(f"{mins}:{secs:02d}")
        self._val_digest.setText(report.digest()[:16] + "…")

        # Lists
        self._missing_list.clear()
        for card, count in sorted(missing.items(), key=lambda x: str(x[0])):
            item = QListWidgetItem(f"  {card}  ×{count}")
            item.setForeground(QColor(COLORS["red"]))
            self._missing_list.addItem(item)

        self._extra_list.clear()
        for card, count in sorted(extra.items(), key=lambda x: str(x[0])):
            item = QListWidgetItem(f"  {card}  ×{count}")
            item.setForeground(QColor(COLORS["amber"]))
            self._extra_list.addItem(item)


# ─────────────────────────────────────────────────────────────────────────────
# MANUAL VALIDATION screen
# ─────────────────────────────────────────────────────────────────────────────

class ManualScreen(QWidget):
    """Operator resolves low-confidence cards manually."""

    override_submitted = pyqtSignal(int, object)   # index, Card
    reset_requested    = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pending: list[tuple[int, str]] = []   # (index, current_card_str)
        self._current_idx = 0
        self._build()

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(48, 24, 48, 24)
        root.setSpacing(16)

        # Header
        header = QHBoxLayout()
        dot = StatusDot(COLORS["blue"], size=16)
        dot.set_pulsing(True)
        header.addWidget(dot, 0, Qt.AlignmentFlag.AlignVCenter)
        header.addSpacing(12)
        title = _label("MANUAL VALIDATION REQUIRED", "label_status")
        title.setStyleSheet(f"color: {COLORS['blue']};")
        header.addWidget(title)
        header.addStretch()
        self._counter_lbl = _label("0 / 0", "label_subtitle")
        header.addWidget(self._counter_lbl)
        root.addLayout(header)
        root.addWidget(_hline())

        # Main area
        main = QHBoxLayout()
        main.setSpacing(32)

        # Left: current card to resolve
        left = QVBoxLayout()
        left.setSpacing(12)

        instr = _label(
            "Recognition confidence was too low for the following card.\n"
            "Please verify and select the correct card.",
            "label_info",
        )
        instr.setWordWrap(True)
        instr.setStyleSheet(f"color: {COLORS['text_secondary']};")
        left.addWidget(instr)
        left.addSpacing(8)

        detected_lbl = _label("DETECTED AS", "label_info")
        detected_lbl.setStyleSheet(f"color: {COLORS['text_dim']}; letter-spacing: 2px;")
        left.addWidget(detected_lbl)

        self._detected_badge = CardBadge()
        left.addWidget(self._detected_badge, 0, Qt.AlignmentFlag.AlignLeft)
        left.addSpacing(16)

        correction_lbl = _label("CORRECT CARD", "label_info")
        correction_lbl.setStyleSheet(f"color: {COLORS['text_dim']}; letter-spacing: 2px;")
        left.addWidget(correction_lbl)

        # Rank + Suit selectors
        sel_row = QHBoxLayout()
        sel_row.setSpacing(12)

        self._rank_combo = QComboBox()
        for r in Rank:
            if r != Rank.JOKER:
                self._rank_combo.addItem(r.value, r)
        self._rank_combo.setMinimumWidth(120)

        self._suit_combo = QComboBox()
        suit_names = {"S": "♠ Spades", "H": "♥ Hearts", "D": "♦ Diamonds", "C": "♣ Clubs"}
        for s in Suit:
            self._suit_combo.addItem(suit_names[s.value], s)

        sel_row.addWidget(self._rank_combo)
        sel_row.addWidget(self._suit_combo)
        sel_row.addStretch()
        left.addLayout(sel_row)

        left.addSpacing(16)
        btn_confirm = QPushButton("CONFIRM CORRECTION")
        btn_confirm.setObjectName("btn_primary")
        btn_confirm.setFixedHeight(52)
        btn_confirm.clicked.connect(self._on_confirm)
        left.addWidget(btn_confirm)

        left.addStretch()
        main.addLayout(left, 2)

        # Right: pending list
        right = QVBoxLayout()
        right.setSpacing(8)

        pending_lbl = _label("PENDING CARDS", "label_info")
        pending_lbl.setStyleSheet(f"color: {COLORS['text_dim']}; letter-spacing: 2px;")
        right.addWidget(pending_lbl)

        self._pending_list = QListWidget()
        right.addWidget(self._pending_list, 1)

        right.addStretch()
        btn_abort = QPushButton("ABORT")
        btn_abort.setObjectName("btn_danger")
        btn_abort.setFixedHeight(44)
        btn_abort.clicked.connect(self.reset_requested.emit)
        right.addWidget(btn_abort)

        main.addLayout(right, 1)
        root.addLayout(main, 1)

    def set_pending(self, items: list[tuple[int, Optional[str]]]) -> None:
        """items: list of (scan_index, card_str_or_None)"""
        self._pending = [(i, s or "??") for i, s in items]
        self._current_idx = 0
        self._refresh_list()
        self._show_current()

    def _show_current(self) -> None:
        if not self._pending:
            return
        idx, card_str = self._pending[self._current_idx]
        self._detected_badge.set_card(card_str)
        total = len(self._pending)
        done  = self._current_idx
        self._counter_lbl.setText(f"{done} / {total}")

    def _refresh_list(self) -> None:
        self._pending_list.clear()
        for i, (idx, card_str) in enumerate(self._pending):
            marker = "→ " if i == self._current_idx else "   "
            item = QListWidgetItem(f"{marker}#{idx:03d}  {card_str}")
            if i == self._current_idx:
                item.setForeground(QColor(COLORS["blue"]))
            self._pending_list.addItem(item)

    def _on_confirm(self) -> None:
        if not self._pending or self._current_idx >= len(self._pending):
            return
        idx, _ = self._pending[self._current_idx]
        rank = self._rank_combo.currentData()
        suit = self._suit_combo.currentData()
        card = Card(rank=rank, suit=suit)
        self.override_submitted.emit(idx, card)

        self._current_idx += 1
        if self._current_idx < len(self._pending):
            self._refresh_list()
            self._show_current()
        else:
            self._counter_lbl.setText(f"{len(self._pending)} / {len(self._pending)}")
