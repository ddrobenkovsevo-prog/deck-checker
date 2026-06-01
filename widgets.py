"""
ui/widgets.py — Reusable custom widgets for the Deck Checker kiosk UI.
"""
from __future__ import annotations

import math
from typing import Optional

from PyQt6.QtCore import (
    QEasingCurve, QPropertyAnimation, QRectF, QSize, Qt, QTimer, pyqtProperty,
)
from PyQt6.QtGui import (
    QColor, QFont, QPainter, QPen, QBrush, QPainterPath,
)
from PyQt6.QtWidgets import QLabel, QWidget

from deck_checker.ui.theme import COLORS, FONTS


# ─────────────────────────────────────────────────────────────────────────────
# Pulsing status indicator (animated dot)
# ─────────────────────────────────────────────────────────────────────────────

class StatusDot(QWidget):
    """Animated coloured dot — pulses when active."""

    def __init__(self, color: str = COLORS["amber"], size: int = 16, parent=None):
        super().__init__(parent)
        self._color = QColor(color)
        self._dot_size = size
        self._opacity = 1.0
        self._pulsing = False
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._phase = 0.0
        self.setFixedSize(size + 8, size + 8)

    def set_color(self, color: str) -> None:
        self._color = QColor(color)
        self.update()

    def set_pulsing(self, pulsing: bool) -> None:
        self._pulsing = pulsing
        if pulsing:
            self._timer.start(50)
        else:
            self._timer.stop()
            self._opacity = 1.0
            self.update()

    def _tick(self) -> None:
        self._phase = (self._phase + 0.15) % (2 * math.pi)
        self._opacity = 0.4 + 0.6 * (0.5 + 0.5 * math.sin(self._phase))
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        color = QColor(self._color)
        # Glow ring
        glow = QColor(self._color)
        glow.setAlpha(int(60 * self._opacity))
        painter.setBrush(QBrush(glow))
        painter.setPen(Qt.PenStyle.NoPen)
        cx = self.width() // 2
        cy = self.height() // 2
        painter.drawEllipse(cx - self._dot_size // 2 - 2, cy - self._dot_size // 2 - 2,
                            self._dot_size + 4, self._dot_size + 4)
        # Main dot
        color.setAlpha(int(255 * self._opacity))
        painter.setBrush(QBrush(color))
        painter.drawEllipse(cx - self._dot_size // 2, cy - self._dot_size // 2,
                            self._dot_size, self._dot_size)


# ─────────────────────────────────────────────────────────────────────────────
# Circular progress ring
# ─────────────────────────────────────────────────────────────────────────────

class RingProgress(QWidget):
    """
    Circular progress ring showing scanned / total cards.
    Centre displays the count in large monospace digits.
    """

    def __init__(self, total: int = 416, parent=None):
        super().__init__(parent)
        self._total   = total
        self._current = 0
        self._color   = QColor(COLORS["amber"])
        self.setFixedSize(220, 220)

    def set_total(self, total: int) -> None:
        self._total = max(1, total)
        self.update()

    def set_current(self, current: int) -> None:
        self._current = current
        # Change colour near completion
        ratio = self._current / self._total
        if ratio >= 1.0:
            self._color = QColor(COLORS["green"])
        elif ratio > 0.5:
            self._color = QColor(COLORS["amber"])
        else:
            self._color = QColor(COLORS["amber"])
        self.update()

    def set_color(self, color: str) -> None:
        self._color = QColor(color)
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = self.width(), self.height()
        margin = 16
        rect = QRectF(margin, margin, w - 2*margin, h - 2*margin)

        # Background ring
        bg_pen = QPen(QColor(COLORS["bg_tertiary"]))
        bg_pen.setWidth(14)
        bg_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(bg_pen)
        painter.drawArc(rect, 0, 360 * 16)

        # Progress arc
        if self._total > 0:
            ratio = min(1.0, self._current / self._total)
            span = int(-ratio * 360 * 16)
            fg_pen = QPen(self._color)
            fg_pen.setWidth(14)
            fg_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(fg_pen)
            painter.drawArc(rect, 90 * 16, span)

        # Centre count
        painter.setPen(QPen(self._color))
        font = QFont(FONTS["mono"], 38, QFont.Weight.Bold)
        painter.setFont(font)
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, str(self._current))

        # Total label below
        painter.setPen(QPen(QColor(COLORS["text_secondary"])))
        font2 = QFont(FONTS["sans"], FONTS["size_xs"])
        painter.setFont(font2)
        label_rect = QRectF(margin, h // 2 + 28, w - 2*margin, 24)
        painter.drawText(label_rect, Qt.AlignmentFlag.AlignCenter,
                         f"of {self._total}")


# ─────────────────────────────────────────────────────────────────────────────
# Card badge (suit + rank display)
# ─────────────────────────────────────────────────────────────────────────────

SUIT_SYMBOLS = {"S": "♠", "H": "♥", "D": "♦", "C": "♣"}
SUIT_COLORS  = {
    "S": COLORS["text_primary"],
    "H": COLORS["red"],
    "D": COLORS["red"],
    "C": COLORS["text_primary"],
}

class CardBadge(QLabel):
    """Compact card label: e.g. A♠  K♥  2♦"""

    def __init__(self, card_str: Optional[str] = None, parent=None):
        super().__init__(parent)
        self.setObjectName("label_card")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        if card_str:
            self.set_card(card_str)
        else:
            self.setText("--")

    def set_card(self, card_str: str) -> None:
        """card_str like 'AS', 'KH', 'TD'"""
        if not card_str or len(card_str) < 2:
            self.setText("--")
            return
        rank = card_str[:-1]
        suit = card_str[-1].upper()
        symbol = SUIT_SYMBOLS.get(suit, suit)
        color  = SUIT_COLORS.get(suit, COLORS["text_primary"])
        self.setText(f"{rank}{symbol}")
        self.setStyleSheet(
            f"color: {color}; background-color: {COLORS['bg_tertiary']};"
            f"border: 1px solid {COLORS['border']}; border-radius: 4px;"
            f"padding: 4px 8px; font-family: {FONTS['mono']};"
            f"font-size: {FONTS['size_md']}px; font-weight: bold;"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Flashing banner (SUCCESS / ERROR)
# ─────────────────────────────────────────────────────────────────────────────

class ResultBanner(QLabel):
    """Large full-width result banner with optional flash animation."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._base_style = ""
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._flash_tick)
        self._flash_count = 0

    def show_success(self, text: str = "SHOE VALID") -> None:
        self._set_style(COLORS["green"], COLORS["green_dim"])
        self.setText(f"✓  {text}")
        self._flash(COLORS["green"])

    def show_error(self, text: str = "SHOE INVALID") -> None:
        self._set_style(COLORS["red"], COLORS["red_dim"])
        self.setText(f"✗  {text}")
        self._flash(COLORS["red"])

    def show_scanning(self, text: str = "SCANNING") -> None:
        self._set_style(COLORS["amber"], COLORS["amber_dim"])
        self.setText(text)
        self._timer.stop()

    def show_idle(self, text: str = "READY") -> None:
        self._set_style(COLORS["text_secondary"], COLORS["bg_secondary"])
        self.setText(text)
        self._timer.stop()

    def _set_style(self, color: str, bg: str) -> None:
        self._base_style = (
            f"color: {color}; background-color: {bg};"
            f"border: 2px solid {color}; border-radius: {8}px;"
            f"font-family: {FONTS['mono']}; font-size: {FONTS['size_lg']}px;"
            f"font-weight: bold; letter-spacing: 4px; padding: 12px;"
        )
        self.setStyleSheet(self._base_style)

    def _flash(self, color: str) -> None:
        self._flash_color = color
        self._flash_count = 0
        self._timer.start(180)

    def _flash_tick(self) -> None:
        self._flash_count += 1
        if self._flash_count > 6:
            self._timer.stop()
            self.setStyleSheet(self._base_style)
            return
        if self._flash_count % 2 == 0:
            self.setStyleSheet(self._base_style)
        else:
            bright = (
                f"color: {COLORS['black']}; background-color: {self._flash_color};"
                f"border: 2px solid {self._flash_color}; border-radius: 8px;"
                f"font-family: {FONTS['mono']}; font-size: {FONTS['size_lg']}px;"
                f"font-weight: bold; letter-spacing: 4px; padding: 12px;"
            )
            self.setStyleSheet(bright)
