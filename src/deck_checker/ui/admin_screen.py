"""
ui/admin_screen.py — Admin Settings screen.

Password protected (default: "evoroot").
Changes are saved to config.json and logged to audit log.
"""
from __future__ import annotations

import logging

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog, QFrame, QGridLayout, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QScrollArea, QSizePolicy,
    QSpinBox, QDoubleSpinBox, QCheckBox,
    QTabWidget, QVBoxLayout, QWidget,
)

from deck_checker.core.config import AppConfig, get_config, save_config
from deck_checker.ui.theme import COLORS, FONTS, STYLESHEET

logger = logging.getLogger(__name__)

ADMIN_PASSWORD = "evoroot"


# ─────────────────────────────────────────────────────────────────────────────
# Password dialog
# ─────────────────────────────────────────────────────────────────────────────

class PasswordDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Admin Access")
        self.setFixedSize(380, 200)
        self.setStyleSheet(STYLESHEET)
        self.setWindowFlags(
            Qt.WindowType.Dialog |
            Qt.WindowType.FramelessWindowHint
        )
        self._ok = False
        self._build()

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(32, 28, 32, 28)
        root.setSpacing(16)

        title = QLabel("ADMIN ACCESS")
        title.setStyleSheet(
            f"color: {COLORS['amber']}; font-family: {FONTS['mono']};"
            f"font-size: 18px; font-weight: bold; letter-spacing: 3px;"
        )
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(title)

        self._pwd_field = QLineEdit()
        self._pwd_field.setEchoMode(QLineEdit.EchoMode.Password)
        self._pwd_field.setPlaceholderText("Enter password")
        self._pwd_field.setStyleSheet(
            f"background: {COLORS['bg_tertiary']}; color: {COLORS['text_primary']};"
            f"border: 1px solid {COLORS['border_bright']}; border-radius: 6px;"
            f"padding: 10px; font-size: 16px; min-height: 42px;"
        )
        self._pwd_field.returnPressed.connect(self._check)
        root.addWidget(self._pwd_field)

        self._error_lbl = QLabel("")
        self._error_lbl.setStyleSheet(f"color: {COLORS['red']}; font-size: 12px;")
        self._error_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(self._error_lbl)

        btns = QHBoxLayout()
        btn_cancel = QPushButton("CANCEL")
        btn_cancel.setFixedHeight(44)
        btn_cancel.clicked.connect(self.reject)
        btns.addWidget(btn_cancel)

        btn_ok = QPushButton("ENTER")
        btn_ok.setObjectName("btn_primary")
        btn_ok.setFixedHeight(44)
        btn_ok.clicked.connect(self._check)
        btns.addWidget(btn_ok)
        root.addLayout(btns)

    def _check(self):
        if self._pwd_field.text() == ADMIN_PASSWORD:
            self.accept()
        else:
            self._error_lbl.setText("Incorrect password")
            self._pwd_field.clear()
            self._pwd_field.setFocus()

    @staticmethod
    def authenticate(parent=None) -> bool:
        dlg = PasswordDialog(parent)
        return dlg.exec() == QDialog.DialogCode.Accepted


# ─────────────────────────────────────────────────────────────────────────────
# Helper widgets
# ─────────────────────────────────────────────────────────────────────────────

def _section_label(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet(
        f"color: {COLORS['amber']}; font-family: {FONTS['mono']};"
        f"font-size: 12px; font-weight: bold; letter-spacing: 3px;"
        f"padding-top: 8px;"
    )
    return lbl


def _field_label(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet(
        f"color: {COLORS['text_secondary']}; font-size: 13px; letter-spacing: 1px;"
    )
    lbl.setMinimumWidth(180)
    return lbl


def _text_input(value: str = "", placeholder: str = "") -> QLineEdit:
    f = QLineEdit(value)
    f.setPlaceholderText(placeholder)
    f.setStyleSheet(
        f"background: {COLORS['bg_tertiary']}; color: {COLORS['text_primary']};"
        f"border: 1px solid {COLORS['border_bright']}; border-radius: 6px;"
        f"padding: 6px 10px; font-family: {FONTS['mono']}; font-size: 14px;"
        f"min-height: 36px;"
    )
    return f


def _int_spin(value: int, min_v: int, max_v: int) -> QSpinBox:
    s = QSpinBox()
    s.setRange(min_v, max_v)
    s.setValue(value)
    s.setStyleSheet(
        f"background: {COLORS['bg_tertiary']}; color: {COLORS['text_primary']};"
        f"border: 1px solid {COLORS['border_bright']}; border-radius: 6px;"
        f"padding: 4px 8px; font-family: {FONTS['mono']}; font-size: 14px;"
        f"min-height: 36px;"
    )
    return s


def _float_spin(value: float, min_v: float, max_v: float, step: float) -> QDoubleSpinBox:
    s = QDoubleSpinBox()
    s.setRange(min_v, max_v)
    s.setSingleStep(step)
    s.setDecimals(2)
    s.setValue(value)
    s.setStyleSheet(
        f"background: {COLORS['bg_tertiary']}; color: {COLORS['text_primary']};"
        f"border: 1px solid {COLORS['border_bright']}; border-radius: 6px;"
        f"padding: 4px 8px; font-family: {FONTS['mono']}; font-size: 14px;"
        f"min-height: 36px;"
    )
    return s


def _checkbox(checked: bool, label: str = "") -> QCheckBox:
    cb = QCheckBox(label)
    cb.setChecked(checked)
    cb.setStyleSheet(
        f"color: {COLORS['text_primary']}; font-size: 14px;"
    )
    return cb


def _hline() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.Shape.HLine)
    line.setStyleSheet(f"background: {COLORS['border']}; max-height: 1px;")
    return line


# ─────────────────────────────────────────────────────────────────────────────
# Admin Screen
# ─────────────────────────────────────────────────────────────────────────────

class AdminScreen(QWidget):
    """Full admin settings screen."""

    close_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cfg = get_config()
        self._fields: dict = {}
        self._build()

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(32, 20, 32, 20)
        root.setSpacing(0)

        # ── Header ────────────────────────────────────────────────────────────
        header = QHBoxLayout()
        title = QLabel("ADMIN SETTINGS")
        title.setStyleSheet(
            f"color: {COLORS['amber']}; font-family: {FONTS['mono']};"
            f"font-size: 28px; font-weight: bold; letter-spacing: 4px;"
        )
        header.addWidget(title)
        header.addStretch()

        btn_close = QPushButton("✕  CLOSE")
        btn_close.setFixedHeight(44)
        btn_close.setFixedWidth(120)
        btn_close.clicked.connect(self._on_close)
        header.addWidget(btn_close)
        root.addLayout(header)

        root.addSpacing(8)
        root.addWidget(_hline())
        root.addSpacing(12)

        # ── Tabs ──────────────────────────────────────────────────────────────
        tabs = QTabWidget()
        tabs.setStyleSheet(f"""
            QTabWidget::pane {{
                border: 1px solid {COLORS['border']};
                border-radius: 8px;
                background: {COLORS['bg_secondary']};
            }}
            QTabBar::tab {{
                background: {COLORS['bg_tertiary']};
                color: {COLORS['text_secondary']};
                padding: 10px 24px;
                font-family: {FONTS['mono']};
                font-size: 12px;
                letter-spacing: 2px;
                border: 1px solid {COLORS['border']};
                border-bottom: none;
                border-radius: 6px 6px 0 0;
            }}
            QTabBar::tab:selected {{
                background: {COLORS['bg_secondary']};
                color: {COLORS['amber']};
                border-color: {COLORS['amber']};
            }}
        """)

        tabs.addTab(self._build_device_tab(),  "DEVICE")
        tabs.addTab(self._build_scan_tab(),    "SCANNING")
        tabs.addTab(self._build_network_tab(), "NETWORK")
        tabs.addTab(self._build_printer_tab(), "PRINTER")
        tabs.addTab(self._build_audit_tab(),   "AUDIT")

        root.addWidget(tabs, 1)
        root.addSpacing(12)

        # ── Footer ────────────────────────────────────────────────────────────
        footer = QHBoxLayout()

        self._status_lbl = QLabel("")
        self._status_lbl.setStyleSheet(f"color: {COLORS['green']}; font-size: 13px;")
        footer.addWidget(self._status_lbl)
        footer.addStretch()

        btn_reset = QPushButton("RESET DEFAULTS")
        btn_reset.setObjectName("btn_danger")
        btn_reset.setFixedHeight(48)
        btn_reset.clicked.connect(self._on_reset)
        footer.addWidget(btn_reset)

        btn_save = QPushButton("SAVE SETTINGS")
        btn_save.setObjectName("btn_success")
        btn_save.setFixedHeight(48)
        btn_save.setMinimumWidth(160)
        btn_save.clicked.connect(self._on_save)
        footer.addWidget(btn_save)

        root.addLayout(footer)

    # ── Tab builders ──────────────────────────────────────────────────────────

    def _scrollable(self, widget: QWidget) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidget(widget)
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("border: none; background: transparent;")
        return scroll

    def _build_device_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(10)

        layout.addWidget(_section_label("DEVICE IDENTIFICATION"))
        layout.addWidget(_hline())

        grid = QGridLayout()
        grid.setSpacing(10)
        grid.setColumnStretch(1, 1)

        rows = [
            ("Device Name",  "device_name",  self._cfg.device.device_name,  "e.g. DECK-CHECKER-01"),
            ("Casino Name",  "casino_name",  self._cfg.device.casino_name,  "e.g. Grand Casino"),
            ("Country",      "country",      self._cfg.device.country,      "e.g. Latvia"),
            ("Location",     "location",     self._cfg.device.location,     "e.g. Floor 2, Pit A"),
            ("Site ID",      "site_id",      self._cfg.device.site_id,      "e.g. SITE-001"),
        ]
        for row_idx, (label, key, value, placeholder) in enumerate(rows):
            grid.addWidget(_field_label(label), row_idx, 0)
            field = _text_input(value, placeholder)
            self._fields[f"device.{key}"] = field
            grid.addWidget(field, row_idx, 1)

        layout.addLayout(grid)
        layout.addStretch()
        return self._scrollable(w)

    def _build_scan_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(10)

        layout.addWidget(_section_label("SCANNING PARAMETERS"))
        layout.addWidget(_hline())

        grid = QGridLayout()
        grid.setSpacing(10)
        grid.setColumnStretch(1, 1)

        # Confidence threshold
        grid.addWidget(_field_label("Confidence Threshold"), 0, 0)
        conf = _float_spin(self._cfg.scan.confidence_threshold, 0.5, 1.0, 0.01)
        self._fields["scan.confidence_threshold"] = conf
        grid.addWidget(conf, 0, 1)

        hint = QLabel("Minimum match score to accept a card (0.82 recommended)")
        hint.setStyleSheet(f"color: {COLORS['text_dim']}; font-size: 11px;")
        grid.addWidget(hint, 1, 1)

        # Auto retry count
        grid.addWidget(_field_label("Auto Retry Count"), 2, 0)
        retry = _int_spin(self._cfg.scan.auto_retry_count, 0, 5)
        self._fields["scan.auto_retry_count"] = retry
        grid.addWidget(retry, 2, 1)

        # Scan timeout
        grid.addWidget(_field_label("Scan Timeout (sec)"), 3, 0)
        timeout = _int_spin(self._cfg.scan.scan_timeout_s, 30, 300)
        self._fields["scan.scan_timeout_s"] = timeout
        grid.addWidget(timeout, 3, 1)

        # Card timeout
        grid.addWidget(_field_label("Card Stall Timeout (sec)"), 4, 0)
        card_timeout = _int_spin(self._cfg.scan.card_timeout_s, 1, 30)
        self._fields["scan.card_timeout_s"] = card_timeout
        grid.addWidget(card_timeout, 4, 1)

        # Default decks
        grid.addWidget(_field_label("Default Decks"), 5, 0)
        decks = _int_spin(self._cfg.scan.default_num_decks, 1, 8)
        self._fields["scan.default_num_decks"] = decks
        grid.addWidget(decks, 5, 1)

        layout.addLayout(grid)
        layout.addStretch()
        return self._scrollable(w)

    def _build_network_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(10)

        layout.addWidget(_section_label("REPORT SYNC"))
        layout.addWidget(_hline())

        sync_cb = _checkbox(self._cfg.network.sync_enabled, "Enable automatic sync")
        self._fields["network.sync_enabled"] = sync_cb
        layout.addWidget(sync_cb)

        layout.addSpacing(8)
        layout.addWidget(_section_label("S3 / CLOUD"))

        grid1 = QGridLayout()
        grid1.setSpacing(10)
        grid1.setColumnStretch(1, 1)
        s3_rows = [
            ("S3 Bucket",  "s3_bucket", self._cfg.network.s3_bucket, "my-casino-reports"),
            ("S3 Region",  "s3_region", self._cfg.network.s3_region, "eu-west-1"),
        ]
        for i, (label, key, value, ph) in enumerate(s3_rows):
            grid1.addWidget(_field_label(label), i, 0)
            f = _text_input(value, ph)
            self._fields[f"network.{key}"] = f
            grid1.addWidget(f, i, 1)
        layout.addLayout(grid1)

        layout.addSpacing(8)
        layout.addWidget(_section_label("SFTP"))

        grid2 = QGridLayout()
        grid2.setSpacing(10)
        grid2.setColumnStretch(1, 1)
        sftp_rows = [
            ("SFTP Host", "sftp_host", self._cfg.network.sftp_host, "sftp.casino.com"),
            ("SFTP User", "sftp_user", self._cfg.network.sftp_user, "username"),
            ("SFTP Path", "sftp_path", self._cfg.network.sftp_path, "/reports/"),
        ]
        for i, (label, key, value, ph) in enumerate(sftp_rows):
            grid2.addWidget(_field_label(label), i, 0)
            f = _text_input(value, ph)
            self._fields[f"network.{key}"] = f
            grid2.addWidget(f, i, 1)

        grid2.addWidget(_field_label("SFTP Port"), 3, 0)
        sftp_port = _int_spin(self._cfg.network.sftp_port, 1, 65535)
        self._fields["network.sftp_port"] = sftp_port
        grid2.addWidget(sftp_port, 3, 1)

        layout.addLayout(grid2)

        layout.addSpacing(8)
        grid3 = QGridLayout()
        grid3.addWidget(_field_label("Sync Interval (sec)"), 0, 0)
        sync_int = _int_spin(self._cfg.network.sync_interval_s, 60, 3600)
        self._fields["network.sync_interval_s"] = sync_int
        grid3.addWidget(sync_int, 0, 1)
        layout.addLayout(grid3)

        layout.addStretch()
        return self._scrollable(w)

    def _build_printer_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(10)

        layout.addWidget(_section_label("THERMAL PRINTER"))
        layout.addWidget(_hline())

        grid = QGridLayout()
        grid.setSpacing(10)
        grid.setColumnStretch(1, 1)

        grid.addWidget(_field_label("Printer Port"), 0, 0)
        port = _text_input(self._cfg.printer.printer_port, "COM1 or /dev/usb/lp0")
        self._fields["printer.printer_port"] = port
        grid.addWidget(port, 0, 1)

        layout.addLayout(grid)
        layout.addSpacing(12)

        auto_print = _checkbox(self._cfg.printer.auto_print,
                               "Auto-print report after each scan")
        self._fields["printer.auto_print"] = auto_print
        layout.addWidget(auto_print)

        print_error = _checkbox(self._cfg.printer.print_on_error,
                                "Print report on ERROR / INVALID shoe")
        self._fields["printer.print_on_error"] = print_error
        layout.addWidget(print_error)

        layout.addStretch()
        return self._scrollable(w)

    def _build_audit_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(10)

        layout.addWidget(_section_label("AUDIT & LOGGING"))
        layout.addWidget(_hline())

        grid = QGridLayout()
        grid.setSpacing(10)
        grid.setColumnStretch(1, 1)

        grid.addWidget(_field_label("Log Retention (days)"), 0, 0)
        retention = _int_spin(self._cfg.audit.log_retention_days, 7, 365)
        self._fields["audit.log_retention_days"] = retention
        grid.addWidget(retention, 0, 1)

        grid.addWidget(_field_label("Export Path"), 1, 0)
        export = _text_input(self._cfg.audit.export_path, "/mnt/usb/reports/")
        self._fields["audit.export_path"] = export
        grid.addWidget(export, 1, 1)

        layout.addLayout(grid)
        layout.addSpacing(12)

        require_insp = _checkbox(self._cfg.audit.require_inspector,
                                 "Require Inspector ID before each scan")
        self._fields["audit.require_inspector"] = require_insp
        layout.addWidget(require_insp)

        layout.addStretch()
        return self._scrollable(w)

    # ── Actions ───────────────────────────────────────────────────────────────

    def _on_save(self) -> None:
        cfg = self._cfg

        def _get(key: str):
            widget = self._fields[key]
            if isinstance(widget, QLineEdit):
                return widget.text().strip()
            elif isinstance(widget, QSpinBox):
                return widget.value()
            elif isinstance(widget, QDoubleSpinBox):
                return widget.value()
            elif isinstance(widget, QCheckBox):
                return widget.isChecked()
            return None

        # Device
        for f in ["device_name","casino_name","country","location","site_id"]:
            old = getattr(cfg.device, f)
            new = _get(f"device.{f}")
            if old != new:
                cfg.log_change("device", f, old, new)
                setattr(cfg.device, f, new)

        # Scan
        for f in ["confidence_threshold","auto_retry_count","scan_timeout_s",
                  "card_timeout_s","default_num_decks"]:
            old = getattr(cfg.scan, f)
            new = _get(f"scan.{f}")
            if old != new:
                cfg.log_change("scan", f, old, new)
                setattr(cfg.scan, f, new)

        # Network
        for f in ["s3_bucket","s3_region","sftp_host","sftp_user","sftp_path",
                  "sftp_port","sync_enabled","sync_interval_s"]:
            old = getattr(cfg.network, f)
            new = _get(f"network.{f}")
            if old != new:
                cfg.log_change("network", f, old, new)
                setattr(cfg.network, f, new)

        # Printer
        for f in ["printer_port","auto_print","print_on_error"]:
            old = getattr(cfg.printer, f)
            new = _get(f"printer.{f}")
            if old != new:
                cfg.log_change("printer", f, old, new)
                setattr(cfg.printer, f, new)

        # Audit
        for f in ["log_retention_days","export_path","require_inspector"]:
            old = getattr(cfg.audit, f)
            new = _get(f"audit.{f}")
            if old != new:
                cfg.log_change("audit", f, old, new)
                setattr(cfg.audit, f, new)

        save_config()
        self._status_lbl.setText("✓  Settings saved")
        logger.info("Admin settings saved")

    def _on_reset(self) -> None:
        from deck_checker.core.config import AppConfig
        new_cfg = AppConfig()
        new_cfg.save()
        self._status_lbl.setText("Settings reset to defaults — please restart")
        self._status_lbl.setStyleSheet(f"color: {COLORS['amber']}; font-size: 13px;")
        logger.warning("Settings reset to defaults by admin")

    def _on_close(self) -> None:
        self.close_requested.emit()
