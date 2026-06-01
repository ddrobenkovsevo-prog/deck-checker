"""
core/config.py — Application configuration.

Stored in config.json next to the executable.
All changes are logged to the audit log.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

CONFIG_FILE = Path(__file__).parent.parent.parent.parent / "config.json"


@dataclass
class DeviceConfig:
    """Device identification."""
    device_name:  str = "DECK-CHECKER-01"
    casino_name:  str = ""
    country:      str = ""
    site_id:      str = ""
    location:     str = ""


@dataclass
class ScanConfig:
    """Scanning operational settings."""
    default_game_type:    str   = "blackjack"
    default_num_decks:    int   = 8
    confidence_threshold: float = 0.82
    auto_retry_count:     int   = 2
    scan_timeout_s:       int   = 120
    card_timeout_s:       int   = 5


@dataclass
class NetworkConfig:
    """Network / sync settings."""
    s3_bucket:      str  = ""
    s3_region:      str  = "eu-west-1"
    sftp_host:      str  = ""
    sftp_port:      int  = 22
    sftp_user:      str  = ""
    sftp_path:      str  = "/deck-checker/reports/"
    sync_enabled:   bool = False
    sync_interval_s:int  = 300


@dataclass
class PrinterConfig:
    """Thermal printer settings."""
    printer_port:    str  = "COM1"
    auto_print:      bool = False
    print_on_error:  bool = True


@dataclass
class AuditConfig:
    """Audit and logging settings."""
    log_retention_days: int  = 90
    export_path:        str  = ""
    require_inspector:  bool = False


@dataclass
class AppConfig:
    device:  DeviceConfig  = field(default_factory=DeviceConfig)
    scan:    ScanConfig    = field(default_factory=ScanConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    printer: PrinterConfig = field(default_factory=PrinterConfig)
    audit:   AuditConfig   = field(default_factory=AuditConfig)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: Path = CONFIG_FILE) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "device":  asdict(self.device),
            "scan":    asdict(self.scan),
            "network": asdict(self.network),
            "printer": asdict(self.printer),
            "audit":   asdict(self.audit),
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        logger.info("Config saved to %s", path)

    @classmethod
    def load(cls, path: Path = CONFIG_FILE) -> "AppConfig":
        path = Path(path)
        if not path.exists():
            logger.info("No config file found at %s — using defaults", path)
            return cls()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            cfg = cls(
                device  = DeviceConfig(**data.get("device",  {})),
                scan    = ScanConfig(**data.get("scan",    {})),
                network = NetworkConfig(**data.get("network", {})),
                printer = PrinterConfig(**data.get("printer", {})),
                audit   = AuditConfig(**data.get("audit",   {})),
            )
            logger.info("Config loaded from %s", path)
            return cfg
        except Exception as exc:
            logger.warning("Failed to load config: %s — using defaults", exc)
            return cls()

    def log_change(self, section: str, field_name: str,
                   old_val, new_val, changed_by: str = "admin") -> None:
        logger.info(
            "CONFIG CHANGE | by=%s | %s.%s: %r → %r",
            changed_by, section, field_name, old_val, new_val,
        )


# Singleton
_config: AppConfig | None = None

def get_config() -> AppConfig:
    global _config
    if _config is None:
        _config = AppConfig.load()
    return _config

def save_config() -> None:
    if _config is not None:
        _config.save()
