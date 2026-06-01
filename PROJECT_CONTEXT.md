# Deck Checker — Project Context & Progress
*Последнее обновление: 2026-06-01*

## Что это
Автоматическая верификация колод для казино. 8 дек (416 карт) за < 60 сек. Blackjack/Baccarat. Reference: TCS John Huxley Deck Argus.

## Hardware
- Raspberry Pi 5 16GB — ✅ настроен, работает
- InnoMaker IMX296 камера — pending
- 10.1" touchscreen 1024×600 — pending
- Coral USB Edge TPU — pending
- IR sensor, мотор, LED strobe, hall sensor — pending

## Текущее состояние

| Фаза | Компонент | Статус |
|------|-----------|--------|
| 1 | models, vision, storage | ✅ |
| 2 | hardware abstraction | ✅ |
| 3 | state machine | ✅ |
| 4 | PyQt6 UI + Admin settings | ✅ |
| 5 | SQLite audit | ⏳ next |
| 6 | S3/SFTP sync | ⏳ |
| 7 | Thermal printer | ⏳ |
| 8 | GLI compliance | ⏳ |

**Тестов: 110/110 на Windows и Pi**

## Структура
```
deck-checker/
├── src/deck_checker/
│   ├── core/
│   │   ├── models.py          # Card, Suit, Rank, ScanReport
│   │   ├── state_machine.py   # INIT→IDLE→SCANNING→SUCCESS/ERROR
│   │   └── config.py          # AppConfig (device/scan/network/printer/audit)
│   ├── vision/                # preprocessing, roi, recognition, learning
│   ├── storage/library.py     # save/load deck profiles
│   ├── hardware/              # base, camera, gpio, strobe (real + mock)
│   └── ui/
│       ├── theme.py           # Evolution purple #1E1040, design tokens
│       ├── widgets.py         # StatusDot, RingProgress, CardBadge, ResultBanner
│       ├── screens.py         # Idle, Scanning, Result, Manual screens
│       ├── admin_screen.py    # Admin settings (password: evoroot)
│       ├── main_window.py     # main kiosk window
│       ├── evolution_logo_white.svg
│       └── settings_icon_white.png
├── tests/                     # 110 tests
├── main.py                    # entry: python main.py --windowed --demo
├── sync.ps1                   # Windows→GitHub→Pi sync
└── pyproject.toml
```

## UI features (Phase 4)
- Evolution purple theme (#1E1040)
- Часы + дата + Evolution logo в правом верхнем углу
- IDLE: форма (Game Type, Decks, Table №, Inspector, PIT, Card Box) + RingProgress
- Game types: Blackjack, Baccarat, Poker, Always 6/7/8
- SCANNING: кольцо прогресса, последние 6 карт, скорость, elapsed
- RESULT: SUCCESS зелёный / ERROR красный, missing/extra карты, digest
- MANUAL VALIDATION: оператор исправляет low-confidence карты
- ADMIN settings (пароль evoroot): 5 вкладок DEVICE/SCANNING/NETWORK/PRINTER/AUDIT
- config.json — все настройки, изменения логируются

## Запуск
```powershell
# Windows dev
python main.py --windowed --demo

# Pi production (fullscreen)
python main.py --demo
```

## Sync workflow
```powershell
.\sync.ps1 "commit message"
```
Автоматом: git push → Pi pull → pytest

## Ключевые детали
- ROI: rows 0:57 rank, 57:105 suit (из 250×350 карты)
- Confidence threshold: 0.82
- Pins (BCM): IR=17, MOTOR=27/22/18, HALL=23, STROBE=24
- Admin password: evoroot
- Pi IP: 192.168.10.2 (Ethernet) / SSH evoroot@192.168.10.2

## Следующий шаг: Phase 5 — SQLite audit
- Таблица scans: timestamp, device, table, inspector, pit, card_box, game_type, num_decks, valid, digest, missing, extra
- История сканов с фильтрами
- Связь с metadata из IDLE формы
