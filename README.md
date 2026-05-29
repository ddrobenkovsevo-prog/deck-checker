# Deck Checker

Casino card deck verification system. Inspects 8-deck shoes (Blackjack / Baccarat) used in back-office, identifies missing, extra, and unrecognised cards.

## Target platform

- Raspberry Pi 5 (16 GB)
- Coral USB Edge TPU
- 2× InnoMaker IMX296 Color global shutter cameras
- 10.1" EDATEC capacitive touchscreen (1024×600)
- LED strobe lighting synchronized via camera strobe pin
- Thermal printer + S3/FTP log sync

## Development strategy

Everything is built and tested on a laptop first, with hardware abstractions and mocks. When the device boots up, only the bottom layer (camera, GPIO) is swapped for real implementations. No "wait for hardware" blockers.

## Quick start

```bash
# Create venv and install
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Run the CV pipeline on a test image
python scripts/test_pipeline.py path/to/card.jpg

# Run tests
pytest
```

## Architecture

See `docs/architecture.md` for the full picture. High level:

```
[ Operator ] → [ UI / state machine ] ─┐
                                       │
[ Cameras ] → [ Capture worker ] ─→ [ Recognition engine ] ─→ [ Storage ] → [ Printer + S3/FTP ]
[ Sensors ] → [ Hardware controller ] ─┘
                          ↓
                   [ Motor + LEDs ]
```

## Status

- [x] Project skeleton
- [x] CV pipeline: preprocessing + ROI extraction
- [ ] Recognition: template matching
- [ ] Recognition: YOLO fallback
- [ ] Card learning (two-pass)
- [ ] Hardware abstractions + mocks
- [ ] State machine
- [ ] PyQt6 UI
- [ ] SQLite storage
- [ ] S3/FTP sync
- [ ] Thermal printer
- [ ] Program Digest (GLI compliance)
