.\sync.ps1 "Add instructions and project context" FOR PUSHING FROM PI AND WIND



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













# Deck Checker — сводка сессии 16.06.2026 (XTR-триггер + цвет)

## Главный результат
Триггерный конвейер работает end-to-end на новой камере:
сенсор PM-T45 → Arduino (XTR-триггер + keepalive) → вспышка в окне
экспозиции → захват по триггеру (picamera2) → правильный цвет →
извлечение индекс-ROI → template matching → метка карты.

**Recognition pass: 51/51 распознано, score 0.90-1.00, margin 0.09-0.32.**
- все метки уникальны и верны
- free-run лотерея потерь устранена: один импульс = один кадр

## РАБОЧАЯ КОНФИГУРАЦИЯ (зафиксировать!)

### Камера (новая, после замены камеры+линзы)
- разъём CSI cam0, i2c-шина 6, сенсор 0x1a
- config.txt: `camera_auto_detect=0`, в конце после `[all]`:
  `dtoverlay=imx296,always-on,cam0`
- триггерный режим включается УТИЛИТОЙ i2c.py (НЕ через rpicam):
  `sudo python3 ~/cam-imx296raw-trigger/i2c-tools-python-eeprom-strobe-trigger/i2c.py trigger on --bus 6`
- регистры триггера ON: 0x300B=0x01, 0x30AE=0x01
- standby-регистры 0x300A/0x3000 иногда залипают в 0x01 после грязного
  выхода (Ctrl+C во время стрима) — расклинить чистым free-run:
  `rpicam-hello -t 2000 --camera 0` (дать закрыться самому)

### Цвет (КЛЮЧЕВЫЕ открытия сессии)
- ColourGains ФИКСИРОВАННЫЕ: **(2.73, 2.13)** — намерены через AWB под
  постоянным FLASHON на белом листе. AWB в боевом режиме НЕ использовать
  (гуляет от кадра к кадру: цветная карта — один баланс, ч/б — другой)
- AnalogueGain: 2.0 (баланс мерян при этом gain — съёмка тем же gain!)
- БЫЛ БАГ: лишний `cv2.cvtColor(frame, COLOR_RGB2BGR)` при сохранении →
  picamera2 с RGB888 уже отдаёт BGR → двойной своп → красное в синее.
  ИСПРАВЛЕНО: сохранять `cv2.imwrite(fn, frame)` без cvtColor.
- камера+ИК-фильтр ИСПРАВНЫ: при комнатном свете и под FLASHON красное
  красное. Проблема была только в коде (своп) + баланс под вспышку.

### Геометрия (после смены камеры индекс сместился)
- DELAY (SENSOR_TO_FOV) = **30000** мкс — ловит карту в идеальной точке:
  индекс с запасом поля слева/сверху/снизу, не у края
- позицию ловим ТАЙМИНГОМ (DELAY), а не движением камеры (сдвиг камеры
  упирается в засветку за машиной)
- окно экстрактора: WX0,WX1,WY0,WY1 = **150, 620, 230, 950**
- индекс в кадре: ранг X276-470 Y333-660, масть X290-440 Y648-850

### Arduino (прошивка strobe_v5_xtr_keepalive.ino)
- idle XTR = HIGH; экспозиция = XTR в LOW; выдержка = длина LOW + 14.26мкс
- DELAY=30000, EXP=500, PRE=60, POST=120 (окно 680 мкс), LOCKOUT=80000
- боевой режим: `KEEPALIVE 400` + `ARM`
  - keepalive 400мс = холостые импульсы (экспозиция без вспышки), держат
    камеру живой между картами (таймаут libcamera 1с не наступает)
  - реальная карта от сенсора = импульс СО вспышкой, сбрасывает таймер
- НЕЛЬЗЯ keepalive 700 (близко к таймауту 1с, не успевает)
- picocom: `picocom -b 115200 --echo --omap crlf /dev/ttyUSB0`
- ВАЖНО: команды по одной с Enter (склеиваются → парсятся криво)

### Захват (trigger_capture.py)
- таймаут камеры задан в коде: FrameDurationLimits (100, 1000000*1000)
  — но он НЕ перекрывает секундный V4L2-таймаут, потому и нужен keepalive
- триггер включается ИЗ скрипта ПОСЛЕ cam.start() (старт сбрасывает режим!)
- порог яркости **60** (--threshold 60) — отсекает тёмные keepalive-кадры,
  карты дают 110-145
- запуск: `python3 trigger_capture.py --out DIR/ --frames 80 --duration 120 --gain 2.0 --threshold 60`

## Грабли сессии (чтобы не повторять)
- делитель XTR был 11k вместо 1.8k → 1В на TRIG+ вместо 3.3В → камера не
  триггерилась. Правильно: D9 →[1.8k]→ TRIG+ →[3.3k]→ GND. Дало ~3В.
- XTR-провод ОТВАЛИВАЛСЯ (плавающий контакт) → камера ждёт и таймаутит.
  TODO: посадить XTR на разъём/пропаять, не на честном слове!
- libcamera v0.7.1 SEGFAULTS на любом внешнем yaml (camera_timeout_value_ms)
  — известный баг, обойдён keepalive-импульсами
- cam.start() в picamera2 СБРАСЫВАЕТ триггерный режим (как rpicam) →
  trigger on давать ПОСЛЕ старта камеры
- Ctrl+Z в rpicam НЕ освобождает камеру (kill %N); Ctrl+C освобождает
- старые card_* кадры в папке мешали тесту (распознались как мусор) —
  чистить папку перед прогоном (rm -rf)

## Файлы (на Pi и в /mnt/user-data/outputs)
- `strobe_v5_xtr_keepalive.ino` — прошивка с keepalive
- `trigger_capture.py` — захват по триггеру (вкл. триггер после start)
- `index_extract_prototype.py` — извлечение, окно 150..620/230..950
- `learn_deck.py` — разметка (--missing/--fix/--append)
- `recognize_run.py` — распознавание TM_CCOEFF_NORMED
- `library_trig/` — 52 эталонных триггерных шаблона, правильный цвет

## Открытые мелочи (доводка, не блокеры)
1. UNRECOGNIZED единичный — кроп иногда цепляет соседнюю карту справа.
   Поджать правую границу окна / отсечь всё правее ранга.
2. Дубли захвата — иногда один кадр двоится, теряется другой (51 из 52).
   Разобраться с дедупом по mtime / порогом склейки.
3. 7C дала повтор метки (trig_00036 и 00052) — следствие дубля захвата.

## ДАЛЬНЕЙШИЙ ПЛАН

### Ближайшее (доводка триггерного стенда)
1. Закрепить XTR-провод физически (разъём/пайка) — критично, плавающий
   контакт уже ронял систему.
2. Утилита калибровки DELAY (идея Дмитрия): интерактивный подбор позиции
   карты в кадре стрелками, визуально, с сохранением константы.
3. Починить единичные промахи кропа (соседняя карта) и дубли захвата.
4. Прогнать колоду 5+ раз, собрать статистику margin (стабильность).

### Перенос в проектный код (Phase 2)
5. index_extract_prototype.py → vision/roi.py (новая геометрия)
6. recognize_run.py (канонизация + NCC) → vision/recognition.py
7. learn_deck.py → vision/learning.py (two-pass)
8. trigger_capture.py → hardware/camera.py (+ мок для оффлайн-тестов)
9. Сверка результата прогона с эталонной колодой → core/ScanReport
   (diff/multiset уже реализован в core/models.py)

### Дальше по роадмапу (Phase 3+)
10. State machine: INIT→IDLE→DECK_LOADED→SCANNING→SUCCESS/ERROR
11. PyQt6 kiosk UI на тачскрине EDATEC 10.1"
12. SQLite audit logging + S3/SFTP sync
13. Thermal printer (python-escpos)
14. Program Digest с ed25519 для GLI
15. Полный прогон 8-колодного шуза (416 карт) под 60 сек

