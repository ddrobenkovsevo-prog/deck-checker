# Инструкция по синхронизации проекта

## Рабочий процесс

Весь код пишется в чате с Claude → копируется на Windows → одной командой летит на GitHub и Pi.

---

## Ежедневная работа (одна команда)

```powershell
cd C:\Users\ddrobenkovs\deck-checker-new
.\sync.ps1 "что сделали"
```

Это автоматически:
1. Коммитит все изменения на Windows
2. Пушит на GitHub
3. Подтягивает на Pi
4. Запускает тесты на Pi

---

## Если Pi не подключён по кабелю

Сначала задай статический IP (слетает после перезагрузки Windows):
```powershell
New-NetIPAddress -InterfaceAlias "Ethernet 2" -IPAddress 192.168.10.1 -PrefixLength 24
```
Потом проверь:
```powershell
ping 192.168.10.2
```

---

## Если забыл активировать venv на Pi

```bash
cd ~/deck-checker
source .venv/bin/activate
pytest tests/ -v
```

---

## Что где лежит

| Место | Путь |
|-------|------|
| Windows (основная папка) | `C:\Users\ddrobenkovs\deck-checker-new` |
| GitHub | `https://github.com/ddrobenkovsevo-prog/deck-checker` |
| Raspberry Pi | `~/deck-checker` |

---

## Если что-то пошло не так с git

**Конфликт при pull:**
```powershell
git merge --abort
git pull
git checkout --theirs <файл>
git add .
git commit -m "resolve conflict"
```

**Посмотреть статус:**
```powershell
git status
git log --oneline -5
```

**Откатить последний коммит (если напортачил):**
```powershell
git revert HEAD
```

---

## Текущее состояние проекта

| Фаза | Компонент | Статус |
|------|-----------|--------|
| 1 | models, vision, storage | ✅ |
| 2 | hardware abstraction | ✅ |
| 3 | state machine | ✅ |
| 4 | PyQt6 UI | ⏳ |
| 5 | SQLite audit | ⏳ |
| 6 | S3/SFTP sync | ⏳ |
| 7 | Thermal printer | ⏳ |
| 8 | GLI compliance | ⏳ |

**Тестов: 110/110 ✅ на Windows и Pi**

---

## Контекст проекта

Если начинаешь новую сессию с Claude — покажи файл `PROJECT_CONTEXT.md` из репозитория, он мгновенно восстановит весь контекст.

---

## Подключение к Pi по SSH

```powershell
ssh evoroot@192.168.10.2
```

Если не подключается — проверь IP на Windows:
```powershell
ipconfig
# Ethernet 2 должен показывать 192.168.10.1
```

---

## Структура репозитория

```
deck-checker/
├── src/deck_checker/
│   ├── core/
│   │   ├── models.py          # Card, Suit, Rank, ScanReport
│   │   └── state_machine.py   # INIT→IDLE→SCANNING→SUCCESS/ERROR
│   ├── vision/
│   │   ├── preprocessing.py   # contour detect, perspective warp
│   │   ├── roi.py             # corner extract, binarise
│   │   ├── recognition.py     # template matching
│   │   └── learning.py        # two-pass calibration
│   ├── storage/
│   │   └── library.py         # save/load deck profiles
│   └── hardware/
│       ├── base.py            # Protocol interfaces
│       ├── camera.py          # IMX296 + MockCamera
│       ├── gpio.py            # IR trigger, motor, hall sensor
│       └── strobe.py          # LED strobe
├── tests/
│   ├── test_phase2.py         # 59 tests — vision + storage
│   ├── test_hardware.py       # 28 tests — hardware mocks
│   └── test_state_machine.py  # 23 tests — state machine
├── demo_windows.py            # standalone demo без железа
├── sync.ps1                   # скрипт синхронизации
├── PROJECT_CONTEXT.md         # контекст для Claude
└── pyproject.toml
```
