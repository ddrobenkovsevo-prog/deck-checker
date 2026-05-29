"""
demo_windows.py — Deck Checker pipeline demo (standalone, no package install)

Works with a flat directory where all .py files are in the same folder.

Usage:
    python demo_windows.py
    python demo_windows.py --errors 3
    python demo_windows.py --save-images
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# ── ANSI colours ─────────────────────────────────────────────────────────────
import os; os.system("")
GREEN  = "\033[92m"; YELLOW = "\033[93m"; RED = "\033[91m"
CYAN   = "\033[96m"; BOLD   = "\033[1m";  RESET = "\033[0m"

# ─────────────────────────────────────────────────────────────────────────────
# MODELS (inline — no import needed)
# ─────────────────────────────────────────────────────────────────────────────

class Suit(Enum):
    SPADES = "S"; HEARTS = "H"; DIAMONDS = "D"; CLUBS = "C"

class Rank(Enum):
    ACE="A"; TWO="2"; THREE="3"; FOUR="4"; FIVE="5"; SIX="6"; SEVEN="7"
    EIGHT="8"; NINE="9"; TEN="T"; JACK="J"; QUEEN="Q"; KING="K"; JOKER="JK"

class GameType(Enum):
    BLACKJACK = "blackjack"; BACCARAT = "baccarat"

@dataclass(frozen=True)
class Card:
    rank: Rank
    suit: Suit
    def __str__(self): return f"{self.rank.value}{self.suit.value}"

@dataclass
class RecognitionResult:
    card: Optional[Card]
    confidence: float
    method: str
    raw_rank: Optional[str] = None
    raw_suit: Optional[str] = None
    @property
    def is_confident(self): return self.confidence >= 0.82
    @property
    def succeeded(self): return self.card is not None and self.is_confident

def _build_shoe(num_decks: int) -> Counter:
    single = Counter(Card(r, s) for r in Rank if r != Rank.JOKER for s in Suit)
    return Counter({c: n * num_decks for c, n in single.items()})

@dataclass
class ScanReport:
    game_type: GameType
    num_decks: int
    scanned: list[RecognitionResult] = field(default_factory=list)
    manual_overrides: dict = field(default_factory=dict)

    @property
    def recognized_cards(self):
        cards = []
        for i, r in enumerate(self.scanned):
            if i in self.manual_overrides: cards.append(self.manual_overrides[i])
            elif r.card is not None: cards.append(r.card)
        return cards

    @property
    def found_counts(self): return Counter(self.recognized_cards)
    @property
    def expected_counts(self): return _build_shoe(self.num_decks)
    @property
    def missing_cards(self): return self.expected_counts - self.found_counts
    @property
    def extra_cards(self): return self.found_counts - self.expected_counts
    @property
    def is_complete(self): return len(self.recognized_cards) == self.num_decks * 52
    @property
    def is_valid(self): return self.is_complete and not self.missing_cards and not self.extra_cards
    @property
    def low_confidence_indices(self):
        return [i for i, r in enumerate(self.scanned)
                if not r.is_confident and i not in self.manual_overrides]
    def digest(self):
        payload = json.dumps([str(c) for c in self.recognized_cards]).encode()
        return hashlib.sha256(payload).hexdigest()

# ─────────────────────────────────────────────────────────────────────────────
# ROI / PREPROCESSING (inline)
# ─────────────────────────────────────────────────────────────────────────────

CARD_W, CARD_H = 250, 350

def binarise(roi: np.ndarray) -> np.ndarray:
    _, binary = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = cv2.findNonZero(binary)
    if coords is None: return binary
    x, y, w, h = cv2.boundingRect(coords)
    pad = 3
    x, y = max(0, x-pad), max(0, y-pad)
    w = min(binary.shape[1]-x, w+2*pad)
    h = min(binary.shape[0]-y, h+2*pad)
    return binary[y:y+h, x:x+w]

def extract_rois(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Corner: rows 0:105, cols 0:50; rank 0:57, suit 57:105
    corner = img[:105, :50]
    rank_raw = corner[:57, :]
    suit_raw = corner[57:, :]
    rank_roi = cv2.resize(rank_raw, (45, 60), interpolation=cv2.INTER_AREA)
    suit_roi = cv2.resize(suit_raw, (40, 40), interpolation=cv2.INTER_AREA)
    return binarise(rank_roi), binarise(suit_roi)

# ─────────────────────────────────────────────────────────────────────────────
# TEMPLATE LIBRARY
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TemplateLibrary:
    rank_templates: dict = field(default_factory=dict)
    suit_templates: dict = field(default_factory=dict)
    def is_ready(self): return bool(self.rank_templates) and bool(self.suit_templates)
    def rank_count(self): return len(self.rank_templates)
    def suit_count(self): return len(self.suit_templates)

# ─────────────────────────────────────────────────────────────────────────────
# RECOGNITION (inline)
# ─────────────────────────────────────────────────────────────────────────────

CONFIDENCE_THRESHOLD = 0.82
RETRY_THRESHOLD = 0.65
MIN_DIM = 8

def _match_one(query: np.ndarray, template: np.ndarray) -> float:
    qh, qw = query.shape[:2]
    th, tw = template.shape[:2]
    if qh < MIN_DIM or qw < MIN_DIM or th < MIN_DIM or tw < MIN_DIM: return 0.0
    if (th, tw) != (qh, qw):
        template = cv2.resize(template, (qw, qh), interpolation=cv2.INTER_AREA)
    result = cv2.matchTemplate(query.astype(np.float32),
                               template.astype(np.float32), cv2.TM_CCOEFF_NORMED)
    return float(result[0, 0])

def _best_match(roi, templates):
    best_key, best_score = None, -1.0
    for key, tmpl in templates.items():
        s = _match_one(roi, tmpl)
        if s > best_score: best_score, best_key = s, key
    if best_key is None: raise ValueError("templates is empty")
    return best_key, best_score

def _conf(rs, ss): return float(np.sqrt(max(rs,0)*max(ss,0)))

def recognise_card(img: np.ndarray, lib: TemplateLibrary) -> RecognitionResult:
    if not lib.is_ready():
        return RecognitionResult(card=None, confidence=0.0, method="template")
    rank_roi, suit_roi = extract_rois(img)
    rank1, rs1 = _best_match(rank_roi, lib.rank_templates)
    suit1, ss1 = _best_match(suit_roi, lib.suit_templates)
    conf1 = _conf(rs1, ss1)
    if conf1 >= CONFIDENCE_THRESHOLD:
        return RecognitionResult(Card(rank1, suit1), conf1, "template",
                                 rank1.value, suit1.value)
    # retry bottom-right corner
    img_rot = cv2.rotate(img, cv2.ROTATE_180)
    rank_roi2, suit_roi2 = extract_rois(img_rot)
    rank2, rs2 = _best_match(rank_roi2, lib.rank_templates)
    suit2, ss2 = _best_match(suit_roi2, lib.suit_templates)
    conf2 = _conf(rs2, ss2)
    if conf2 > conf1:
        return RecognitionResult(Card(rank2, suit2), conf2, "template",
                                 rank2.value, suit2.value)
    return RecognitionResult(Card(rank1, suit1), conf1, "template",
                             rank1.value, suit1.value)

def recognise_batch(images, lib):
    return [recognise_card(img, lib) for img in images]

# ─────────────────────────────────────────────────────────────────────────────
# SYNTHETIC IMAGE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _rank_template(rank: Rank) -> np.ndarray:
    img = np.full((60, 45), 255, dtype=np.uint8)
    ordinal = list(Rank).index(rank)
    for i in range(ordinal + 1):
        x = 4 + (i % 5) * 7
        y = 4 + (i // 5) * 12
        img[y:y+8, x:x+6] = 0
    return img

def _suit_template(suit: Suit) -> np.ndarray:
    img = np.full((40, 40), 255, dtype=np.uint8)
    o = list(Suit).index(suit)
    if o == 0:
        cv2.fillPoly(img, [np.array([[20,5],[5,35],[35,35]], np.int32)], 0)
    elif o == 1:
        cv2.circle(img, (20,20), 14, 0, -1)
    elif o == 2:
        cv2.fillPoly(img, [np.array([[20,4],[36,20],[20,36],[4,20]], np.int32)], 0)
    else:
        cv2.circle(img,(20,28),10,0,-1); cv2.circle(img,(12,18),9,0,-1); cv2.circle(img,(28,18),9,0,-1)
    return img

def make_card_image(card: Card, noise: int = 0) -> np.ndarray:
    img = np.full((350, 250), 200, dtype=np.uint8)
    img[0:57,  0:45] = _rank_template(card.rank)[:57, :]
    img[57:97, 0:40] = _suit_template(card.suit)
    if noise:
        layer = np.random.randint(-noise, noise+1, img.shape, dtype=np.int16)
        img = np.clip(img.astype(np.int16)+layer, 0, 255).astype(np.uint8)
    return img

def build_library() -> TemplateLibrary:
    lib = TemplateLibrary()
    for rank in Rank:
        if rank != Rank.JOKER:
            lib.rank_templates[rank] = binarise(_rank_template(rank))
    for suit in Suit:
        lib.suit_templates[suit] = binarise(_suit_template(suit))
    return lib

# ─────────────────────────────────────────────────────────────────────────────
# DEMO SECTIONS
# ─────────────────────────────────────────────────────────────────────────────

def section(title):
    print(f"\n{CYAN}{BOLD}{'─'*60}{RESET}")
    print(f"{CYAN}{BOLD}  {title}{RESET}")
    print(f"{CYAN}{'─'*60}{RESET}")

def demo_single_card(lib, save_dir):
    section("1. Single card recognition")
    cards = [Card(Rank.ACE,Suit.SPADES), Card(Rank.KING,Suit.HEARTS),
             Card(Rank.QUEEN,Suit.DIAMONDS), Card(Rank.TWO,Suit.CLUBS),
             Card(Rank.TEN,Suit.SPADES), Card(Rank.JACK,Suit.HEARTS)]
    print(f"\n  {'Card':<8} {'Recognised':<12} {'Confidence':>10}  Status")
    print(f"  {'────':<8} {'──────────':<12} {'──────────':>10}  ──────")
    for card in cards:
        img = make_card_image(card, noise=5)
        r = recognise_card(img, lib)
        ok = r.card == card
        status = f"{GREEN}✓ OK{RESET}" if ok else f"{RED}✗ WRONG{RESET}"
        cc = GREEN if r.confidence >= 0.82 else YELLOW
        print(f"  {str(card):<8} {str(r.card):<12} {cc}{r.confidence:>9.1%}{RESET}  {status}")
        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(save_dir / f"card_{card}.png"), img)
    print()

def demo_full_shoe(lib, num_decks=8, num_errors=0):
    section(f"2. Full {num_decks}-deck shoe scan ({num_decks*52} cards)")
    shoe = [Card(r,s) for _ in range(num_decks)
            for s in Suit for r in Rank if r != Rank.JOKER]
    random.shuffle(shoe)

    if num_errors:
        all_cards = [Card(r,s) for r in Rank if r!=Rank.JOKER for s in Suit]
        idxs = random.sample(range(len(shoe)), k=min(num_errors, len(shoe)))
        for i in idxs: shoe[i] = random.choice(all_cards)
        print(f"\n  {YELLOW}⚠ Injected {num_errors} error(s){RESET}")

    print(f"\n  Generating {len(shoe)} card images… ", end="", flush=True)
    t0 = time.perf_counter()
    images = [make_card_image(c, noise=3) for c in shoe]
    print(f"done ({time.perf_counter()-t0:.2f}s)")

    print(f"  Running recognition… ", end="", flush=True)
    t0 = time.perf_counter()
    results = recognise_batch(images, lib)
    elapsed = time.perf_counter()-t0
    print(f"done ({elapsed:.2f}s  →  {len(shoe)/elapsed:.0f} cards/sec)")

    report = ScanReport(GameType.BLACKJACK, num_decks)
    report.scanned = results
    correct = sum(1 for r,e in zip(results,shoe) if r.card==e)
    total = len(shoe)
    lc = len(report.low_confidence_indices)

    print(f"\n  {'Cards scanned:':<28} {total}")
    print(f"  {'Correct recognitions:':<28} {GREEN}{correct}{RESET}/{total} ({correct/total:.1%})")
    print(f"  {'Low confidence:':<28} {YELLOW if lc else GREEN}{lc}{RESET}")

    for label, diff in [("Missing", report.missing_cards), ("Extra", report.extra_cards)]:
        if diff:
            items = ', '.join(f'{c}×{n}' for c,n in list(diff.items())[:5])
            print(f"  {label+' cards:':<28} {RED}{sum(diff.values())}{RESET} ({items})")

    verdict = f"{GREEN}✓ SHOE VALID{RESET}" if report.is_valid else f"{RED}✗ SHOE INVALID{RESET}"
    print(f"\n  Verdict: {verdict}")
    print(f"  Audit digest: {report.digest()[:32]}…\n")

def demo_timing(lib):
    section("3. Performance benchmark")
    print(f"\n  {'Cards':<8} {'Time (s)':>10} {'Cards/sec':>12} {'< 60s':>8}")
    print(f"  {'─────':<8} {'────────':>10} {'─────────':>12} {'─────':>8}")
    for n in [52, 208, 416]:
        cards = [Card(list(Rank)[i%13], list(Suit)[i%4]) for i in range(n)]
        images = [make_card_image(c) for c in cards]
        t0 = time.perf_counter()
        recognise_batch(images, lib)
        elapsed = time.perf_counter()-t0
        ok = f"{GREEN}✓{RESET}" if elapsed < 60 else f"{RED}✗{RESET}"
        print(f"  {n:<8} {elapsed:>10.3f} {n/elapsed:>12.0f} {ok:>8}")
    print()

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-images", action="store_true")
    parser.add_argument("--errors", type=int, default=0)
    args = parser.parse_args()

    print(f"\n{BOLD}{CYAN}╔══════════════════════════════════════════════════════╗{RESET}")
    print(f"{BOLD}{CYAN}║          DECK CHECKER — Windows Demo                 ║{RESET}")
    print(f"{BOLD}{CYAN}║          (standalone, no package install needed)     ║{RESET}")
    print(f"{BOLD}{CYAN}╚══════════════════════════════════════════════════════╝{RESET}")

    section("0. Building template library")
    t0 = time.perf_counter()
    lib = build_library()
    print(f"\n  {GREEN}✓{RESET} {lib.rank_count()} rank + {lib.suit_count()} suit templates "
          f"in {(time.perf_counter()-t0)*1000:.0f} ms\n")

    save_dir = Path("demo_output") if args.save_images else None
    demo_single_card(lib, save_dir)
    demo_full_shoe(lib, num_decks=8, num_errors=args.errors)
    demo_timing(lib)

    print(f"{CYAN}{'─'*60}{RESET}")
    print(f"{BOLD}{GREEN}  All demos complete.{RESET}\n")

if __name__ == "__main__":
    main()
