"""Core data models for the Deck Checker system."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Optional


class Suit(Enum):
    SPADES = "S"
    HEARTS = "H"
    DIAMONDS = "D"
    CLUBS = "C"


class Rank(Enum):
    ACE = "A"
    TWO = "2"
    THREE = "3"
    FOUR = "4"
    FIVE = "5"
    SIX = "6"
    SEVEN = "7"
    EIGHT = "8"
    NINE = "9"
    TEN = "T"
    JACK = "J"
    QUEEN = "Q"
    KING = "K"
    JOKER = "JK"


class GameType(Enum):
    BLACKJACK = "blackjack"
    BACCARAT = "baccarat"


class ScanState(Enum):
    INIT = auto()
    IDLE = auto()
    DECK_LOADED = auto()
    SCANNING = auto()
    SUCCESS = auto()
    ERROR = auto()
    MANUAL_VALIDATION = auto()


@dataclass(frozen=True)
class Card:
    rank: Rank
    suit: Suit

    def __str__(self) -> str:
        return f"{self.rank.value}{self.suit.value}"

    def __repr__(self) -> str:
        return f"Card({self.rank.value}{self.suit.value})"

    @classmethod
    def from_string(cls, s: str) -> "Card":
        if len(s) < 2:
            raise ValueError(f"Invalid card string: {s!r}")
        rank_str = s[:-1]
        suit_str = s[-1]
        rank = Rank(rank_str)
        suit = Suit(suit_str)
        return cls(rank=rank, suit=suit)


@dataclass
class RecognitionResult:
    card: Optional[Card]
    confidence: float
    method: str
    raw_rank: Optional[str] = None
    raw_suit: Optional[str] = None

    @property
    def is_confident(self) -> bool:
        return self.confidence >= 0.82

    @property
    def succeeded(self) -> bool:
        return self.card is not None and self.is_confident


def build_standard_shoe(num_decks: int = 8) -> Counter:
    single = Counter(
        Card(rank=r, suit=s)
        for r in Rank
        if r != Rank.JOKER
        for s in Suit
    )
    return Counter({card: count * num_decks for card, count in single.items()})


@dataclass
class ScanReport:
    game_type: GameType
    num_decks: int
    scanned: list[RecognitionResult] = field(default_factory=list)
    manual_overrides: dict[int, Card] = field(default_factory=dict)

    @property
    def recognized_cards(self) -> list[Card]:
        cards = []
        for i, r in enumerate(self.scanned):
            if i in self.manual_overrides:
                cards.append(self.manual_overrides[i])
            elif r.card is not None:
                cards.append(r.card)
        return cards

    @property
    def found_counts(self) -> Counter:
        return Counter(self.recognized_cards)

    @property
    def expected_counts(self) -> Counter:
        return build_standard_shoe(self.num_decks)

    @property
    def missing_cards(self) -> Counter:
        return self.expected_counts - self.found_counts

    @property
    def extra_cards(self) -> Counter:
        return self.found_counts - self.expected_counts

    @property
    def is_complete(self) -> bool:
        return len(self.recognized_cards) == self.num_decks * 52

    @property
    def is_valid(self) -> bool:
        return self.is_complete and not self.missing_cards and not self.extra_cards

    @property
    def low_confidence_indices(self) -> list[int]:
        return [
            i for i, r in enumerate(self.scanned)
            if not r.is_confident and i not in self.manual_overrides
        ]

    def digest(self) -> str:
        payload = json.dumps(
            [str(c) for c in self.recognized_cards], sort_keys=True
        ).encode()
        return hashlib.sha256(payload).hexdigest()
