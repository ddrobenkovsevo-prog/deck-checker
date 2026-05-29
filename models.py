"""Domain models for cards, decks, and recognition results.

These models are deliberately framework-free — no OpenCV, no PyQt, no GPIO.
That keeps the domain testable and lets us swap any layer above without rewriting
the contracts the recognition engine and state machine speak.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


class Suit(str, Enum):
    """Card suit. Joker has no suit (None)."""

    SPADES = "S"
    HEARTS = "H"
    DIAMONDS = "D"
    CLUBS = "C"

    @property
    def is_red(self) -> bool:
        return self in (Suit.HEARTS, Suit.DIAMONDS)


class Rank(str, Enum):
    """Card rank, including a sentinel for the joker."""

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
    ACE = "A"
    JOKER = "JK"


@dataclass(frozen=True, slots=True)
class Card:
    """A single playing card. Suit is None only for jokers."""

    rank: Rank
    suit: Suit | None = None

    def __post_init__(self) -> None:
        if self.rank is Rank.JOKER and self.suit is not None:
            raise ValueError("Joker must not have a suit")
        if self.rank is not Rank.JOKER and self.suit is None:
            raise ValueError(f"Non-joker rank {self.rank} requires a suit")

    @property
    def code(self) -> str:
        """Short canonical identifier: 'AS', 'KH', 'TC', 'JK'."""
        if self.rank is Rank.JOKER:
            return "JK"
        assert self.suit is not None
        return f"{self.rank.value}{self.suit.value}"

    @classmethod
    def from_code(cls, code: str) -> Card:
        """Parse a card from its canonical code (e.g. 'AS', 'TH', 'JK')."""
        code = code.upper().strip()
        if code == "JK":
            return cls(rank=Rank.JOKER)
        if len(code) != 2:
            raise ValueError(f"Invalid card code: {code!r}")
        try:
            rank = Rank(code[0])
            suit = Suit(code[1])
        except ValueError as e:
            raise ValueError(f"Invalid card code: {code!r}") from e
        return cls(rank=rank, suit=suit)

    def __str__(self) -> str:
        return self.code


def standard_deck() -> list[Card]:
    """The 52 cards of a standard deck, no joker."""
    return [
        Card(rank=r, suit=s)
        for s in Suit
        for r in Rank
        if r is not Rank.JOKER
    ]


def deck_with_jokers(jokers: int = 1) -> list[Card]:
    """A 52-card deck plus N jokers."""
    return standard_deck() + [Card(rank=Rank.JOKER) for _ in range(jokers)]


class GameType(str, Enum):
    """Game configuration determines deck count and expected card multiset."""

    BLACKJACK_8 = "blackjack_8"
    BACCARAT_8 = "baccarat_8"

    @property
    def deck_count(self) -> int:
        return 8

    @property
    def total_cards(self) -> int:
        return self.deck_count * 52  # no jokers in either game

    def expected_multiset(self) -> dict[Card, int]:
        """How many copies of each card should be present in the shoe."""
        return {card: self.deck_count for card in standard_deck()}


@dataclass(slots=True)
class LearnedCard:
    """One card's learned template data, stored on disk per deck profile."""

    card: Card
    rank_template: "np.ndarray"
    suit_template: "np.ndarray"
    # Mean intensity of the rank glyph; used as a quick sanity gate.
    rank_mean: float = 0.0
    suit_mean: float = 0.0


@dataclass(slots=True)
class RecognitionResult:
    """Output of the recognition engine for one card pass."""

    card: Card | None
    confidence: float
    rank_roi: "np.ndarray | None" = None
    suit_roi: "np.ndarray | None" = None
    error_reason: str | None = None
    timestamp_ms: int = 0

    @property
    def recognized(self) -> bool:
        return self.card is not None


@dataclass(slots=True)
class ScanReport:
    """End-of-scan summary: what was expected vs what we saw."""

    game: GameType
    seen: list[Card] = field(default_factory=list)
    unrecognized_count: int = 0
    started_at_ms: int = 0
    finished_at_ms: int = 0

    @property
    def total_cards_seen(self) -> int:
        return len(self.seen) + self.unrecognized_count

    def diff(self) -> tuple[dict[Card, int], dict[Card, int]]:
        """Compare to expected multiset. Returns (missing, extra)."""
        expected = self.game.expected_multiset()
        actual: dict[Card, int] = {}
        for c in self.seen:
            actual[c] = actual.get(c, 0) + 1

        missing: dict[Card, int] = {}
        extra: dict[Card, int] = {}
        all_cards = set(expected) | set(actual)
        for card in all_cards:
            e = expected.get(card, 0)
            a = actual.get(card, 0)
            if a < e:
                missing[card] = e - a
            elif a > e:
                extra[card] = a - e
        return missing, extra

    @property
    def is_valid(self) -> bool:
        missing, extra = self.diff()
        return (
            not missing
            and not extra
            and self.unrecognized_count == 0
            and self.total_cards_seen == self.game.total_cards
        )
