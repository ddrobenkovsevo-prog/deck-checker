"""Tests for core domain models — no CV or hardware dependencies."""

from __future__ import annotations

import pytest

from deck_checker.core import Card, GameType, Rank, ScanReport, Suit, standard_deck


class TestCard:
    def test_normal_card_code(self) -> None:
        assert Card(Rank.ACE, Suit.SPADES).code == "AS"
        assert Card(Rank.TEN, Suit.HEARTS).code == "TH"

    def test_joker(self) -> None:
        joker = Card(Rank.JOKER)
        assert joker.code == "JK"
        assert joker.suit is None

    def test_joker_with_suit_rejected(self) -> None:
        with pytest.raises(ValueError):
            Card(Rank.JOKER, Suit.SPADES)

    def test_non_joker_without_suit_rejected(self) -> None:
        with pytest.raises(ValueError):
            Card(Rank.ACE)  # missing suit

    @pytest.mark.parametrize(
        "code,expected",
        [
            ("AS", Card(Rank.ACE, Suit.SPADES)),
            ("kh", Card(Rank.KING, Suit.HEARTS)),
            ("TD", Card(Rank.TEN, Suit.DIAMONDS)),
            ("JK", Card(Rank.JOKER)),
        ],
    )
    def test_from_code_roundtrip(self, code: str, expected: Card) -> None:
        assert Card.from_code(code) == expected
        assert Card.from_code(expected.code) == expected

    @pytest.mark.parametrize("bad", ["", "A", "ABC", "ZZ", "AX"])
    def test_from_code_rejects_garbage(self, bad: str) -> None:
        with pytest.raises(ValueError):
            Card.from_code(bad)

    def test_red_suits(self) -> None:
        assert Suit.HEARTS.is_red
        assert Suit.DIAMONDS.is_red
        assert not Suit.SPADES.is_red
        assert not Suit.CLUBS.is_red

    def test_cards_hashable_and_unique(self) -> None:
        deck = set(standard_deck())
        assert len(deck) == 52


class TestGameType:
    def test_blackjack_8_expects_8_of_each(self) -> None:
        expected = GameType.BLACKJACK_8.expected_multiset()
        assert len(expected) == 52
        assert all(count == 8 for count in expected.values())
        assert sum(expected.values()) == 416

    def test_total_cards(self) -> None:
        assert GameType.BLACKJACK_8.total_cards == 416
        assert GameType.BACCARAT_8.total_cards == 416


class TestScanReport:
    def test_perfect_scan(self) -> None:
        report = ScanReport(game=GameType.BLACKJACK_8)
        for card, n in GameType.BLACKJACK_8.expected_multiset().items():
            report.seen.extend([card] * n)

        missing, extra = report.diff()
        assert missing == {}
        assert extra == {}
        assert report.is_valid

    def test_missing_card_detected(self) -> None:
        report = ScanReport(game=GameType.BLACKJACK_8)
        cards = []
        for card, n in GameType.BLACKJACK_8.expected_multiset().items():
            cards.extend([card] * n)
        cards.pop()  # drop one card

        report.seen = cards
        missing, extra = report.diff()
        assert sum(missing.values()) == 1
        assert extra == {}
        assert not report.is_valid

    def test_extra_card_detected(self) -> None:
        report = ScanReport(game=GameType.BLACKJACK_8)
        cards = []
        for card, n in GameType.BLACKJACK_8.expected_multiset().items():
            cards.extend([card] * n)
        cards.append(Card(Rank.ACE, Suit.SPADES))  # one extra ace

        report.seen = cards
        missing, extra = report.diff()
        assert missing == {}
        assert extra == {Card(Rank.ACE, Suit.SPADES): 1}
        assert not report.is_valid
