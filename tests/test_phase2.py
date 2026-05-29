"""
Tests for vision/recognition.py, vision/learning.py, storage/library.py.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from deck_checker.core.models import Card, GameType, Rank, Suit
from deck_checker.vision.recognition import (
    CONFIDENCE_THRESHOLD, TemplateLibrary,
    _best_rank, _best_suit, _combined_confidence,
    _match_one, recognise_batch, recognise_card,
)
from deck_checker.vision.learning import (
    LearningError, calibrate_exposure, learn_card, run_learning_pass,
)
from deck_checker.storage.library import (
    list_profiles, load_library, profile_summary, save_library,
)
from deck_checker.vision.roi import binarise


def _rank_tmpl(rank):
    img = np.full((60, 45), 255, dtype=np.uint8)
    ordinal = list(Rank).index(rank)
    for i in range(ordinal + 1):
        x = 4 + (i % 5) * 7
        y = 4 + (i // 5) * 12
        img[y:y+8, x:x+6] = 0
    return img

def _suit_tmpl(suit):
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

def _make_lib():
    lib = TemplateLibrary()
    for r in Rank:
        if r != Rank.JOKER:
            lib.rank_templates[r] = binarise(_rank_tmpl(r))
    for s in Suit:
        lib.suit_templates[s] = binarise(_suit_tmpl(s))
    return lib

def _card_img(card, noise=0):
    img = np.full((350, 250), 200, dtype=np.uint8)
    img[0:57, 0:45] = _rank_tmpl(card.rank)[:57, :]
    img[57:97, 0:40] = _suit_tmpl(card.suit)
    if noise:
        layer = np.random.randint(-noise, noise+1, img.shape, dtype=np.int16)
        img = np.clip(img.astype(np.int16)+layer, 0, 255).astype(np.uint8)
    return img


class TestMatchOne:
    def test_identical_images_return_one(self):
        img = np.full((60, 45), 128, dtype=np.uint8)
        assert _match_one(img, img) == pytest.approx(1.0, abs=0.01)
    def test_tiny_query_returns_zero(self):
        assert _match_one(np.full((3,3),128,dtype=np.uint8), np.full((60,45),128,dtype=np.uint8)) == 0.0
    def test_tiny_template_returns_zero(self):
        assert _match_one(np.full((60,45),128,dtype=np.uint8), np.full((3,3),128,dtype=np.uint8)) == 0.0
    def test_template_resized(self):
        q = _rank_tmpl(Rank.ACE)
        assert _match_one(q, cv2.resize(q,(30,20))) > 0.7

class TestBestRankSuit:
    def setup_method(self):
        self.lib = _make_lib()
    def test_ace_found(self):
        r, s = _best_rank(binarise(_rank_tmpl(Rank.ACE)), self.lib.rank_templates)
        assert r == Rank.ACE and s > 0.9
    def test_spades_found(self):
        s, sc = _best_suit(binarise(_suit_tmpl(Suit.SPADES)), self.lib.suit_templates)
        assert s == Suit.SPADES and sc > 0.9
    @pytest.mark.parametrize("rank", [r for r in Rank if r != Rank.JOKER])
    def test_all_ranks(self, rank):
        found, _ = _best_rank(binarise(_rank_tmpl(rank)), self.lib.rank_templates)
        assert found == rank
    @pytest.mark.parametrize("suit", list(Suit))
    def test_all_suits(self, suit):
        found, _ = _best_suit(binarise(_suit_tmpl(suit)), self.lib.suit_templates)
        assert found == suit
    def test_empty_rank_raises(self):
        with pytest.raises(ValueError): _best_rank(_rank_tmpl(Rank.ACE), {})
    def test_empty_suit_raises(self):
        with pytest.raises(ValueError): _best_suit(_suit_tmpl(Suit.SPADES), {})

class TestCombinedConfidence:
    def test_both_one(self):
        assert _combined_confidence(1.0, 1.0) == pytest.approx(1.0)
    def test_one_zero(self):
        assert _combined_confidence(1.0, 0.0) == pytest.approx(0.0)
    def test_negative_clamped(self):
        assert _combined_confidence(-0.5, 0.9) == pytest.approx(0.0)

class TestRecogniseCard:
    def setup_method(self):
        self.lib = _make_lib()
    def test_empty_lib(self):
        r = recognise_card(_card_img(Card(Rank.ACE, Suit.SPADES)), TemplateLibrary())
        assert r.card is None
    @pytest.mark.parametrize("card", [
        Card(Rank.ACE, Suit.SPADES), Card(Rank.KING, Suit.HEARTS),
        Card(Rank.QUEEN, Suit.DIAMONDS), Card(Rank.TWO, Suit.CLUBS),
        Card(Rank.TEN, Suit.SPADES), Card(Rank.JACK, Suit.HEARTS),
    ])
    def test_recognise(self, card):
        r = recognise_card(_card_img(card), self.lib)
        assert r.card == card
        assert r.confidence >= CONFIDENCE_THRESHOLD

class TestRecogniseBatch:
    def test_empty(self):
        assert recognise_batch([], _make_lib()) == []
    def test_batch(self):
        lib = _make_lib()
        cards = [Card(Rank.ACE, Suit.SPADES), Card(Rank.KING, Suit.HEARTS)]
        results = recognise_batch([_card_img(c) for c in cards], lib)
        for e, r in zip(cards, results):
            assert r.card == e

class TestLearning:
    def test_learn_card_fails_gracefully(self):
        with pytest.raises(LearningError):
            learn_card(Card(Rank.ACE, Suit.SPADES),
                       lambda: np.zeros((480,640,3),dtype=np.uint8), max_retries=1)
    def test_empty_learning_pass(self):
        lib = run_learning_pass(lambda: np.zeros((480,640,3),dtype=np.uint8),
                                calibrate=False, cards_to_learn=[])
        assert lib.rank_count() == 0

class TestLibraryPersistence:
    def _min_lib(self):
        lib = TemplateLibrary()
        lib.rank_templates[Rank.ACE] = binarise(_rank_tmpl(Rank.ACE))
        lib.suit_templates[Suit.SPADES] = binarise(_suit_tmpl(Suit.SPADES))
        return lib

    def test_save_load_roundtrip(self, tmp_path):
        lib = _make_lib()
        save_library(lib, tmp_path/"p", game_type=GameType.BLACKJACK, num_decks=8)
        loaded, meta = load_library(tmp_path/"p")
        assert loaded.rank_count() == lib.rank_count()
        assert meta["game_type"] == "blackjack"

    def test_overwrite_false_raises(self, tmp_path):
        lib = self._min_lib()
        save_library(lib, tmp_path/"p")
        with pytest.raises(FileExistsError):
            save_library(lib, tmp_path/"p", overwrite=False)

    def test_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_library(tmp_path/"ghost")

    def test_list_profiles(self, tmp_path):
        lib = self._min_lib()
        for n in ["a","b","c"]:
            save_library(lib, tmp_path/n)
        assert len(list_profiles(tmp_path)) == 3

    def test_summary(self, tmp_path):
        lib = self._min_lib()
        save_library(lib, tmp_path/"p", game_type=GameType.BACCARAT)
        s = profile_summary(tmp_path/"p")
        assert s["game_type"] == "baccarat"

    def test_summary_missing(self, tmp_path):
        assert profile_summary(tmp_path/"ghost") is None
