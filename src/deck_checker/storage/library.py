"""
Deck profile persistence.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from deck_checker.core.models import GameType, Rank, Suit
from deck_checker.vision.recognition import TemplateLibrary

logger = logging.getLogger(__name__)

PROFILE_FILENAME = "profile.json"
RANKS_DIR = "templates/ranks"
SUITS_DIR = "templates/suits"


def save_library(
    library: TemplateLibrary,
    profile_dir: Path,
    *,
    game_type: GameType = GameType.BLACKJACK,
    num_decks: int = 8,
    exposure_value: float = 0.5,
    overwrite: bool = False,
) -> Path:
    profile_dir = Path(profile_dir).resolve()
    profile_json = profile_dir / PROFILE_FILENAME
    if profile_json.exists() and not overwrite:
        raise FileExistsError(
            f"Profile already exists at {profile_json}. "
            "Pass overwrite=True to replace it."
        )
    ranks_dir = profile_dir / RANKS_DIR
    suits_dir = profile_dir / SUITS_DIR
    ranks_dir.mkdir(parents=True, exist_ok=True)
    suits_dir.mkdir(parents=True, exist_ok=True)
    rank_files: dict[str, str] = {}
    for rank, template in library.rank_templates.items():
        filename = f"{rank.value}.png"
        path = ranks_dir / filename
        cv2.imwrite(str(path), template)
        rank_files[rank.value] = str(path.relative_to(profile_dir))
    suit_files: dict[str, str] = {}
    for suit, template in library.suit_templates.items():
        filename = f"{suit.value}.png"
        path = suits_dir / filename
        cv2.imwrite(str(path), template)
        suit_files[suit.value] = str(path.relative_to(profile_dir))
    metadata = {
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "game_type": game_type.value,
        "num_decks": num_decks,
        "exposure_value": exposure_value,
        "rank_templates": rank_files,
        "suit_templates": suit_files,
    }
    profile_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info("Saved deck profile to %s", profile_json)
    return profile_json


def load_library(profile_dir: Path) -> tuple[TemplateLibrary, dict]:
    profile_dir = Path(profile_dir).resolve()
    profile_json = profile_dir / PROFILE_FILENAME
    if not profile_json.exists():
        raise FileNotFoundError(f"No profile found at {profile_json}")
    metadata = json.loads(profile_json.read_text(encoding="utf-8"))
    _validate_metadata(metadata)
    library = TemplateLibrary()
    for rank_value, rel_path in metadata["rank_templates"].items():
        path = profile_dir / rel_path
        if not path.exists():
            raise FileNotFoundError(f"Rank template missing: {path}")
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not read rank template: {path}")
        library.rank_templates[Rank(rank_value)] = img
    for suit_value, rel_path in metadata["suit_templates"].items():
        path = profile_dir / rel_path
        if not path.exists():
            raise FileNotFoundError(f"Suit template missing: {path}")
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not read suit template: {path}")
        library.suit_templates[Suit(suit_value)] = img
    logger.info("Loaded deck profile from %s: %d ranks, %d suits",
                profile_dir, library.rank_count(), library.suit_count())
    return library, metadata


def _validate_metadata(metadata: dict) -> None:
    required_keys = {"version", "game_type", "num_decks", "rank_templates", "suit_templates"}
    missing = required_keys - metadata.keys()
    if missing:
        raise ValueError(f"profile.json missing keys: {missing}")
    if metadata.get("version") != 1:
        raise ValueError(f"Unsupported profile version: {metadata.get('version')}")


def list_profiles(base_dir: Path) -> list[Path]:
    base_dir = Path(base_dir)
    return sorted(p.parent for p in base_dir.rglob(PROFILE_FILENAME))


def profile_summary(profile_dir: Path) -> Optional[dict]:
    try:
        profile_json = Path(profile_dir) / PROFILE_FILENAME
        metadata = json.loads(profile_json.read_text(encoding="utf-8"))
        return {
            "path": str(profile_dir),
            "game_type": metadata.get("game_type"),
            "num_decks": metadata.get("num_decks"),
            "created_at": metadata.get("created_at"),
            "rank_count": len(metadata.get("rank_templates", {})),
            "suit_count": len(metadata.get("suit_templates", {})),
        }
    except Exception as exc:
        logger.warning("Could not read profile at %s: %s", profile_dir, exc)
        return None
