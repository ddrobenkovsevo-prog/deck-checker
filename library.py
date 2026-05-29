"""
Deck profile persistence.

A "deck profile" is a directory containing:

    profile.json          — metadata (game type, creation time, exposure, etc.)
    templates/
        ranks/
            A.png  2.png  ... K.png
        suits/
            S.png  H.png  D.png  C.png

The JSON and PNG files are the source of truth; the TemplateLibrary is an
in-memory view that the recogniser uses at runtime.

Layout is intentionally simple so casino IT can inspect / backup profiles
with standard filesystem tools, satisfying GLI audit requirements.
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


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_library(
    library: TemplateLibrary,
    profile_dir: Path,
    *,
    game_type: GameType = GameType.BLACKJACK,
    num_decks: int = 8,
    exposure_value: float = 0.5,
    overwrite: bool = False,
) -> Path:
    """
    Persist a TemplateLibrary to *profile_dir*.

    Parameters
    ----------
    library:       Populated TemplateLibrary from the learning pass.
    profile_dir:   Destination directory (created if absent).
    game_type:     Stored in metadata only.
    num_decks:     Stored in metadata only.
    exposure_value: Calibrated exposure from Pass 1.
    overwrite:     If False and profile_dir already contains profile.json,
                   raise FileExistsError.

    Returns
    -------
    Absolute path to the saved profile.json.
    """
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

    # Write rank PNGs
    rank_files: dict[str, str] = {}
    for rank, template in library.rank_templates.items():
        filename = f"{rank.value}.png"
        path = ranks_dir / filename
        cv2.imwrite(str(path), template)
        rank_files[rank.value] = str(path.relative_to(profile_dir))
        logger.debug("Saved rank template: %s", path)

    # Write suit PNGs
    suit_files: dict[str, str] = {}
    for suit, template in library.suit_templates.items():
        filename = f"{suit.value}.png"
        path = suits_dir / filename
        cv2.imwrite(str(path), template)
        suit_files[suit.value] = str(path.relative_to(profile_dir))
        logger.debug("Saved suit template: %s", path)

    # Write metadata JSON
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


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_library(profile_dir: Path) -> tuple[TemplateLibrary, dict]:
    """
    Load a TemplateLibrary from *profile_dir*.

    Returns
    -------
    (library, metadata_dict)

    Raises
    ------
    FileNotFoundError  if profile.json or any referenced template is missing.
    ValueError         if profile.json is malformed.
    """
    profile_dir = Path(profile_dir).resolve()
    profile_json = profile_dir / PROFILE_FILENAME

    if not profile_json.exists():
        raise FileNotFoundError(f"No profile found at {profile_json}")

    metadata = json.loads(profile_json.read_text(encoding="utf-8"))
    _validate_metadata(metadata)

    library = TemplateLibrary()

    # Load rank templates
    for rank_value, rel_path in metadata["rank_templates"].items():
        path = profile_dir / rel_path
        if not path.exists():
            raise FileNotFoundError(f"Rank template missing: {path}")
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not read rank template: {path}")
        rank = Rank(rank_value)
        library.rank_templates[rank] = img
        logger.debug("Loaded rank template: %s (%s)", rank, path)

    # Load suit templates
    for suit_value, rel_path in metadata["suit_templates"].items():
        path = profile_dir / rel_path
        if not path.exists():
            raise FileNotFoundError(f"Suit template missing: {path}")
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not read suit template: {path}")
        suit = Suit(suit_value)
        library.suit_templates[suit] = img
        logger.debug("Loaded suit template: %s (%s)", suit, path)

    logger.info(
        "Loaded deck profile from %s: %d ranks, %d suits",
        profile_dir, library.rank_count(), library.suit_count(),
    )
    return library, metadata


def _validate_metadata(metadata: dict) -> None:
    required_keys = {"version", "game_type", "num_decks", "rank_templates", "suit_templates"}
    missing = required_keys - metadata.keys()
    if missing:
        raise ValueError(f"profile.json missing keys: {missing}")
    if metadata.get("version") != 1:
        raise ValueError(f"Unsupported profile version: {metadata.get('version')}")


# ---------------------------------------------------------------------------
# Convenience: list available profiles
# ---------------------------------------------------------------------------

def list_profiles(base_dir: Path) -> list[Path]:
    """
    Return a sorted list of directories under *base_dir* that contain a
    valid profile.json.
    """
    base_dir = Path(base_dir)
    profiles = sorted(
        p.parent
        for p in base_dir.rglob(PROFILE_FILENAME)
    )
    return profiles


def profile_summary(profile_dir: Path) -> Optional[dict]:
    """
    Return lightweight metadata dict for display in the UI, or None
    if the profile is unreadable.
    """
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
