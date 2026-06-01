"""
main.py — Deck Checker kiosk entry point.

Usage:
    python main.py                  # fullscreen kiosk (production)
    python main.py --windowed       # windowed mode (development)
    python main.py --windowed --demo # windowed + synthetic library
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

def main() -> None:
    parser = argparse.ArgumentParser(description="Deck Checker Kiosk")
    parser.add_argument("--windowed", action="store_true",
                        help="Run in window instead of fullscreen")
    parser.add_argument("--demo", action="store_true",
                        help="Use synthetic template library (no real camera learning needed)")
    parser.add_argument("--profile", type=str, default="",
                        help="Path to saved deck profile directory")
    args = parser.parse_args()

    # Must import PyQt6 after argument parsing
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import Qt

    app = QApplication(sys.argv)
    app.setApplicationName("Deck Checker")

    # ── Load or build template library ────────────────────────────────────────
    from deck_checker.vision.recognition import TemplateLibrary

    library = TemplateLibrary()

    if args.profile:
        from deck_checker.storage.library import load_library
        profile_path = Path(args.profile)
        if profile_path.exists():
            library, meta = load_library(profile_path)
            logging.info("Loaded profile from %s: %d ranks, %d suits",
                         profile_path, library.rank_count(), library.suit_count())
        else:
            logging.warning("Profile not found: %s — using empty library", profile_path)

    elif args.demo:
        # Build synthetic library for demo/testing
        import cv2
        import numpy as np
        from deck_checker.core.models import Rank, Suit
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
                cv2.circle(img,(20,28),10,0,-1)
                cv2.circle(img,(12,18),9,0,-1)
                cv2.circle(img,(28,18),9,0,-1)
            return img

        for r in Rank:
            if r != Rank.JOKER:
                library.rank_templates[r] = binarise(_rank_tmpl(r))
        for s in Suit:
            library.suit_templates[s] = binarise(_suit_tmpl(s))
        logging.info("Demo library built: %d ranks, %d suits",
                     library.rank_count(), library.suit_count())

    # ── Launch main window ─────────────────────────────────────────────────────
    from deck_checker.ui.main_window import MainWindow

    window = MainWindow(
        library=library,
        kiosk=not args.windowed,
    )

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
