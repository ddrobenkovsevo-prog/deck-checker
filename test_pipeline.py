"""Run the CV preprocessing pipeline on a single card image.

Saves intermediate artifacts to a debug directory so you can eyeball each step.

Usage:
    python scripts/test_pipeline.py path/to/card.jpg
    python scripts/test_pipeline.py path/to/card.jpg --out /tmp/debug
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

from deck_checker.vision import preprocessing, roi


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Path to an image of a single card")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("debug_output"),
        help="Directory for intermediate artifacts (default: ./debug_output)",
    )
    args = parser.parse_args()

    if not args.image.exists():
        print(f"error: {args.image} does not exist", file=sys.stderr)
        return 1

    image = cv2.imread(str(args.image))
    if image is None:
        print(f"error: cannot decode {args.image}", file=sys.stderr)
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"input: {args.image} ({image.shape[1]}×{image.shape[0]})")
    print(f"output: {args.out}/")

    # Stage 1: card detection
    corners = preprocessing.detect_card_contour(image)
    if corners is None:
        print("✗ no card detected")
        return 2

    overlay = image.copy()
    cv2.drawContours(overlay, [corners.astype(int)], -1, (0, 255, 0), 3)
    for i, pt in enumerate(preprocessing.order_corners(corners)):
        cv2.circle(overlay, tuple(pt.astype(int)), 8, (0, 0, 255), -1)
        cv2.putText(
            overlay,
            str(i),
            tuple(pt.astype(int) + (10, 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )
    cv2.imwrite(str(args.out / "01_corners.jpg"), overlay)
    print("✓ stage 1: corners detected")

    # Stage 2: perspective correction
    warped = preprocessing.perspective_correct(image, corners)
    cv2.imwrite(str(args.out / "02_warped.jpg"), warped)
    print(f"✓ stage 2: warped to {warped.shape[1]}×{warped.shape[0]}")

    # Stage 3: illumination normalization
    normalized = preprocessing.normalize_illumination(warped)
    cv2.imwrite(str(args.out / "03_normalized.jpg"), normalized)
    print("✓ stage 3: illumination normalized")

    # Stage 4: corner ROI extraction
    corner_tl = roi.extract_corner(normalized, "top-left")
    corner_br = roi.extract_corner(normalized, "bottom-right")
    cv2.imwrite(str(args.out / "04a_corner_topleft.jpg"), corner_tl)
    cv2.imwrite(str(args.out / "04b_corner_bottomright.jpg"), corner_br)
    print("✓ stage 4: both corners extracted")

    # Stage 5: rank/suit split + binarization
    rank, suit = roi.split_rank_suit(corner_tl)
    rank_bin = roi.tightly_crop(roi.to_grayscale_clean(rank))
    suit_bin = roi.tightly_crop(roi.to_grayscale_clean(suit))
    cv2.imwrite(str(args.out / "05a_rank_raw.jpg"), rank)
    cv2.imwrite(str(args.out / "05b_suit_raw.jpg"), suit)
    cv2.imwrite(str(args.out / "05c_rank_binary.jpg"), rank_bin)
    cv2.imwrite(str(args.out / "05d_suit_binary.jpg"), suit_bin)
    print(
        f"✓ stage 5: rank {rank_bin.shape[1]}×{rank_bin.shape[0]}, "
        f"suit {suit_bin.shape[1]}×{suit_bin.shape[0]}"
    )

    print(f"\ndone. open {args.out}/ to inspect all stages.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
