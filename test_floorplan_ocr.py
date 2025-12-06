# tests/test_floorplan_ocr.py
"""
Test script to verify your full OCR pipeline (pytesseract + EasyOCR)
on your actual floorplan: data/raw/floorplan6.png

Run with:
    python tests/test_floorplan_ocr.py
"""

from pathlib import Path
from PIL import Image, ImageDraw
import json
import os

from src.ocr_labels import run_ocr_combined

IMAGE_PATH = Path("data/raw/floorplan6.png")
OUT_DIR = Path("tests/ocr_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def draw_debug(image_path, labels, out_path):
    """Draw bounding boxes + label text for debugging."""
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    for lab in labels:
        x, y, w, h = lab["bbox"]
        cx, cy = lab["centroid"]
        text = f"{lab['text']} ({lab['confidence']:.0f})"

        draw.rectangle([x, y, x+w, y+h], outline=(0,255,0), width=2)
        draw.ellipse([(cx-3, cy-3), (cx+3, cy+3)], outline=(255,0,0), width=2)
        draw.text((x, max(0, y-14)), text, fill=(255,0,0))

    img.save(out_path)
    print("Debug image saved:", out_path)


def validate_labels(labels):
    """Basic structural validation for OCR outputs."""
    assert isinstance(labels, list)
    for l in labels:
        assert isinstance(l["text"], str)
        assert isinstance(l["confidence"], (int, float))
        assert len(l["bbox"]) == 4
        assert len(l["centroid"]) == 2
        assert 0 <= l["confidence"] <= 100
        # bbox sanity check
        x,y,w,h = l["bbox"]
        assert w >= 0 and h >= 0


def main():
    print("=== Testing OCR on:", IMAGE_PATH, "===")

    if not IMAGE_PATH.exists():
        raise FileNotFoundError(f"Image not found at {IMAGE_PATH}")

    # Run full OCR pipeline (Tesseract + EasyOCR)
    labels = run_ocr_combined(
        str(IMAGE_PATH),
        use_easyocr=True,   # set False if you want tesseract-only
        tesseract_min_conf=10
    )

    print(f"\nFound {len(labels)} labels:\n")
    for lab in labels:
        print(lab)

    # Validate fields
    validate_labels(labels)

    # Save JSON
    json_path = OUT_DIR / "floorplan6_ocr.json"
    with open(json_path, "w") as f:
        json.dump(labels, f, indent=2)
    print("\nSaved JSON:", json_path)

    # Save debug visualization
    debug_path = OUT_DIR / "floorplan6_ocr_debug.png"
    draw_debug(IMAGE_PATH, labels, debug_path)

    print("\nOCR test COMPLETE.\nResults saved in", OUT_DIR)


if __name__ == "__main__":
    main()
