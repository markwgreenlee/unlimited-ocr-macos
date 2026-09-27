#!/usr/bin/env python3
"""Generate a small synthetic text page for smoke-testing the install."""
from PIL import Image, ImageDraw, ImageFont
import pathlib

LINES = [
    (44, "Unlimited-OCR Install Check"),
    (30, "This page exercises transcription, layout typing and numerals."),
    (30, "Measured threshold: 0.42 units/s (SD 0.08, N = 24)."),
    (26, "Table 1: latency by condition"),
    (26, "control 2.1 s    condition A 1.4 s    condition B 1.9 s"),
]
CANDIDATE_FONTS = [
    "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
]


def load_font(size):
    for path in CANDIDATE_FONTS:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def main():
    img = Image.new("RGB", (1000, 320), "white")
    draw = ImageDraw.Draw(img)
    y = 30
    for size, text in LINES:
        draw.text((40, y), text, font=load_font(size), fill="black")
        y += size + 22
    out = pathlib.Path(__file__).resolve().parent / "synthetic_page.png"
    img.save(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
