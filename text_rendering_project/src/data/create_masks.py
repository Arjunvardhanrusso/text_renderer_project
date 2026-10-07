import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
METADATA_PATH = PROCESSED_DIR / "metadata.csv"

LAYOUT_DIR = PROCESSED_DIR / "layout_masks"
GLYPH_DIR = PROCESSED_DIR / "glyph_masks"

LAYOUT_DIR.mkdir(parents=True, exist_ok=True)
GLYPH_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 128


def create_layout_mask(points):
    """
    Creates a binary mask showing WHERE the text should appear.

    Black = background
    White = text region
    """

    mask = np.zeros(
        (IMAGE_SIZE, IMAGE_SIZE),
        dtype=np.uint8
    )

    polygon = np.array(points, dtype=np.int32)

    cv2.fillPoly(
        mask,
        [polygon],
        color=255
    )

    return mask


def get_font(font_size):
    """
    Try to use DejaVuSans.
    Fall back to Pillow's default font if unavailable.
    """

    try:
        return ImageFont.truetype(
            "DejaVuSans.ttf",
            font_size
        )
    except OSError:
        return ImageFont.load_default()


def find_best_font(text, box_width, box_height):
    """
    Find the largest font that fits inside the
    annotated text bounding box.
    """

    best_font = get_font(10)

    max_size = max(10, box_height * 2)

    for size in range(8, max_size + 1):
        font = get_font(size)

        temp = Image.new("L", (IMAGE_SIZE, IMAGE_SIZE))
        draw = ImageDraw.Draw(temp)

        bbox = draw.textbbox(
            (0, 0),
            text,
            font=font
        )

        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        if (
            text_width <= box_width
            and text_height <= box_height
        ):
            best_font = font
        else:
            break

    return best_font


def create_glyph_mask(text, points):
    """
    Creates a mask showing WHAT text should appear.
    """

    image = Image.new(
        "L",
        (IMAGE_SIZE, IMAGE_SIZE),
        color=0
    )

    draw = ImageDraw.Draw(image)

    xs = [point[0] for point in points]
    ys = [point[1] for point in points]

    x_min = max(min(xs), 0)
    x_max = min(max(xs), IMAGE_SIZE - 1)

    y_min = max(min(ys), 0)
    y_max = min(max(ys), IMAGE_SIZE - 1)

    box_width = max(x_max - x_min, 1)
    box_height = max(y_max - y_min, 1)

    font = find_best_font(
        text,
        box_width,
        box_height
    )

    text_bbox = draw.textbbox(
        (0, 0),
        text,
        font=font
    )

    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    # Center text inside the ICDAR bounding box
    x = x_min + (box_width - text_width) // 2
    y = y_min + (box_height - text_height) // 2

    # Correct Pillow's font offset
    x -= text_bbox[0]
    y -= text_bbox[1]

    draw.text(
        (x, y),
        text,
        fill=255,
        font=font
    )

    return np.array(image)


def main():

    metadata = pd.read_csv(METADATA_PATH)

    print(f"Samples found: {len(metadata)}")

    for index, row in metadata.iterrows():

        sample_id = row["sample_id"]
        text = str(row["text"])

        points = json.loads(row["points"])

        # -------------------------
        # Layout mask
        # -------------------------

        layout_mask = create_layout_mask(points)

        layout_path = (
            LAYOUT_DIR
            / f"{sample_id}.png"
        )

        cv2.imwrite(
            str(layout_path),
            layout_mask
        )

        # -------------------------
        # Glyph mask
        # -------------------------

        glyph_mask = create_glyph_mask(
            text,
            points
        )

        glyph_path = (
            GLYPH_DIR
            / f"{sample_id}.png"
        )

        cv2.imwrite(
            str(glyph_path),
            glyph_mask
        )

        if (index + 1) % 500 == 0:
            print(
                f"Processed {index + 1}/{len(metadata)}"
            )

    print("=" * 50)
    print("MASK GENERATION COMPLETE")
    print("=" * 50)

    print(f"Layout masks: {LAYOUT_DIR}")
    print(f"Glyph masks:  {GLYPH_DIR}")


if __name__ == "__main__":
    main()