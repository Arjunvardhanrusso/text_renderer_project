import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
METADATA_PATH = PROCESSED_DIR / "metadata.csv"

OUTPUT_DIR = PROCESSED_DIR / "glyph_masks_warped"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 128


def get_font(size):
    try:
        return ImageFont.truetype(
            "DejaVuSans.ttf",
            size
        )
    except OSError:
        return ImageFont.load_default()


def find_best_font(text, width, height):

    best_font = get_font(8)

    for size in range(8, max(height * 2, 10)):

        font = get_font(size)

        temp = Image.new(
            "L",
            (max(width, 1), max(height, 1)),
            0
        )

        draw = ImageDraw.Draw(temp)

        bbox = draw.textbbox(
            (0, 0),
            text,
            font=font
        )

        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        if (
            text_width <= width * 0.9
            and text_height <= height * 0.85
        ):
            best_font = font
        else:
            break

    return best_font


def distance(p1, p2):
    return np.linalg.norm(
        np.array(p1, dtype=np.float32)
        - np.array(p2, dtype=np.float32)
    )


def create_warped_glyph(text, points):

    points = np.array(
        points,
        dtype=np.float32
    )

    # ICDAR points:
    # top-left, top-right,
    # bottom-right, bottom-left

    width_top = distance(
        points[0],
        points[1]
    )

    width_bottom = distance(
        points[3],
        points[2]
    )

    height_left = distance(
        points[0],
        points[3]
    )

    height_right = distance(
        points[1],
        points[2]
    )

    width = max(
        int(max(width_top, width_bottom)),
        8
    )

    height = max(
        int(max(height_left, height_right)),
        8
    )

    # ---------------------------------
    # Render text on flat rectangle
    # ---------------------------------

    glyph_image = Image.new(
        "L",
        (width, height),
        color=0
    )

    draw = ImageDraw.Draw(glyph_image)

    font = find_best_font(
        text,
        width,
        height
    )

    bbox = draw.textbbox(
        (0, 0),
        text,
        font=font
    )

    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    x = (width - text_width) // 2
    y = (height - text_height) // 2

    x -= bbox[0]
    y -= bbox[1]

    draw.text(
        (x, y),
        text,
        font=font,
        fill=255
    )

    glyph_array = np.array(
        glyph_image,
        dtype=np.uint8
    )

    # ---------------------------------
    # Perspective transform
    # ---------------------------------

    source_points = np.array(
        [
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ],
        dtype=np.float32
    )

    transform = cv2.getPerspectiveTransform(
        source_points,
        points
    )

    warped = cv2.warpPerspective(
        glyph_array,
        transform,
        (IMAGE_SIZE, IMAGE_SIZE),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    return warped


def main():

    metadata = pd.read_csv(
        METADATA_PATH
    )

    print("=" * 60)
    print("WARPED GLYPH MASK GENERATION")
    print("=" * 60)

    print(
        f"Samples found: {len(metadata)}"
    )

    for index, row in metadata.iterrows():

        sample_id = row["sample_id"]
        text = str(row["text"])

        points = json.loads(
            row["points"]
        )

        glyph = create_warped_glyph(
            text,
            points
        )

        output_path = (
            OUTPUT_DIR
            / f"{sample_id}.png"
        )

        cv2.imwrite(
            str(output_path),
            glyph
        )

        if (index + 1) % 500 == 0:
            print(
                f"Processed "
                f"{index + 1}/{len(metadata)}"
            )

    print("\n" + "=" * 60)
    print("WARPED GLYPH GENERATION COMPLETE")
    print("=" * 60)

    print(
        f"Output: {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()