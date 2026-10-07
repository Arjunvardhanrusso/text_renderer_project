import cv2
import json
import pandas as pd
from pathlib import Path

from parse_annotations import load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

ORIGINAL_DIR = OUTPUT_DIR / "original"
MASKED_DIR = OUTPUT_DIR / "masked"

ORIGINAL_DIR.mkdir(parents=True, exist_ok=True)
MASKED_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 128
PADDING = 20


def process_sample(image, annotation, sample_id):
    points = annotation["points"]
    text = annotation["text"]

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    x_min = max(min(xs) - PADDING, 0)
    y_min = max(min(ys) - PADDING, 0)

    x_max = min(max(xs) + PADDING, image.shape[1])
    y_max = min(max(ys) + PADDING, image.shape[0])

    if x_max <= x_min or y_max <= y_min:
        return None

    crop = image[y_min:y_max, x_min:x_max].copy()

    if crop.size == 0:
        return None

    # Convert original points to coordinates inside the crop
    local_points = [
        (x - x_min, y - y_min)
        for x, y in points
    ]

    masked_crop = crop.copy()

    # Hide the text region
    polygon = [
        [x - x_min, y - y_min]
        for x, y in points
    ]

    import numpy as np

    polygon = np.array(polygon, dtype=np.int32)

    cv2.fillPoly(
        masked_crop,
        [polygon],
        color=(0, 0, 0)
    )

    # Resize
    original_resized = cv2.resize(
        crop,
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    masked_resized = cv2.resize(
        masked_crop,
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    original_filename = f"{sample_id}.jpg"
    masked_filename = f"{sample_id}.jpg"

    cv2.imwrite(
        str(ORIGINAL_DIR / original_filename),
        original_resized
    )

    cv2.imwrite(
        str(MASKED_DIR / masked_filename),
        masked_resized
    )

    # Scale coordinates to 128x128
    scale_x = IMAGE_SIZE / crop.shape[1]
    scale_y = IMAGE_SIZE / crop.shape[0]

    resized_points = [
        [
            int(x * scale_x),
            int(y * scale_y)
        ]
        for x, y in local_points
    ]

    return {
        "sample_id": sample_id,
        "text": text,
        "original_image": str(
            ORIGINAL_DIR / original_filename
        ),
        "masked_image": str(
            MASKED_DIR / masked_filename
        ),
        "points": json.dumps(resized_points)
    }


def main():
    dataset = load_dataset()

    metadata = []

    sample_number = 0

    for item in dataset:

        image = cv2.imread(item["image_path"])

        if image is None:
            continue

        for annotation in item["annotations"]:

            sample_id = f"sample_{sample_number:05d}"

            result = process_sample(
                image,
                annotation,
                sample_id
            )

            if result is not None:
                result["source_image"] = Path(item["image_path"]).name

                metadata.append(result)
                sample_number += 1

    metadata_df = pd.DataFrame(metadata)

    metadata_path = OUTPUT_DIR / "metadata.csv"

    metadata_df.to_csv(
        metadata_path,
        index=False
    )

    print("=" * 50)
    print("PREPROCESSING COMPLETE")
    print("=" * 50)

    print(f"Samples created: {len(metadata)}")
    print(f"Original images: {ORIGINAL_DIR}")
    print(f"Masked images:   {MASKED_DIR}")
    print(f"Metadata:        {metadata_path}")


if __name__ == "__main__":
    main()