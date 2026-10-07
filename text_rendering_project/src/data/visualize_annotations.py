import cv2
import numpy as np
from pathlib import Path

from parse_annotations import load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "annotation_checks"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def draw_annotations(image_path, annotations):
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    for annotation in annotations:
        text = annotation["text"]
        points = np.array(annotation["points"], dtype=np.int32)

        # Draw bounding box
        cv2.polylines(
            image,
            [points],
            isClosed=True,
            color=(0, 255, 0),
            thickness=2,
        )

        # Text label position
        x, y = points[0]

        cv2.putText(
            image,
            text,
            (x, max(y - 5, 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

    return image


def main():
    dataset = load_dataset()

    print(f"Dataset images: {len(dataset)}")

    # Check first 5 images
    for index, sample in enumerate(dataset[:5], start=1):

        image = draw_annotations(
            sample["image_path"],
            sample["annotations"],
        )

        output_path = OUTPUT_DIR / f"check_{index}.jpg"

        cv2.imwrite(str(output_path), image)

        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()