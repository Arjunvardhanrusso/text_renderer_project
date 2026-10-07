from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

IMAGE_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "icdar2015"
    / "ch4_training_images"
)

ANNOTATION_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "icdar2015"
    / "ch4_training_localization_transcription_gt"
)


def parse_annotation_file(annotation_path):
    """
    Read one ICDAR 2015 annotation file.

    Each line looks like:
    x1,y1,x2,y2,x3,y3,x4,y4,text
    """

    annotations = []

    with open(annotation_path, "r", encoding="utf-8-sig") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            # Split only first 8 commas.
            # This prevents problems if transcription contains commas.
            parts = line.split(",", 8)

            if len(parts) < 9:
                continue

            coordinates = list(map(int, parts[:8]))
            text = parts[8].strip()

            # ICDAR uses ### for unreadable text
            if text == "###":
                continue

            points = [
                (coordinates[0], coordinates[1]),
                (coordinates[2], coordinates[3]),
                (coordinates[4], coordinates[5]),
                (coordinates[6], coordinates[7]),
            ]

            annotations.append(
                {
                    "text": text,
                    "points": points,
                }
            )

    return annotations


def load_dataset():
    dataset = []

    for annotation_file in sorted(ANNOTATION_DIR.glob("gt_img_*.txt")):

        # gt_img_1.txt -> img_1.jpg
        image_name = annotation_file.name.replace("gt_", "").replace(
            ".txt", ".jpg"
        )

        image_path = IMAGE_DIR / image_name

        if not image_path.exists():
            print(f"Image not found: {image_path}")
            continue

        annotations = parse_annotation_file(annotation_file)

        dataset.append(
            {
                "image_path": str(image_path),
                "annotations": annotations,
            }
        )

    return dataset


if __name__ == "__main__":

    dataset = load_dataset()

    print("=" * 50)
    print("ICDAR 2015 DATASET CHECK")
    print("=" * 50)

    print(f"Images loaded: {len(dataset)}")

    total_words = sum(
        len(item["annotations"])
        for item in dataset
    )

    print(f"Readable text instances: {total_words}")

    if dataset:
        sample = dataset[0]

        print("\nSample image:")
        print(sample["image_path"])

        print("\nAnnotations:")

        for annotation in sample["annotations"][:10]:
            print(
                f"Text: {annotation['text']} | "
                f"Points: {annotation['points']}"
            )