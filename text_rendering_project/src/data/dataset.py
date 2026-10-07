from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

METADATA_PATH = PROCESSED_DIR / "metadata.csv"


# --------------------------------------------------
# PyTorch Dataset
# --------------------------------------------------

class TextRenderingDataset(Dataset):

    def __init__(self, metadata_df):
        self.metadata = metadata_df.reset_index(drop=True)

        self.rgb_transform = transforms.ToTensor()
        self.mask_transform = transforms.ToTensor()

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):

        row = self.metadata.iloc[index]

        sample_id = row["sample_id"]

        original_path = (
            PROCESSED_DIR
            / "original"
            / f"{sample_id}.jpg"
        )

        masked_path = (
            PROCESSED_DIR
            / "masked"
            / f"{sample_id}.jpg"
        )

        layout_path = (
            PROCESSED_DIR
            / "layout_masks"
            / f"{sample_id}.png"
        )

        glyph_path = (
            PROCESSED_DIR
            / "glyph_masks"
            / f"{sample_id}.png"
        )

        warped_glyph_path = (
                PROCESSED_DIR
                / "glyph_masks_warped"
                / f"{sample_id}.png"
        )

        original = Image.open(original_path).convert("RGB")
        masked = Image.open(masked_path).convert("RGB")
        layout = Image.open(layout_path).convert("L")
        glyph = Image.open(glyph_path).convert("L")
        warped_glyph = Image.open(warped_glyph_path).convert("L")

        original = self.rgb_transform(original)
        masked = self.rgb_transform(masked)
        layout = self.mask_transform(layout)
        glyph = self.mask_transform(glyph)
        warped_glyph = self.mask_transform(warped_glyph)

        return {
            "sample_id": sample_id,
            "text": str(row["text"]),
            "original": original,
            "masked": masked,
            "layout": layout,
            "glyph": glyph,
            "warped_glyph": warped_glyph,
        }


# --------------------------------------------------
# Train / Validation / Test split
# --------------------------------------------------

def load_splits(
    train_ratio=0.8,
    val_ratio=0.1,
    seed=42
):

    metadata = pd.read_csv(METADATA_PATH)

    # Split using ORIGINAL ICDAR images
    # This prevents words from the same photograph
    # appearing in both training and testing.

    source_images = (
        metadata["source_image"]
        .drop_duplicates()
        .sample(frac=1, random_state=seed)
        .tolist()
    )

    total_images = len(source_images)

    train_end = int(total_images * train_ratio)
    val_end = train_end + int(total_images * val_ratio)

    train_images = set(
        source_images[:train_end]
    )

    val_images = set(
        source_images[train_end:val_end]
    )

    test_images = set(
        source_images[val_end:]
    )

    train_df = metadata[
        metadata["source_image"].isin(train_images)
    ]

    val_df = metadata[
        metadata["source_image"].isin(val_images)
    ]

    test_df = metadata[
        metadata["source_image"].isin(test_images)
    ]

    train_dataset = TextRenderingDataset(train_df)
    val_dataset = TextRenderingDataset(val_df)
    test_dataset = TextRenderingDataset(test_df)

    return (
        train_dataset,
        val_dataset,
        test_dataset
    )


# --------------------------------------------------
# Dataset Test
# --------------------------------------------------

if __name__ == "__main__":

    train_dataset, val_dataset, test_dataset = load_splits()

    print("=" * 50)
    print("DATASET CHECK")
    print("=" * 50)

    print(
        f"Training samples:   {len(train_dataset)}"
    )

    print(
        f"Validation samples: {len(val_dataset)}"
    )

    print(
        f"Testing samples:    {len(test_dataset)}"
    )

    print("\nOriginal image split:")

    print(
        "Train images:",
        train_dataset.metadata["source_image"].nunique()
    )

    print(
        "Val images:",
        val_dataset.metadata["source_image"].nunique()
    )

    print(
        "Test images:",
        test_dataset.metadata["source_image"].nunique()
    )

    sample = train_dataset[0]

    print("\nSample ID:", sample["sample_id"])
    print("Text:", sample["text"])

    print("\nTensor shapes:")

    print(
        "Original:",
        sample["original"].shape
    )

    print(
        "Masked:",
        sample["masked"].shape
    )

    print(
        "Layout:",
        sample["layout"].shape
    )

    print(
        "Glyph:",
        sample["glyph"].shape
    )

    print(
        "Warped Glyph:",
        sample["warped_glyph"].shape
    )

    print("\nValue ranges:")

    print(
        "Original:",
        sample["original"].min().item(),
        sample["original"].max().item()
    )

    print(
        "Masked:",
        sample["masked"].min().item(),
        sample["masked"].max().item()
    )

    print(
        "Layout:",
        sample["layout"].min().item(),
        sample["layout"].max().item()
    )

    print(
        "Glyph:",
        sample["glyph"].min().item(),
        sample["glyph"].max().item()
    )

