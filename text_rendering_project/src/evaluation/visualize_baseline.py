from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image

from ..data.dataset import load_splits
from ..models.unet import UNet


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "checkpoints"
    / "baseline_best.pt"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "generated_images"
    / "baseline"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_device():

    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def main():

    device = get_device()

    print("=" * 60)
    print("BASELINE VISUALIZATION")
    print("=" * 60)

    print("Device:", device)

    # -----------------------------
    # Test dataset
    # -----------------------------

    _, _, test_dataset = load_splits()

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    # -----------------------------
    # Load model
    # -----------------------------

    model = UNet(
        input_channels=3,
        output_channels=3,
        base_channels=32
    ).to(device)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print(
        "Loaded checkpoint from epoch:",
        checkpoint["epoch"]
    )

    print(
        "Validation loss:",
        checkpoint["val_loss"]
    )

    # -----------------------------
    # Generate samples
    # -----------------------------

    number_of_samples = 10

    fig, axes = plt.subplots(
        number_of_samples,
        3,
        figsize=(10, number_of_samples * 3)
    )

    with torch.no_grad():

        for index, batch in enumerate(test_loader):

            if index >= number_of_samples:
                break

            masked = batch["masked"].to(device)
            target = batch["original"].to(device)

            text = batch["text"][0]
            sample_id = batch["sample_id"][0]

            prediction = model(masked)

            prediction = prediction.cpu().squeeze(0)
            masked_cpu = masked.cpu().squeeze(0)
            target_cpu = target.cpu().squeeze(0)

            # -------------------------
            # Save predicted image
            # -------------------------

            prediction_image = to_pil_image(
                prediction.clamp(0, 1)
            )

            prediction_image.save(
                OUTPUT_DIR
                / f"{sample_id}_prediction.png"
            )

            # -------------------------
            # Plot comparison
            # -------------------------

            axes[index, 0].imshow(
                to_pil_image(masked_cpu)
            )

            axes[index, 0].set_title(
                f"Masked\n{text}"
            )

            axes[index, 1].imshow(
                prediction_image
            )

            axes[index, 1].set_title(
                "Baseline Prediction"
            )

            axes[index, 2].imshow(
                to_pil_image(target_cpu)
            )

            axes[index, 2].set_title(
                "Ground Truth"
            )

            for column in range(3):
                axes[index, column].axis("off")

    plt.tight_layout()

    comparison_path = (
        OUTPUT_DIR
        / "baseline_comparison.png"
    )

    plt.savefig(
        comparison_path,
        dpi=150
    )

    plt.close()

    print("\nSaved predictions to:")
    print(OUTPUT_DIR)

    print("\nComparison image:")
    print(comparison_path)


if __name__ == "__main__":
    main()