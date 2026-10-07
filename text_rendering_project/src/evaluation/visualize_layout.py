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
    / "layout_best.pt"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "generated_images"
    / "layout"
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
    print("LAYOUT MODEL VISUALIZATION")
    print("=" * 60)

    _, _, test_dataset = load_splits()

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    model = UNet(
        input_channels=4,
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

    print("Loaded epoch:", checkpoint["epoch"])
    print("Validation loss:", checkpoint["val_loss"])

    number_of_samples = 10

    fig, axes = plt.subplots(
        number_of_samples,
        4,
        figsize=(12, number_of_samples * 3)
    )

    with torch.no_grad():

        for index, batch in enumerate(test_loader):

            if index >= number_of_samples:
                break

            masked = batch["masked"].to(device)
            layout = batch["layout"].to(device)
            target = batch["original"].to(device)

            text = batch["text"][0]
            sample_id = batch["sample_id"][0]

            model_input = torch.cat(
                [masked, layout],
                dim=1
            )

            prediction = model(model_input)

            prediction_cpu = (
                prediction
                .cpu()
                .squeeze(0)
                .clamp(0, 1)
            )

            masked_cpu = masked.cpu().squeeze(0)
            layout_cpu = layout.cpu().squeeze(0)
            target_cpu = target.cpu().squeeze(0)

            prediction_image = to_pil_image(
                prediction_cpu
            )

            prediction_image.save(
                OUTPUT_DIR
                / f"{sample_id}_prediction.png"
            )

            axes[index, 0].imshow(
                to_pil_image(masked_cpu)
            )

            axes[index, 0].set_title(
                f"Masked\n{text}"
            )

            axes[index, 1].imshow(
                layout_cpu.squeeze(0),
                cmap="gray"
            )

            axes[index, 1].set_title(
                "Layout Mask"
            )

            axes[index, 2].imshow(
                prediction_image
            )

            axes[index, 2].set_title(
                "Layout Prediction"
            )

            axes[index, 3].imshow(
                to_pil_image(target_cpu)
            )

            axes[index, 3].set_title(
                "Ground Truth"
            )

            for column in range(4):
                axes[index, column].axis("off")

    plt.tight_layout()

    comparison_path = (
        OUTPUT_DIR
        / "layout_comparison.png"
    )

    plt.savefig(
        comparison_path,
        dpi=150
    )

    plt.close()

    print("\nSaved:")
    print(comparison_path)


if __name__ == "__main__":
    main()