from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image

from ..data.dataset import load_splits
from ..models.unet import UNet


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHECKPOINT_DIR = PROJECT_ROOT / "outputs" / "checkpoints"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "generated_images"
    / "comparison"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def load_model(input_channels, checkpoint_name, device):

    model = UNet(
        input_channels=input_channels,
        output_channels=3,
        base_channels=32
    ).to(device)

    checkpoint = torch.load(
        CHECKPOINT_DIR / checkpoint_name,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model


def main():

    device = get_device()

    print("=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    print("Device:", device)

    _, _, test_dataset = load_splits()

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    # --------------------------------
    # Load all three models
    # --------------------------------

    baseline_model = load_model(
        3,
        "baseline_best.pt",
        device
    )

    layout_model = load_model(
        4,
        "layout_best.pt",
        device
    )

    glyph_model = load_model(
        5,
        "glyph_layout_best.pt",
        device
    )

    number_of_samples = 10

    fig, axes = plt.subplots(
        number_of_samples,
        6,
        figsize=(18, number_of_samples * 3)
    )

    with torch.no_grad():

        for index, batch in enumerate(test_loader):

            if index >= number_of_samples:
                break

            masked = batch["masked"].to(device)
            layout = batch["layout"].to(device)
            glyph = batch["glyph"].to(device)
            target = batch["original"].to(device)

            text = batch["text"][0]

            # --------------------------------
            # Baseline
            # --------------------------------

            baseline_prediction = baseline_model(
                masked
            )

            # --------------------------------
            # Layout
            # --------------------------------

            layout_input = torch.cat(
                [masked, layout],
                dim=1
            )

            layout_prediction = layout_model(
                layout_input
            )

            # --------------------------------
            # Layout + Glyph
            # --------------------------------

            glyph_input = torch.cat(
                [masked, layout, glyph],
                dim=1
            )

            glyph_prediction = glyph_model(
                glyph_input
            )

            # Move to CPU
            masked_cpu = masked.cpu().squeeze(0)
            glyph_cpu = glyph.cpu().squeeze(0)
            target_cpu = target.cpu().squeeze(0)

            baseline_cpu = (
                baseline_prediction
                .cpu()
                .squeeze(0)
                .clamp(0, 1)
            )

            layout_cpu = (
                layout_prediction
                .cpu()
                .squeeze(0)
                .clamp(0, 1)
            )

            glyph_prediction_cpu = (
                glyph_prediction
                .cpu()
                .squeeze(0)
                .clamp(0, 1)
            )

            # --------------------------------
            # Plot
            # --------------------------------

            axes[index, 0].imshow(
                to_pil_image(masked_cpu)
            )

            axes[index, 0].set_title(
                f"Masked\n{text}"
            )

            axes[index, 1].imshow(
                glyph_cpu.squeeze(0),
                cmap="gray"
            )

            axes[index, 1].set_title(
                "Glyph Mask"
            )

            axes[index, 2].imshow(
                to_pil_image(baseline_cpu)
            )

            axes[index, 2].set_title(
                "Baseline"
            )

            axes[index, 3].imshow(
                to_pil_image(layout_cpu)
            )

            axes[index, 3].set_title(
                "Layout"
            )

            axes[index, 4].imshow(
                to_pil_image(glyph_prediction_cpu)
            )

            axes[index, 4].set_title(
                "Layout + Glyph"
            )

            axes[index, 5].imshow(
                to_pil_image(target_cpu)
            )

            axes[index, 5].set_title(
                "Ground Truth"
            )

            for column in range(6):
                axes[index, column].axis("off")

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR
        / "all_models_comparison.png"
    )

    plt.savefig(
        output_path,
        dpi=150
    )

    plt.close()

    print("\nSaved comparison:")
    print(output_path)


if __name__ == "__main__":
    main()