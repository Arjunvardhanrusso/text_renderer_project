from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image

from ..data.dataset import load_splits
from ..models.unet import UNet
from ..models.conditioned_unet import ConditionedUNet


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHECKPOINT_DIR = PROJECT_ROOT / "outputs" / "checkpoints"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "generated_images"
    / "edge_comparison"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_device():

    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def load_unet(input_channels, checkpoint_name, device):

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


def load_edge_model(device):

    model = ConditionedUNet(
        image_channels=3,
        condition_channels=2,
        output_channels=3,
        base_channels=32
    ).to(device)

    checkpoint = torch.load(
        CHECKPOINT_DIR / "conditioned_edge_best.pt",
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
    print("EDGE LOSS MODEL COMPARISON")
    print("=" * 60)

    _, _, test_dataset = load_splits()

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    baseline_model = load_unet(
        3,
        "baseline_best.pt",
        device
    )

    naive_glyph_model = load_unet(
        5,
        "glyph_layout_best.pt",
        device
    )

    edge_model = load_edge_model(device)

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
            warped_glyph = batch["warped_glyph"].to(device)
            target = batch["original"].to(device)

            text = batch["text"][0]

            # Baseline
            baseline_prediction = baseline_model(masked)

            # Naive glyph model
            naive_input = torch.cat(
                [masked, layout, glyph],
                dim=1
            )

            naive_prediction = naive_glyph_model(
                naive_input
            )

            # Edge conditioned model
            condition = torch.cat(
                [layout, warped_glyph],
                dim=1
            )

            edge_prediction = edge_model(
                masked,
                condition
            )

            # Move to CPU
            masked_cpu = masked.cpu().squeeze(0)
            warped_glyph_cpu = warped_glyph.cpu().squeeze(0)
            target_cpu = target.cpu().squeeze(0)

            baseline_cpu = (
                baseline_prediction
                .cpu()
                .squeeze(0)
                .clamp(0, 1)
            )

            naive_cpu = (
                naive_prediction
                .cpu()
                .squeeze(0)
                .clamp(0, 1)
            )

            edge_cpu = (
                edge_prediction
                .cpu()
                .squeeze(0)
                .clamp(0, 1)
            )

            # Plot
            axes[index, 0].imshow(
                to_pil_image(masked_cpu)
            )
            axes[index, 0].set_title(
                f"Masked\n{text}"
            )

            axes[index, 1].imshow(
                warped_glyph_cpu.squeeze(0),
                cmap="gray"
            )
            axes[index, 1].set_title(
                "Warped Glyph"
            )

            axes[index, 2].imshow(
                to_pil_image(baseline_cpu)
            )
            axes[index, 2].set_title(
                "Baseline"
            )

            axes[index, 3].imshow(
                to_pil_image(naive_cpu)
            )
            axes[index, 3].set_title(
                "Naive Glyph"
            )

            axes[index, 4].imshow(
                to_pil_image(edge_cpu)
            )
            axes[index, 4].set_title(
                "Conditioned + Edge"
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
        / "edge_model_comparison.png"
    )

    plt.savefig(
        output_path,
        dpi=150
    )

    plt.close()

    print("\nSaved:")
    print(output_path)


if __name__ == "__main__":
    main()