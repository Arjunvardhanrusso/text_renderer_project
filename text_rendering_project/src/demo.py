from pathlib import Path
import sys

import matplotlib.pyplot as plt
import torch
from torchvision.transforms.functional import to_pil_image

from .data.dataset import load_splits
from .models.unet import UNet
from .models.glyph_renderer import GlyphRenderer


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "checkpoints"
)


def get_device():

    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def main():

    device = get_device()

    # Allow:
    # python -m src.demo 5
    sample_index = 0

    if len(sys.argv) > 1:
        sample_index = int(
            sys.argv[1]
        )

    _, _, test_dataset = load_splits()

    if sample_index >= len(test_dataset):
        raise ValueError(
            f"Index must be below {len(test_dataset)}"
        )

    sample = test_dataset[
        sample_index
    ]

    masked = (
        sample["masked"]
        .unsqueeze(0)
        .to(device)
    )

    layout = (
        sample["layout"]
        .unsqueeze(0)
        .to(device)
    )

    glyph = (
        sample["warped_glyph"]
        .unsqueeze(0)
        .to(device)
    )

    target = sample["original"]

    text = sample["text"]

    # --------------------------------
    # Baseline
    # --------------------------------

    baseline = UNet(
        input_channels=3,
        output_channels=3,
        base_channels=32
    ).to(device)

    baseline_checkpoint = torch.load(
        CHECKPOINT_DIR
        / "baseline_best.pt",
        map_location=device
    )

    baseline.load_state_dict(
        baseline_checkpoint[
            "model_state_dict"
        ]
    )

    baseline.eval()

    # --------------------------------
    # Renderer
    # --------------------------------

    renderer = GlyphRenderer().to(
        device
    )

    renderer_checkpoint = torch.load(
        CHECKPOINT_DIR
        / "glyph_renderer_best.pt",
        map_location=device
    )

    renderer.load_state_dict(
        renderer_checkpoint[
            "model_state_dict"
        ]
    )

    renderer.eval()

    # --------------------------------
    # Prediction
    # --------------------------------

    with torch.no_grad():

        reconstructed = baseline(
            masked
        )

        rendered = renderer(
            masked,
            reconstructed,
            layout,
            glyph
        )

    reconstructed = (
        reconstructed
        .cpu()
        .squeeze(0)
        .clamp(0, 1)
    )

    rendered = (
        rendered
        .cpu()
        .squeeze(0)
        .clamp(0, 1)
    )

    # --------------------------------
    # Display
    # --------------------------------

    fig, axes = plt.subplots(
        1,
        5,
        figsize=(16, 4)
    )

    axes[0].imshow(
        to_pil_image(
            sample["masked"]
        )
    )
    axes[0].set_title(
        "Masked Input"
    )

    axes[1].imshow(
        sample[
            "warped_glyph"
        ].squeeze(),
        cmap="gray"
    )
    axes[1].set_title(
        f"Glyph: {text}"
    )

    axes[2].imshow(
        to_pil_image(
            reconstructed
        )
    )
    axes[2].set_title(
        "Baseline"
    )

    axes[3].imshow(
        to_pil_image(
            rendered
        )
    )
    axes[3].set_title(
        "Final Model"
    )

    axes[4].imshow(
        to_pil_image(
            target
        )
    )
    axes[4].set_title(
        "Ground Truth"
    )

    for axis in axes:
        axis.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()