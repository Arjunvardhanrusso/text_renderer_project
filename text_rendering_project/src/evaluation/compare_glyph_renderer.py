from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image

from ..data.dataset import load_splits
from ..models.unet import UNet
from ..models.glyph_renderer import GlyphRenderer


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHECKPOINT_DIR = PROJECT_ROOT / "outputs" / "checkpoints"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "generated_images"
    / "glyph_renderer"
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
    print("GLYPH RENDERER COMPARISON")
    print("=" * 60)

    _, _, test_dataset = load_splits()

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    # ----------------------------------------
    # Load frozen baseline
    # ----------------------------------------

    baseline = UNet(
        input_channels=3,
        output_channels=3,
        base_channels=32
    ).to(device)

    baseline_checkpoint = torch.load(
        CHECKPOINT_DIR / "baseline_best.pt",
        map_location=device
    )

    baseline.load_state_dict(
        baseline_checkpoint["model_state_dict"]
    )

    baseline.eval()

    # ----------------------------------------
    # Load glyph renderer
    # ----------------------------------------

    renderer = GlyphRenderer().to(device)

    renderer_checkpoint = torch.load(
        CHECKPOINT_DIR / "glyph_renderer_best.pt",
        map_location=device
    )

    renderer.load_state_dict(
        renderer_checkpoint["model_state_dict"]
    )

    renderer.eval()

    print(
        "Renderer loaded from epoch:",
        renderer_checkpoint["epoch"]
    )

    number_of_samples = 10

    fig, axes = plt.subplots(
        number_of_samples,
        5,
        figsize=(16, number_of_samples * 3)
    )

    with torch.no_grad():

        for index, batch in enumerate(test_loader):

            if index >= number_of_samples:
                break

            masked = batch["masked"].to(device)
            layout = batch["layout"].to(device)
            glyph = batch["warped_glyph"].to(device)
            target = batch["original"].to(device)

            text = batch["text"][0]
            sample_id = batch["sample_id"][0]

            # Baseline reconstruction
            reconstructed = baseline(masked)

            # Glyph renderer output
            rendered = renderer(
                masked,
                reconstructed,
                layout,
                glyph
            )

            masked_cpu = masked.cpu().squeeze(0)
            glyph_cpu = glyph.cpu().squeeze(0)

            reconstructed_cpu = (
                reconstructed
                .cpu()
                .squeeze(0)
                .clamp(0, 1)
            )

            rendered_cpu = (
                rendered
                .cpu()
                .squeeze(0)
                .clamp(0, 1)
            )

            target_cpu = target.cpu().squeeze(0)

            # Save rendered result
            rendered_image = to_pil_image(
                rendered_cpu
            )

            rendered_image.save(
                OUTPUT_DIR
                / f"{sample_id}_rendered.png"
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
                "Warped Glyph"
            )

            axes[index, 2].imshow(
                to_pil_image(reconstructed_cpu)
            )

            axes[index, 2].set_title(
                "Baseline"
            )

            axes[index, 3].imshow(
                rendered_image
            )

            axes[index, 3].set_title(
                "Glyph Renderer"
            )

            axes[index, 4].imshow(
                to_pil_image(target_cpu)
            )

            axes[index, 4].set_title(
                "Ground Truth"
            )

            for column in range(5):
                axes[index, column].axis("off")

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR
        / "glyph_renderer_comparison.png"
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