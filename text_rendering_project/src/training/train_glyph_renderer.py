from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..data.dataset import load_splits
from ..models.unet import UNet
from ..models.glyph_renderer import GlyphRenderer


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHECKPOINT_DIR = PROJECT_ROOT / "outputs" / "checkpoints"
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

BASELINE_CHECKPOINT = CHECKPOINT_DIR / "baseline_best.pt"

BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 1e-3


def get_device():

    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def renderer_loss(output, target, reconstructed, glyph):
    """
    1. Match the target image inside glyph pixels.
    2. Encourage visible contrast from reconstructed background.
    """

    glyph_rgb = glyph.expand_as(target)

    # ---------------------------------
    # Text reconstruction loss
    # ---------------------------------

    difference = (
        torch.abs(output - target)
        * glyph_rgb
    )

    glyph_pixels = glyph_rgb.sum().clamp(min=1.0)

    text_loss = difference.sum() / glyph_pixels

    # ---------------------------------
    # Contrast loss
    # Prevent renderer from making
    # glyph identical to background.
    # ---------------------------------

    contrast = (
        torch.abs(output - reconstructed)
        * glyph_rgb
    ).sum() / glyph_pixels

    minimum_contrast = 0.20

    contrast_loss = F.relu(
        minimum_contrast - contrast
    )

    total_loss = (
        text_loss
        + 0.30 * contrast_loss
    )

    return total_loss


def train():

    device = get_device()

    print("=" * 60)
    print("GLYPH RENDERER TRAINING")
    print("=" * 60)

    print("Device:", device)

    train_dataset, val_dataset, _ = load_splits()

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    print(f"Training samples:   {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # ==========================================
    # Load and FREEZE baseline U-Net
    # ==========================================

    baseline = UNet(
        input_channels=3,
        output_channels=3,
        base_channels=32
    ).to(device)

    checkpoint = torch.load(
        BASELINE_CHECKPOINT,
        map_location=device
    )

    baseline.load_state_dict(
        checkpoint["model_state_dict"]
    )

    baseline.eval()

    for parameter in baseline.parameters():
        parameter.requires_grad = False

    print(
        "Baseline loaded from epoch:",
        checkpoint["epoch"]
    )

    # ==========================================
    # Glyph renderer
    # ==========================================

    renderer = GlyphRenderer().to(device)

    optimizer = torch.optim.Adam(
        renderer.parameters(),
        lr=LEARNING_RATE
    )

    best_val_loss = float("inf")
    history = []

    # ==========================================
    # Training
    # ==========================================

    for epoch in range(1, EPOCHS + 1):

        renderer.train()

        train_loss = 0.0

        for batch in train_loader:

            masked = batch["masked"].to(device)
            layout = batch["layout"].to(device)
            glyph = batch["warped_glyph"].to(device)
            target = batch["original"].to(device)

            # Baseline reconstruction stays frozen
            with torch.no_grad():

                reconstructed = baseline(
                    masked
                )

            optimizer.zero_grad()

            output = renderer(
                masked,
                reconstructed,
                layout,
                glyph
            )

            loss = renderer_loss(
                output,
                target,
                reconstructed,
                glyph
            )

            loss.backward()

            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ======================================
        # Validation
        # ======================================

        renderer.eval()

        val_loss = 0.0

        with torch.no_grad():

            for batch in val_loader:

                masked = batch["masked"].to(device)
                layout = batch["layout"].to(device)
                glyph = batch["warped_glyph"].to(device)
                target = batch["original"].to(device)

                reconstructed = baseline(
                    masked
                )

                output = renderer(
                    masked,
                    reconstructed,
                    layout,
                    glyph
                )

                loss = renderer_loss(
                    output,
                    target,
                    reconstructed,
                    glyph
                )

                val_loss += loss.item()

        val_loss /= len(val_loader)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss
        })

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f}"
        )

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            checkpoint_path = (
                CHECKPOINT_DIR
                / "glyph_renderer_best.pt"
            )

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict":
                        renderer.state_dict(),
                    "optimizer_state_dict":
                        optimizer.state_dict(),
                    "val_loss": val_loss
                },
                checkpoint_path
            )

            print(
                f"  -> Saved best renderer "
                f"(val loss {val_loss:.6f})"
            )

    # ==========================================
    # Save history
    # ==========================================

    history_df = pd.DataFrame(history)

    history_path = (
        METRICS_DIR
        / "glyph_renderer_training_history.csv"
    )

    history_df.to_csv(
        history_path,
        index=False
    )

    print("\n" + "=" * 60)
    print("GLYPH RENDERER TRAINING COMPLETE")
    print("=" * 60)

    print(
        f"Best validation loss: "
        f"{best_val_loss:.6f}"
    )

    print(
        "Checkpoint:",
        CHECKPOINT_DIR / "glyph_renderer_best.pt"
    )


if __name__ == "__main__":
    train()