from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..data.dataset import load_splits
from ..models.conditioned_unet import ConditionedUNet


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHECKPOINT_DIR = PROJECT_ROOT / "outputs" / "checkpoints"
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 1e-3


def get_device():

    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def reconstruction_loss(prediction, target, layout):

    global_loss = F.l1_loss(
        prediction,
        target
    )

    text_mask = layout.expand_as(target)

    text_difference = (
        torch.abs(prediction - target)
        * text_mask
    )

    mask_pixels = text_mask.sum().clamp(min=1.0)

    text_loss = (
        text_difference.sum()
        / mask_pixels
    )

    return (
        0.2 * global_loss
        + 0.8 * text_loss
    )


def train():

    device = get_device()

    print("=" * 60)
    print("IMPROVED CONDITIONED U-NET TRAINING")
    print("=" * 60)

    print(f"Device: {device}")

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

    model = ConditionedUNet(
        image_channels=3,
        condition_channels=2,
        output_channels=3,
        base_channels=32
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    best_val_loss = float("inf")

    history = []

    for epoch in range(1, EPOCHS + 1):

        # -------------------------
        # Training
        # -------------------------

        model.train()

        train_loss = 0.0

        for batch in train_loader:

            masked = batch["masked"].to(device)
            layout = batch["layout"].to(device)
            warped_glyph = batch["warped_glyph"].to(device)
            target = batch["original"].to(device)

            condition = torch.cat(
                [layout, warped_glyph],
                dim=1
            )

            optimizer.zero_grad()

            prediction = model(
                masked,
                condition
            )

            loss = reconstruction_loss(
                prediction,
                target,
                layout
            )

            loss.backward()

            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # -------------------------
        # Validation
        # -------------------------

        model.eval()

        val_loss = 0.0

        with torch.no_grad():

            for batch in val_loader:

                masked = batch["masked"].to(device)
                layout = batch["layout"].to(device)
                warped_glyph = batch["warped_glyph"].to(device)
                target = batch["original"].to(device)

                condition = torch.cat(
                    [layout, warped_glyph],
                    dim=1
                )

                prediction = model(
                    masked,
                    condition
                )

                loss = reconstruction_loss(
                    prediction,
                    target,
                    layout
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
                / "conditioned_best.pt"
            )

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict":
                        optimizer.state_dict(),
                    "val_loss": val_loss,
                },
                checkpoint_path
            )

            print(
                f"  -> Saved best model "
                f"(val loss {val_loss:.6f})"
            )

    # -------------------------
    # Save history
    # -------------------------

    history_df = pd.DataFrame(history)

    history_path = (
        METRICS_DIR
        / "conditioned_training_history.csv"
    )

    history_df.to_csv(
        history_path,
        index=False
    )

    print("\n" + "=" * 60)
    print("IMPROVED CONDITIONED TRAINING COMPLETE")
    print("=" * 60)

    print(
        f"Best validation loss: "
        f"{best_val_loss:.6f}"
    )

    print(
        f"Checkpoint: "
        f"{CHECKPOINT_DIR / 'conditioned_best.pt'}"
    )

    print(
        f"Training history: "
        f"{history_path}"
    )


if __name__ == "__main__":
    train()