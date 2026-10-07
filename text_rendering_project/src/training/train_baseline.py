from pathlib import Path


import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..data.dataset import load_splits
from ..models.unet import UNet

# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHECKPOINT_DIR = PROJECT_ROOT / "outputs" / "checkpoints"
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 1e-3


# --------------------------------------------------
# Device
# --------------------------------------------------

def get_device():

    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


# --------------------------------------------------
# Loss
# --------------------------------------------------

def reconstruction_loss(prediction, target, layout):
    """
    Global reconstruction loss +
    extra emphasis on the text region.

    Layout is used only for calculating loss.
    It is NOT given to the baseline model as input.
    """

    global_loss = F.l1_loss(
        prediction,
        target
    )

    # Expand:
    # [B,1,H,W] -> [B,3,H,W]
    text_mask = layout.expand_as(target)

    text_difference = (
        torch.abs(prediction - target)
        * text_mask
    )

    mask_pixels = text_mask.sum().clamp(min=1.0)

    text_loss = text_difference.sum() / mask_pixels

    return (
        0.2 * global_loss
        + 0.8 * text_loss
    )


# --------------------------------------------------
# Training
# --------------------------------------------------

def train():

    device = get_device()

    print("=" * 60)
    print("BASELINE U-NET TRAINING")
    print("=" * 60)

    print(f"Device: {device}")

    # Dataset
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

    # Baseline = 3-channel masked RGB only
    model = UNet(
        input_channels=3,
        output_channels=3,
        base_channels=32
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    best_val_loss = float("inf")

    history = []

    # --------------------------------------------------
    # Epoch loop
    # --------------------------------------------------

    for epoch in range(1, EPOCHS + 1):

        # ==========================
        # Training
        # ==========================

        model.train()

        train_loss = 0.0

        for batch in train_loader:

            masked = batch["masked"].to(device)
            target = batch["original"].to(device)
            layout = batch["layout"].to(device)

            optimizer.zero_grad()

            prediction = model(masked)

            loss = reconstruction_loss(
                prediction,
                target,
                layout
            )

            loss.backward()

            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ==========================
        # Validation
        # ==========================

        model.eval()

        val_loss = 0.0

        with torch.no_grad():

            for batch in val_loader:

                masked = batch["masked"].to(device)
                target = batch["original"].to(device)
                layout = batch["layout"].to(device)

                prediction = model(masked)

                loss = reconstruction_loss(
                    prediction,
                    target,
                    layout
                )

                val_loss += loss.item()

        val_loss /= len(val_loader)

        # Save history
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

        # ==========================
        # Best checkpoint
        # ==========================

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            checkpoint_path = (
                CHECKPOINT_DIR
                / "baseline_best.pt"
            )

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                },
                checkpoint_path
            )

            print(
                f"  -> Saved best model "
                f"(val loss {val_loss:.6f})"
            )

    # --------------------------------------------------
    # Save training history
    # --------------------------------------------------

    history_df = pd.DataFrame(history)

    history_path = (
        METRICS_DIR
        / "baseline_training_history.csv"
    )

    history_df.to_csv(
        history_path,
        index=False
    )

    print("\n" + "=" * 60)
    print("BASELINE TRAINING COMPLETE")
    print("=" * 60)

    print(
        f"Best validation loss: "
        f"{best_val_loss:.6f}"
    )

    print(
        f"Checkpoint: "
        f"{CHECKPOINT_DIR / 'baseline_best.pt'}"
    )

    print(
        f"Training history: "
        f"{history_path}"
    )


if __name__ == "__main__":
    train()