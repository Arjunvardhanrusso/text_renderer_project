from pathlib import Path
import re

import cv2
import easyocr
import numpy as np
import pandas as pd
import torch

from skimage.metrics import (
    structural_similarity,
    peak_signal_noise_ratio,
)

from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data.dataset import load_splits
from ..models.unet import UNet
from ..models.glyph_renderer import GlyphRenderer


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "checkpoints"
)

METRICS_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "metrics"
)

METRICS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


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
# Text normalization
# --------------------------------------------------

def normalize_text(text):

    text = str(text).lower()

    # Keep letters and numbers
    text = re.sub(
        r"[^a-z0-9]",
        "",
        text
    )

    return text


# --------------------------------------------------
# Levenshtein distance
# --------------------------------------------------

def levenshtein(a, b):

    if len(a) < len(b):
        return levenshtein(b, a)

    if len(b) == 0:
        return len(a)

    previous = list(
        range(len(b) + 1)
    )

    for i, char_a in enumerate(
        a,
        start=1
    ):

        current = [i]

        for j, char_b in enumerate(
            b,
            start=1
        ):

            insertion = (
                current[j - 1] + 1
            )

            deletion = (
                previous[j] + 1
            )

            substitution = (
                previous[j - 1]
                + (char_a != char_b)
            )

            current.append(
                min(
                    insertion,
                    deletion,
                    substitution
                )
            )

        previous = current

    return previous[-1]


def calculate_cer(
    ground_truth,
    prediction
):

    gt = normalize_text(
        ground_truth
    )

    pred = normalize_text(
        prediction
    )

    if len(gt) == 0:
        return 0.0

    distance = levenshtein(
        gt,
        pred
    )

    return distance / len(gt)


# --------------------------------------------------
# Convert tensor to NumPy RGB image
# --------------------------------------------------

def tensor_to_numpy(tensor):

    image = (
        tensor
        .detach()
        .cpu()
        .squeeze(0)
        .permute(1, 2, 0)
        .numpy()
    )

    image = np.clip(
        image,
        0,
        1
    )

    return image


# --------------------------------------------------
# Crop only the text region using layout mask
# --------------------------------------------------

def crop_text_region(
    image,
    layout
):

    mask = (
        layout
        .detach()
        .cpu()
        .squeeze()
        .numpy()
    )

    coords = np.argwhere(
        mask > 0.5
    )

    if len(coords) == 0:
        return image

    y_min, x_min = coords.min(
        axis=0
    )

    y_max, x_max = coords.max(
        axis=0
    )

    # Small padding around the word
    padding = 4

    y_min = max(
        int(y_min) - padding,
        0
    )

    x_min = max(
        int(x_min) - padding,
        0
    )

    y_max = min(
        int(y_max) + padding,
        image.shape[0] - 1
    )

    x_max = min(
        int(x_max) + padding,
        image.shape[1] - 1
    )

    crop = image[
        y_min:y_max + 1,
        x_min:x_max + 1
    ]

    # OCR works better on larger text
    crop = cv2.resize(
        crop,
        None,
        fx=4,
        fy=4,
        interpolation=cv2.INTER_CUBIC
    )

    return crop


# --------------------------------------------------
# OCR
# --------------------------------------------------

def run_ocr(
    reader,
    image,
    layout
):

    crop = crop_text_region(
        image,
        layout
    )

    # Convert 0-1 RGB to uint8
    crop_uint8 = (
        crop * 255
    ).astype(
        np.uint8
    )

    results = reader.readtext(
        crop_uint8,
        detail=1,
        paragraph=False
    )

    if not results:
        return "", 0.0

    # Choose highest-confidence result
    best = max(
        results,
        key=lambda item: item[2]
    )

    text = best[1]
    confidence = float(best[2])

    return text, confidence


# --------------------------------------------------
# Image metrics
# --------------------------------------------------

def calculate_image_metrics(
    prediction,
    target
):

    ssim = structural_similarity(
        target,
        prediction,
        channel_axis=2,
        data_range=1.0
    )

    psnr = peak_signal_noise_ratio(
        target,
        prediction,
        data_range=1.0
    )

    return ssim, psnr


# --------------------------------------------------
# Main evaluation
# --------------------------------------------------

def main():

    device = get_device()

    print("=" * 60)
    print("FINAL MODEL EVALUATION")
    print("=" * 60)

    print(
        "Neural-network device:",
        device
    )

    # EasyOCR doesn't use Apple MPS,
    # so use CPU for OCR.
    print(
        "Loading EasyOCR..."
    )

    reader = easyocr.Reader(
        ["en"],
        gpu=False
    )

    _, _, test_dataset = load_splits()

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    print(
        f"Test samples: {len(test_dataset)}"
    )

    # --------------------------------------------------
    # Baseline model
    # --------------------------------------------------

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

    # --------------------------------------------------
    # Glyph renderer
    # --------------------------------------------------

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

    results = []

    # For a quick test first, change to 50.
    # Set to None for all 473.
    MAX_SAMPLES = None

    with torch.no_grad():

        for index, batch in enumerate(
            tqdm(
                test_loader,
                desc="Evaluating"
            )
        ):

            if (
                MAX_SAMPLES is not None
                and index >= MAX_SAMPLES
            ):
                break

            masked = batch[
                "masked"
            ].to(device)

            layout = batch[
                "layout"
            ].to(device)

            glyph = batch[
                "warped_glyph"
            ].to(device)

            target = batch[
                "original"
            ].to(device)

            ground_truth = batch[
                "text"
            ][0]

            sample_id = batch[
                "sample_id"
            ][0]

            # ------------------------------------------
            # Baseline
            # ------------------------------------------

            baseline_output = baseline(
                masked
            )

            # ------------------------------------------
            # Final renderer
            # ------------------------------------------

            rendered_output = renderer(
                masked,
                baseline_output,
                layout,
                glyph
            )

            # ------------------------------------------
            # Convert to NumPy
            # ------------------------------------------

            baseline_np = tensor_to_numpy(
                baseline_output
            )

            rendered_np = tensor_to_numpy(
                rendered_output
            )

            target_np = tensor_to_numpy(
                target
            )

            # ------------------------------------------
            # OCR
            # ------------------------------------------

            baseline_text, baseline_conf = (
                run_ocr(
                    reader,
                    baseline_np,
                    layout
                )
            )

            renderer_text, renderer_conf = (
                run_ocr(
                    reader,
                    rendered_np,
                    layout
                )
            )

            # ------------------------------------------
            # CER
            # ------------------------------------------

            baseline_cer = calculate_cer(
                ground_truth,
                baseline_text
            )

            renderer_cer = calculate_cer(
                ground_truth,
                renderer_text
            )

            # Exact word match
            gt_normalized = normalize_text(
                ground_truth
            )

            baseline_correct = int(
                normalize_text(
                    baseline_text
                )
                == gt_normalized
            )

            renderer_correct = int(
                normalize_text(
                    renderer_text
                )
                == gt_normalized
            )

            # ------------------------------------------
            # Image metrics
            # ------------------------------------------

            baseline_ssim, baseline_psnr = (
                calculate_image_metrics(
                    baseline_np,
                    target_np
                )
            )

            renderer_ssim, renderer_psnr = (
                calculate_image_metrics(
                    rendered_np,
                    target_np
                )
            )

            results.append({
                "sample_id":
                    sample_id,

                "ground_truth":
                    ground_truth,

                "baseline_ocr":
                    baseline_text,

                "renderer_ocr":
                    renderer_text,

                "baseline_correct":
                    baseline_correct,

                "renderer_correct":
                    renderer_correct,

                "baseline_cer":
                    baseline_cer,

                "renderer_cer":
                    renderer_cer,

                "baseline_ocr_confidence":
                    baseline_conf,

                "renderer_ocr_confidence":
                    renderer_conf,

                "baseline_ssim":
                    baseline_ssim,

                "renderer_ssim":
                    renderer_ssim,

                "baseline_psnr":
                    baseline_psnr,

                "renderer_psnr":
                    renderer_psnr,
            })

    # --------------------------------------------------
    # Save individual results
    # --------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    results_path = (
        METRICS_DIR
        / "final_model_evaluation.csv"
    )

    results_df.to_csv(
        results_path,
        index=False
    )

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    summary = {
        "samples":
            len(results_df),

        "baseline_word_accuracy":
            results_df[
                "baseline_correct"
            ].mean(),

        "renderer_word_accuracy":
            results_df[
                "renderer_correct"
            ].mean(),

        "baseline_cer":
            results_df[
                "baseline_cer"
            ].mean(),

        "renderer_cer":
            results_df[
                "renderer_cer"
            ].mean(),

        "baseline_ocr_confidence":
            results_df[
                "baseline_ocr_confidence"
            ].mean(),

        "renderer_ocr_confidence":
            results_df[
                "renderer_ocr_confidence"
            ].mean(),

        "baseline_ssim":
            results_df[
                "baseline_ssim"
            ].mean(),

        "renderer_ssim":
            results_df[
                "renderer_ssim"
            ].mean(),

        "baseline_psnr":
            results_df[
                "baseline_psnr"
            ].mean(),

        "renderer_psnr":
            results_df[
                "renderer_psnr"
            ].mean(),
    }

    summary_df = pd.DataFrame(
        [summary]
    )

    summary_path = (
        METRICS_DIR
        / "final_model_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False
    )

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    print(
        f"Samples evaluated: "
        f"{summary['samples']}"
    )

    print("\nWORD ACCURACY")

    print(
        f"Baseline: "
        f"{summary['baseline_word_accuracy']:.4f}"
    )

    print(
        f"Renderer: "
        f"{summary['renderer_word_accuracy']:.4f}"
    )

    print("\nCHARACTER ERROR RATE (lower is better)")

    print(
        f"Baseline: "
        f"{summary['baseline_cer']:.4f}"
    )

    print(
        f"Renderer: "
        f"{summary['renderer_cer']:.4f}"
    )

    print("\nOCR CONFIDENCE")

    print(
        f"Baseline: "
        f"{summary['baseline_ocr_confidence']:.4f}"
    )

    print(
        f"Renderer: "
        f"{summary['renderer_ocr_confidence']:.4f}"
    )

    print("\nSSIM")

    print(
        f"Baseline: "
        f"{summary['baseline_ssim']:.4f}"
    )

    print(
        f"Renderer: "
        f"{summary['renderer_ssim']:.4f}"
    )

    print("\nPSNR")

    print(
        f"Baseline: "
        f"{summary['baseline_psnr']:.4f}"
    )

    print(
        f"Renderer: "
        f"{summary['renderer_psnr']:.4f}"
    )

    print("\nSaved:")
    print(results_path)
    print(summary_path)


if __name__ == "__main__":
    main()