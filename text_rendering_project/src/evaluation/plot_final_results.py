from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SUMMARY_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "metrics"
    / "final_model_summary.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "plots"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def main():

    summary = pd.read_csv(
        SUMMARY_PATH
    ).iloc[0]

    # ---------------------------------
    # Text readability metrics
    # ---------------------------------

    metrics = [
        "Word Accuracy",
        "OCR Confidence"
    ]

    baseline = [
        summary["baseline_word_accuracy"],
        summary["baseline_ocr_confidence"]
    ]

    renderer = [
        summary["renderer_word_accuracy"],
        summary["renderer_ocr_confidence"]
    ]

    x = range(len(metrics))

    plt.figure(figsize=(8, 5))

    width = 0.35

    plt.bar(
        [i - width / 2 for i in x],
        baseline,
        width,
        label="Baseline"
    )

    plt.bar(
        [i + width / 2 for i in x],
        renderer,
        width,
        label="Glyph Renderer"
    )

    plt.xticks(x, metrics)

    plt.ylabel("Score")
    plt.title("Text Readability Comparison")

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "text_readability_comparison.png",
        dpi=200
    )

    plt.close()

    # ---------------------------------
    # CER
    # ---------------------------------

    plt.figure(figsize=(6, 5))

    plt.bar(
        ["Baseline", "Glyph Renderer"],
        [
            summary["baseline_cer"],
            summary["renderer_cer"]
        ]
    )

    plt.ylabel(
        "Character Error Rate"
    )

    plt.title(
        "Character Error Rate (Lower is Better)"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "cer_comparison.png",
        dpi=200
    )

    plt.close()

    # ---------------------------------
    # Image quality
    # ---------------------------------

    image_metrics = [
        "SSIM",
        "PSNR / 30"
    ]

    baseline_image = [
        summary["baseline_ssim"],
        summary["baseline_psnr"] / 30
    ]

    renderer_image = [
        summary["renderer_ssim"],
        summary["renderer_psnr"] / 30
    ]

    x = range(len(image_metrics))

    plt.figure(figsize=(8, 5))

    plt.bar(
        [i - width / 2 for i in x],
        baseline_image,
        width,
        label="Baseline"
    )

    plt.bar(
        [i + width / 2 for i in x],
        renderer_image,
        width,
        label="Glyph Renderer"
    )

    plt.xticks(
        x,
        image_metrics
    )

    plt.ylabel("Normalized Score")

    plt.title(
        "Image Quality Comparison"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "image_quality_comparison.png",
        dpi=200
    )

    plt.close()

    print("Final plots saved to:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()