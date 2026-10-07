# Lightweight Glyph-Guided Text Rendering in Images

A lightweight computer-vision project inspired by the text-conditioning ideas behind **TextDiffuser/TextDiffuser2**.

The original goal was to improve text rendering inside generated images using Stable Diffusion. Because Stable Diffusion/TextDiffuser2 fine-tuning required model weights, storage, GPU memory, and datasets that were too large for the available local hardware, the project was redesigned as a lightweight PyTorch system that can be trained and evaluated locally.

The final system combines:

1. A **lightweight U-Net** that reconstructs a masked text region.
2. A **perspective-aligned glyph representation** that explicitly provides the requested text structure.
3. A small **neural glyph renderer** that learns the appearance of the text while preserving the glyph as a hard structural constraint.
4. OCR-based evaluation to measure whether the final image actually contains readable text.

---

## Final Result

The final model was evaluated on **473 unseen test samples** from ICDAR 2015.

| Metric | Baseline U-Net | Final Glyph Renderer |
|---|---:|---:|
| Word Accuracy ↑ | 0.0021 | **0.2875** |
| Character Error Rate ↓ | 0.9954 | **0.4372** |
| OCR Confidence ↑ | 0.0113 | **0.3715** |
| SSIM ↑ | **0.7721** | 0.7710 |
| PSNR ↑ | **23.1075** | 23.0576 |

The final glyph-guided renderer substantially improves text readability while keeping overall image similarity almost unchanged.

---

# Project Architecture

The final pipeline is:

```text
ICDAR 2015 Image
        |
        v
Text Annotation + Bounding Polygon
        |
        v
Crop / Resize to 128 x 128
        |
        +----------------------+
        |                      |
        v                      v
Masked Image             Text Transcription
        |                      |
        v                      v
Baseline U-Net             Glyph Mask
        |                      |
        v                      v
Background              Perspective Warp
Reconstruction                |
        |                      |
        +----------+-----------+
                   |
                   v
             Glyph Renderer
                   |
                   v
        Final Image with Text
                   |
                   v
        OCR / CER / SSIM / PSNR
```

The final renderer uses the glyph as a **hard structural constraint**, so the model cannot solve the problem by simply producing a smooth blank region.

---

# Repository Structure

```text
text_rendering_project/
|
├── data/
│   ├── raw/
│   │   └── icdar2015/
│   │       ├── ch4_training_images/
│   │       └── ch4_training_localization_transcription_gt/
│   │
│   └── processed/
│       ├── original/
│       ├── masked/
│       ├── layout_masks/
│       ├── glyph_masks/
│       ├── glyph_masks_warped/
│       └── metadata.csv
│
├── src/
│   ├── __init__.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── parse_annotations.py
│   │   ├── visualize_annotations.py
│   │   ├── preprocess.py
│   │   ├── create_masks.py
│   │   ├── create_warped_glyph_masks.py
│   │   └── dataset.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── unet.py
│   │   ├── conditioned_unet.py
│   │   └── glyph_renderer.py
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train_baseline.py
│   │   ├── train_layout.py
│   │   ├── train_glyph_layout.py
│   │   ├── train_conditioned.py
│   │   ├── train_conditioned_edge.py
│   │   └── train_glyph_renderer.py
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── visualize_baseline.py
│   │   ├── visualize_layout.py
│   │   ├── compare_predictions.py
│   │   ├── compare_improved.py
│   │   ├── visualize_edge_model.py
│   │   ├── compare_glyph_renderer.py
│   │   ├── evaluate_final_models.py
│   │   └── plot_final_results.py
│   │
│   └── demo.py
│
├── outputs/
│   ├── checkpoints/
│   ├── generated_images/
│   ├── metrics/
│   ├── plots/
│   └── annotation_checks/
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

# Requirements

Recommended:

- Python 3.10+
- macOS, Linux, or Windows
- PyTorch
- CPU is supported
- Apple Silicon Macs can use **MPS**
- NVIDIA GPUs can use **CUDA**

Main Python packages:

```text
torch
torchvision
numpy
opencv-python
Pillow
matplotlib
pandas
scikit-image
scikit-learn
easyocr
PyYAML
tqdm
```

---

# 1. Clone the Repository

Replace the URL below with the actual GitHub repository URL.

```bash
git clone <YOUR_GITHUB_REPO_URL>
cd text_rendering_project
```

---

# 2. Create a Virtual Environment

## macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

## Windows

```bash
python -m venv venv
venv\Scripts\activate
```

After activation, the terminal should show something similar to:

```text
(venv)
```

Confirm the interpreter:

```bash
which python
```

On Windows:

```bash
where python
```

The path should point inside the project's `venv` directory.

---

# 3. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Verify PyTorch:

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('MPS:', torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False)"
```

---

# 4. Download the ICDAR 2015 Dataset

This project uses the **ICDAR 2015 Incidental Scene Text Dataset**.

From **Task 4.1: Text Localization (2015 edition)**, download only:

1. **Training Set Images**
2. **Training Set Localisation and Transcription Ground Truth**

The dataset is available from the Robust Reading Competition website:

https://rrc.cvc.uab.es/?ch=4&com=downloads

After extracting the downloads, place the folders exactly as follows:

```text
data/
└── raw/
    └── icdar2015/
        ├── ch4_training_images/
        │   ├── img_1.jpg
        │   ├── img_2.jpg
        │   └── ...
        │
        └── ch4_training_localization_transcription_gt/
            ├── gt_img_1.txt
            ├── gt_img_2.txt
            └── ...
```

Do not rename the individual image or annotation files.

---

# 5. Parse and Verify the Dataset

Run:

```bash
python src/data/parse_annotations.py
```

Expected output should be approximately:

```text
==================================================
ICDAR 2015 DATASET CHECK
==================================================
Images loaded: 1000
Readable text instances: 4468
```

Unreadable ICDAR annotations marked with `###` are ignored.

---

# 6. Visually Verify Bounding Boxes

Run:

```bash
python src/data/visualize_annotations.py
```

This creates files under:

```text
outputs/annotation_checks/
```

Open several generated images and verify that the annotation polygons surround the correct text.

Do not continue if the bounding boxes are obviously incorrect.

---

# 7. Preprocess the Dataset

Run:

```bash
python src/data/preprocess.py
```

The preprocessing step:

- crops each readable text instance
- keeps surrounding visual context
- resizes samples to `128 x 128`
- saves the original crop
- masks the text region
- records metadata
- keeps track of the original source image

Expected processed folders:

```text
data/processed/
├── original/
├── masked/
└── metadata.csv
```

The `source_image` field is important because it is later used to prevent train/test leakage.

---

# 8. Generate Layout and Glyph Masks

Run:

```bash
python src/data/create_masks.py
```

This creates:

```text
data/processed/layout_masks/
data/processed/glyph_masks/
```

The layout mask tells the model **where** text belongs.

The glyph mask tells the model **what text** should appear.

For a quick manual verification, compare matching files such as:

```text
data/processed/original/sample_00000.jpg
data/processed/masked/sample_00000.jpg
data/processed/layout_masks/sample_00000.png
data/processed/glyph_masks/sample_00000.png
```

---

# 9. Generate Perspective-Aligned Glyph Masks

Run:

```bash
python -m src.data.create_warped_glyph_masks
```

This creates:

```text
data/processed/glyph_masks_warped/
```

The text glyph is warped to approximately follow the ICDAR quadrilateral orientation.

---

# 10. Verify the PyTorch Dataset Loader

Run:

```bash
python src/data/dataset.py
```

Expected output is similar to:

```text
==================================================
DATASET CHECK
==================================================
Training samples:   3537
Validation samples: 458
Testing samples:    473

Original image split:
Train images: 783
Val images:   97
Test images:  99

Tensor shapes:
Original:      torch.Size([3, 128, 128])
Masked:        torch.Size([3, 128, 128])
Layout:        torch.Size([1, 128, 128])
Glyph:         torch.Size([1, 128, 128])
Warped Glyph:  torch.Size([1, 128, 128])
```

The dataset is split by **original source image**, not by individual word crop. This prevents words from the same photograph from appearing in both training and testing.

---

# 11. Verify the Baseline U-Net

Run:

```bash
python -m src.models.unet
```

Expected:

```text
Baseline output: torch.Size([2, 3, 128, 128])
Layout output: torch.Size([2, 3, 128, 128])
Layout + Glyph output: torch.Size([2, 3, 128, 128])
```

The lightweight U-Net contains approximately:

```text
1.93 million trainable parameters
```

---

# 12. Train the Baseline U-Net

Run:

```bash
python -m src.training.train_baseline
```

Training configuration:

```text
Epochs:        10
Batch size:    16
Learning rate: 0.001
Optimizer:     Adam
```

Best checkpoint:

```text
outputs/checkpoints/baseline_best.pt
```

Training history:

```text
outputs/metrics/baseline_training_history.csv
```

The baseline reconstructs the masked image region but typically produces blurry patches instead of readable text.

---

# 13. Optional Experimental Models

The repository contains several experiments performed before arriving at the final design.

They are useful for reproducing the development process but are **not required to run the final demo**.

## Layout-conditioned model

```bash
python -m src.training.train_layout
```

Input:

```text
Masked RGB + Layout
```

Checkpoint:

```text
outputs/checkpoints/layout_best.pt
```

## Naive layout + glyph model

```bash
python -m src.training.train_glyph_layout
```

Input:

```text
Masked RGB + Layout + Glyph
```

Checkpoint:

```text
outputs/checkpoints/glyph_layout_best.pt
```

## Separate conditioning encoder

Verify architecture:

```bash
python -m src.models.conditioned_unet
```

Train:

```bash
python -m src.training.train_conditioned
```

## Edge-loss experiment

```bash
python -m src.training.train_conditioned_edge
```

These experiments showed that simply giving the U-Net glyph information did not guarantee that the network would actually preserve character structure.

---

# 14. Final Glyph Renderer Architecture

The final model is:

```text
Masked Image
    |
    v
Frozen Baseline U-Net
    |
    v
Background Reconstruction
    |
    +----------------------+
                           |
Warped Glyph -------------+
                           |
Layout -------------------+
                           |
                           v
                    Glyph Renderer
                           |
                           v
                  Final Rendered Image
```

The glyph is used as a hard structural mask:

```python
output = reconstructed * (1 - glyph_rgb) + text_rgb * glyph_rgb
```

This guarantees that glyph pixels are represented in the final output.

The glyph renderer has only approximately:

```text
39,395 trainable parameters
```

Verify it with:

```bash
python -m src.models.glyph_renderer
```

---

# 15. Train the Final Glyph Renderer

The baseline U-Net must be trained first because the final renderer loads and freezes:

```text
outputs/checkpoints/baseline_best.pt
```

Then run:

```bash
python -m src.training.train_glyph_renderer
```

Training configuration:

```text
Epochs:        5
Batch size:    16
Learning rate: 0.001
Optimizer:     Adam
```

Best checkpoint:

```text
outputs/checkpoints/glyph_renderer_best.pt
```

Training history:

```text
outputs/metrics/glyph_renderer_training_history.csv
```

---

# 16. Visualize the Final Renderer

Run:

```bash
python -m src.evaluation.compare_glyph_renderer
```

Output:

```text
outputs/generated_images/glyph_renderer/
```

The comparison displays:

```text
Masked Input
Warped Glyph
Baseline Reconstruction
Glyph Renderer
Ground Truth
```

---

# 17. Run Final Evaluation

Run:

```bash
python -m src.evaluation.evaluate_final_models
```

The script evaluates the baseline and final glyph renderer across the unseen test set using:

- exact word accuracy
- Character Error Rate (CER)
- OCR confidence
- SSIM
- PSNR

EasyOCR runs on CPU while the neural networks can use MPS/CUDA.

Expected final results from the completed experiment:

```text
Samples evaluated: 473

WORD ACCURACY
Baseline: 0.0021
Renderer: 0.2875

CHARACTER ERROR RATE
Baseline: 0.9954
Renderer: 0.4372

OCR CONFIDENCE
Baseline: 0.0113
Renderer: 0.3715

SSIM
Baseline: 0.7721
Renderer: 0.7710

PSNR
Baseline: 23.1075
Renderer: 23.0576
```

Evaluation files:

```text
outputs/metrics/final_model_evaluation.csv
outputs/metrics/final_model_summary.csv
```

---

# 18. Generate Result Plots

Run:

```bash
python -m src.evaluation.plot_final_results
```

Generated plots:

```text
outputs/plots/
├── text_readability_comparison.png
├── cer_comparison.png
└── image_quality_comparison.png
```

---

# 19. Run the Final Demo

The demo uses an **unseen test sample** and displays:

```text
Masked Input
    |
    v
Warped Glyph
    |
    v
Baseline Reconstruction
    |
    v
Final Glyph Renderer
    |
    v
Ground Truth
```

Run:

```bash
python -m src.demo 0
```

Try different test samples:

```bash
python -m src.demo 5
python -m src.demo 15
python -m src.demo 50
python -m src.demo 100
```

The argument is the test-set sample index.

Valid indexes are from:

```text
0 to 472
```

---

# Quick Start for Running the Demo After Cloning

If trained checkpoints are included in the GitHub repository, the shortest setup is:

```bash
git clone <YOUR_GITHUB_REPO_URL>
cd text_rendering_project

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

Download the two ICDAR 2015 training files and place them under:

```text
data/raw/icdar2015/
```

Then prepare the data:

```bash
python src/data/preprocess.py
python src/data/create_masks.py
python -m src.data.create_warped_glyph_masks
```

Verify:

```bash
python src/data/dataset.py
```

Then run:

```bash
python -m src.demo 0
```

### Important

The demo requires these trained checkpoints:

```text
outputs/checkpoints/baseline_best.pt
outputs/checkpoints/glyph_renderer_best.pt
```

If these files are **not included in the repository**, train them first:

```bash
python -m src.training.train_baseline
python -m src.training.train_glyph_renderer
```

Then:

```bash
python -m src.demo 0
```

---

# Complete Reproduction Sequence

For a complete run from a fresh clone:

```bash
# 1. Clone
git clone <YOUR_GITHUB_REPO_URL>
cd text_rendering_project

# 2. Environment
python3 -m venv venv
source venv/bin/activate

# 3. Dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 4. Download ICDAR 2015 manually and place it in data/raw/icdar2015/

# 5. Verify raw annotations
python src/data/parse_annotations.py

# 6. Visual annotation check
python src/data/visualize_annotations.py

# 7. Preprocess
python src/data/preprocess.py

# 8. Create masks
python src/data/create_masks.py

# 9. Create warped glyphs
python -m src.data.create_warped_glyph_masks

# 10. Verify data loader
python src/data/dataset.py

# 11. Verify U-Net
python -m src.models.unet

# 12. Train baseline
python -m src.training.train_baseline

# 13. Verify glyph renderer
python -m src.models.glyph_renderer

# 14. Train glyph renderer
python -m src.training.train_glyph_renderer

# 15. Visual comparison
python -m src.evaluation.compare_glyph_renderer

# 16. Final evaluation
python -m src.evaluation.evaluate_final_models

# 17. Result plots
python -m src.evaluation.plot_final_results

# 18. Demo
python -m src.demo 0
```

---

# Development Experiments

The project did not jump directly to the final architecture. Several lightweight alternatives were tested.

| Experiment | Main Idea | Outcome |
|---|---|---|
| Baseline U-Net | Masked RGB → RGB reconstruction | Good background reconstruction, unreadable text |
| Layout U-Net | Add text-position mask | Minimal improvement |
| Naive Glyph U-Net | Add layout + glyph as input channels | Glyph mostly ignored |
| Conditioned U-Net | Separate condition encoder and feature fusion | Small improvement |
| Edge-loss model | Penalize missing character edges | Still insufficient |
| **Final Glyph Renderer** | Hard glyph constraint + learned text appearance | **Readable text achieved** |

This development process showed that **glyph information must be structurally enforced**, not merely provided as an optional input channel.

---

# Key Design Decisions

## Why ICDAR 2015?

It is small enough for local experimentation and provides:

- 1,000 training images
- word-level text transcriptions
- polygon/bounding-box locations

After filtering unreadable annotations, the project produced:

```text
4468 readable text instances
```

## Why 128 x 128?

The project was intended to run locally. `128 x 128` provides a practical balance between:

- training speed
- memory requirements
- text visibility
- model size

## Why split by original image?

Splitting individual word crops randomly can cause train/test leakage.

Instead, all words from the same original ICDAR image remain in the same split.

## Why a hard glyph constraint?

Earlier models could ignore glyph conditioning and minimize pixel reconstruction loss by creating a smooth region.

The final renderer prevents this by explicitly using the glyph in the output composition.

---

# Limitations

The final output improves text readability but does **not** perfectly reproduce the original scene text appearance.

Known limitations include:

- generic glyph font may differ from the original font
- generated text can appear pasted onto the reconstructed background
- background reconstruction can be blurry
- text shadows are not modeled explicitly
- exact text color may differ
- curved or highly stylized text remains difficult
- complex perspective may not be reproduced perfectly
- the final renderer depends on an explicit text transcription/glyph

Therefore, the project should be described as:

> **Glyph-guided text preservation and rendering**

rather than:

> **Fully photorealistic free-form text generation.**

---

# Final Conclusion

A standard lightweight U-Net was able to reconstruct masked image regions but failed to recover readable text.

Adding layout information, naive glyph conditioning, separate conditioning encoders, and edge-aware losses produced only marginal improvements.

The final solution introduced a **hard glyph-guided neural renderer** that separates:

```text
background reconstruction
```

from:

```text
text structure + text appearance
```

On 473 unseen test samples, the final renderer improved exact OCR word accuracy from:

```text
0.21%
```

to:

```text
28.75%
```

and reduced Character Error Rate from:

```text
0.9954
```

to:

```text
0.4372
```

while SSIM and PSNR remained almost unchanged.

This demonstrates that explicit glyph guidance can significantly improve textual readability in a lightweight local image-reconstruction pipeline.

---

# Suggested GitHub Notes

Large/generated project files should generally not be committed unless intentionally required for the demo.

A typical `.gitignore` should exclude:

```text
venv/
__pycache__/
*.pyc
.DS_Store

data/raw/
data/processed/

outputs/generated_images/
```

You may choose to **keep the two final trained checkpoints in GitHub** if their sizes are reasonable:

```text
outputs/checkpoints/baseline_best.pt
outputs/checkpoints/glyph_renderer_best.pt
```

Keeping those checkpoints makes it easier for another user to reproduce the final demo without retraining.

If checkpoint files are too large for normal GitHub storage, use Git LFS or provide them separately.

---

## License / Dataset Notice

The ICDAR 2015 dataset is not created by this project. Users should download it from its official distribution source and follow the dataset's own terms of use.

---

## Author

Add your name, university, course, and project information here.

```text
Name: Arjun Akkena

```
