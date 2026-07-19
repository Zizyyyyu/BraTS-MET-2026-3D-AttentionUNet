# 3D Attention U-Net for Brain Metastasis Segmentation (BraTS-MET 2026)

![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch)
![MICCAI](https://img.shields.io/badge/MICCAI-BraTS--MET%202026-blue)
![3D Segmentation](https://img.shields.io/badge/Task-3D%20Medical%20Image%20Segmentation-green)

This project implements a **3D Attention U-Net** for brain tumor (metastasis) segmentation using multi-modal MRI scans, developed for the **BraTS-MET 2026 Challenge** at MICCAI. It segments brain metastases into four tumor sub-regions from four MRI sequences (T1-native, T1-contrast-enhanced, T2-weighted, and T2-FLAIR).

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Training](#training)
- [Checkpoints & Resuming](#checkpoints--resuming)
- [Outputs](#outputs)
- [Project Structure](#project-structure)
- [License](#license)

## Overview

The pipeline covers the full 3D medical image segmentation workflow:

- **Data loading** — reads NIfTI (`.nii.gz`) multi-modal MRI volumes with class-balanced patch sampling
- **Model** — 3D Attention U-Net with attention-gated skip connections
- **Loss function** — combined Generalized Dice Loss + weighted Cross-Entropy Loss
- **Training** — mixed precision (AMP), gradient clipping, learning rate scheduling, atomic checkpointing
- **Evaluation** — per-class Dice scores for four foreground classes (NETC, SNFH, ET, RC)
- **Visualization** — loss curves, mean Dice curves, and per-class Dice curves saved as PNG plots

## Architecture

The model is a **3D Attention U-Net** (`U3D_UNet.py`) inspired by [Oktay et al., Attention U-Net](https://arxiv.org/abs/1804.03999):

```
Encoder                           Decoder
┌─────────┐    ┌─────────────────────────────┐
│ Conv×2  │ ──→│      Attention Block        │
│ (4→16)  │    │  ┌───┐   ┌───┐   ┌───────┐ │
└────┬────┘    │  │ g │ + │ReLU│ → │σ(x)×f│ │
     │↓        │  └───┘   └───┘   └───────┘ │
┌─────────┐    └──────────┬──────────────────┘
│ Conv×2  │    ┌──────────┘
│ (16→32) │ ──→│   Attention Block
└────┬────┘    └──────────┐
     │↓                   │
┌─────────┐    ┌──────────┘
│ Conv×2  │ ──→│   Attention Block
│ (32→64) │    └──────────┐
└────┬────┘               │
     │↓                   │
┌─────────┐    ┌──────────┘
│ Conv×2  │ ──→│   Attention Block
│ (64→128)│    └──────────┐
└────┬────┘               │
     │↓              ┌────┴────┐
┌────────────┐       │ Conv×2  │
│ Conv×2     │       │(128→16) │
│(128→256)   │ ←──── │ + Conv1 │
└────────────┘       └─────────┘
  Bottleneck           Output (5)
```

**Encoder path:** 4 levels of double `Conv3d + InstanceNorm3d + ReLU` blocks with `MaxPool3d(2)` downsampling (16 → 32 → 64 → 128 → 256 base channels at bottleneck).

**Decoder path:** 4 levels with `ConvTranspose3d` upsampling and **attention gates** that learn to suppress irrelevant background regions in the skip connections. Each attention gate computes a gating signal from the decoder features to weight the encoder features via sigmoid attention.

**Output:** 1×1×1 convolution mapping 16 channels to 5 output logits (background + 4 foreground classes).

## Dataset

**BraTS-MET 2026 Challenge** — multi-modal brain MRI scans for metastasis segmentation.

### Input Modalities (4 channels)

| Modality | Suffix | Description |
|----------|--------|-------------|
| T1-native | `t1n` | T1-weighted without contrast |
| T1-CE | `t1c` | T1-weighted with contrast enhancement |
| T2-weighted | `t2w` | T2-weighted |
| T2-FLAIR | `t2f` | T2 Fluid Attenuated Inversion Recovery |

### Segmentation Labels (5 classes)

| Label | Name | Tissue |
|-------|------|--------|
| 0 | Background | Non-tumor |
| 1 | NETC | Necrosis |
| 2 | SNFH | Peritumoral edema |
| 3 | ET | Enhancing tumor |
| 4 | RC | Resection cavity |

### Data Directory Structure

```
/path/to/brats_data/
├── BraTS-MET-XXXXX-XXX/
│   ├── BraTS-MET-XXXXX-XXX-t1n.nii.gz
│   ├── BraTS-MET-XXXXX-XXX-t1c.nii.gz
│   ├── BraTS-MET-XXXXX-XXX-t2w.nii.gz
│   ├── BraTS-MET-XXXXX-XXX-t2f.nii.gz
│   ├── BraTS-MET-XXXXX-XXX-seg.nii.gz
│   └── ...
└── ...
```

### Class-Balanced Patch Sampling

During training, patches of size `(128, 128, 128)` are cropped with **85% probability near a tumor voxel** from a randomly selected foreground class to mitigate severe class imbalance. During validation, cropping is centered on the tumor region. Known corrupted patients (`BraTS-MET-01094-002` and `BraTS-MET-01184-002`) are automatically excluded.

## Requirements

- **Python ≥ 3.8**
- **PyTorch ≥ 2.0** (CUDA recommended)
- **Nibabel** (NIfTI I/O)
- **NumPy**
- **Matplotlib**

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd MICCAI

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install nibabel numpy matplotlib
```

> **Note:** The project does not include a `requirements.txt` yet — install the packages listed above manually. CUDA version should match your system's driver (adjust the PyTorch install command accordingly).

## Quick Start

### 1. Prepare the Dataset

Download the BraTS-MET 2026 training dataset and ensure your data directory follows the structure above.

### 2. Update the Data Path

Edit the `data_dir` variable in `train.py` (line 229) to point to your dataset:

```python
data_dir = "/path/to/your/brats_data"
```

### 3. Run Training

```bash
python train.py
```

Training automatically creates `./checkpoints/` and `./training_curves/` directories.

## Configuration

All hyperparameters are defined as constants inside `train.py:main()`. Below are the key settings:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `patch_size` | `(128, 128, 128)` | 3D patch dimensions for training |
| `validation_ratio` | `0.2` | Fraction of patients held out for validation |
| `batch_size` | `8` | Mini-batch size |
| `num_workers` | `14` | DataLoader worker processes |
| `prefetch_factor` | `1` | Samples prefetched per worker |
| `in_channels` | `4` | Input MRI modalities |
| `num_classes` | `5` | Segmentation classes (1 background + 4 foreground) |
| `base_channels` | `16` | Initial feature channels (doubled at each encoder level) |
| `num_epochs` | `75` | Total training epochs |
| `learning_rate` | `1e-4` | Initial learning rate (AdamW) |
| `weight_decay` | `1e-5` | AdamW weight decay |
| `seed` | `2026` | Random seed for reproducibility |
| `class_crop_probability` | `0.85` | Probability of class-aware patch cropping |

Modify these directly in `train.py` or refactor them into a config file as needed.

## Training

The training pipeline includes the following features:

### Reproducibility
- Seeds are set for `random`, `numpy`, `torch`, and CUDA operations.
- A custom `seed_worker` ensures DataLoader workers produce deterministic orders.
- Full random states are saved with each checkpoint for exact resumption.

### Mixed Precision (AMP)
- Automatic Mixed Precision via `torch.cuda.amp` (`GradScaler` + `autocast`) is enabled when CUDA is available, speeding up training and reducing memory usage.

### Optimization
- **Optimizer:** AdamW (`lr=1e-4`, `weight_decay=1e-5`)
- **Scheduler:** `ReduceLROnPlateau` — halves the learning rate (`factor=0.5`) when validation mean Dice plateaus (`patience=4`)
- **Gradient clipping:** Max norm of 1.0

### Loss Function

The combined loss (`DiceCELoss`) consists of:

1. **Generalized Dice Loss** (weight `1.0`) — multi-class Dice loss with inverse-squared-frequency weighting per class; excludes background (class 0) and only considers valid foreground classes present in the batch.

2. **Weighted Cross-Entropy Loss** (weight `1.0`) with class weights:
   - Background: `0.1` (heavily down-weighted)
   - All foreground classes: `1.0`

### Logging

Every 10 training steps, the current running loss and mean Dice are printed. At the end of each epoch, full per-class metrics for both training and validation are reported:

```
Epoch [1/75]
Learning rate: 0.00010000
Train Loss: 0.3521    Train Mean Dice: 0.8245
Train NETC Dice: 0.7102    Train SNFH Dice: 0.8543
Train ET Dice: 0.8910      Train RC Dice: 0.8425
Validation Loss: 0.4023    Validation Mean Dice: 0.8012
Validation NETC Dice: 0.6801  Validation SNFH Dice: 0.8310
Validation ET Dice: 0.8723    Validation RC Dice: 0.8214
Epoch time: 12.34 minutes
```

## Checkpoints & Resuming

### Checkpoint Files

All checkpoints are saved to `./checkpoints/`:

| File | Content | Usage |
|------|---------|-------|
| `last_training_checkpoint.pth` | Full training state | Resume training from last epoch |
| `best_attention_unet_3d.pth` | Model with best val Dice | Inference / evaluation |
| `final_attention_unet_3d.pth` | Model after all epochs | Final export |

### What's Saved in a Full Checkpoint

- Model, optimizer, scheduler, and scaler states
- Best validation Dice score
- Complete training history (all metrics per epoch)
- Training/validation split indices and patient IDs
- DataLoader generator state
- Python, NumPy, and PyTorch/CUDA random states

### Resume Training

By default, `resume_training = True`. If `last_training_checkpoint.pth` exists, training automatically resumes from the saved epoch. The script validates dataset consistency (patient ID list must match) before resuming. If the dataset has changed, delete or move the checkpoint and set `resume_training = False`.

### Atomic Saves

All checkpoints are written to a `.tmp` file first, then atomically renamed via `.replace()`, preventing corruption from interrupted writes.

## Outputs

### Training Curves

After each epoch, updated plots are saved to `./training_curves/`:

| File | Content |
|------|---------|
| `loss_curve.png` | Training and validation loss over epochs |
| `mean_dice_curve.png` | Training and validation mean Dice score |
| `train_class_dice_curve.png` | Per-class Dice for each training class |
| `validation_class_dice_curve.png` | Per-class Dice for validation |
| `training_history.csv` | Full tabular history of all metrics |

### Plot Example

The loss and Dice curves are saved at 200 DPI and include:
- Grid lines for readability
- Markers at each epoch
- Dice plots scaled `[0.0, 1.0]` with legend
- Tight layout for clean margins

## Project Structure

```
MICCAI/
├── train.py                 # Main training script — orchestrates entire pipeline
├── Dataset.py               # BraTSDataset — NIfTI data loader with patch sampling
├── U3D_UNet.py              # AttentionUNet — 3D model architecture with attention gates
├── losses.py                # GeneralizedDiceLoss & DiceCELoss functions
├── checkpoints/             # Saved model weights and training checkpoints
│   ├── last_training_checkpoint.pth
│   ├── best_attention_unet_3d.pth
│   └── final_attention_unet_3d.pth
├── training_curves/         # Loss/Dice PNG plots and CSV history
│   ├── loss_curve.png
│   ├── mean_dice_curve.png
│   ├── train_class_dice_curve.png
│   ├── validation_class_dice_curve.png
│   └── training_history.csv
└── README.md                # This file
```

## Notes

- The data directory path is currently hard-coded to `/root/autodl-tmp/brats_raw/training_extracted/MICCAI-LH-BraTS2025-MET-Challenge-Training` (an AutoDL cloud GPU path). Update it for your environment.
- Two known corrupted patients are excluded automatically. If other corrupted scans are encountered, add them to the `excluded_patient_ids` set in `Dataset.py`.
- The model targets 5 output channels (1 background + 4 foreground). For inference, apply `argmax` along the channel dimension to obtain per-voxel class predictions.

## License

This project is developed for the BraTS-MET 2026 Challenge at MICCAI. All rights reserved.
