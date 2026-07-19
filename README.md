# 3D Attention U-Net for Brain Metastasis Segmentation (BraTS-METS 2026)

[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![BraTS-METS](https://img.shields.io/badge/Challenge-BraTS--METS%202026-blue)](https://www.synapse.org/Synapse:syn74274097/wiki/639600)

This repository contains a PyTorch training pipeline for five-class brain metastasis segmentation using a 3D Attention U-Net and multi-modal MRI patches. It was developed for the BraTS-METS 2026 setting and follows the `BraTS-MET-*` subject and file naming convention.

- Repository: [Zizyyyyu/BraTS-MET-2026-3D-AttentionUNet](https://github.com/Zizyyyyu/BraTS-MET-2026-3D-AttentionUNet)
- Challenge: [BraTS 2026 Task 1 data page](https://www.synapse.org/Synapse:syn74274097/wiki/639600)

## Scope

The current code implements:

- recursive discovery and loading of labeled NIfTI subjects;
- four-channel MRI patch construction;
- class-aware training-patch sampling;
- a 3D Attention U-Net;
- combined Generalized Dice and weighted cross-entropy loss;
- patch-based training and validation;
- foreground class Dice reporting;
- checkpointing, training resumption, CSV logging, and curve plotting.

The repository currently does **not** implement full-volume sliding-window inference, prediction NIfTI export, challenge submission packaging, or the official BraTS-METS lesion-wise evaluation pipeline. The Dice values printed by `train.py` are patch-based, voxel-wise class Dice values used for local training monitoring.

## Project Structure

```text
MICCAI/
├── train.py          # Training, validation, checkpointing, logging, and plotting
├── Dataset.py        # NIfTI loading, normalization, and patch selection
├── U3D_UNet.py       # 3D Attention U-Net
├── losses.py         # Generalized Dice loss and combined Dice/CE loss
├── README.md
├── .gitignore
├── checkpoints/      # Generated checkpoints; ignored by Git
└── training_curves/  # Generated plots and CSV history; ignored by Git
```

## Model Architecture

`U3D_UNet.py` defines `AttentionUNet` with four encoder levels, one bottleneck, four attention-gated decoder levels, and a final `1 × 1 × 1` convolution.

With the defaults `in_channels=4`, `base_channels=16`, and `num_classes=5`, the channel flow is:

```text
Input: 4 channels

Encoder 1:     4 → 16
Encoder 2:    16 → 32
Encoder 3:    32 → 64
Encoder 4:    64 → 128
Bottleneck:  128 → 256

Decoder 4:   256 → 128, with attention-gated Encoder 4 skip
Decoder 3:   128 → 64,  with attention-gated Encoder 3 skip
Decoder 2:    64 → 32,  with attention-gated Encoder 2 skip
Decoder 1:    32 → 16,  with attention-gated Encoder 1 skip

Output:       16 → 5 logits
```

Each encoder and post-concatenation decoder block contains two repetitions of:

```text
Conv3d(kernel_size=3, padding=1, bias=False)
→ InstanceNorm3d
→ ReLU
```

Downsampling uses `MaxPool3d(kernel_size=2, stride=2)`. Upsampling uses `ConvTranspose3d(kernel_size=2, stride=2)`.

Each attention block transforms the encoder and decoder features with separate `1 × 1 × 1` convolutions and instance normalization, adds them, applies ReLU, produces a one-channel sigmoid attention map, and multiplies that map with the encoder features before concatenation.

Because the network downsamples four times, patch dimensions should be divisible by 16. The configured `(128, 128, 128)` patch satisfies this requirement.

## Dataset

### Expected Files

`BraTSDataset` recursively searches `data_dir` for directories whose names begin with `BraTS-MET-`. For a subject named `BraTS-MET-XXXXX-XXX`, labeled training expects:

```text
/path/to/data/
└── BraTS-MET-XXXXX-XXX/
    ├── BraTS-MET-XXXXX-XXX-t1n.nii.gz
    ├── BraTS-MET-XXXXX-XXX-t1c.nii.gz
    ├── BraTS-MET-XXXXX-XXX-t2w.nii.gz
    ├── BraTS-MET-XXXXX-XXX-t2f.nii.gz
    └── BraTS-MET-XXXXX-XXX-seg.nii.gz
```

The implementation currently requires all four modality files for every discovered subject.

### Input Channels

The four channels are stacked in this fixed order:

| Channel | Suffix | MRI sequence |
|---:|---|---|
| 0 | `t1n` | T1-weighted, non-contrast |
| 1 | `t1c` | T1-weighted, contrast-enhanced |
| 2 | `t2w` | T2-weighted |
| 3 | `t2f` | T2-FLAIR |

### Segmentation Labels

The code accepts labels `0` through `4` and maps any value outside this range to background (`0`). The class names used for logging are:

| Label | Logged name | Meaning |
|---:|---|---|
| 0 | Background | Non-target voxels |
| 1 | NETC | Non-enhancing tumor core |
| 2 | SNFH | Surrounding non-enhancing FLAIR hyperintensity |
| 3 | ET | Enhancing tumor |
| 4 | RC | Resection cavity |

The model therefore outputs five logits per voxel.

### Excluded Subjects

`Dataset.py` excludes these subject IDs:

```text
BraTS-MET-01094-002
BraTS-MET-01184-002
```

### Patch Selection

The default patch size is `(128, 128, 128)`.

During training:

- the foreground classes present in the subject are identified;
- with probability `0.85`, one present foreground class is selected and a voxel from that class is used as the crop center;
- otherwise, the crop start is sampled uniformly from the valid spatial range.

During validation:

- if foreground voxels exist, the patch is centered on the midpoint of the foreground bounding box;
- otherwise, it is centered on the image.

If an image dimension is smaller than the patch dimension, zero-padding is applied at the end of that dimension.

### Normalization

Normalization is performed separately for each modality after patch extraction. Nonzero voxels are standardized using the patch foreground mean and standard deviation. Zero-valued background voxels remain zero.

## Requirements

- Python 3.8 or newer
- PyTorch
- NumPy
- NiBabel
- Matplotlib

The repository does not currently include a `requirements.txt` or environment file.

## Installation

```bash
git clone https://github.com/Zizyyyyu/BraTS-MET-2026-3D-AttentionUNet.git
cd BraTS-MET-2026-3D-AttentionUNet

pip install torch numpy nibabel matplotlib
```

Install a CUDA-enabled PyTorch build appropriate for the local GPU driver when GPU training is required.

## Training

### 1. Configure the Dataset Path

In `train.py`, update:

```python
data_dir = "/root/autodl-tmp/brats_raw/training_extracted/MICCAI-LH-BraTS2025-MET-Challenge-Training"
```

### 2. Start Training

Run the command from the repository root so that relative output paths point to the intended directories:

```bash
python train.py
```

### Default Configuration

The current defaults are defined directly inside `train.py:main()`:

| Setting | Default |
|---|---:|
| Patch size | `(128, 128, 128)` |
| Validation ratio | `0.2` |
| Batch size | `8` |
| DataLoader workers | `14` |
| Prefetch factor | `1` |
| Input channels | `4` |
| Output classes | `5` |
| Base channels | `16` |
| Epochs | `75` |
| Learning rate | `1e-4` |
| Weight decay | `1e-5` |
| Random seed | `2026` |
| Class-aware crop probability | `0.85` |
| Resume training | `True` |

Subjects are shuffled with a NumPy generator seeded with `2026`. The validation size is `max(1, int(dataset_size * 0.2))`; all remaining subjects are used for training. When resuming, the saved training and validation indices are reused.

## Optimization

### Loss

`DiceCELoss` adds two equally weighted terms:

```text
loss = 1.0 × GeneralizedDiceLoss + 1.0 × WeightedCrossEntropyLoss
```

The Generalized Dice component:

- applies softmax to the logits;
- excludes background;
- ignores foreground classes absent from the current batch targets;
- uses inverse-squared target-volume class weights.

Cross-entropy uses these class weights:

```text
[0.1, 1.0, 1.0, 1.0, 1.0]
```

### Optimizer and Scheduler

- Optimizer: `AdamW(lr=1e-4, weight_decay=1e-5)`
- Scheduler: `ReduceLROnPlateau(mode="max", factor=0.5, patience=4)`
- Scheduler target: validation mean Dice
- Gradient clipping: maximum norm `1.0`

### Mixed Precision

CUDA is used when available. Automatic mixed precision with `torch.cuda.amp.autocast` and `GradScaler` is enabled only on CUDA. DataLoader pinned memory is also enabled only when CUDA is used.

## Local Dice Metrics

After applying `argmax` to the five output logits, `train.py` accumulates voxel intersections and denominators for labels `1` through `4` over all patches in an epoch. For each class:

```text
Dice = (2 × intersection + 1e-5) / (prediction voxels + target voxels + 1e-5)
```

The reported mean Dice is the mean over foreground classes whose accumulated denominator is greater than zero. These metrics are reported for both training and validation.

Every 10 training batches, the script prints the running loss and mean Dice. At the end of each epoch, it prints loss, mean Dice, and the four class Dice values.

These are not the official BraTS-METS lesion-wise ranking metrics. For challenge-compatible evaluation, use full-volume NIfTI predictions and the separate [official BraTS evaluation package](https://github.com/BraTS/BraTS_evaluation).

## Checkpoints and Resuming

The following files are generated in `./checkpoints/`:

| File | Contents |
|---|---|
| `last_training_checkpoint.pth` | Model, optimizer, scheduler, scaler, history, split indices, subject IDs, and saved random states |
| `best_attention_unet_3d.pth` | Model state and metadata from the highest local validation mean Dice |
| `final_attention_unet_3d.pth` | Model state and architecture metadata after the final configured epoch |

Checkpoint writes use a temporary file followed by `Path.replace()`.

With `resume_training=True`, the script loads `last_training_checkpoint.pth` when it exists. It verifies the saved ordered subject-ID list, restores the saved split, and resumes at the next epoch. If the ordered subject list differs, training stops with an error.

The checkpoint stores Python, NumPy, PyTorch, CUDA, and DataLoader-generator states. This improves continuity across restarts, but the code does not promise bit-for-bit deterministic resumption: cuDNN benchmarking is enabled on CUDA, deterministic algorithms are not forced, and persistent DataLoader worker RNG states are not checkpointed.

## Training Outputs

After each completed epoch, the script updates:

```text
training_curves/
├── loss_curve.png
├── mean_dice_curve.png
├── train_class_dice_curve.png
├── validation_class_dice_curve.png
└── training_history.csv
```

The PNG files are saved at 200 DPI. The CSV includes epoch number, learning rate, training and validation losses, mean Dice values, four class Dice values for each split, and epoch duration in minutes.

The `checkpoints/` and `training_curves/` directories are ignored by Git because their contents are generated during training.

## Current Limitations

- The dataset path and training settings are hard-coded inside `train.py`.
- Every accepted subject must contain all four modality files.
- Training and validation operate on one patch per subject access rather than on full reconstructed volumes.
- There is no data augmentation beyond random/class-aware cropping.
- There is no inference or NIfTI prediction export script.
- There is no challenge submission or containerization pipeline.
- Local model selection uses patch-based foreground class Dice rather than official lesion-wise BraTS-METS metrics.

## License

This repository currently does not include a license file.
