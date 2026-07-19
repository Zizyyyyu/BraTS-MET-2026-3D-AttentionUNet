import csv
import random
import time
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset
from Dataset import BraTSDataset
from U3D_UNet import AttentionUNet
from losses import DiceCELoss


def set_seed(seed: int):
    """Set random seed for reproducibility across random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int):
    """Initialize DataLoader worker seed from PyTorch initial seed."""
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def create_empty_history():
    """Return an empty training history dictionary with all metric keys."""
    return {
        "epoch": [],
        "learning_rate": [],
        "train_loss": [],
        "validation_loss": [],
        "train_mean_dice": [],
        "validation_mean_dice": [],
        "train_NETC_dice": [],
        "train_SNFH_dice": [],
        "train_ET_dice": [],
        "train_RC_dice": [],
        "validation_NETC_dice": [],
        "validation_SNFH_dice": [],
        "validation_ET_dice": [],
        "validation_RC_dice": [],
        "epoch_minutes": []
    }


def update_dice_statistics(logits: torch.Tensor, target: torch.Tensor, num_classes: int):
    """Update Dice score accumulators (intersections and denominators) for each class."""
    prediction = torch.argmax(logits, dim=1)
    intersections = torch.zeros(num_classes - 1, dtype=torch.float64)
    denominators = torch.zeros(num_classes - 1, dtype=torch.float64)
    for class_index in range(1, num_classes):
        prediction_mask = prediction == class_index
        target_mask = target == class_index
        intersection = torch.sum(prediction_mask & target_mask).double()
        denominator = torch.sum(prediction_mask).double() + torch.sum(target_mask).double()
        intersections[class_index - 1] = intersection.cpu()
        denominators[class_index - 1] = denominator.cpu()
    return intersections, denominators


def calculate_dice(intersections: torch.Tensor, denominators: torch.Tensor, smooth: float = 1e-5):
    """Calculate Dice scores from accumulated intersections and denominators."""
    dice_scores = torch.zeros_like(intersections)
    valid_classes = denominators > 0
    if valid_classes.any():
        dice_scores[valid_classes] = (2.0 * intersections[valid_classes] + smooth) / (denominators[valid_classes] + smooth)
        mean_dice = dice_scores[valid_classes].mean().item()
    else:
        mean_dice = 0.0
    return dice_scores.tolist(), mean_dice


def save_training_history(history: dict, csv_path: Path):
    """Save training history to CSV file atomically using a temporary file."""
    field_names = list(history.keys())
    epoch_count = len(history["epoch"])
    temporary_path = csv_path.with_suffix(".csv.tmp")
    with open(temporary_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=field_names)
        writer.writeheader()
        for index in range(epoch_count):
            row = {key: history[key][index] for key in field_names}
            writer.writerow(row)
    temporary_path.replace(csv_path)


def plot_training_history(history: dict, class_names: tuple, output_dir: Path):
    """Plot and save training/validation loss and Dice curves to PNG files."""
    epochs = history["epoch"]
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, history["train_loss"], marker="o", label="Train Loss")
    plt.plot(epochs, history["validation_loss"], marker="o", label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png", dpi=200, bbox_inches="tight")
    plt.close()
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, history["train_mean_dice"], marker="o", label="Train Mean Dice")
    plt.plot(epochs, history["validation_mean_dice"], marker="o", label="Validation Mean Dice")
    plt.xlabel("Epoch")
    plt.ylabel("Dice")
    plt.title("Training and Validation Mean Dice")
    plt.ylim(0.0, 1.0)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "mean_dice_curve.png", dpi=200, bbox_inches="tight")
    plt.close()
    plt.figure(figsize=(9, 6))
    for class_name in class_names:
        plt.plot(epochs, history[f"train_{class_name}_dice"], marker="o", label=f"Train {class_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Dice")
    plt.title("Training Dice for Each Class")
    plt.ylim(0.0, 1.0)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "train_class_dice_curve.png", dpi=200, bbox_inches="tight")
    plt.close()
    plt.figure(figsize=(9, 6))
    for class_name in class_names:
        plt.plot(epochs, history[f"validation_{class_name}_dice"], marker="o", label=f"Validation {class_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Dice")
    plt.title("Validation Dice for Each Class")
    plt.ylim(0.0, 1.0)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "validation_class_dice_curve.png", dpi=200, bbox_inches="tight")
    plt.close()


def atomic_torch_save(data: dict, path: Path):
    """Save torch checkpoint atomically by writing to temporary file then replacing."""
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(data, temporary_path)
    temporary_path.replace(path)


def move_optimizer_to_device(optimizer: torch.optim.Optimizer, device: torch.device):
    """Move optimizer state tensors to the specified device."""
    for optimizer_state in optimizer.state.values():
        for key, value in optimizer_state.items():
            if torch.is_tensor(value):
                optimizer_state[key] = value.to(device)


def restore_random_states(checkpoint: dict):
    """Restore Python, NumPy, and PyTorch random states from checkpoint dict."""
    if "python_random_state" in checkpoint:
        random.setstate(checkpoint["python_random_state"])
    if "numpy_random_state" in checkpoint:
        np.random.set_state(checkpoint["numpy_random_state"])
    if "torch_random_state" in checkpoint:
        torch.set_rng_state(checkpoint["torch_random_state"])
    if torch.cuda.is_available() and checkpoint.get("cuda_random_state") is not None:
        torch.cuda.set_rng_state_all(checkpoint["cuda_random_state"])


def train_one_epoch(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer, scaler: GradScaler, device: torch.device, epoch: int, num_classes: int, use_amp: bool):
    """Train model for one epoch and return mean loss and Dice scores."""
    model.train()
    total_loss = 0.0
    total_intersections = torch.zeros(num_classes - 1, dtype=torch.float64)
    total_denominators = torch.zeros(num_classes - 1, dtype=torch.float64)
    for batch_index, (images, targets) in enumerate(dataloader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, targets)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        intersections, denominators = update_dice_statistics(logits.detach(), targets, num_classes)
        total_intersections += intersections
        total_denominators += denominators
        total_loss += loss.item()
        if (batch_index + 1) % 10 == 0:
            _, current_dice = calculate_dice(total_intersections, total_denominators)
            current_loss = total_loss / (batch_index + 1)
            print(f"Epoch [{epoch}] Step [{batch_index + 1}/{len(dataloader)}] Loss: {current_loss:.4f} Dice: {current_dice:.4f}")
    mean_loss = total_loss / len(dataloader)
    dice_scores, mean_dice = calculate_dice(total_intersections, total_denominators)
    return mean_loss, mean_dice, dice_scores


def validate_one_epoch(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device, num_classes: int, use_amp: bool):
    """Validate model for one epoch and return mean loss and Dice scores."""
    model.eval()
    total_loss = 0.0
    total_intersections = torch.zeros(num_classes - 1, dtype=torch.float64)
    total_denominators = torch.zeros(num_classes - 1, dtype=torch.float64)
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with autocast(enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, targets)
            intersections, denominators = update_dice_statistics(logits, targets, num_classes)
            total_intersections += intersections
            total_denominators += denominators
            total_loss += loss.item()
    mean_loss = total_loss / len(dataloader)
    dice_scores, mean_dice = calculate_dice(total_intersections, total_denominators)
    return mean_loss, mean_dice, dice_scores


def main():
    """Main training loop: setup datasets, model, optimizer, train/validate, save checkpoints."""
    data_dir = "/root/autodl-tmp/brats_raw/training_extracted/MICCAI-LH-BraTS2025-MET-Challenge-Training"
    patch_size = (128, 128, 128)
    validation_ratio = 0.2
    batch_size = 8
    num_workers = 14
    prefetch_factor = 1
    in_channels = 4
    num_classes = 5
    base_channels = 16
    num_epochs = 75
    learning_rate = 1e-4
    weight_decay = 1e-5
    seed = 2026
    resume_training = True
    class_names = ("NETC", "SNFH", "ET", "RC")
    checkpoint_dir = Path("./checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    curve_dir = Path("./training_curves")
    curve_dir.mkdir(parents=True, exist_ok=True)
    last_checkpoint_path = checkpoint_dir / "last_training_checkpoint.pth"
    best_model_path = checkpoint_dir / "best_attention_unet_3d.pth"
    final_model_path = checkpoint_dir / "final_attention_unet_3d.pth"
    history_csv_path = curve_dir / "training_history.csv"
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"Device: {device}")
    print(f"batch_size:{batch_size}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    training_base_dataset = BraTSDataset(data_dir=data_dir, patch_size=patch_size, training=True, with_label=True, class_crop_probability=0.85)
    validation_base_dataset = BraTSDataset(data_dir=data_dir, patch_size=patch_size, training=False, with_label=True, class_crop_probability=0.0)
    if training_base_dataset.patient_dirs != validation_base_dataset.patient_dirs:
        raise RuntimeError("Training and validation patient lists do not match")
    dataset_size = len(training_base_dataset)
    patient_ids = [patient_dir.name for patient_dir in training_base_dataset.patient_dirs]
    resume_checkpoint = None
    if resume_training and last_checkpoint_path.exists():
        print(f"Loading checkpoint: {last_checkpoint_path}")
        resume_checkpoint = torch.load(last_checkpoint_path, map_location="cpu")
        saved_patient_ids = resume_checkpoint.get("patient_ids")
        if saved_patient_ids is not None and saved_patient_ids != patient_ids:
            raise RuntimeError("Dataset patient list has changed since the checkpoint was created")
        training_indices = resume_checkpoint["training_indices"]
        validation_indices = resume_checkpoint["validation_indices"]
    else:
        indices = np.arange(dataset_size)
        random_generator = np.random.default_rng(seed)
        random_generator.shuffle(indices)
        validation_size = max(1, int(dataset_size * validation_ratio))
        validation_indices = indices[:validation_size].tolist()
        training_indices = indices[validation_size:].tolist()
    training_dataset = Subset(training_base_dataset, training_indices)
    validation_dataset = Subset(validation_base_dataset, validation_indices)
    print(f"Valid patients: {dataset_size}")
    print(f"Training patients: {len(training_dataset)}")
    print(f"Validation patients: {len(validation_dataset)}")
    print("Balanced class-aware patch sampling enabled")
    dataloader_generator = torch.Generator()
    if resume_checkpoint is not None and "dataloader_generator_state" in resume_checkpoint:
        dataloader_generator.set_state(resume_checkpoint["dataloader_generator_state"])
    else:
        dataloader_generator.manual_seed(seed)
    training_dataloader = DataLoader(dataset=training_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=use_amp, persistent_workers=num_workers > 0, prefetch_factor=prefetch_factor, worker_init_fn=seed_worker, generator=dataloader_generator)
    validation_dataloader = DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=use_amp, persistent_workers=num_workers > 0, prefetch_factor=prefetch_factor, worker_init_fn=seed_worker)
    model = AttentionUNet(in_channels=in_channels, num_classes=num_classes, base_channels=base_channels)
    model = model.to(device)
    criterion = DiceCELoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode="max", factor=0.5, patience=4)
    scaler = GradScaler(enabled=use_amp)
    start_epoch = 1
    best_validation_dice = -1.0
    history = create_empty_history()
    if resume_checkpoint is not None:
        model.load_state_dict(resume_checkpoint["model_state_dict"])
        optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
        move_optimizer_to_device(optimizer, device)
        scheduler.load_state_dict(resume_checkpoint["scheduler_state_dict"])
        scaler.load_state_dict(resume_checkpoint["scaler_state_dict"])
        start_epoch = resume_checkpoint["epoch"] + 1
        best_validation_dice = resume_checkpoint.get("best_validation_dice", -1.0)
        history = resume_checkpoint.get("history", create_empty_history())
        restore_random_states(resume_checkpoint)
        print(f"Checkpoint epoch: {resume_checkpoint['epoch']}")
        print(f"Resume from epoch: {start_epoch}")
        print(f"Best validation Dice: {best_validation_dice:.4f}")
        if len(history["epoch"]) > 0:
            save_training_history(history, history_csv_path)
            plot_training_history(history, class_names, curve_dir)
    if start_epoch > num_epochs:
        print(f"Training has already reached epoch {start_epoch - 1}")
        print(f"Configured total epochs: {num_epochs}")
        return
    for epoch in range(start_epoch, num_epochs + 1):
        epoch_start_time = time.time()
        current_learning_rate = optimizer.param_groups[0]["lr"]
        print("=" * 60)
        print(f"Epoch [{epoch}/{num_epochs}]")
        print(f"Learning rate: {current_learning_rate:.8f}")
        training_loss, training_dice, training_class_dice = train_one_epoch(model=model, dataloader=training_dataloader, criterion=criterion, optimizer=optimizer, scaler=scaler, device=device, epoch=epoch, num_classes=num_classes, use_amp=use_amp)
        validation_loss, validation_dice, validation_class_dice = validate_one_epoch(model=model, dataloader=validation_dataloader, criterion=criterion, device=device, num_classes=num_classes, use_amp=use_amp)
        scheduler.step(validation_dice)
        print(f"Train Loss: {training_loss:.4f}")
        print(f"Train Mean Dice: {training_dice:.4f}")
        for class_name, dice_score in zip(class_names, training_class_dice):
            print(f"Train {class_name} Dice: {dice_score:.4f}")
        print(f"Validation Loss: {validation_loss:.4f}")
        print(f"Validation Mean Dice: {validation_dice:.4f}")
        for class_name, dice_score in zip(class_names, validation_class_dice):
            print(f"Validation {class_name} Dice: {dice_score:.4f}")
        epoch_minutes = (time.time() - epoch_start_time) / 60.0
        print(f"Epoch time: {epoch_minutes:.2f} minutes")
        history["epoch"].append(epoch)
        history["learning_rate"].append(current_learning_rate)
        history["train_loss"].append(training_loss)
        history["validation_loss"].append(validation_loss)
        history["train_mean_dice"].append(training_dice)
        history["validation_mean_dice"].append(validation_dice)
        history["epoch_minutes"].append(epoch_minutes)
        for class_index, class_name in enumerate(class_names):
            history[f"train_{class_name}_dice"].append(training_class_dice[class_index])
            history[f"validation_{class_name}_dice"].append(validation_class_dice[class_index])
        save_training_history(history, history_csv_path)
        plot_training_history(history, class_names, curve_dir)
        print(f"Training curves updated: {curve_dir}")
        if validation_dice > best_validation_dice:
            best_validation_dice = validation_dice
            best_model_data = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "validation_loss": validation_loss,
                "validation_dice": validation_dice,
                "validation_class_dice": validation_class_dice,
                "num_classes": num_classes,
                "in_channels": in_channels,
                "base_channels": base_channels,
                "patch_size": patch_size
            }
            atomic_torch_save(best_model_data, best_model_path)
            print(f"Best model saved: {best_model_path}")
        training_checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_validation_dice": best_validation_dice,
            "history": history,
            "training_indices": training_indices,
            "validation_indices": validation_indices,
            "patient_ids": patient_ids,
            "dataloader_generator_state": dataloader_generator.get_state(),
            "python_random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
            "torch_random_state": torch.get_rng_state(),
            "cuda_random_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "num_classes": num_classes,
            "in_channels": in_channels,
            "base_channels": base_channels,
            "patch_size": patch_size,
            "batch_size": batch_size
        }
        atomic_torch_save(training_checkpoint, last_checkpoint_path)
        print(f"Training checkpoint saved: {last_checkpoint_path}")
    final_model_data = {
        "epoch": num_epochs,
        "model_state_dict": model.state_dict(),
        "num_classes": num_classes,
        "in_channels": in_channels,
        "base_channels": base_channels,
        "patch_size": patch_size
    }
    atomic_torch_save(final_model_data, final_model_path)
    print("Training completed.")
    print(f"Best validation Dice: {best_validation_dice:.4f}")
    print(f"Best model: {best_model_path}")
    print(f"Final model: {final_model_path}")
    print(f"Training checkpoint: {last_checkpoint_path}")
    print(f"Training curves: {curve_dir}")
    print(f"Training history: {history_csv_path}")


if __name__ == "__main__":
    time0 = time.time()
    print(time0)
    main()
    time1 = time.time()
    print(f"run time is {time1-time0}s")
