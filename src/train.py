"""
Colab-Optimized Mixed-Precision Training Pipeline for ResNet-101 and DenseNet-201.
Implements sequential model training, early stopping, and automatic memory cleanup.
"""

import os
import gc
import json
import time
import argparse
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.amp import autocast, GradScaler

from dataset import create_dataloaders
from models import build_resnet101, build_densenet201, freeze_backbone, unfreeze_backbone

def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device
) -> Tuple[float, float]:
    """Train for a single epoch with Automatic Mixed Precision (AMP)."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, targets, _ in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        use_cuda = (device.type == "cuda")
        with autocast(device_type=device.type, enabled=use_cuda):
            outputs = model(images)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(targets).sum().item()
        total += targets.size(0)

    epoch_loss = total_loss / (total + 1e-9)
    epoch_acc = correct / (total + 1e-9)
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """Evaluate loss and accuracy on validation or test loader."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    use_cuda = (device.type == "cuda")
    for images, targets, _ in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with autocast(device_type=device.type, enabled=use_cuda):
            outputs = model(images)
            loss = criterion(outputs, targets)

        total_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(targets).sum().item()
        total += targets.size(0)

    epoch_loss = total_loss / (total + 1e-9)
    epoch_acc = correct / (total + 1e-9)
    return epoch_loss, epoch_acc


def train_model(
    model_name: str,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    num_classes: int = 2,
    epochs: int = 20,
    lr: float = 1e-4,
    weight_decay: float = 1e-2,
    checkpoint_dir: str = "./checkpoints",
    device: Optional[torch.device] = None
) -> Tuple[nn.Module, Dict]:
    """
    Train and fine-tune a single backbone with Early Stopping and Checkpointing.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(checkpoint_dir, exist_ok=True)
    best_weight_path = os.path.join(checkpoint_dir, f"best_{model_name}.pth")

    print("\n" + "=" * 70)
    print(f"STARTING TRAINING: {model_name.upper()} on {device}")
    print("=" * 70)

    # Initialize model
    if "resnet" in model_name.lower():
        model = build_resnet101(num_classes=num_classes, pretrained=True)
    elif "densenet" in model_name.lower():
        model = build_densenet201(num_classes=num_classes, pretrained=True)
    else:
        raise ValueError(f"Unknown architecture: {model_name}")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    scaler = GradScaler(device.type, enabled=(device.type == "cuda"))

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
        "best_epoch": 0, "best_val_loss": float("inf")
    }

    best_val_loss = float("inf")
    patience = 6
    patience_counter = 0

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss, val_acc = evaluate_epoch(model, val_loader, criterion, device)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        epoch_time = time.time() - epoch_start
        print(f"Epoch [{epoch:2d}/{epochs:2d}] ({epoch_time:.1f}s) | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc*100:5.2f}% | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc*100:5.2f}% | LR: {current_lr:.1e}")

        # Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            history["best_epoch"] = epoch
            history["best_val_loss"] = best_val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc
            }, best_weight_path)
            print(f"  --> Checkpoint saved: {best_weight_path} (Val Loss: {best_val_loss:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch} (no improvement for {patience} epochs)")
                break

        # Colab GPU memory cleanup
        if device.type == "cuda":
            torch.cuda.empty_cache()

    total_time = time.time() - start_time
    print(f"Finished training {model_name} in {total_time/60:.2f} minutes.")
    print(f"Best Val Loss: {history['best_val_loss']:.4f} at epoch {history['best_epoch']}")

    # Load best weights before returning
    checkpoint = torch.load(best_weight_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, history


def run_sequential_pipeline(
    manifest_dir: str = "./manifests",
    data_root: str = ".",
    checkpoint_dir: str = "./checkpoints",
    epochs: int = 20,
    batch_size: int = 16,
    lr: float = 1e-4
):
    """
    Executes sequential training of ResNet-101 and DenseNet-201 with explicit VRAM purging.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing Sequential Training on: {device}")

    train_loader, val_loader, test_loader = create_dataloaders(
        manifest_dir=manifest_dir,
        data_root=data_root,
        batch_size=batch_size
    )

    # 1. Train ResNet-101
    model_resnet, resnet_history = train_model(
        model_name="resnet101",
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=lr,
        checkpoint_dir=checkpoint_dir,
        device=device
    )

    # Clear VRAM completely before DenseNet
    del model_resnet
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print("Purged ResNet-101 from memory to ensure zero VRAM spikes.")

    # 2. Train DenseNet-201
    model_densenet, dense_history = train_model(
        model_name="densenet201",
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=lr,
        checkpoint_dir=checkpoint_dir,
        device=device
    )

    del model_densenet
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Save training history summary
    history_file = os.path.join(checkpoint_dir, "training_history.json")
    with open(history_file, "w") as f:
        json.dump({"resnet101": resnet_history, "densenet201": dense_history}, f, indent=2)
    print(f"\nSaved combined training history to: {history_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ResNet-101 and DenseNet-201 sequentially")
    parser.add_argument("--manifest_dir", type=str, default="./manifests")
    parser.add_argument("--data_root", type=str, default=".")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    run_sequential_pipeline(
        manifest_dir=args.manifest_dir,
        data_root=args.data_root,
        checkpoint_dir=args.checkpoint_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )
