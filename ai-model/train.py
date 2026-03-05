"""Training script for SmartMine AI Safety Detection (ResNet-101)."""

import argparse
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Allow imports from sibling package when run as a script
sys.path.insert(0, str(Path(__file__).parent))
from models.resnet101 import get_model


# ---------------------------------------------------------------------------
# Data transforms
# ---------------------------------------------------------------------------

def build_transforms():
    """Return (train_transform, val_transform) as a tuple."""
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])

    return train_transform, val_transform


# ---------------------------------------------------------------------------
# Training loop helpers
# ---------------------------------------------------------------------------

def run_epoch(model, loader, criterion, optimizer, scaler, device, training: bool):
    """Run one full epoch and return (avg_loss, accuracy).

    Args:
        optimizer: Required when *training* is ``True``; pass ``None`` for
            validation/evaluation passes where no parameter updates occur.
    """
    model.train(training)
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            if training:
                optimizer.zero_grad()
                with autocast(enabled=scaler is not None):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            else:
                with autocast(enabled=scaler is not None):
                    outputs = model(images)
                    loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += images.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def plot_curves(train_losses, val_losses, train_accs, val_accs, save_path: str) -> None:
    """Save a two-panel loss + accuracy training curve figure."""
    epochs = range(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, train_losses, label="Train Loss")
    ax1.plot(epochs, val_losses, label="Val Loss")
    ax1.set_title("Loss over epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()

    ax2.plot(epochs, train_accs, label="Train Accuracy")
    ax2.plot(epochs, val_accs, label="Val Accuracy")
    ax2.set_title("Accuracy over epochs")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Training curves saved to {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train SmartMine ResNet-101 model")
    parser.add_argument("--data_dir", type=str, default="ai-model/dataset",
                        help="Root dataset directory (must contain train/ and val/ sub-dirs)")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--save_path", type=str,
                        default="ai-model/models/resnet101_smartmine.pth")
    return parser.parse_args()


def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ------------------------------------------------------------------
    # Datasets & loaders
    # ------------------------------------------------------------------
    train_transform, val_transform = build_transforms()

    train_dir = os.path.join(args.data_dir, "train")
    val_dir = os.path.join(args.data_dir, "val")

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

    class_names = train_dataset.classes
    print(f"Classes ({len(class_names)}): {class_names}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # ------------------------------------------------------------------
    # Model, loss, optimiser, scheduler
    # ------------------------------------------------------------------
    model = get_model(num_classes=args.num_classes, device=device)
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=1e-4,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler() if device == "cuda" else None

    # ------------------------------------------------------------------
    # Training loop with early stopping
    # ------------------------------------------------------------------
    best_val_loss = float("inf")
    patience = 7
    patience_counter = 0

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    print(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Train Acc':>9}  {'Val Loss':>8}  {'Val Acc':>7}  {'Time':>6}")
    print("-" * 60)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, scaler, device, training=True)
        vl_loss, vl_acc = run_epoch(model, val_loader, criterion, None, scaler, device, training=False)

        scheduler.step()

        train_losses.append(tr_loss)
        val_losses.append(vl_loss)
        train_accs.append(tr_acc)
        val_accs.append(vl_acc)

        elapsed = time.time() - t0
        print(f"{epoch:>6}  {tr_loss:>10.4f}  {tr_acc:>9.4f}  {vl_loss:>8.4f}  {vl_acc:>7.4f}  {elapsed:>5.1f}s")

        # Save best checkpoint
        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            patience_counter = 0
            os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
            torch.save({
                "epoch": epoch,
                "val_acc": vl_acc,
                "state_dict": model.state_dict(),
                "class_names": class_names,
            }, args.save_path)
            print(f"  → Best model saved (val_loss={vl_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch} epochs (patience={patience}).")
                break

    # ------------------------------------------------------------------
    # Final outputs
    # ------------------------------------------------------------------
    plot_curves(train_losses, val_losses, train_accs, val_accs, "training_curves.png")

    print("\n=== Training Summary ===")
    print(f"  Best Val Loss : {best_val_loss:.4f}")
    print(f"  Best Val Acc  : {max(val_accs):.4f}")
    print(f"  Model saved to: {args.save_path}")


if __name__ == "__main__":
    main()
