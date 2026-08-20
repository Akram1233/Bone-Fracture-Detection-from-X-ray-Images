"""
Training pipeline for Bone Fracture Detection Deep Learning model.
"""

import os
import time
import argparse
import json
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.model import build_model
from src.dataset import get_dataloaders


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels.data).item()
        total += labels.size(0)

    epoch_loss = running_loss / max(total, 1)
    epoch_acc = correct / max(total, 1)
    return epoch_loss, epoch_acc


def validate_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validation", leave=False):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data).item()
            total += labels.size(0)

    epoch_loss = running_loss / max(total, 1)
    epoch_acc = correct / max(total, 1)
    return epoch_loss, epoch_acc


def train_model(
    data_dir: str = "data",
    architecture: str = "resnet50",
    epochs: int = 15,
    batch_size: int = 16,
    learning_rate: float = 1e-4,
    save_dir: str = "models",
    patience: int = 7,
    progress_callback = None
) -> dict:
    """
    Main training function that runs the complete training loop, validation, and model saving.
    """
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training on device: {device} | Architecture: {architecture}")

    train_loader, val_loader, _ = get_dataloaders(data_dir, batch_size=batch_size)
    
    if len(train_loader.dataset) == 0:
        raise ValueError(f"No training data found in '{os.path.join(data_dir, 'train')}'. Please generate or add dataset.")

    model = build_model(architecture=architecture, num_classes=2, pretrained=True, device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "architecture": architecture,
        "epochs": epochs
    }

    best_val_acc = 0.0
    best_model_path = os.path.join(save_dir, "best_model.pth")
    epochs_no_improve = 0

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch [{epoch:02d}/{epochs:02d}] "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc*100:.2f}%")

        if progress_callback:
            progress_callback(epoch, epochs, train_loss, train_acc, val_loss, val_acc)

        # Check for improvement
        if val_acc > best_val_acc or epoch == 1:
            best_val_acc = val_acc
            epochs_no_improve = 0
            # Save checkpoint
            checkpoint = {
                "epoch": epoch,
                "architecture": architecture,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
                "classes": ["normal", "fractured"]
            }
            torch.save(checkpoint, best_model_path)
            print(f" -> Checkpoint saved! Best Val Acc: {best_val_acc*100:.2f}%")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"[!] Early stopping triggered at epoch {epoch}.")
                break

    elapsed = time.time() - start_time
    history["training_time_seconds"] = round(elapsed, 2)
    history["best_val_acc"] = best_val_acc

    # Save history json
    with open(os.path.join(save_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=4)

    # Plot and save curves
    plot_training_curves(history, os.path.join(save_dir, "learning_curves.png"))

    print(f"[DONE] Training finished in {elapsed:.1f}s. Best Model: {best_model_path}")
    return history


def plot_training_curves(history: dict, save_path: str):
    """Saves loss and accuracy comparison plots."""
    epochs = range(1, len(history["train_loss"]) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Loss Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], 'b-o', label='Train Loss')
    plt.plot(epochs, history["val_loss"], 'r-o', label='Val Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Cross-Entropy Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Accuracy Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, [a * 100 for a in history["train_acc"]], 'b-o', label='Train Accuracy')
    plt.plot(epochs, [a * 100 for a in history["val_acc"]], 'r-o', label='Val Accuracy')
    plt.title('Accuracy Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Bone Fracture Detection Model")
    parser.add_argument("--data_dir", type=str, default="data", help="Path to data directory")
    parser.add_argument("--arch", type=str, default="resnet50", choices=["resnet50", "resnet34", "resnet18", "densenet121", "efficientnet_b0", "custom_cnn"])
    parser.add_argument("--epochs", type=int, default=15, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--save_dir", type=str, default="models", help="Directory to save models")
    parser.add_argument("--patience", type=int, default=7, help="Early stopping patience")
    
    args = parser.parse_args()
    train_model(
        data_dir=args.data_dir,
        architecture=args.arch,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        save_dir=args.save_dir,
        patience=args.patience
    )
