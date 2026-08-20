"""
Model evaluation module generating Confusion Matrix, Classification Report,
and ROC-AUC metrics for Bone Fracture Detection.
"""

import os
import argparse
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    f1_score,
    accuracy_score
)

from src.model import build_model
from src.dataset import get_dataloaders


def evaluate_model(
    model_path: str = "models/best_model.pth",
    data_dir: str = "data",
    split: str = "val",
    output_dir: str = "models/evaluation"
) -> dict:
    """
    Evaluates the model on validation or test split and saves diagnostic plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)
    arch = checkpoint.get("architecture", "resnet50")
    classes = checkpoint.get("classes", ["normal", "fractured"])

    model = build_model(architecture=arch, num_classes=len(classes), pretrained=False, device=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    train_loader, val_loader, test_loader = get_dataloaders(data_dir, batch_size=16)
    
    loader = test_loader if split == "test" and test_loader is not None else val_loader

    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)

            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.numpy())
            all_probs.extend(probs[:, 1].cpu().numpy()) # Probability of class 'fractured'

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)

    # Compute metrics
    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, zero_division=0)
    
    # Handle single class edge case in test subset
    if len(np.unique(all_targets)) > 1:
        roc_auc = roc_auc_score(all_targets, all_probs)
    else:
        roc_auc = 1.0

    report = classification_report(all_targets, all_preds, target_names=classes, output_dict=True, zero_division=0)
    cm = confusion_matrix(all_targets, all_preds)

    # 1. Plot Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(f'Confusion Matrix ({split.capitalize()} Set)')
    plt.xlabel('Predicted Label')
    plt.ylabel('Ground Truth Label')
    plt.tight_layout()
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()

    # 2. Plot ROC Curve
    if len(np.unique(all_targets)) > 1:
        fpr, tpr, _ = roc_curve(all_targets, all_probs)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (1 - Specificity)')
        plt.ylabel('True Positive Rate (Sensitivity)')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        roc_path = os.path.join(output_dir, "roc_curve.png")
        plt.savefig(roc_path, dpi=300)
        plt.close()
    else:
        roc_path = None

    metrics_summary = {
        "architecture": arch,
        "split": split,
        "accuracy": round(acc, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "confusion_matrix_plot": cm_path,
        "roc_curve_plot": roc_path
    }

    with open(os.path.join(output_dir, "evaluation_metrics.json"), "w") as f:
        json.dump(metrics_summary, f, indent=4)

    print(f"\n===== EVALUATION SUMMARY ({split.upper()}) =====")
    print(f"Accuracy : {acc*100:.2f}%")
    print(f"F1-Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")
    print(f"Plots saved to: {output_dir}")
    print("==========================================\n")

    return metrics_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Bone Fracture Model")
    parser.add_argument("--model_path", type=str, default="models/best_model.pth")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"])
    parser.add_argument("--output_dir", type=str, default="models/evaluation")
    args = parser.parse_args()

    evaluate_model(
        model_path=args.model_path,
        data_dir=args.data_dir,
        split=args.split,
        output_dir=args.output_dir
    )
