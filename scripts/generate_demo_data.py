"""
Setup and Demo Data Generation Script for Bone Fracture Detection.
Generates sample X-rays, builds dataset directory hierarchy, and pre-trains a starter model.
"""

import os
import sys

# Ensure root directory is in Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils import generate_dataset_structure
from src.train import train_model
from src.evaluate import evaluate_model


def setup_project():
    print("=" * 60)
    print(" Bone Fracture Detection - Project Setup & Initialization")
    print("=" * 60)

    # 1. Generate synthetic dataset structure and demo X-rays
    print("\n[Step 1/3] Generating synthetic medical X-ray dataset...")
    generate_dataset_structure(
        base_dir="data",
        num_train_per_class=40,
        num_val_per_class=10,
        num_test_per_class=10
    )

    # 2. Train baseline deep learning model
    print("\n[Step 2/3] Training baseline deep learning model (ResNet18 / ResNet50)...")
    train_model(
        data_dir="data",
        architecture="resnet18",
        epochs=8,
        batch_size=16,
        learning_rate=3e-4,
        save_dir="models",
        patience=5
    )

    # 3. Evaluate trained model
    print("\n[Step 3/3] Evaluating baseline model and generating diagnostic plots...")
    evaluate_model(
        model_path="models/best_model.pth",
        data_dir="data",
        split="val",
        output_dir="models/evaluation"
    )

    print("\n" + "=" * 60)
    print(" [SUCCESS] Project is fully initialized and ready to run!")
    print(" - Run Web App: streamlit run app.py")
    print(" - Run REST API: uvicorn api:app --reload --port 8000")
    print("=" * 60)


if __name__ == "__main__":
    setup_project()
