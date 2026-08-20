"""
Dataset loading and data augmentation pipeline for Bone Fracture X-ray classification.
"""

import os
from typing import Tuple, List, Optional
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# Standard ImageNet normalization parameters
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]


def get_train_transforms(img_size: int = 224) -> transforms.Compose:
    """
    Data augmentations tailored for medical bone X-ray imaging.
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
    ])


def get_val_transforms(img_size: int = 224) -> transforms.Compose:
    """
    Deterministic preprocessing transforms for validation and testing.
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
    ])


class BoneFractureDataset(Dataset):
    """
    PyTorch Dataset loading X-ray images from disk.
    Directory structure:
        root_dir/
            fractured/
            normal/
    """
    def __init__(self, root_dir: str, transform: Optional[transforms.Compose] = None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples: List[Tuple[str, int]] = []
        self.classes = ["normal", "fractured"]
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        if os.path.exists(root_dir):
            for cls_name in self.classes:
                cls_dir = os.path.join(root_dir, cls_name)
                if not os.path.exists(cls_dir):
                    # Also check capitalized or alternate casing
                    for d in os.listdir(root_dir):
                        if d.lower() == cls_name.lower():
                            cls_dir = os.path.join(root_dir, d)
                            break

                if os.path.exists(cls_dir):
                    for fname in os.listdir(cls_dir):
                        if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                            self.samples.append((os.path.join(cls_dir, fname), self.class_to_idx[cls_name]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


def get_dataloaders(
    data_dir: str,
    batch_size: int = 16,
    img_size: int = 224,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """
    Creates train, val, and optional test DataLoaders.
    """
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    test_dir = os.path.join(data_dir, "test")

    train_dataset = BoneFractureDataset(train_dir, transform=get_train_transforms(img_size))
    val_dataset = BoneFractureDataset(val_dir, transform=get_val_transforms(img_size))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = None
    if os.path.exists(test_dir):
        test_dataset = BoneFractureDataset(test_dir, transform=get_val_transforms(img_size))
        if len(test_dataset) > 0:
            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers
            )

    return train_loader, val_loader, test_loader
