"""
Deep Learning Model Architecture for Bone Fracture Detection.
Supports ResNet (18, 34, 50), DenseNet121, EfficientNet-B0, and Custom CNN.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class CustomBoneCNN(nn.Module):
    """
    Lightweight, custom Convolutional Neural Network designed for X-ray feature extraction.
    """
    def __init__(self, num_classes: int = 2, dropout_rate: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), # 112x112
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), # 56x56

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), # 28x28

            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # 14x14
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x

    def get_target_layer_for_gradcam(self):
        # Return the last convolutional layer
        return self.features[-2]


class BoneFractureClassifier(nn.Module):
    """
    Transfer Learning classifier for Bone Fracture Detection.
    """
    def __init__(
        self,
        architecture: str = "resnet50",
        num_classes: int = 2,
        pretrained: bool = True,
        dropout_rate: float = 0.3
    ):
        super().__init__()
        self.architecture = architecture.lower()
        self.num_classes = num_classes

        if self.architecture == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            backbone = models.resnet50(weights=weights)
            in_features = backbone.fc.in_features
            backbone.fc = nn.Identity()
            self.backbone = backbone
            self.classifier = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_rate),
                nn.Linear(256, num_classes)
            )

        elif self.architecture == "resnet34":
            weights = models.ResNet34_Weights.DEFAULT if pretrained else None
            backbone = models.resnet34(weights=weights)
            in_features = backbone.fc.in_features
            backbone.fc = nn.Identity()
            self.backbone = backbone
            self.classifier = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_rate),
                nn.Linear(256, num_classes)
            )

        elif self.architecture == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            backbone = models.resnet18(weights=weights)
            in_features = backbone.fc.in_features
            backbone.fc = nn.Identity()
            self.backbone = backbone
            self.classifier = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, num_classes)
            )

        elif self.architecture == "densenet121":
            weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
            backbone = models.densenet121(weights=weights)
            in_features = backbone.classifier.in_features
            backbone.classifier = nn.Identity()
            self.backbone = backbone
            self.classifier = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_rate),
                nn.Linear(256, num_classes)
            )

        elif self.architecture == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            backbone = models.efficientnet_b0(weights=weights)
            in_features = backbone.classifier[1].in_features
            backbone.classifier = nn.Identity()
            self.backbone = backbone
            self.classifier = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_rate),
                nn.Linear(256, num_classes)
            )

        elif self.architecture == "custom_cnn":
            self.custom_model = CustomBoneCNN(num_classes=num_classes, dropout_rate=dropout_rate)
            self.backbone = None
            self.classifier = None

        else:
            raise ValueError(f"Unsupported architecture: {architecture}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.architecture == "custom_cnn":
            return self.custom_model(x)
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits

    def get_target_layer_for_gradcam(self):
        """
        Returns the layer to attach Grad-CAM hooks to based on the architecture.
        """
        if self.architecture.startswith("resnet"):
            # Last bottleneck layer of ResNet
            return self.backbone.layer4[-1]
        elif self.architecture == "densenet121":
            return self.backbone.features.denseblock4[-1]
        elif self.architecture == "efficientnet_b0":
            return self.backbone.features[-1]
        elif self.architecture == "custom_cnn":
            return self.custom_model.get_target_layer_for_gradcam()
        raise ValueError("Unknown target layer for architecture")


def build_model(
    architecture: str = "resnet50",
    num_classes: int = 2,
    pretrained: bool = True,
    device: torch.device = None
) -> nn.Module:
    """Factory function to build and move model to device."""
    model = BoneFractureClassifier(
        architecture=architecture,
        num_classes=num_classes,
        pretrained=pretrained
    )
    if device:
        model = model.to(device)
    return model
