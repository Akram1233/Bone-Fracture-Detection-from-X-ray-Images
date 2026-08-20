"""
Unified Prediction & Explainability Engine for Bone Fracture Detection.
"""

import os
from typing import Union, Dict, Any, Optional
from PIL import Image
import numpy as np
import torch
import cv2
from torchvision import transforms

from src.model import build_model
from src.gradcam import GradCAM, overlay_gradcam, draw_bounding_boxes, COLORMAP_DICT
from src.utils import apply_clahe, generate_diagnostic_report
from src.dataset import NORMALIZE_MEAN, NORMALIZE_STD


class BoneFracturePredictor:
    """
    High-level predictor that handles image preprocessing, deep learning inference,
    Grad-CAM heatmap generation, and ROI localization.
    """
    def __init__(self, model_path: str = "models/best_model.pth", device: Optional[str] = None):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model_path = model_path
        self.classes = ["normal", "fractured"]
        self.architecture = "resnet50"
        self.model = None
        self.gradcam = None
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
        ])

        self._load_model()

    def _load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model checkpoint '{self.model_path}' not found! "
                "Please run 'python scripts/generate_demo_data.py' or train a model first."
            )

        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.architecture = checkpoint.get("architecture", "resnet50")
        self.classes = checkpoint.get("classes", ["normal", "fractured"])

        self.model = build_model(
            architecture=self.architecture,
            num_classes=len(self.classes),
            pretrained=False,
            device=self.device
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        # Initialize Grad-CAM with target layer
        target_layer = self.model.get_target_layer_for_gradcam()
        self.gradcam = GradCAM(self.model, target_layer)

    def predict(
        self,
        image_input: Union[str, Image.Image, np.ndarray],
        enhance_contrast: bool = True,
        cam_threshold: float = 0.25,
        colormap_name: str = "TURBO (High Contrast)",
        alpha: float = 0.45
    ) -> Dict[str, Any]:
        """
        Run fracture detection diagnosis and generate Grad-CAM visualization.
        """
        # 1. Normalize input image to PIL RGB and Numpy
        if isinstance(image_input, str):
            orig_pil = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, np.ndarray):
            orig_pil = Image.fromarray(image_input).convert("RGB")
        else:
            orig_pil = image_input.convert("RGB")

        orig_np = np.array(orig_pil)

        # 2. Contrast Enhancement (CLAHE)
        if enhance_contrast:
            enhanced_np = apply_clahe(orig_np)
            enhanced_pil = Image.fromarray(enhanced_np)
        else:
            enhanced_np = orig_np.copy()
            enhanced_pil = orig_pil.copy()

        # 3. Model Preprocessing & Tensor
        tensor_img = self.transform(enhanced_pil).unsqueeze(0).to(self.device) # (1, 3, 224, 224)

        # 4. Inference & Grad-CAM Heatmap
        tensor_img.requires_grad = True
        logits = self.model(tensor_img)
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]
        pred_idx = int(np.argmax(probs))
        pred_label = self.classes[pred_idx]
        confidence = float(probs[pred_idx])

        # Target class for GradCAM
        fracture_idx = self.classes.index("fractured") if "fractured" in self.classes else pred_idx
        cam_mask = self.gradcam.generate_cam(tensor_img, target_class=fracture_idx)

        # 5. Overlays and Bounding Boxes
        cv2_colormap = COLORMAP_DICT.get(colormap_name, cv2.COLORMAP_TURBO)
        overlaid_np, heatmap_np, bounding_boxes = overlay_gradcam(
            original_image=enhanced_np,
            cam_mask=cam_mask,
            colormap=cv2_colormap,
            alpha=alpha,
            threshold=cam_threshold
        )

        box_color = (255, 65, 84) if pred_label == "fractured" else (50, 205, 50)
        localized_np = draw_bounding_boxes(
            overlaid_np,
            bounding_boxes,
            label="Fracture Zone" if pred_label == "fractured" else "Anatomical ROI",
            box_color=box_color
        )

        # 6. Generate Clinical Diagnostic Report
        report = generate_diagnostic_report(
            prediction=pred_label,
            confidence=confidence,
            bounding_boxes=bounding_boxes,
            model_name=self.architecture.upper()
        )

        return {
            "prediction": pred_label,
            "confidence": confidence,
            "probabilities": {self.classes[i]: float(probs[i]) for i in range(len(self.classes))},
            "is_fractured": (pred_label.lower() == "fractured"),
            "original_image": orig_pil,
            "enhanced_image": enhanced_pil,
            "heatmap_image": Image.fromarray(heatmap_np),
            "gradcam_overlay": Image.fromarray(overlaid_np),
            "localized_image": Image.fromarray(localized_np),
            "bounding_boxes": bounding_boxes,
            "report": report,
            "architecture": self.architecture,
            "cam_mask": cam_mask,
            "enhanced_np": enhanced_np
        }
