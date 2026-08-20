"""
Automated unit and integration tests for Bone Fracture Detection pipeline.
"""

import os
import unittest
import numpy as np
import torch
from PIL import Image

from src.utils import generate_synthetic_bone_xray, apply_clahe
from src.model import build_model
from src.gradcam import GradCAM, overlay_gradcam
from src.predictor import BoneFracturePredictor


class TestBoneFracturePipeline(unittest.TestCase):

    def test_synthetic_image_generation(self):
        """Test generating normal and fractured synthetic bone X-rays."""
        img_normal = generate_synthetic_bone_xray(is_fractured=False, size=(128, 128))
        img_fractured = generate_synthetic_bone_xray(is_fractured=True, size=(128, 128))

        self.assertIsInstance(img_normal, Image.Image)
        self.assertIsInstance(img_fractured, Image.Image)
        self.assertEqual(img_normal.size, (128, 128))
        self.assertEqual(img_fractured.size, (128, 128))

    def test_clahe_enhancement(self):
        """Test CLAHE contrast enhancement."""
        test_np = (np.random.rand(128, 128, 3) * 255).astype(np.uint8)
        enhanced = apply_clahe(test_np)
        self.assertEqual(enhanced.shape, (128, 128, 3))
        self.assertEqual(enhanced.dtype, np.uint8)

    def test_model_forward_and_gradcam(self):
        """Test model creation, forward pass, and Grad-CAM gradient generation."""
        model = build_model(architecture="resnet18", num_classes=2, pretrained=False)
        model.eval()

        dummy_tensor = torch.randn(1, 3, 224, 224, requires_grad=True)
        target_layer = model.get_target_layer_for_gradcam()

        gradcam = GradCAM(model, target_layer)
        cam = gradcam.generate_cam(dummy_tensor, target_class=1)

        self.assertEqual(cam.shape, (7, 7)) # Spatial dims of resnet18 layer4
        self.assertTrue(0.0 <= np.min(cam) <= np.max(cam) <= 1.0)

        # Test overlay
        orig_np = np.zeros((224, 224, 3), dtype=np.uint8)
        overlaid, heatmap, bboxes = overlay_gradcam(orig_np, cam)
        self.assertEqual(overlaid.shape, (224, 224, 3))
        self.assertEqual(heatmap.shape, (224, 224, 3))


if __name__ == "__main__":
    unittest.main()
