"""
Grad-CAM (Gradient-weighted Class Activation Mapping) and Visual Explainability
for Bone Fracture Detection.
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


COLORMAP_DICT = {
    "JET (Classic Medical)": cv2.COLORMAP_JET,
    "TURBO (High Contrast)": cv2.COLORMAP_TURBO,
    "HOT (Thermal Glow)": cv2.COLORMAP_HOT,
    "MAGMA (Deep Heat)": cv2.COLORMAP_MAGMA,
    "VIRIDIS (Perceptual)": cv2.COLORMAP_VIRIDIS,
    "PLASMA (Neon Radiation)": cv2.COLORMAP_PLASMA,
    "INFERNO (Intense Trauma)": cv2.COLORMAP_INFERNO,
    "BONE (Radiological Grayscale)": cv2.COLORMAP_BONE,
}


class GradCAM:
    """
    Computes Grad-CAM heatmaps for a given CNN model and target layer.
    """
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_class: int = None
    ) -> np.ndarray:
        """
        Generate raw Grad-CAM heatmap for the input tensor.
        """
        self.model.eval()
        self.model.zero_grad()

        # Forward pass
        logits = self.model(input_tensor)

        if target_class is None:
            target_class = torch.argmax(logits, dim=1).item()

        # Backward pass for target class score
        score = logits[0, target_class]
        score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Grad-CAM hooks failed to capture gradients/activations.")

        # Global average pooling of gradients
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True) # (1, C, 1, 1)

        # Weighted combination of activation maps
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True) # (1, 1, H', W')
        cam = F.relu(cam) # ReLU to keep only features that contribute positively

        # Normalize CAM
        cam = cam.squeeze().cpu().numpy()
        if np.max(cam) - np.min(cam) > 1e-8:
            cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))
        else:
            cam = np.zeros_like(cam)

        return cam


def overlay_gradcam(
    original_image: np.ndarray,
    cam_mask: np.ndarray,
    colormap: int = cv2.COLORMAP_JET,
    alpha: float = 0.5,
    threshold: float = 0.2
) -> tuple[np.ndarray, np.ndarray, list]:
    """
    Overlays Grad-CAM heatmap onto the original image and computes bounding boxes for ROI.
    """
    h, w = original_image.shape[:2]

    # Resize CAM to match original image dimensions
    cam_resized = cv2.resize(cam_mask, (w, h), interpolation=cv2.INTER_LINEAR)
    
    # Convert to 8-bit heatmap
    heatmap_uint8 = np.uint8(255 * cam_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Superimpose heatmap onto original image
    overlaid = (alpha * heatmap_colored.astype(float) + (1.0 - alpha) * original_image.astype(float)).astype(np.uint8)

    # Detect high activation regions (Potential Fracture ROI Bounding Boxes)
    binary_mask = (cam_resized >= threshold).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bounding_boxes = []
    min_area = (h * w) * 0.005 # Filter out noise specs

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= min_area:
            x, y, bw, bh = cv2.boundingRect(cnt)
            # Calculate mean activation in this ROI
            roi_cam = cam_resized[y:y+bh, x:x+bw]
            mean_act = float(np.mean(roi_cam))
            bounding_boxes.append({
                "x": int(x),
                "y": int(y),
                "w": int(bw),
                "h": int(bh),
                "activation": round(mean_act, 3)
            })

    return overlaid, heatmap_colored, bounding_boxes


def draw_bounding_boxes(
    image: np.ndarray,
    bounding_boxes: list,
    label: str = "Suspicious Zone",
    box_color: tuple = (255, 60, 60)
) -> np.ndarray:
    """
    Draws highlighted futuristic bounding boxes with glowing tag badges.
    """
    img_with_boxes = image.copy()
    for i, box in enumerate(bounding_boxes, 1):
        x, y, w, h = box["x"], box["y"], box["w"], box["h"]
        act = box["activation"]
        
        # Outer neon corner brackets
        line_len = min(w, h) // 4
        thickness = 2
        
        # Draw main rectangle
        cv2.rectangle(img_with_boxes, (x, y), (x + w, y + h), box_color, thickness)
        
        # Tag background
        tag = f"ROI #{i}: {int(act*100)}% Alert"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        tag_y = max(y - 6, th + 6)
        
        # Draw dark contrast box for text
        cv2.rectangle(img_with_boxes, (x, tag_y - th - 4), (x + tw + 6, tag_y + 2), (15, 23, 42), -1)
        cv2.rectangle(img_with_boxes, (x, tag_y - th - 4), (x + tw + 6, tag_y + 2), box_color, 1)
        cv2.putText(img_with_boxes, tag, (x + 3, tag_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        
    return img_with_boxes
