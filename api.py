"""
FastAPI REST API for Headless Bone Fracture Detection & Grad-CAM Inference.
"""

import os
import io
import base64
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
from PIL import Image

from src.predictor import BoneFracturePredictor

app = FastAPI(
    title="Bone Fracture Detection API",
    description="Deep Learning REST API with Grad-CAM explainability for medical X-ray fracture diagnosis.",
    version="1.0.0"
)

# Enable CORS for cross-origin frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy loading of predictor
_predictor: Optional[BoneFracturePredictor] = None


def get_predictor() -> BoneFracturePredictor:
    global _predictor
    if _predictor is None:
        model_path = "models/best_model.pth"
        if not os.path.exists(model_path):
            raise HTTPException(
                status_code=503,
                detail="Model checkpoint not found. Run 'python generate_demo_data.py' first."
            )
        _predictor = BoneFracturePredictor(model_path=model_path)
    return _predictor


@app.get("/")
def root():
    return {
        "service": "Bone Fracture Detection AI API",
        "status": "online",
        "endpoints": {
            "health": "/health",
            "predict": "/predict (POST multipart/form-data)",
            "model_info": "/model-info"
        }
    }


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.get("/model-info")
def model_info():
    predictor = get_predictor()
    return {
        "architecture": predictor.architecture,
        "classes": predictor.classes,
        "device": str(predictor.device)
    }


@app.post("/predict")
async def predict_xray(
    file: UploadFile = File(...),
    enhance_contrast: bool = Query(True, description="Apply CLAHE contrast enhancement"),
    cam_threshold: float = Query(0.25, description="Threshold for localization bounding boxes"),
    include_base64_images: bool = Query(True, description="Return base64 encoded Grad-CAM images")
):
    """
    Receives an X-ray image file and returns deep learning fracture classification and Grad-CAM ROI.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image (PNG, JPG, JPEG).")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to decode image: {e}")

    predictor = get_predictor()
    results = predictor.predict(
        image_input=image,
        enhance_contrast=enhance_contrast,
        cam_threshold=cam_threshold
    )

    response = {
        "filename": file.filename,
        "prediction": results["prediction"],
        "is_fractured": results["is_fractured"],
        "confidence": round(results["confidence"], 4),
        "probabilities": {k: round(v, 4) for k, v in results["probabilities"].items()},
        "suspicious_rois": results["bounding_boxes"],
        "diagnostic_report": results["report"]
    }

    if include_base64_images:
        # Convert images to base64
        def pil_to_b64(pil_img):
            buf = io.BytesIO()
            pil_img.save(buf, format="JPEG")
            return base64.b64encode(buf.getvalue()).decode("utf-8")

        response["images"] = {
            "gradcam_overlay": pil_to_b64(results["gradcam_overlay"]),
            "heatmap": pil_to_b64(results["heatmap_image"]),
            "localized": pil_to_b64(results["localized_image"])
        }

    return response


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
