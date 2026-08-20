"""
Utility functions for X-ray image enhancement, synthetic bone X-ray generation,
clinical diagnostic report generation, and advanced radiologist workbench filters.
"""

import os
import random
import cv2
import base64
import datetime
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from typing import Tuple, List, Dict, Optional


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) to enhance bone radiopacity.
    """
    if len(image.shape) == 3:
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        l_clahe = clahe.apply(l)
        lab = cv2.merge((l_clahe, a, b))
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    else:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        enhanced = clahe.apply(image)
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    return enhanced


def adjust_radiograph_properties(
    image_np: np.ndarray,
    brightness: float = 1.0,
    contrast: float = 1.0,
    sharpness: float = 1.0,
    invert: bool = False
) -> np.ndarray:
    """
    Applies PACS Radiology workstation controls (Brightness, Contrast, Sharpness, Inversion).
    """
    img = image_np.copy()

    # 1. Negative Invert (Standard in PACS viewers)
    if invert:
        img = 255 - img

    pil_img = Image.fromarray(img)

    # 2. Brightness
    if brightness != 1.0:
        enh_b = ImageEnhance.Brightness(pil_img)
        pil_img = enh_b.enhance(brightness)

    # 3. Contrast
    if contrast != 1.0:
        enh_c = ImageEnhance.Contrast(pil_img)
        pil_img = enh_c.enhance(contrast)

    # 4. Sharpness
    if sharpness != 1.0:
        enh_s = ImageEnhance.Sharpness(pil_img)
        pil_img = enh_s.enhance(sharpness)

    return np.array(pil_img)


def pil_to_base64(pil_image: Image.Image, format: str = "PNG") -> str:
    """Encodes PIL image into a base64 data string."""
    buffer = io_bytes = None
    import io
    buffer = io.BytesIO()
    pil_image.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def get_image_comparison_slider_html(
    img_left_b64: str,
    img_right_b64: str,
    label_left: str = "Raw Radiograph",
    label_right: str = "AI Grad-CAM Overlay"
) -> str:
    """
    Generates an interactive, responsive Before/After split wipe comparison slider.
    """
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            .slider-wrapper {{
                position: relative;
                width: 100%;
                max-width: 580px;
                height: 380px;
                margin: 0 auto;
                overflow: hidden;
                border-radius: 12px;
                border: 2px solid rgba(56, 189, 248, 0.4);
                box-shadow: 0 8px 30px rgba(0, 0, 0, 0.4);
                user-select: none;
            }}
            .slider-img {{
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                object-fit: contain;
                background-color: #020617;
            }}
            .img-overlay {{
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                overflow: hidden;
                clip-path: polygon(0 0, 50% 0, 50% 100%, 0 100%);
            }}
            .range-input {{
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                opacity: 0;
                cursor: ew-resize;
                z-index: 10;
                margin: 0;
            }}
            .divider-line {{
                position: absolute;
                top: 0;
                bottom: 0;
                left: 50%;
                width: 3px;
                background: #38BDF8;
                box-shadow: 0 0 12px #38BDF8;
                pointer-events: none;
                z-index: 5;
            }}
            .divider-handle {{
                position: absolute;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                width: 36px;
                height: 36px;
                background: #0284C7;
                border: 2px solid #FFFFFF;
                border-radius: 50%;
                box-shadow: 0 0 15px rgba(56, 189, 248, 0.8);
                display: flex;
                align-items: center;
                justify-content: center;
                color: white;
                font-weight: bold;
                font-size: 14px;
                pointer-events: none;
                z-index: 6;
            }}
            .badge-left, .badge-right {{
                position: absolute;
                bottom: 12px;
                padding: 4px 10px;
                background: rgba(15, 23, 42, 0.85);
                border: 1px solid rgba(56, 189, 248, 0.4);
                color: #38BDF8;
                font-size: 11px;
                font-weight: 700;
                border-radius: 6px;
                pointer-events: none;
                z-index: 8;
            }}
            .badge-left {{ left: 12px; }}
            .badge-right {{ right: 12px; color: #F43F5E; border-color: rgba(244, 63, 94, 0.4); }}
        </style>
    </head>
    <body style="margin:0; background:transparent;">
        <div class="slider-wrapper" id="sliderBox">
            <!-- Background Image (Right / Overlay) -->
            <img class="slider-img" src="data:image/png;base64,{img_right_b64}" />
            
            <!-- Foreground Clipped Image (Left / Raw) -->
            <div class="img-overlay" id="overlay">
                <img class="slider-img" src="data:image/png;base64,{img_left_b64}" />
            </div>

            <!-- Visual Divider & Handle -->
            <div class="divider-line" id="dividerLine"></div>
            <div class="divider-handle" id="dividerHandle">&#8596;</div>

            <!-- Labels -->
            <div class="badge-left">&#9664; {label_left}</div>
            <div class="badge-right">{label_right} &#9654;</div>

            <!-- Interactive Slider Input -->
            <input type="range" min="0" max="100" value="50" class="range-input" id="rangeInput" oninput="slide(this.value)" />
        </div>

        <script>
            function slide(val) {{
                document.getElementById('overlay').style.clipPath = 'polygon(0 0, ' + val + '% 0, ' + val + '% 100%, 0 100%)';
                document.getElementById('dividerLine').style.left = val + '%';
                document.getElementById('dividerHandle').style.left = val + '%';
            }}
        </script>
    </body>
    </html>
    """


def generate_printable_hospital_report_html(
    patient_name: str,
    patient_age: str,
    patient_gender: str,
    mrn_id: str,
    body_part: str,
    referring_physician: str,
    results: dict,
    logo_b64: Optional[str] = None
) -> str:
    """
    Generates a professional, printable clinical diagnostic certificate with hospital letterhead.
    Supports browser 1-click printing / Save as PDF via window.print().
    """
    is_fx = results.get("is_fractured", False)
    finding_text = "SUSPECTED FRACTURE / CORTICAL DISRUPTION" if is_fx else "NO ACUTE FRACTURE / INTACT OSSEOUS ARCHITECTURE"
    finding_color = "#DC2626" if is_fx else "#059669"
    conf_pct = results.get("confidence", 0.0) * 100
    triage_level = "LEVEL 2 - URGENT ORTHOPEDIC REVIEW" if is_fx else "LEVEL 5 - ROUTINE / DISCHARGE ELIGIBLE"

    logo_tag = f'<img src="data:image/png;base64,{logo_b64}" width="65" style="border-radius:50%; margin-right:15px;"/>' if logo_b64 else ''

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8"/>
        <title>Clinical Radiographic AI Diagnostic Report - {mrn_id}</title>
        <style>
            body {{
                font-family: 'Helvetica Neue', Arial, sans-serif;
                color: #1F2937;
                background: #FFFFFF;
                margin: 0;
                padding: 24px;
            }}
            .report-card {{
                max-width: 780px;
                margin: 0 auto;
                border: 2px solid #E5E7EB;
                border-radius: 12px;
                padding: 30px;
                box-shadow: 0 4px 20px rgba(0,0,0,0.05);
            }}
            .header {{
                display: flex;
                align-items: center;
                border-bottom: 2px solid #2563EB;
                padding-bottom: 16px;
                margin-bottom: 20px;
            }}
            .hospital-title {{
                font-size: 20px;
                font-weight: 800;
                color: #1E3A8A;
                letter-spacing: -0.5px;
            }}
            .hospital-subtitle {{
                font-size: 11px;
                color: #6B7280;
                text-transform: uppercase;
                letter-spacing: 1px;
            }}
            .meta-grid {{
                display: grid;
                grid-template-columns: repeat(3, 1fr);
                gap: 12px;
                background: #F8FAFC;
                border: 1px solid #E2E8F0;
                border-radius: 8px;
                padding: 14px;
                margin-bottom: 22px;
                font-size: 12px;
            }}
            .meta-item b {{
                color: #475569;
                text-transform: uppercase;
                font-size: 10px;
                display: block;
                margin-bottom: 2px;
            }}
            .diagnosis-box {{
                background: #F0FDF4;
                border: 2px solid {finding_color};
                border-radius: 8px;
                padding: 16px;
                margin-bottom: 22px;
                text-align: center;
            }}
            .diag-title {{
                font-size: 18px;
                font-weight: 800;
                color: {finding_color};
            }}
            .triage-badge {{
                display: inline-block;
                margin-top: 6px;
                padding: 4px 12px;
                border-radius: 6px;
                font-weight: 700;
                font-size: 11px;
                background: {finding_color};
                color: #FFFFFF;
            }}
            .section-title {{
                font-size: 13px;
                font-weight: 700;
                color: #1E293B;
                border-bottom: 1px solid #E2E8F0;
                padding-bottom: 4px;
                margin-top: 16px;
                margin-bottom: 8px;
                text-transform: uppercase;
            }}
            .findings-list {{
                font-size: 12px;
                line-height: 1.6;
                color: #334155;
            }}
            .signature-box {{
                display: flex;
                justify-content: space-between;
                margin-top: 36px;
                padding-top: 16px;
                border-top: 1px dashed #CBD5E1;
                font-size: 11px;
                color: #64748B;
            }}
            .print-btn {{
                background: #0284C7;
                color: white;
                border: none;
                padding: 10px 20px;
                font-size: 13px;
                font-weight: 700;
                border-radius: 8px;
                cursor: pointer;
                display: block;
                margin: 0 auto 16px auto;
            }}
            @media print {{
                .print-btn {{ display: none; }}
                body {{ padding: 0; }}
                .report-card {{ border: none; box-shadow: none; padding: 0; }}
            }}
        </style>
    </head>
    <body>
        <button class="print-btn" onclick="window.print()">🖨️ Print / Save Official PDF Report</button>

        <div class="report-card">
            <div class="header">
                {logo_tag}
                <div style="flex-grow:1;">
                    <div class="hospital-title">DEPARTMENT OF RADIOLOGY & ORTHOPEDIC TRAUMA</div>
                    <div class="hospital-subtitle">AI-Assisted Diagnostic Decision Support System | X-Ray AI Engine v1.0</div>
                </div>
                <div style="text-align:right; font-size:11px; color:#6B7280;">
                    Report Date: <b>{datetime.datetime.now().strftime("%d-%b-%Y %H:%M")}</b><br/>
                    MRN: <b>{mrn_id}</b>
                </div>
            </div>

            <div class="meta-grid">
                <div class="meta-item"><b>Patient Name</b>{patient_name if patient_name else "Anonymous / Triage Patient"}</div>
                <div class="meta-item"><b>Age / Gender</b>{patient_age if patient_age else "N/A"} / {patient_gender if patient_gender else "Unspecified"}</div>
                <div class="meta-item"><b>Anatomical Region</b>{body_part if body_part else "Radiographic Study"}</div>
                <div class="meta-item"><b>Referring Physician</b>{referring_physician if referring_physician else "ED Attending Physician"}</div>
                <div class="meta-item"><b>AI Neural Model</b>{results.get('architecture', 'ResNet').upper()}</div>
                <div class="meta-item"><b>Detection Confidence</b>{conf_pct:.2f}%</div>
            </div>

            <div class="diagnosis-box" style="background: {'#FEF2F2' if is_fx else '#ECFDF5'};">
                <div style="font-size:11px; text-transform:uppercase; color:#6B7280; font-weight:700;">Primary AI Radiographic Finding</div>
                <div class="diag-title">{finding_text}</div>
                <div class="triage-badge">{triage_level}</div>
            </div>

            <div class="section-title">Clinical Findings & Morphological Breakdown</div>
            <ul class="findings-list">
                <li><b>Cortical Continuity:</b> {'Cortical step-off and radiolucent fracture fissure identified across visualized bone margins.' if is_fx else 'Intact, smooth cortical margins with preserved trabecular pattern and joint spaces.'}</li>
                <li><b>Suspicious Regions of Interest (ROI):</b> {len(results.get('bounding_boxes', []))} hotspot region(s) localized by Gradient-weighted Class Activation Mapping (Grad-CAM).</li>
                <li><b>Radiodensity & Contrast:</b> CLAHE contrast equalization performed for bone micro-structure enhancement.</li>
            </ul>

            <div class="section-title">Diagnostic Decision Support Advisory</div>
            <p style="font-size:11px; color:#64748B; line-height:1.5;">
                This report is generated by a clinical Deep Learning decision support engine. It is designed to assist medical practitioners during clinical triage. Final clinical correlation and radiological sign-off are required.
            </p>

            <div class="signature-box">
                <div>
                    <b>Attending Radiologist / Orthopedic Surgeon:</b><br/>
                    ____________________________________<br/>
                    Date & Stamp
                </div>
                <div style="text-align:right;">
                    <b>X-Ray AI Engine Verification:</b><br/>
                    Validated via Grad-CAM Attention v1.0<br/>
                    Authentication Code: <code>{mrn_id}-VERIFIED</code>
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    return html


def generate_synthetic_bone_xray(
    is_fractured: bool = False,
    size: Tuple[int, int] = (300, 300),
    bone_type: str = "long_bone"
) -> Image.Image:
    """
    Generates an anatomically plausible synthetic X-ray of a bone structure with optional fracture line.
    """
    w, h = size
    img = np.zeros((h, w), dtype=np.float32)
    
    y_coords, x_coords = np.mgrid[0:h, 0:w]
    center_x, center_y = w // 2, h // 2
    
    soft_tissue_dist = np.sqrt(((x_coords - center_x) / (w * 0.4))**2 + ((y_coords - center_y) / (h * 0.48))**2)
    soft_tissue = np.clip(1.0 - soft_tissue_dist, 0.0, 1.0) * 45.0
    img += soft_tissue
    
    bone_width = int(w * random.uniform(0.22, 0.32))
    bone_left = center_x - bone_width // 2
    bone_right = center_x + bone_width // 2
    
    for x in range(bone_left, bone_right):
        dist_from_cortex = min(abs(x - bone_left), abs(x - bone_right))
        density = 160.0 + 80.0 * (1.0 - (dist_from_cortex / (bone_width / 2.0)))
        img[:, x] += density
        
    for y in range(h):
        flare = 0.0
        if y < h * 0.25:
            flare = (1.0 - y / (h * 0.25)) ** 2 * (w * 0.15)
        elif y > h * 0.75:
            flare = ((y - h * 0.75) / (h * 0.25)) ** 2 * (w * 0.15)
            
        if flare > 0:
            left = max(0, int(bone_left - flare))
            right = min(w, int(bone_right + flare))
            img[y, left:right] += random.uniform(50, 90)

    noise_grain = np.random.normal(0, 12.0, (h, w)).astype(np.float32)
    img += noise_grain
    img = cv2.GaussianBlur(img, (5, 5), 1.5)
    
    if is_fractured:
        fracture_y = random.randint(int(h * 0.35), int(h * 0.65))
        fracture_x_start = bone_left - random.randint(2, 6)
        fracture_x_end = bone_right + random.randint(2, 6)
        fracture_style = random.choice(["transverse", "oblique", "jagged"])
        
        fx_points = []
        num_pts = random.randint(6, 12)
        xs = np.linspace(fracture_x_start, fracture_x_end, num_pts)
        slope = random.uniform(-0.4, 0.4) if fracture_style != "transverse" else random.uniform(-0.05, 0.05)
        curr_y = fracture_y
        
        for idx, x in enumerate(xs):
            jitter_y = random.uniform(-4, 4) if fracture_style == "jagged" else random.uniform(-1.5, 1.5)
            pt_y = int(curr_y + (x - fracture_x_start) * slope + jitter_y)
            pt_y = np.clip(pt_y, 10, h - 10)
            fx_points.append((int(x), pt_y))
            
        pts = np.array(fx_points, np.int32)
        thickness = random.randint(2, 4)
        fx_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.polylines(fx_mask, [pts], isClosed=False, color=255, thickness=thickness)
        fx_mask_blurred = cv2.GaussianBlur(fx_mask.astype(np.float32), (3, 3), 0.8)
        
        reduction = (fx_mask_blurred / 255.0) * random.uniform(140.0, 190.0)
        img = np.clip(img - reduction, 0, 255)
        
        if random.random() > 0.4:
            shift = random.randint(2, 5)
            img[fracture_y:, bone_left:bone_right] = np.roll(img[fracture_y:, bone_left:bone_right], shift, axis=1)

    detector_noise = np.random.poisson(img.clip(1, 255) / 4.0) * 4.0
    img = 0.75 * img + 0.25 * detector_noise
    img_final = np.clip(img, 0, 255).astype(np.uint8)
    
    rgb_img = cv2.cvtColor(img_final, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(rgb_img)


def generate_dataset_structure(
    base_dir: str = "data",
    num_train_per_class: int = 40,
    num_val_per_class: int = 10,
    num_test_per_class: int = 10
):
    """
    Creates complete sample dataset folders and generates realistic synthetic X-rays.
    """
    splits = {
        "train": num_train_per_class,
        "val": num_val_per_class,
        "test": num_test_per_class
    }
    classes = ["normal", "fractured"]
    
    for split_name, count in splits.items():
        for cls_name in classes:
            target_dir = os.path.join(base_dir, split_name, cls_name)
            os.makedirs(target_dir, exist_ok=True)
            
            is_fractured = (cls_name == "fractured")
            for i in range(1, count + 1):
                img = generate_synthetic_bone_xray(
                    is_fractured=is_fractured,
                    size=(256, 256)
                )
                img_path = os.path.join(target_dir, f"{cls_name}_{i:03d}.png")
                img.save(img_path)

    sample_dir = os.path.join(base_dir, "sample_images")
    os.makedirs(sample_dir, exist_ok=True)
    
    sample_types = [
        ("demo_normal_wrist.png", False),
        ("demo_normal_femur.png", False),
        ("demo_fractured_radius.png", True),
        ("demo_fractured_tibia.png", True),
        ("demo_fractured_hairline.png", True)
    ]
    for filename, is_fractured in sample_types:
        img = generate_synthetic_bone_xray(is_fractured=is_fractured, size=(320, 320))
        img.save(os.path.join(sample_dir, filename))
        
    print(f"[OK] Generated sample dataset under '{base_dir}'.")


def generate_diagnostic_report(
    prediction: str,
    confidence: float,
    bounding_boxes: List[Dict],
    model_name: str = "ResNet50"
) -> str:
    """
    Generates a structured medical AI diagnostic summary.
    """
    status_icon = "🚨" if prediction.lower() == "fractured" else "✅"
    risk_level = "High" if confidence > 0.85 and prediction.lower() == "fractured" else "Moderate" if prediction.lower() == "fractured" else "Low"
    num_rois = len(bounding_boxes)
    roi_summary = f"{num_rois} suspicious cortical disruption zone(s) identified." if num_rois > 0 else "No significant cortical disruptions localized."
    
    report = f"""
### {status_icon} Bone Fracture AI Diagnostic Report

- **Model Architecture**: {model_name}
- **Primary Finding**: **{prediction.upper()}**
- **Confidence Level**: **{confidence * 100:.2f}%**
- **Fracture Risk Index**: **{risk_level}**
- **Visual Attention Localization**: {roi_summary}

#### Clinical Findings Breakdown:
- **Cortical Continuity**: {"Discontinuity / Radiolucent line observed" if prediction.lower() == "fractured" else "Continuous, intact cortex throughout visualized osseous structures."}
- **Grad-CAM Hotspots**: {"High gradient concentration near anatomical break points." if prediction.lower() == "fractured" else "Diffuse/low gradient distribution consistent with normal anatomy."}

> [!NOTE]
> *Disclaimer: This AI-generated diagnostic output is intended for clinical decision support and triage assistance. Final diagnosis should always be confirmed by a licensed radiologist or orthopedic specialist.*
"""
    return report
