"""
Bone Fracture AI Diagnostic Studio - X-Ray AI Engine
Advanced Clinical Vision Platform with PACS Image Workbench, Interactive Split Slider,
Audio Readout, and Official Printable Hospital Certification.
"""

import os
import io
import base64
import time
import datetime
import random
import pandas as pd
import numpy as np
from PIL import Image
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
import plotly.express as px

from src.predictor import BoneFracturePredictor
from src.gradcam import COLORMAP_DICT
from src.utils import (
    generate_dataset_structure,
    apply_clahe,
    adjust_radiograph_properties,
    pil_to_base64,
    get_image_comparison_slider_html,
    generate_printable_hospital_report_html
)
from src.train import train_model
from src.evaluate import evaluate_model


# Load custom project logo
LOGO_PATH = "assets/logo.png"
logo_image = Image.open(LOGO_PATH) if os.path.exists(LOGO_PATH) else None


def get_logo_base64():
    if os.path.exists(LOGO_PATH):
        with open(LOGO_PATH, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    return ""


# Page Configuration
st.set_page_config(
    page_title="X-Ray AI Engine | Clinical Diagnostic Studio",
    page_icon=logo_image if logo_image else "🩻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Luxury Medical Styling (Glassmorphism, Glowing Badges, Radiology Console)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Gradient Brand Header */
    .hero-container {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 58, 138, 0.85) 50%, rgba(15, 23, 42, 0.95) 100%);
        border: 1px solid rgba(56, 189, 248, 0.35);
        border-radius: 16px;
        padding: 24px 32px;
        margin-bottom: 24px;
        box-shadow: 0 12px 35px -10px rgba(2, 132, 199, 0.35);
        backdrop-filter: blur(14px);
    }
    
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38BDF8 0%, #818CF8 50%, #C084FC 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        letter-spacing: -0.5px;
    }
    
    .hero-subtitle {
        color: #94A3B8;
        font-size: 1.02rem;
        margin-top: 6px;
        font-weight: 400;
    }

    /* Luxury Status Badges */
    .badge-normal {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(5, 150, 105, 0.25) 100%);
        border: 2px solid #10B981;
        color: #10B981;
        padding: 16px 24px;
        border-radius: 14px;
        font-weight: 800;
        font-size: 1.35rem;
        text-align: center;
        box-shadow: 0 0 25px rgba(16, 185, 129, 0.25);
        animation: pulseGreen 2.5s infinite;
    }

    .badge-fractured {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(185, 28, 28, 0.25) 100%);
        border: 2px solid #EF4444;
        color: #EF4444;
        padding: 16px 24px;
        border-radius: 14px;
        font-weight: 800;
        font-size: 1.35rem;
        text-align: center;
        box-shadow: 0 0 25px rgba(239, 68, 68, 0.25);
        animation: pulseRed 2s infinite;
    }

    @keyframes pulseGreen {
        0%, 100% { box-shadow: 0 0 15px rgba(16, 185, 129, 0.2); }
        50% { box-shadow: 0 0 30px rgba(16, 185, 129, 0.45); }
    }

    @keyframes pulseRed {
        0%, 100% { box-shadow: 0 0 15px rgba(239, 68, 68, 0.2); }
        50% { box-shadow: 0 0 35px rgba(239, 68, 68, 0.5); }
    }

    /* PACS Console Card */
    .pacs-card {
        background: rgba(15, 23, 42, 0.8);
        border: 1px solid rgba(56, 189, 248, 0.3);
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 16px;
    }

    /* Triage Priority Tag */
    .triage-tag-urgent {
        background: #DC2626;
        color: white;
        padding: 4px 12px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 12px;
        display: inline-block;
        box-shadow: 0 0 12px rgba(220, 38, 38, 0.4);
    }
    .triage-tag-routine {
        background: #059669;
        color: white;
        padding: 4px 12px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 12px;
        display: inline-block;
        box-shadow: 0 0 12px rgba(5, 150, 105, 0.4);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_predictor(model_path="models/best_model.pth"):
    if not os.path.exists(model_path):
        return None
    try:
        return BoneFracturePredictor(model_path=model_path)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None


def create_gauge_chart(prob_fracture: float, confidence: float, is_fractured: bool):
    """Generates an ultra-clean Plotly radial gauge chart."""
    color = "#EF4444" if is_fractured else "#10B981"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=prob_fracture * 100,
        number={'suffix': "%", 'font': {'size': 32, 'color': color, 'family': "Plus Jakarta Sans"}},
        title={'text': "Fracture Risk Index", 'font': {'size': 15, 'color': "#94A3B8"}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#475569"},
            'bar': {'color': color, 'thickness': 0.3},
            'bgcolor': "rgba(15, 23, 42, 0.4)",
            'borderwidth': 1,
            'bordercolor': "#334155",
            'steps': [
                {'range': [0, 30], 'color': 'rgba(16, 185, 129, 0.2)'},
                {'range': [30, 70], 'color': 'rgba(234, 179, 8, 0.2)'},
                {'range': [70, 100], 'color': 'rgba(239, 68, 68, 0.25)'}
            ],
            'threshold': {
                'line': {'color': color, 'width': 4},
                'thickness': 0.8,
                'value': prob_fracture * 100
            }
        }
    ))
    fig.update_layout(
        height=210,
        margin=dict(l=15, r=15, t=25, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E2E8F0")
    )
    return fig


def create_probability_bar_chart(probabilities: dict):
    """Horizontal sleek probability comparison bar chart."""
    categories = ["Normal / Intact", "Fractured"]
    values = [probabilities.get("normal", 0.0) * 100, probabilities.get("fractured", 0.0) * 100]
    colors = ["#10B981", "#EF4444"]

    fig = go.Figure(go.Bar(
        x=values,
        y=categories,
        orientation='h',
        marker=dict(
            color=colors,
            line=dict(color='rgba(255, 255, 255, 0.2)', width=1)
        ),
        text=[f"{v:.1f}%" for v in values],
        textposition='auto',
        textfont=dict(size=14, color="white", family="Plus Jakarta Sans")
    ))
    fig.update_layout(
        title="Class Probability Distribution",
        title_font=dict(size=13, color="#94A3B8"),
        xaxis=dict(range=[0, 100], showgrid=True, gridcolor="rgba(255,255,255,0.05)", title="Confidence (%)"),
        yaxis=dict(autorange="reversed"),
        height=170,
        margin=dict(l=10, r=20, t=30, b=15),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E2E8F0")
    )
    return fig


def render_voice_diagnosis_button(text_to_speak: str):
    """Embeds HTML5 Web Speech API for real-time audible medical voice readouts."""
    safe_text = text_to_speak.replace('"', '\\"').replace('\n', ' ')
    speech_html = f"""
    <script>
        function speakDiagnosis() {{
            if ('speechSynthesis' in window) {{
                window.speechSynthesis.cancel();
                var msg = new SpeechSynthesisUtterance("{safe_text}");
                msg.rate = 0.95;
                msg.pitch = 1.0;
                window.speechSynthesis.speak(msg);
            }} else {{
                alert("Speech Synthesis not supported by this browser.");
            }}
        }}
    </script>
    <button onclick="speakDiagnosis()" style="
        background: linear-gradient(135deg, #0284C7 0%, #0369A1 100%);
        border: 1px solid #38BDF8;
        color: white;
        font-weight: 700;
        padding: 9px 18px;
        border-radius: 8px;
        cursor: pointer;
        font-size: 13px;
        box-shadow: 0 0 15px rgba(56, 189, 248, 0.4);
        transition: all 0.2s;
        margin-top: 8px;
    ">
        🔊 Listen to AI Clinical Voice Diagnosis
    </button>
    """
    components.html(speech_html, height=52)


def main():
    logo_b64 = get_logo_base64()

    # Sidebar
    if logo_b64:
        st.sidebar.markdown(f"""
        <div style="text-align: center; margin-bottom: 16px;">
            <img src="data:image/png;base64,{logo_b64}" width="140" style="border-radius: 50%; border: 3px solid #38BDF8; box-shadow: 0 0 25px rgba(56, 189, 248, 0.4); margin-bottom: 8px;"/>
            <h2 style="margin-top: 4px; font-weight: 800; color: #38BDF8; font-size: 1.35rem; letter-spacing: -0.5px;">X-Ray AI Engine</h2>
            <p style="font-size: 0.82rem; color: #94A3B8; margin: 0;">Clinical Radiology Vision Platform</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.sidebar.markdown("""
        <div style="text-align: center; margin-bottom: 20px;">
            <h2 style="margin-top: 8px; font-weight: 800; color: #38BDF8; font-size: 1.4rem;">X-Ray AI Engine</h2>
            <p style="font-size: 0.85rem; color: #94A3B8;">Next-Gen Orthopedic Vision</p>
        </div>
        """, unsafe_allow_html=True)

    st.sidebar.markdown("---")
    st.sidebar.subheader("🎛️ Grad-CAM & Heatmap Controls")
    
    colormap_choice = st.sidebar.selectbox(
        "Thermal Colormap Palette:",
        list(COLORMAP_DICT.keys()),
        index=1
    )
    alpha_slider = st.sidebar.slider("Heatmap Opacity (Alpha)", 0.1, 0.9, 0.45, 0.05)
    cam_threshold = st.sidebar.slider("ROI Fracture Threshold", 0.10, 0.60, 0.25, 0.05)
    enhance_contrast = st.sidebar.toggle("CLAHE Radiopacity Enhancement", value=True)

    # PACS Radiology Workbench Controls in Sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔬 PACS Radiologist Console")
    invert_negative = st.sidebar.toggle("Invert / Negative Radiograph View", value=False)
    pacs_brightness = st.sidebar.slider("Window Level (Brightness)", 0.5, 1.8, 1.0, 0.05)
    pacs_contrast = st.sidebar.slider("Window Width (Contrast)", 0.5, 2.2, 1.0, 0.05)
    pacs_sharpness = st.sidebar.slider("Edge Detail / Sharpness", 0.5, 3.0, 1.0, 0.1)

    # Model status badge in sidebar
    model_path = "models/best_model.pth"
    predictor = load_predictor(model_path)

    st.sidebar.markdown("---")
    if predictor is None:
        st.sidebar.warning("⚠️ No Model Loaded")
        if st.sidebar.button("🚀 Auto-Initialize Demo Model", use_container_width=True):
            with st.spinner("Generating dataset and training model..."):
                from scripts.generate_demo_data import setup_project
                setup_project()
                st.cache_resource.clear()
                st.rerun()
    else:
        st.sidebar.markdown(f"""
        <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(56, 189, 248, 0.3); border-radius: 10px; padding: 12px;">
            <div style="font-size: 0.8rem; color: #94A3B8;">ACTIVE NEURAL ENGINE</div>
            <div style="font-size: 1.1rem; font-weight: 700; color: #38BDF8;">{predictor.architecture.upper()}</div>
            <div style="font-size: 0.75rem; color: #10B981;">⚡ Accelerated on {str(predictor.device).upper()}</div>
        </div>
        """, unsafe_allow_html=True)

    # Top Hero Banner
    if logo_b64:
        st.markdown(f"""
        <div class="hero-container" style="display: flex; align-items: center; gap: 24px;">
            <img src="data:image/png;base64,{logo_b64}" width="100" style="border-radius: 50%; border: 3px solid #38BDF8; box-shadow: 0 0 25px rgba(56, 189, 248, 0.5); flex-shrink: 0;"/>
            <div>
                <h1 class="hero-title">🩻 Bone Fracture Detection — X-Ray AI Engine</h1>
                <div class="hero-subtitle">Advanced Clinical Vision Platform with PACS Image Processing, Grad-CAM Explainability & Official Reporting</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="hero-container">
            <h1 class="hero-title">🩻 X-Ray AI Engine Studio</h1>
            <div class="hero-subtitle">Advanced Clinical Vision Platform with PACS Image Processing, Grad-CAM Explainability & Official Reporting</div>
        </div>
        """, unsafe_allow_html=True)

    # Navigation Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "✨ Diagnostic Cinema",
        "📂 Batch Triage Screening",
        "📊 Clinical Analytics & ROC",
        "⚙️ Transfer Learning Studio"
    ])

    # ---------------- TAB 1: Diagnostic Cinema ----------------
    with tab1:
        col_left, col_right = st.columns([1, 2], gap="large")

        with col_left:
            st.markdown("### 1. Select Radiograph")
            input_mode = st.radio("Source Mode:", ["Upload Scan", "Preset Clinical Scans"], horizontal=True)

            selected_image = None
            sample_name = "custom_upload.png"

            if input_mode == "Upload Scan":
                uploaded_file = st.file_uploader("Upload X-Ray (PNG, JPG, DICOM preview)", type=["png", "jpg", "jpeg", "bmp"])
                if uploaded_file is not None:
                    selected_image = Image.open(uploaded_file)
                    sample_name = uploaded_file.name
            else:
                sample_dir = "data/sample_images"
                if os.path.exists(sample_dir):
                    sample_files = [f for f in os.listdir(sample_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
                    if sample_files:
                        chosen_sample = st.selectbox("Select clinical sample scan:", sample_files)
                        sample_path = os.path.join(sample_dir, chosen_sample)
                        selected_image = Image.open(sample_path)
                        sample_name = chosen_sample

            if selected_image is not None:
                # Apply PACS adjustments live
                img_np_raw = np.array(selected_image.convert("RGB"))
                img_np_pacs = adjust_radiograph_properties(
                    img_np_raw,
                    brightness=pacs_brightness,
                    contrast=pacs_contrast,
                    sharpness=pacs_sharpness,
                    invert=invert_negative
                )
                selected_image = Image.fromarray(img_np_pacs)

                st.image(selected_image, caption=f"Scan: {sample_name} (PACS Adjusted)", use_container_width=True)
                run_btn = st.button("⚡ Run AI Diagnostic Scan", type="primary", use_container_width=True)
            else:
                run_btn = False

        with col_right:
            st.markdown("### 2. Comprehensive AI Diagnosis & Triage")

            if predictor is None:
                st.warning("Please click 'Auto-Initialize Demo Model' in the sidebar to activate the AI engine.")
            elif selected_image is not None and run_btn:
                with st.spinner("Analyzing radiograph & computing deep gradient tensors..."):
                    results = predictor.predict(
                        image_input=selected_image,
                        enhance_contrast=enhance_contrast,
                        cam_threshold=cam_threshold,
                        colormap_name=colormap_choice,
                        alpha=alpha_slider
                    )

                is_fx = results["is_fractured"]
                conf_pct = results["confidence"] * 100
                prob_fx = results["probabilities"].get("fractured", 0.0)

                # Happy Delight Animation on Normal or Alert on Fracture
                if not is_fx:
                    st.balloons()
                    st.markdown(f"""
                    <div class="badge-normal">
                        ✨ NORMAL / HEALTHY — NO FRACTURE DETECTED ({conf_pct:.1f}% Confidence)
                    </div>
                    <div class="happy-box" style="margin-top: 10px;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <h4 style="margin:0; font-size: 1.15rem; color: #A7F3D0;">🎉 Clean & Intact Bone Architecture</h4>
                                <p style="margin: 4px 0 0 0; font-size: 0.92rem;">0 suspicious cortical disruption zones detected. Intact osseous cortex throughout.</p>
                            </div>
                            <span class="triage-tag-routine">PRIORITY: ROUTINE (LEVEL 5)</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    voice_msg = f"Diagnostic Result: Normal radiograph. No bone fracture detected with {conf_pct:.1f} percent confidence."
                else:
                    st.markdown(f"""
                    <div class="badge-fractured">
                        🚨 SUSPECTED FRACTURE DETECTED ({conf_pct:.1f}% Confidence)
                    </div>
                    <div class="alert-box" style="margin-top: 10px;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <h4 style="margin:0; font-size: 1.15rem; color: #FECACA;">⚠️ Orthopedic Triage Alert</h4>
                                <p style="margin: 4px 0 0 0; font-size: 0.92rem;">Suspicious cortical discontinuity localized in {len(results['bounding_boxes'])} anatomical zone(s).</p>
                            </div>
                            <span class="triage-tag-urgent">PRIORITY: EMERGENT (LEVEL 2)</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    voice_msg = f"Clinical Alert: Suspected bone fracture detected with {conf_pct:.1f} percent confidence. Orthopedic triage review recommended."

                # Audio Voice Readout Button
                render_voice_diagnosis_button(voice_msg)

                st.write("")

                # Gauges and Probability Bars
                g_col1, g_col2 = st.columns([1, 1])
                with g_col1:
                    st.plotly_chart(create_gauge_chart(prob_fx, results["confidence"], is_fx), use_container_width=True)
                with g_col2:
                    st.plotly_chart(create_probability_bar_chart(results["probabilities"]), use_container_width=True)

                st.markdown("---")

                # Interactive Before/After Split Slider
                st.markdown("### 🎚️ Interactive Before / After Radiograph Split Comparison")
                st.caption("Drag the glowing cyan handle horizontally to reveal the AI Grad-CAM attention heatmap directly over the raw radiograph:")

                raw_b64 = pil_to_base64(results["original_image"])
                overlay_b64 = pil_to_base64(results["gradcam_overlay"])
                slider_html = get_image_comparison_slider_html(raw_b64, overlay_b64)
                components.html(slider_html, height=400)

                st.markdown("---")
                st.markdown("### 🔬 4-Panel Clinical Radiographic Grid")

                p1, p2 = st.columns(2)
                with p1:
                    st.markdown("**1. Raw Input Radiograph**")
                    st.image(results["original_image"], use_container_width=True)
                with p2:
                    st.markdown("**2. CLAHE Contrast Enhanced**")
                    st.image(results["enhanced_image"], use_container_width=True)

                p3, p4 = st.columns(2)
                with p3:
                    st.markdown(f"**3. Grad-CAM Attention Heatmap ({colormap_choice.split()[0]})**")
                    st.image(results["heatmap_image"], use_container_width=True)
                with p4:
                    st.markdown("**4. Holographic Fracture Localization ROI**")
                    st.image(results["localized_image"], use_container_width=True)

                # Suspicious ROI Hotspot Coordinate Table
                if results["bounding_boxes"]:
                    st.markdown("---")
                    st.markdown("#### 🎯 Localized Fracture Hotspots (ROI Breakdown)")
                    roi_data = []
                    for idx, box in enumerate(results["bounding_boxes"], 1):
                        roi_data.append({
                            "ROI ID": f"Zone #{idx}",
                            "X Pixel": box["x"],
                            "Y Pixel": box["y"],
                            "Width (px)": box["w"],
                            "Height (px)": box["h"],
                            "Area (px²)": box["w"] * box["h"],
                            "Attention Intensity": f"{int(box['activation']*100)}%"
                        })
                    st.dataframe(pd.DataFrame(roi_data), use_container_width=True)

                # Official Patient Demographics & Printable Medical PDF Certificate
                st.markdown("---")
                st.markdown("### 🏥 Official Clinical Report & Printable PDF")

                with st.expander("📝 Fill Patient Demographics for Official PDF", expanded=False):
                    c_pt1, c_pt2, c_pt3 = st.columns(3)
                    with c_pt1:
                        p_name = st.text_input("Patient Name", value="Jane Doe")
                        p_age = st.text_input("Patient Age", value="34")
                    with c_pt2:
                        p_gender = st.selectbox("Gender", ["Female", "Male", "Other"], index=0)
                        p_mrn = st.text_input("Medical Record Number (MRN)", value=f"MRN-{random.randint(100000, 999999)}")
                    with c_pt3:
                        p_body = st.text_input("Anatomical Region", value="Right Wrist / Forearm (AP/Lateral)")
                        p_doc = st.text_input("Referring Physician", value="Dr. A. Sharma, MD (Orthopedics)")

                # Generate Hospital HTML Certificate
                report_html = generate_printable_hospital_report_html(
                    patient_name=p_name if 'p_name' in locals() else "Patient",
                    patient_age=p_age if 'p_age' in locals() else "34",
                    patient_gender=p_gender if 'p_gender' in locals() else "Female",
                    mrn_id=p_mrn if 'p_mrn' in locals() else "MRN-100234",
                    body_part=p_body if 'p_body' in locals() else "Radiographic Study",
                    referring_physician=p_doc if 'p_doc' in locals() else "Dr. Sharma",
                    results=results,
                    logo_b64=logo_b64
                )
                components.html(report_html, height=520, scrolling=True)

                st.write("")
                st.download_button(
                    label="📥 Download Diagnostic Certificate (Markdown)",
                    data=results["report"],
                    file_name=f"bone_fracture_diagnosis_{sample_name.replace('.png', '')}.md",
                    mime="text/markdown",
                    use_container_width=True
                )
            elif selected_image is None:
                st.info("👈 Upload an X-ray or choose a sample scan from the left panel to begin diagnostic analysis.")

    # ---------------- TAB 2: Batch Processing ----------------
    with tab2:
        st.markdown("### ⚡ High-Throughput Batch Triage Screening")
        st.markdown("Rapidly scan large batches of radiographic studies for emergency department triage.")

        batch_files = st.file_uploader("Upload Multiple X-Rays for Rapid Screening", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

        if batch_files and predictor is not None:
            if st.button(f"🚀 Analyze All {len(batch_files)} Scans", type="primary", use_container_width=True):
                p_bar = st.progress(0)
                batch_data = []

                for idx, b_file in enumerate(batch_files):
                    img = Image.open(b_file)
                    res = predictor.predict(
                        img,
                        enhance_contrast=enhance_contrast,
                        cam_threshold=cam_threshold,
                        colormap_name=colormap_choice,
                        alpha=alpha_slider
                    )
                    batch_data.append({
                        "Filename": b_file.name,
                        "Diagnosis": "🚨 FRACTURE" if res["is_fractured"] else "✅ NORMAL",
                        "Confidence (%)": round(res["confidence"] * 100, 1),
                        "Fracture Probability (%)": round(res["probabilities"].get("fractured", 0) * 100, 1),
                        "Normal Probability (%)": round(res["probabilities"].get("normal", 0) * 100, 1),
                        "Suspicious Zones": len(res["bounding_boxes"])
                    })
                    p_bar.progress((idx + 1) / len(batch_files))

                df_batch = pd.DataFrame(batch_data)
                
                # Metrics overview
                num_fractures = sum(1 for d in batch_data if "FRACTURE" in d["Diagnosis"])
                num_normal = len(batch_data) - num_fractures

                m1, m2, m3 = st.columns(3)
                m1.metric("Total Studies Processed", len(batch_data))
                m2.metric("Fractures Flagged", num_fractures, delta=f"{num_fractures/len(batch_data)*100:.1f}%", delta_color="inverse")
                m3.metric("Normal / Clean Scans", num_normal, delta=f"{num_normal/len(batch_data)*100:.1f}%")

                st.dataframe(df_batch, use_container_width=True)

                csv = df_batch.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Export Triage Report as CSV",
                    csv,
                    "batch_orthopedic_triage_report.csv",
                    "text/csv",
                    use_container_width=True
                )

    # ---------------- TAB 3: Model Analytics ----------------
    with tab3:
        st.markdown("### 📊 Performance Analytics & Validation Metrics")
        
        eval_dir = "models/evaluation"
        cm_path = os.path.join(eval_dir, "confusion_matrix.png")
        roc_path = os.path.join(eval_dir, "roc_curve.png")
        curves_path = "models/learning_curves.png"

        ac1, ac2 = st.columns(2)
        with ac1:
            if os.path.exists(cm_path):
                st.image(cm_path, caption="Confusion Matrix (Validation Split)", use_container_width=True)
            else:
                st.info("Run evaluation to generate Confusion Matrix.")

        with ac2:
            if os.path.exists(roc_path):
                st.image(roc_path, caption="ROC-AUC Curve (Sensitivity vs 1 - Specificity)", use_container_width=True)
            else:
                st.info("Run evaluation to generate ROC curve.")

        if os.path.exists(curves_path):
            st.image(curves_path, caption="Convergence Curves (Cross-Entropy Loss & Accuracy)", use_container_width=True)

        if st.button("🔄 Recompute Validation Performance", use_container_width=True):
            with st.spinner("Evaluating validation split..."):
                metrics = evaluate_model()
                st.success(f"Validation Accuracy: {metrics['accuracy']*100:.2f}% | F1-Score: {metrics['f1_score']:.4f} | ROC-AUC: {metrics['roc_auc']:.4f}")
                st.rerun()

    # ---------------- TAB 4: Transfer Learning Studio ----------------
    with tab4:
        st.markdown("### ⚙️ Transfer Learning & Dataset Management")

        # Dataset inspection
        data_base = "data"
        counts = {"train": {}, "val": {}, "test": {}}
        for split in ["train", "val", "test"]:
            for cls in ["normal", "fractured"]:
                p = os.path.join(data_base, split, cls)
                if os.path.exists(p):
                    counts[split][cls] = len([f for f in os.listdir(p) if f.endswith(('.png', '.jpg', '.jpeg'))])
                else:
                    counts[split][cls] = 0

        df_counts = pd.DataFrame(counts).T
        st.markdown("#### Dataset Distribution")
        st.dataframe(df_counts, use_container_width=True)

        col_tl1, col_tl2 = st.columns(2)

        with col_tl1:
            st.markdown("#### 🔬 Synthetic Medical Data Generator")
            st.caption("Generates simulated X-rays with cortical bone radiopacity, trabecular texture, and hairline/transverse fractures.")
            num_train_gen = st.slider("Training Scans per Class", 20, 150, 40)
            if st.button("Generate Synthetic Dataset", use_container_width=True):
                with st.spinner("Synthesizing anatomical bone radiographs..."):
                    generate_dataset_structure(num_train_per_class=num_train_gen, num_val_per_class=10, num_test_per_class=10)
                    st.success("New synthetic dataset generated successfully!")
                    st.rerun()

        with col_tl2:
            st.markdown("#### 🧠 Neural Network Retraining")
            arch_select = st.selectbox(
                "Select Model Backbone:",
                ["resnet18", "resnet34", "resnet50", "densenet121", "efficientnet_b0", "custom_cnn"]
            )
            epochs_select = st.slider("Epochs", 3, 30, 10)
            lr_select = st.select_slider("Learning Rate", [1e-4, 3e-4, 5e-4, 1e-3], value=3e-4)

            if st.button("🚀 Train Model on Active Dataset", type="primary", use_container_width=True):
                prog_bar = st.progress(0)
                status_box = st.empty()

                def train_callback(epoch, total_epochs, t_loss, t_acc, v_loss, v_acc):
                    prog_bar.progress(epoch / total_epochs)
                    status_box.markdown(f"**Epoch {epoch}/{total_epochs}** — Train Acc: `{t_acc*100:.1f}%` | Val Acc: `{v_acc*100:.1f}%` | Val Loss: `{v_loss:.4f}`")

                with st.spinner(f"Fine-tuning {arch_select.upper()} on dataset..."):
                    try:
                        hist = train_model(
                            data_dir="data",
                            architecture=arch_select,
                            epochs=epochs_select,
                            learning_rate=lr_select,
                            progress_callback=train_callback
                        )
                        evaluate_model()
                        st.success(f"Retraining Complete! Best Validation Accuracy: {hist['best_val_acc']*100:.2f}%")
                        st.cache_resource.clear()
                        st.rerun()
                    except Exception as e:
                        st.error(f"Training failed: {e}")


if __name__ == "__main__":
    main()
