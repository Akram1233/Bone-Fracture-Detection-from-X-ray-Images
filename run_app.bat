@echo off
title Bone Fracture AI Diagnostic Studio
echo =======================================================
echo   Launching Bone Fracture AI Diagnostic Studio
echo =======================================================
echo.

:: Check if models exist, if not run initialization
if not exist "models\best_model.pth" (
    echo [!] Model checkpoint not found. Initializing dataset and baseline model...
    python generate_demo_data.py
)

echo.
echo [*] Starting Streamlit web application...
echo [*] Open your browser at http://localhost:8501
streamlit run app.py

pause
