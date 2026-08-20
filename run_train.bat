@echo off
title Bone Fracture Model Trainer
echo =======================================================
echo   Bone Fracture Deep Learning Model Training
echo =======================================================
echo.

python -m src.train --data_dir data --arch resnet50 --epochs 15 --batch_size 16 --lr 0.0001
echo.
echo [*] Evaluating model on validation split...
python -m src.evaluate --model_path models/best_model.pth --data_dir data --split val

echo.
echo [DONE] Training & Evaluation complete! Check 'models/' folder for metrics and plots.
pause
