"""
Helper script with instructions and automated methods to download real Bone Fracture X-ray datasets.
"""

import os
import shutil
import zipfile


def download_kaggle_instructions():
    print("""
========================================================================
  HOW TO DOWNLOAD REAL BONE FRACTURE X-RAY DATASETS (Kaggle / Online)
========================================================================

Option A: Kaggle Web Download (Recommended & Easiest)
------------------------------------------------------
1. Open your web browser and go to one of these top datasets:
   - Kaggle Bone Fracture Multi-Region Dataset:
     https://www.kaggle.com/datasets/bmadushanirodrigo/fracture-multi-region-x-ray-data
   - Kaggle Bone Fracture Detection:
     https://www.kaggle.com/datasets/vuppalaadithyasairam/bone-fracture-detection-using-xrays

2. Click the 'Download' button on Kaggle (saves a .zip file).

3. Extract the downloaded ZIP file.

4. Place your images into the project's 'data/' folder:
     data/
     ├── train/
     │   ├── fractured/   (Put fractured X-ray images here)
     │   └── normal/      (Put normal/healthy X-ray images here)
     └── val/
         ├── fractured/
         └── normal/

5. Run 'run_train.bat' to train the deep learning model on your dataset!
========================================================================
""")


def organize_downloaded_folder(source_dir: str, target_dir: str = "data"):
    """
    Utility to organize uncompressed images into train/val structure.
    """
    if not os.path.exists(source_dir):
        print(f"[!] Source folder '{source_dir}' does not exist.")
        return

    print(f"[*] Organizing images from '{source_dir}' into '{target_dir}'...")
    # Supports common folder names in Kaggle datasets
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                src_path = os.path.join(root, file)
                parent_name = os.path.basename(root).lower()
                
                cls = "fractured" if ("fracture" in parent_name or "fractured" in parent_name) else "normal"
                dest_dir = os.path.join(target_dir, "train", cls)
                os.makedirs(dest_dir, exist_ok=True)
                
                shutil.copy2(src_path, os.path.join(dest_dir, file))
                
    print(f"[SUCCESS] Dataset organized into '{target_dir}'.")


if __name__ == "__main__":
    download_kaggle_instructions()
