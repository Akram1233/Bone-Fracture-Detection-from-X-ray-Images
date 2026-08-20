"""
Utility script to create a clean ZIP archive of the entire Bone Fracture project.
"""

import os
import zipfile
import datetime


def zip_project(output_filename: str = None):
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    if output_filename is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = os.path.join(root_dir, f"bone_fracture_project_{timestamp}.zip")

    # Exclude temporary or cache files
    exclude_dirs = {"__pycache__", ".git", ".pytest_cache", ".gemini", ".system_generated"}
    exclude_extensions = {".pyc", ".pyo"}

    print(f"[*] Packaging project into: {output_filename} ...")
    file_count = 0

    with zipfile.ZipFile(output_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(root_dir):
            # Prune excluded directories
            dirs[:] = [d for d in dirs if d not in exclude_dirs and not d.startswith(".")]
            
            for file in files:
                ext = os.path.splitext(file)[1]
                if ext in exclude_extensions or file.endswith(".zip"):
                    continue

                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, root_dir)
                zipf.write(full_path, rel_path)
                file_count += 1

    file_size_mb = os.path.getsize(output_filename) / (1024 * 1024)
    print(f"[SUCCESS] Packaged {file_count} files ({file_size_mb:.2f} MB) into '{output_filename}'")
    return output_filename


if __name__ == "__main__":
    zip_project()
