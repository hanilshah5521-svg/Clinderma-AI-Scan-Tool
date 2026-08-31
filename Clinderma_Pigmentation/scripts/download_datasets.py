"""
Clinderma Stage 3 — Dataset Acquisition Launcher

Run this from Google Colab.

It downloads/acquires raw datasets into temporary Colab storage:
    /content/clinderma_workspace/raw/

It does NOT train anything.
It does NOT normalize anything.
It does NOT modify your Google Drive datasets.
"""

from pathlib import Path
import sys
import subprocess

# Install only the dataset-access dependency if necessary.
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "roboflow"],
    check=True,
)

PROJECT = Path("/content/drive/MyDrive/Clinderma_Pigmentation")
sys.path.insert(0, str(PROJECT))

from src.data.dataset_sources import acquire_all

manifest = acquire_all(project_drive=str(PROJECT))

print("\n" + "=" * 80)
print("CLINDERMA STAGE 3 — DATASET ACQUISITION")
print("=" * 80)
print("Workspace:", "/content/clinderma_workspace")
print("Drive project:", PROJECT)

for name, result in manifest["datasets"].items():
    print("\n" + "-" * 80)
    print(name)
    print("Status:", result.get("status"))
    print("Path:", result.get("path"))
    if result.get("message"):
        print("Message:", result["message"])

print("\n" + "=" * 80)
print("IMPORTANT")
print("=" * 80)
print("No normalization was performed.")
print("No training was started.")
print("No Drive dataset was deleted.")
print("No missing/private Roboflow dataset was silently replaced.")
print("\nManifest:", "/content/clinderma_workspace/dataset_manifest.json")
