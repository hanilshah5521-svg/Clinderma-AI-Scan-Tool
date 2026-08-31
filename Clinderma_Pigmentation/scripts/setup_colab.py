"""
Clinderma Pigmentation — Stage 2
Safe Google Colab <-> Google Drive setup.

This script ONLY:
- mounts Google Drive
- locates Clinderma_Pigmentation
- verifies/creates required directories
- checks available Drive storage
- checks CUDA/GPU
- prints a clear environment report

It does NOT:
- download datasets
- normalize images
- create train/val/test splits
- train models
- delete or overwrite project data
"""

from pathlib import Path
import shutil
import sys
import os

# ------------------------------------------------------------
# 1. Mount Google Drive
# ------------------------------------------------------------
try:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)
except Exception as exc:
    raise RuntimeError(
        "Google Drive could not be mounted. "
        "Run this script from Google Colab."
    ) from exc

MYDRIVE = Path("/content/drive/MyDrive")

# ------------------------------------------------------------
# 2. Locate existing Clinderma_Pigmentation
# ------------------------------------------------------------
PROJECT = MYDRIVE / "Clinderma_Pigmentation"

if not PROJECT.exists():
    raise FileNotFoundError(
        "\nClinderma_Pigmentation was not found.\n"
        f"Expected location:\n{PROJECT}\n\n"
        "Please make sure the folder exists directly inside "
        "Google Drive → My Drive."
    )

if not PROJECT.is_dir():
    raise RuntimeError(
        f"{PROJECT} exists but is not a directory."
    )

# ------------------------------------------------------------
# 3. Required project structure
# ------------------------------------------------------------
REQUIRED_DIRS = [
    "src",
    "src/data",
    "src/training",
    "src/evaluation",
    "src/utils",
    "data",
    "data/raw",
    "data/normalized",
    "data/splits",
    "experiments",
    "runs",
    "reports",
    "exports",
    "config",
    "scripts",
]

created = []
already_present = []

for relative in REQUIRED_DIRS:
    path = PROJECT / relative

    if path.exists():
        already_present.append(relative)
    else:
        path.mkdir(parents=True, exist_ok=True)
        created.append(relative)

# ------------------------------------------------------------
# 4. Storage report
# ------------------------------------------------------------
total, used, free = shutil.disk_usage(MYDRIVE)

def gb(value):
    return value / (1024 ** 3)

# ------------------------------------------------------------
# 5. CUDA / GPU report
# ------------------------------------------------------------
cuda_available = False
gpu_name = "N/A"
cuda_version = "N/A"

try:
    import torch

    cuda_available = torch.cuda.is_available()

    if cuda_available:
        gpu_name = torch.cuda.get_device_name(0)
        cuda_version = torch.version.cuda or "N/A"
except ImportError:
    torch = None

# ------------------------------------------------------------
# 6. Project size
# ------------------------------------------------------------
def directory_size(path):
    total_size = 0

    for item in path.rglob("*"):
        try:
            if item.is_file():
                total_size += item.stat().st_size
        except (FileNotFoundError, PermissionError):
            pass

    return total_size

project_size = directory_size(PROJECT)

# ------------------------------------------------------------
# 7. Existing data/checkpoint indicators
# ------------------------------------------------------------
def count_files(path):
    if not path.exists():
        return 0

    count = 0

    for item in path.rglob("*"):
        try:
            if item.is_file():
                count += 1
        except (FileNotFoundError, PermissionError):
            pass

    return count

raw_count = count_files(PROJECT / "data" / "raw")
normalized_count = count_files(PROJECT / "data" / "normalized")
split_count = count_files(PROJECT / "data" / "splits")
run_count = count_files(PROJECT / "runs")
export_count = count_files(PROJECT / "exports")

# ------------------------------------------------------------
# 8. Final report
# ------------------------------------------------------------
print("\n" + "=" * 72)
print("CLINDERMA PIGMENTATION — COLAB ENVIRONMENT SETUP")
print("=" * 72)

print("\nPROJECT")
print("-" * 72)
print("Project:", PROJECT)
print("Project exists:", PROJECT.exists())
print("Project size: %.3f GB" % gb(project_size))

print("\nGOOGLE DRIVE")
print("-" * 72)
print("Total: %.2f GB" % gb(total))
print("Used:  %.2f GB" % gb(used))
print("Free:  %.2f GB" % gb(free))

print("\nGPU")
print("-" * 72)
print("CUDA available:", cuda_available)
print("GPU:", gpu_name)
print("CUDA version:", cuda_version)

print("\nEXISTING PROJECT CONTENT")
print("-" * 72)
print("Raw files:", raw_count)
print("Normalized files:", normalized_count)
print("Split files:", split_count)
print("Run files:", run_count)
print("Export files:", export_count)

print("\nDIRECTORIES CREATED")
print("-" * 72)

if created:
    for item in created:
        print("  +", item)
else:
    print("  None — structure already existed.")

print("\nDIRECTORIES ALREADY PRESENT")
print("-" * 72)

for item in already_present:
    print("  ✓", item)

print("\nSAFETY")
print("-" * 72)
print("✓ No datasets downloaded")
print("✓ No images modified")
print("✓ No splits created")
print("✓ No checkpoints modified")
print("✓ No files deleted")
print("✓ No model downloaded")
print("✓ No training started")

if not cuda_available:
    print("\nWARNING:")
    print("CUDA is currently unavailable.")
    print("This is NOT an error for Stage 2.")
    print("A GPU will be required when we start SegFormer training.")

print("\n" + "=" * 72)
print("STAGE 2 SETUP COMPLETE")
print("=" * 72)
