# Clinderma AI Scan Tool — Pigmentation Segmentation

Semantic segmentation of facial pigmentation for the **Clinderma AI Scan Tool**, an image-based
skin assessment system that converts facial photographs into structured, dermatologist-checkable
scores.

This repository holds the **pigmentation branch**: dataset acquisition, multi-source dataset
construction, SegFormer-B2 training, evaluation and exported model artifacts. Face detection,
region extraction and acne detection live in the companion repository
[Clinderma (preprocessing and acne)](https://github.com/Harsh2517-coder/Clinderma).

---

## Results

Two training runs were completed on a Tesla T4 (Google Colab), both fine-tuning
`nvidia/mit-b2` (SegFormer-B2) at 512 x 512 with combined cross-entropy and Dice loss.

| Run | Classes | Test mIoU | Test Dice | Best val Dice (epoch) |
|---|---|---|---|---|
| `segformer_b2_pie_melasma` | background, PIE, Melasma | **0.585** | **0.690** | 0.665 (epoch 6) |
| `segformer_b2_multisource_v2` | background, PIH, PIE, Scar, Dark spot, Melasma | 0.307 | 0.364 | 0.367 (epoch 12) |

### Per-class test IoU

| Class | 3-class run | 6-class run |
|---|---|---|
| background | 0.979 | 0.980 |
| PIE | 0.232 | 0.336 |
| Melasma | 0.544 | 0.525 |
| PIH | — | 0.000 |
| Scar | — | 0.000 |
| Dark spot | — | 0.000 |

**Read this honestly.** The six-class run never produces a single PIH, scar or dark-spot
prediction — those classes have too few annotated pixels in the assembled data, so the model
collapses them into background. The three-class run (PIE and melasma only) is therefore the
usable model, and melasma is the class it segments best.

Data split for the three-class run: **831 train / 123 validation / 85 test** images.

Full metric histories, confusion matrices and configurations:
`Clinderma_Pigmentation/training/results/`.

---

## Repository layout

```
Clinderma_Pigmentation/
├── src/data/dataset_discovery_download.py   Multi-source dataset acquisition (Roboflow API, Figshare)
├── scripts/                                 Colab setup and dataset download helpers
├── training/
│   ├── checkpoints/<run>/                   best_model.pt, last_state.pt, training_config.json
│   ├── results/<run>_history.json           Per-epoch loss, mIoU, Dice, confusion matrices
│   ├── results/<run>_final_results.json     Test-set metrics
│   └── stage7a_environment.json             Workspace and path configuration
├── exports/segformer_b2_pie_melasma/        Inference-ready model (safetensors + configs)
├── data/{raw,normalized,splits}/            Dataset working directories (not committed)
└── reports/, experiments/, runs/            Reserved for QC output and experiment definitions

ModelTraining_MoreDatasets (1).ipynb         End-to-end Colab notebook (stages 8A–8L)
```

---

## Model

- **Architecture:** SegFormer-B2 (`nvidia/mit-b2`), `SegformerForSemanticSegmentation`
- **Input:** 512 x 512 RGB, ImageNet normalisation
- **Output:** per-pixel class logits, upsampled to input resolution
- **Exported classes:** `0 = background`, `1 = PIE`, `2 = MELASMA`
- **Format:** Hugging Face `safetensors` with `config.json` and `preprocessor_config.json`

### Loading the exported model

```python
import torch
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

path = "Clinderma_Pigmentation/exports/segformer_b2_pie_melasma"
model = SegformerForSemanticSegmentation.from_pretrained(path).eval()
processor = SegformerImageProcessor.from_pretrained(path)

image = Image.open("face.jpg").convert("RGB")
inputs = processor(images=image, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits                      # (1, 3, H/4, W/4)

logits = torch.nn.functional.interpolate(
    logits, size=image.size[::-1], mode="bilinear", align_corners=False
)
mask = logits.argmax(dim=1)[0].cpu().numpy()             # 0 background, 1 PIE, 2 melasma
```

---

## Dataset pipeline

The notebook builds the training set in stages rather than using a single public dataset:

1. **Acquisition** — pigmentation datasets pulled from the Roboflow API and the MEMI-DS
   collection on Figshare; nothing is normalised or deleted at this stage.
2. **Audit** — each source is inventoried for class names, annotation format and image counts.
3. **Class mapping** — heterogeneous source labels are mapped onto one semantic scheme.
4. **Dataset construction** — COCO and polygon annotations are rasterised into paired
   image/mask files.
5. **Balancing and augmentation** — minority classes receive synchronised image/mask augmentation.
6. **Integrity checks** — mask/annotation verification, orientation and resolution safety.
7. **Source-aware splitting** — splits are made per source so one dataset cannot dominate a split.
8. **Training** — SegFormer-B2 fine-tuning with orientation- and resolution-safe metrics.

---

## Training configuration

| Setting | 3-class run | 6-class run |
|---|---|---|
| Backbone | `nvidia/mit-b2` | `nvidia/mit-b2` |
| Image size | 512 | 512 |
| Batch size | 2 (effective 8 via gradient accumulation) | 2 (effective 8) |
| Learning rate | 6e-5 | 6e-5 |
| Weight decay | 0.01 | 0.01 |
| Warmup | 2 epochs | 2 epochs |
| Max epochs | 30 (13 run) | 30 (19 run) |
| Loss | 0.5 x CE + 0.5 x Dice | CE + Dice |
| Mixed precision | Yes | Yes |
| Early stopping patience | — | 7 |
| Seed | 42 | 42 |
| GPU | Tesla T4 | Tesla T4 |

---

## Getting started

```bash
git clone https://github.com/hanilshah5521-svg/Clinderma-AI-Scan-Tool.git
cd Clinderma-AI-Scan-Tool

# Model weights are tracked with Git LFS and are NOT included in a plain clone
git lfs install
git lfs pull

pip install torch torchvision transformers safetensors pillow numpy
```

Training is designed to run in Google Colab with Google Drive mounted for checkpoints —
open `ModelTraining_MoreDatasets (1).ipynb` and run the stages in order. Dataset acquisition
requires a `ROBOFLOW_API_KEY` in Colab Secrets or the environment.

---

## Known limitations

- **Three of six classes are unlearned.** PIH, scar and dark spot score 0.000 IoU in the
  six-class run. They need substantially more annotated pixels before they are usable.
- **Trained on external data.** No Clinderma clinical images are in the training set yet, so
  results are not validated against dermatologist grading on Indian skin tones.
- **PIE is weak** (0.232 IoU) even in the three-class run; melasma carries the model.
- **Checkpoints are Git LFS pointers.** A clone without `git lfs pull` yields 134-byte stub
  files rather than weights.
- **No zone mapping yet.** Predicted masks are not yet intersected with the five facial zones
  (forehead, cheeks, nose, chin) that the scoring layer expects.

---

## Roadmap

- Re-train once Clinderma's annotated clinical images are available
- Recover PIH, scar and dark spot with targeted annotation rather than augmentation alone
- Map predicted masks onto the five facial zones and report per-zone coverage
- Feed pigmentation coverage into the composite C-Score alongside acne, texture and hydration

---

## Disclaimer

Research and development work for an academic project. Outputs are computer-vision estimates
and are **not a medical diagnosis**.
