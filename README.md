
#  SAR-to-EO Translation and Cloud & Shadow Segmentation Pipeline

This repository presents a complete Earth Observation (EO) processing pipeline with two stages:

1. **SAR to EO Translation** using CycleGAN  
2. **Cloud & Shadow Segmentation** on EO imagery using U-Net

---

##  Development Rationale & Tech Choices

Earth Observation (EO) data is critical for applications like agriculture, environment monitoring, and land classification. However, Sentinel-2 EO imagery is often obscured by cloud cover, making it unusable. Synthetic Aperture Radar (SAR), like Sentinel-1, is weather-agnostic but difficult to visually interpret.

To solve this:

- We used **CycleGAN** for **unpaired domain translation** — ideal for converting SAR to EO without exact matching tiles.
- **EfficientNet-B0** was chosen as the encoder for its lightweight and accurate design, making it suitable for 256×256 tiles on a single GPU.
- A **U-Net** was employed for segmentation due to its encoder-decoder structure and **skip connections**, which are great for capturing fine-grained spatial features like cloud boundaries.
- **TacoReader** helped us work with the CloudSEN12 and Sen12MS datasets using STAC-compliant metadata.
- We used **mixed precision (AMP)** and **gradient clipping** to stabilize training and optimize GPU usage.

---

##  Project Structure

\`\`\`
.
├── Cloud_Mask/
│   ├── output/
│   ├── Readme.md
│   ├── best_model_weights.pth
│   ├── cloud_segmentation_readme.docx
│   ├── code.ipynb
│   ├── final_model_weights.pth
│   └── requirements.txt
│
├── SAR_to_EO/
│   ├── checkpoints/
│   ├── samples/
│   ├── Readme.md
│   ├── requirements.txt
│   ├── sar-to-eo-2.ipynb
│
├── .gitattributes
└── Readme.md
\`\`\`

---

## Part 1: SAR → EO Translation using CycleGAN

###  Objective

Translate Sentinel-1 SAR images into Sentinel-2-like EO images using CycleGAN variants trained on a filtered subset of Sen12MS.

###  EO Target Configurations

- **RGB**: B4 (R), B3 (G), B2 (B)
- **RGB + NIR**: B4, B3, B2, B8
- **NIR–SWIR–RE**: B8 (NIR), B11 (SWIR), B5 (Red Edge)

###  How to Run

\`\`\`bash
pip install torch torchvision rasterio matplotlib scikit-learn pillow torchmetrics
\`\`\`

Then run \`sar-to-eo-2.ipynb\` in order.

---

###  Preprocessing Workflow

- Pair SAR and EO tiles using filename matching
- Read and stack SAR bands: VV, VH, VV–VH, |∇VV| (Sobel)
- Resize tiles to 256×256
- Normalize using percentile clipping → scale to [-1, 1]
- Cache tensors as \`.pt\` files for fast loading

---

###  Model Architecture

#### CycleGAN with EfficientNet-B0 Backbone

- **Generators (G, F)**:
  - G: SAR (4-band) → EO (3/4-band)
  - F: EO → SAR (4-band)
- **Discriminators**: PatchGAN (70×70)
- **Losses**:
  - \`𝓛_G = 𝓛_adv + λ_cyc 𝓛_cyc + λ_sup 𝓛_sup (+ λ_id 𝓛_id)\`
  - \`𝓛_D = MSE(D_real, 0.9) + MSE(D_fake, 0)\`

###  Training Features

- Mixed precision (AMP)
- Gradient clipping (‖g‖₂ ≤ 1.0)
- Label smoothing (real = 0.9)
- Checkpointing every 10 epochs

---

###  Results (After 25 Epochs)

#### RGB_NIR
- **PSNR**: 15.92–21.89 dB
- **SSIM**: 0.378–0.750
- **NDVI MAE**: 0.153

#### NIR_SWIR_RE
- **PSNR**: up to 43.81 dB
- **SSIM**: up to 0.954

#### RGB
- **PSNR**: ~15 dB
- **SSIM**: Low, needs more epochs

---

##  Part 2: Cloud & Shadow Segmentation on EO Images

###  Project Overview

We use a U-Net model to segment EO images into three classes:
- \`0\`: Clear
- \`1\`: Cloud / Thin Cloud
- \`2\`: Shadow

###  Dataset

Filtered CloudSEN12 tiles over Mexico using TacoReader.

### How to Run

\`\`\`bash
pip install torch torchvision rasterio matplotlib scikit-learn geopandas tacoreader
\`\`\`

Prepare dataset:
\`\`\`python
import tacoreader
taco_df = tacoreader.load("mini.taco")
\`\`\`

Train:
\`\`\`bash
python train_unet.py
\`\`\`

---

### Preprocessing Summary

- Selected 9 bands: \`[B2, B3, B4, B5, B6, B7, B8, B11, B12]\`
- Normalized to [0, 1] by dividing by 10000
- Converted 8-class ground truth to 3-class
- Ignored invalid labels (\`255\`)

---

###  U-Net Model Details

- 4 encoder-decoder blocks
- BatchNorm + ReLU activations
- 9 input channels → 3 output classes
- Loss: \`CrossEntropyLoss (ignore_index=255)\`
- Optimizer: \`AdamW\`
- Metrics: Accuracy, F1 (macro), IoU (macro)

---

###  Results

#### Validation (Epoch 35/40)
- **Accuracy**: 0.888
- **F1 (macro)**: 0.837
- **IoU (macro)**: 0.754

#### Test Set
- **Accuracy**: 0.8606
- **F1 (macro)**: 0.7423
- **IoU (macro)**: 0.6527

---

###  Sample Predictions

Stored under \`sample_outputs/\`:
- \`input_.png\` → RGB Input
- \`gt_.png\` → Ground Truth
- \`pred_.png\` → Prediction

---

##  Pipeline Flow

\`\`\`mermaid
graph LR
A[SAR Sentinel-1] --> B[CycleGAN G: SAR → EO]
B --> C[EO (RGB/NIR/SWIR)]
C --> D[U-Net: Cloud & Shadow Segmentation]
D --> E[Cloud Mask Outputs]
\`\`\`

---

##  Tools & Libraries Used

- \`PyTorch\`, \`torchvision\` – DL Framework
- \`TacoReader\` – Dataset filtering & STAC support
- \`Rasterio\` – Read GeoTIFFs
- \`Matplotlib\`, \`Pillow\` – Visualization
- \`scikit-learn\`, \`torchmetrics\` – Evaluation Metrics
- \`AMP\`, \`GradScaler\`, \`Mixed Precision\` – Performance optimization

---

##  Notes

- This modular pipeline allows EO image restoration and segmentation, even under extreme cloud cover.
- CycleGAN + U-Net combination generalizes well across scenes.
- The project is optimized for reproducibility, interpretability, and extensibility.

---

