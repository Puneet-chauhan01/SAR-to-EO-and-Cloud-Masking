### Team Members :- Puneet Chauhan(2K23/IT/124) pschauhan6149@gmail.com, Raghav Garg (2K23/IT/126) raghavgarg_23it126@dtu.ac.in ###

https://www.kaggle.com/code/puneetchauhan01/sar-to-eo


Overview:-
Satellite Synthetic Aperture Radar (SAR) data are weather- and daylight-independent but hard for analysts to interpret. Optical (EO) imagery is intuitive yet often unavailable due to clouds.
The goal of this project is to translate 10 m dual-polarised Sentinel-1 SAR tiles into Sentinel-2-like EO mosaics using CycleGAN variants trained on the Sen12MS winter subset.

Objective:-
Build three domain-specific models usimg cycleGAN

RGB (B4 / B3 / B2)

RGB + NIR (B4 / B3 / B2 / B8)

NIR–SWIR–RE (B8 / B11 / B5)


how to run:-

Simply Run code cells in order with headings signifying their objective.



Description of Code:-
Preprocessing steps:-

## Pre-processing Workflow (Sen12MS → CycleGAN)


### 1. Dataset discovery  
1. Scan the SAR folder for files matching `*_s1_*.tif*`.  
2. Replace `"_s1_" → "_s2_"` in the filename and search the EO folder to locate the matching optical tile.  
3. Store valid `(SAR path, EO path)` pairs for later indexing.

### 2. Tile loading  
* Open each GeoTIFF with `rasterio`.  
* Read all bands into memory and transpose to **(H × W × C)** `float32` arrays.

### 3. Spectral band selection  
| Band-set | EO channels kept | Use-case |
|----------|-----------------|----------|
| `RGB`            | B4 (R), B3 (G), B2 (B)        | natural-colour |
| `RGB_NIR`        | B4, B3, B2, B8 (NIR)          | NDVI, false colour |
| `NIR_SWIR_RE`    | B8 (NIR), B11 (SWIR), B5 (RE) | moisture / vegetation stress |

### 4. preprocessing steps:-
Here’s a streamlined **Pre-processing** section for your README, highlighting only the key steps and the recent caching enhancement:

---

## Pre-processing Workflow (Sen12MS → CycleGAN)

1. **Pair discovery**

   * Scan `sentinel1/` for `*_s1_*.tif*`, replace `"_s1_" → "_s2_"` to find matching EO tile in `sentinel2/`.
   * Build list of valid (SAR, EO) pairs.

2. **Tile loading & band selection**

   * Read GeoTIFFs with `rasterio` → H × W × C `float32` arrays.
   * Select EO channels per configuration:

     * **RGB**: B4, B3, B2
     * **RGB\_NIR**: B4, B3, B2, B8
     * **NIR\_SWIR\_RE**: B8, B11, B5

3. **Refined lee filtering**
 * Apply 7×7 Refined Lee filter to SAR VV & VH.

4. **feature expansion**

   * Stack four SAR features:

     1. VV
     2. VH
     3. VV – VH
     4. |∇VV| (Sobel magnitude)

4. **Resizing & normalization**

   * Resize all bands to 256×256 px (Lanczos).
   * Percentile clip each channel to the 0.1–99.9% range.
   * Linearly scale to \[0, 1], then map to \[–1, 1].

5. **Tensor conversion**

   * Ensure channel-first (`C×H×W`), contiguous arrays.
   * Convert to `torch.float32` tensors.

6. **Persistent caching**

   * On first run, save each processed sample as an FP16 `.pt` in `cache/`.
   * Subsequent loads skip all I/O/preprocessing in ≈ 0.3 ms per sample.

7. **Dataset splitting**
* 70% training, 15% validation, 15% test (stratified by file index).  
* Splits are fixed by setting the random seed to `42` for reproducibility.

### 10. Loader settings  
* `batch_size` = 1 (default for CycleGAN; can be raised to 4 on a 16 GB GPU).  

**Result:** every mini-batch that reaches the CycleGAN contains  
```text
'sar': (4, 256, 256)   in [-1, 1]
'eo' : (k, 256, 256)   in [-1, 1]   where k ∈ {3,4,3}
```


Model Architecture:-
---

## 🌐 Model Architecture – CycleGAN for SAR ↔ EO Translation

This CycleGAN is designed to perform image translation between SAR (Synthetic Aperture Radar) and EO (Electro-Optical) domains. It includes two generators and two discriminators.

---

### 🔁 Generators

There are two generators:

* **G (SAR → EO)**: Translates SAR images (with 2 channels: VV and VH) into EO images. The output can be:

  * RGB (3 channels)
  * RGB + NIR (4 channels)
  * NIR-SWIR-RE (3 channels, alternative spectral bands)

* **F (EO → SAR)**: Translates EO images back to SAR. The input channel size matches the output of G, and it always outputs 2 channels (VV, VH).

## 🔧 Model Architecture  

- **Dual Generators (`G` and `F`)**  
  - **Backbone:** *EfficientNet-B0* encoder pre-trained on ImageNet.  
  - **Input/Output shapes** (C ×256×256):  
    - `G : SAR (4 bands) → EO (k bands)` *where k = 3,4 or 3*  
    - `F : EO (k bands) → SAR (4 bands)`  
  - **Decoder:** four bilinear up-sampling stages with 1 × 1 convolutions and skip connections, followed by a 3-layer head ending in **tanh** ( -1 to 1 range).  
 
  - **Capacity:** ≈3.6 M parameters per generator → lightweight enough for 256×256 tiles on a single T4 GPU.  

- **Paired Discriminators (`Dᴇ` and `Dˢ`)**  
  - **Design:** *Tiny 70 × 70 PatchGAN* (3 spectral-norm 4 × 4 convolutions: 32→64→1 filters).  
  - **Purpose:** delivers a matrix of realism scores, enabling fine-grained texture judgement while keeping the network under 0.25 M parameters.  

- **Parallel Support**  
  - All four networks can be wrapped in `DataParallel` if multiple GPUs are present; otherwise they run on a single CUDA device or CPU.  





**Total generator loss**  
$$ \mathcal{L}_G = \mathcal{L}_{adv} + \lambda_{cyc}\,\mathcal{L}_{cyc} + \lambda_{sup}\,\mathcal{L}_{sup} (+ \lambda_{id}\,\mathcal{L}_{id}) $$

**Discriminator loss**  
$$ \mathcal{L}_D = \text{MSE}(D(real),0.9) + \text{MSE}(D(fake),0) $$

### Implementation Notes  

- **MS-SSIM term** is computed as `(1 – MS-SSIM)` so lower is better and remains in the ∥0,1∥ interval.  
- **Gradient clipping:**  ‖g‖₂ ≤ 1.0 for all networks to avoid exploding updates under AMP.  
- **Mixed precision (AMP):**  
  - Inputs held in FP16, model weights stay in FP32.  
  - `GradScaler` manages dynamic loss scaling; no NaNs observed after this change.  
- **Label smoothing** (0.9 instead of 1) helps the discriminators generalise and prevents quick saturation.  
- **Learning rate & schedule:** Adam, LR = 1×10⁻⁴ for both generators and discriminators, linear decay after epoch 100 (total 200 epochs).  

## ⚙️ Training Workflow at a Glance  

1. **Forward pass**:  
   - `G` generates EO from SAR → feeds `Dᴇ`.  
   - `F` generates SAR from EO → feeds `Dˢ`.  
   - Reconstructed images compute cycle losses.  
2. **Generator step**: minimise 𝓛G.  
3. **Discriminator step**: minimise 𝓛D separately.  
4. **Checkpointing** every 10 epochs; early checkpoints at epoch 15 resume seamlessly.  

With this architecture–loss combination the model reaches stable training (no divergence) and shows steadily improving PSNR/SSIM after only a few epochs, while remaining compact enough for rapid experimentation.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/53750707/e111913a-f548-46f3-a18c-2845a860bc26/sar-to-eo-2-19.ipynb


Key Areas:-


Caching impact – first pass builds a FP16 cache in /working; subsequent epochs read tensors in ≈0.3 ms, pushing GPU utilisation above 95%.

Mixed-precision rule-of-thumb – keep model weights FP32, cast inputs to FP16, and convert outputs back to FP32 for matplotlib; this removed the “Attempting to unscale FP16 gradients” and “Unsupported dtype” errors.

Visual checks – after 20 epochs edges and large shapes begin to appear but output is blurred more training may increase sharpnss.


---

## 📊 Results

> ⚠️ The current results are based on **only 25 epochs** of training, as even on a small subset of the dataset, training is resource-intensive. This draft was created to build the core pipeline, so that multiple models can later be implemented and compared using the same structure.

---





=== RGB_NIR validation metrics ===
Per-band PSNR : ['15.92 dB', '20.37 dB', '21.89 dB', '12.14 dB']
Per-band SSIM : ['0.378', '0.672', '0.750', '0.064']
NDVI MAE      : 0.153

=== NIR_SWIR_RE validation metrics ===
Per-band PSNR : ['17.09 dB', '43.81 dB', '20.98 dB']
Per-band SSIM : ['0.297', '0.954', '0.332']


=== RGB validation metrics ===
Per-band PSNR : ['11.55 dB', '17.05 dB', '15.48 dB']
Per-band SSIM : ['0.019', '0.307', '0.096']





Tools & Frameworks
PyTorch 2.x – core DL library, AMP & GradScaler.

torchvision – residual block helper functions.

torchmetrics – PSNR & SSIM.

rasterio – GeoTIFF I/O.

Pillow – per-band resizing.

NumPy, scikit-learn – array ops & deterministic splits.

matplotlib – quick RGB / NDVI visualisation.

pytorch_msssim
