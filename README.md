# Task-Based MRI Lesion Detectability under k-Space Noise and Undersampling

A simulation pipeline for measuring how MRI acquisition settings affect a *detection task* rather than pixel-level image quality. Synthetic elliptical lesions are inserted into white matter of 11,000 Human Connectome Project brain slices. Acquisition is simulated with complex Gaussian k-space noise and Cartesian undersampling (1x / 2x / 4x), and three model observers are trained to detect the lesions in 64 × 64 patches.

**Course:** BIOE 580, Department of Bioengineering, University of Illinois Urbana-Champaign
**Authors:** Ryan Leuba, Zhang Weijian, Yameng Cai
**Report:** [`report/BIOE580_Final_Report.pdf`](report/BIOE580_Final_Report.pdf)

## Method

- **Forward model:** x̂ = |F⁻¹(M · (F x + n))|, with n ~ CN(0, σ²I) and M a Cartesian sampling mask at R ∈ {1, 2, 4}
- **Lesion:** a 2-D Gaussian ellipse (σₓ = 4, σᵧ = 2, θ = 30°), with amplitudes 0.05, 0.10, and 0.20, placed at valid white-matter locations
- **Task:** signal-present vs. signal-absent 64 × 64 patches, 10,000 for training and 1,000 for testing
- **Observers:** regularized Hotelling observer (HO), Gabor channelized HO, and Laguerre–Gauss channelized HO
- **Figures of merit:** ROC AUC and detectability index d′, with noise levels swept to find low-, moderate-, and high-detectability operating points

## Results

- HO and Gabor CHO perform similarly at moderate-to-high detectability, while Laguerre–Gauss CHO is more robust at low detectability.
- 4x undersampling sharply reduces observer performance, while 2x largely preserves detectability.
- At 2x undersampling and normalized noise 0.10 (raw k-space σ ≈ 20.11), HO test AUC rises with lesion amplitude: **0.726** (0.05), **0.910** (0.10), and **0.997** (0.20).
- Original-background (signal-absent) baselines give HO AUC ≈ 0.48–0.49, as expected for chance level.

## Pipeline

Run the four steps in order. Each one reads the files the previous step saved to `MRI/`.

| Step | Script | What it does |
|------|--------|--------------|
| 0 | `c0_add_lesion.py` | Loads the HCP slices and white-matter segmentations, picks lesion and matched normal ROI centers, and saves lesioned image volumes for each amplitude |
| 1 | `c1_add_noise.py` | Simulates k-space noise and undersampling, saves reconstructed volumes and MSE / SSIM metrics, and checks the measured image-space noise against theory |
| 2 | `c2_get_patch_datasets.py` | Extracts signal-present and signal-absent 64 × 64 patches from the reconstructions |
| 3 | `c3_train_observers.py` | Trains HO / Gabor CHO / LG CHO, calibrates the detectability bands, and summarizes the sampling and lesion-amplitude studies |

```bash
pip install -r requirements.txt
python c0_add_lesion.py
python c1_add_noise.py
python c2_get_patch_datasets.py
python c3_train_observers.py
```

### Data

The HCP slices were provided for the course and are not redistributed here. Place them in `MRI/` next to the scripts:

```
MRI/HCP_selected_ones2.npy          # N x 260 x 311 brain slices
MRI/HCP_selected_ones2_wmparc.npy   # matching wmparc segmentations
```

The 2x and 4x Cartesian sampling masks are included in `masks/`.

**Disk space:** each reconstructed volume is 11,000 × 260 × 311 float32, about 3.6 GB. Step 1 writes 20 of them (about 71 GB), and step 0 writes about 11 GB of lesioned volumes.

## Repository layout

```
c0_add_lesion.py … c3_train_observers.py   pipeline scripts (cells marked with # %%)
notebooks/                                 original notebooks with their rendered outputs
masks/                                     2x / 4x Cartesian undersampling masks
report/                                    final report (PDF)
```

## Known issues

These come from the original notebooks and are documented rather than silently changed:

1. **`c1_add_noise` was repaired.** In the committed notebook only the first cell had been updated. The later cells called an older API and relied on variables left over from an earlier kernel session (`max_metric_images`, `mask_lookup`, `dataset_shape`, ...), so it could not run top to bottom. In the script, those cells use the first cell's helpers. Its case list now also generates every reconstruction that step 2 reads, including signal-absent backgrounds at each noise level. Steps 0 → 1 → 2 were verified end to end on synthetic data.
2. **Step 3 can request patches that step 2 never extracts.** `c3` chooses its lesion amplitude and its "moderate" noise level from the data at runtime. It then asks for that amplitude across the whole noise sweep, for lesion patches at 4x undersampling, and for all three amplitudes at the chosen noise level. `c2` extracts only amplitude 0.10 across the noise sweep, only 0.05 and 0.20 at noise 0.10, and no lesion patches at 4x. Unless the data happens to pick those exact settings, `c3` stops with a `FileNotFoundError` naming the missing patch file. The reported results came from patch files already on disk from earlier runs.
3. **The 0.005 and 0.01 noise levels share a file.** File names format the noise level with two decimals, so 0.005 is written as `0.01`. The "0.005" calibration point in steps 2 and 3 therefore reuses the 0.01 data.
