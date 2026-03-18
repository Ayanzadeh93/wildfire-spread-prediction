# TS-SatFire Training Guide - Step by Step

## Prerequisites Check

### Step 1: Verify Environment
```powershell
conda activate wildfire
python -c "import torch, monai, rasterio, numpy, pandas; print('✓ All packages OK')"
```

### Step 2: Set Environment Variables
```powershell
$env:TS_SATFIRE_DATA = "C:\Tim\Wildfire_physics\ts-satfire"
$env:TS_SATFIRE_ROOT = "C:\Tim\Wildfire_physics\dataset"
$env:WANDB_DISABLED = "true"  # Disable wandb if you don't have account
```

---

## Dataset Generation

### Step 3: Generate Training Dataset (Active Fire Detection)
```powershell
python dataset_gen_afba.py -mode train -ts 10 -it 3 -uc af
```
**Expected output:** Creates `dataset/dataset_train/af_train_img_seqtoseq_alll_10i_3.npy` and label file
**Time:** ~30-60 minutes depending on data size

### Step 4: Generate Validation Dataset (Active Fire Detection)
```powershell
python dataset_gen_afba.py -mode val -ts 10 -it 3 -uc af
```
**Expected output:** Creates `dataset/dataset_val/af_val_img_seqtoseq_alll_10i_3.npy` and label file
**Time:** ~5-15 minutes

### Step 5: Generate Test Dataset (Active Fire Detection) - Optional
```powershell
python dataset_gen_afba.py -mode test -ts 10 -it 3 -uc af
```
**Expected output:** Creates individual test files for each fire location
**Time:** ~10-30 minutes per fire

---

## Training Models

### Step 6: Train 3D Spatial-Temporal Model (SwinUNETR) - Active Fire Detection

**Command:**
```powershell
python run_spatial_temp_model.py -m swinunetr3d -mode af -b 1 -r 0 -lr 1e-4 -nh 4 -ed 96 -nc 8 -ts 10 -it 3 -av v1
```

**Parameters explained:**
- `-m swinunetr3d`: Model architecture (SwinUNETR 3D)
- `-mode af`: Task mode (Active Fire detection)
- `-b 1`: Batch size (reduce if OOM errors)
- `-r 0`: Run number (for reproducibility)
- `-lr 1e-4`: Learning rate
- `-nh 4`: Number of attention heads
- `-ed 96`: Embedding dimension
- `-nc 8`: Number of input channels
- `-ts 10`: Time-series length
- `-it 3`: Interval between samples
- `-av v1`: Attention version

**Expected behavior:**
- Training for 100 epochs
- Saves top 3 checkpoints based on validation loss (after epoch 50)
- Checkpoints saved to `saved_models/` directory
- Progress bars showing loss per epoch

**Time:** ~2-6 hours depending on GPU/CPU

### Step 7: Train 3D Spatial-Temporal Model (SwinUNETR) - Burned Area Mapping

**Command:**
```powershell
python run_spatial_temp_model.py -m swinunetr3d -mode ba -b 1 -r 0 -lr 1e-4 -nh 4 -ed 96 -nc 8 -ts 10 -it 3 -av v1
```

**Note:** Requires BA dataset generation first:
```powershell
python dataset_gen_afba.py -mode train -ts 10 -it 3 -uc ba
python dataset_gen_afba.py -mode val -ts 10 -it 3 -uc ba
```

### Step 8: Train 2D Spatial Model (UNet/SwinUNETR) - Active Fire Detection

**Command:**
```powershell
python run_spatial_model.py -m swinunetr2d -mode af -b 2 -r 0 -lr 1e-4 -nh 4 -ed 96 -nc 8 -ts 10 -it 3
```

**Available 2D models:**
- `unet`: Standard UNet
- `attunet`: Attention UNet
- `unetr2d`: UNETR 2D
- `swinunetr2d`: SwinUNETR 2D

### Step 9: Train Prediction Model (Next-Day Fire Progression)

**First, generate prediction datasets:**
```powershell
python dataset_gen_pred.py -mode train -ts 10 -it 3
python dataset_gen_pred.py -mode val -ts 10 -it 3
```

**Then train:**
```powershell
python run_spatial_temp_model_pred.py -m swinunetr3d -mode pred -b 1 -r 0 -lr 1e-4 -nh 4 -ed 96 -nc 27 -ts 10 -it 3
```

**Note:** Prediction uses 27 channels (8 satellite + 19 auxiliary features)

---

## Testing/Inference

### Step 10: Run Inference on Test Set

**For Active Fire Detection:**
```powershell
python run_spatial_temp_model.py -m swinunetr3d -mode af -b 1 -r 0 -lr 1e-4 -nh 4 -ed 96 -nc 8 -ts 10 -it 3 -av v1 -test -epoch 50
```

**Parameters:**
- `-test`: Enable test mode
- `-epoch 50`: Load checkpoint from epoch 50 (adjust to your best checkpoint)

**Expected output:**
- F1 scores and IoU scores per test fire
- Visualization plots in `evaluation_plot/` directory

---

## Monitoring Training

### Check Training Progress

1. **Watch console output:**
   - Train loss per epoch
   - Validation loss, Mean IoU, Mean Dice
   - Best checkpoint saves

2. **Check saved models:**
   ```powershell
   ls saved_models/
   ```

3. **View evaluation plots:**
   ```powershell
   ls evaluation_plot/
   ```

---

## Troubleshooting

### Out of Memory (OOM) Errors
- Reduce batch size: `-b 1` → `-b 0.5` (if supported) or use gradient accumulation
- Reduce model size: `-ed 96` → `-ed 48`
- Reduce time-series length: `-ts 10` → `-ts 6`

### Missing Dataset Files
- Verify dataset generation completed successfully
- Check files exist: `ls dataset/dataset_train/`

### WandB Errors
- Set `$env:WANDB_DISABLED = "true"` to disable (already done in guide)

### Slow Training
- Use GPU if available (code auto-detects CUDA)
- Reduce batch size if CPU-only
- Consider smaller model for faster iteration

---

## Quick Start Summary

**Minimum viable training (Active Fire Detection):**

```powershell
# 1. Activate environment
conda activate wildfire

# 2. Set paths
$env:TS_SATFIRE_DATA = "C:\Tim\Wildfire_physics\ts-satfire"
$env:TS_SATFIRE_ROOT = "C:\Tim\Wildfire_physics\dataset"
$env:WANDB_DISABLED = "true"

# 3. Generate datasets (if not done)
python dataset_gen_afba.py -mode train -ts 10 -it 3 -uc af
python dataset_gen_afba.py -mode val -ts 10 -it 3 -uc af

# 4. Train model
python run_spatial_temp_model.py -m swinunetr3d -mode af -b 1 -r 0 -lr 1e-4 -nh 4 -ed 96 -nc 8 -ts 10 -it 3 -av v1
```

---

## Model Architecture Options

### 3D Spatial-Temporal Models (for AF/BA):
- `swinunetr3d`: SwinUNETR 3D (recommended)
- `unetr3d`: UNETR 3D
- `unet3d`: 3D UNet
- `attunet3d`: 3D Attention UNet

### 2D Spatial Models (for AF/BA):
- `swinunetr2d`: SwinUNETR 2D
- `unetr2d`: UNETR 2D
- `unet`: Standard UNet
- `attunet`: Attention UNet

### Temporal Models (requires TensorFlow):
- `t4fire`: T4Fire transformer
- `gru_custom`: Custom GRU
- `lstm_custom`: Custom LSTM

---

## Expected File Structure After Training

```
C:\Tim\Wildfire_physics\
├── dataset/
│   ├── dataset_train/
│   │   ├── af_train_img_seqtoseq_alll_10i_3.npy
│   │   └── af_train_label_seqtoseq_alll_10i_3.npy
│   ├── dataset_val/
│   │   ├── af_val_img_seqtoseq_alll_10i_3.npy
│   │   └── af_val_label_seqtoseq_alll_10i_3.npy
│   └── dataset_test/
│       └── (individual fire test files)
├── saved_models/
│   └── (checkpoint files)
└── evaluation_plot/
    └── (visualization images)
```

