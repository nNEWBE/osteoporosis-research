# 02. Google Colab Free Tier Optimization & Execution Guide

## 1. Google Colab Free Tier Specifications & Constraints

Google Colab Free Tier provides powerful cloud computing resources, but it is bounded by strict constraints:

| Resource | Free Tier Allocation | System Implications |
|---|---|---|
| **Accelerator** | 1x NVIDIA Tesla T4 (15 GB VRAM) | Ample for inference; training deep 101/201 backbones requires memory discipline. |
| **System RAM** | ~12.7 GB | High CPU image caching or multi-worker loading can trigger Out-Of-Memory (OOM) kills. |
| **Disk Space** | ~78 GB temporary storage | Ephemeral: erased once runtime disconnects. |
| **Idle Timeout** | ~60–90 minutes | Disconnects if browser tab goes inactive or receives no heartbeat. |
| **Max Runtime** | 12 hours | Session terminated and wiped after 12 continuous hours. |

---

## 2. Step-by-Step Setup Guide

### Step 1: Enable GPU Acceleration
1. Open Google Colab: [colab.research.google.com](https://colab.research.google.com).
2. Click **Runtime** in the top menu bar $\to$ **Change runtime type**.
3. Under *Hardware accelerator*, select **T4 GPU**.
4. Click **Save**. Verify by running:
   ```python
   !nvidia-smi
   ```

### Step 2: Upload Your Dataset
There are two reliable ways to load the knee X-ray dataset into Colab:

#### Option A: Direct ZIP Upload (Fastest for single sessions)
1. On your computer, zip your dataset folder (containing `Normal/` and `Osteoporosis/`) into `knee_osteoporosis_data.zip`.
2. In Colab, open the file browser on the left pane and drag-and-drop the zip file into the root directory.
3. Unzip the archive directly in the notebook:
   ```python
   !unzip -q knee_osteoporosis_data.zip -d ./data
   ```

#### Option B: Google Drive Mounting (Recommended for persistence)
1. Upload your dataset zip or folder to your Google Drive (e.g. `Colab Notebooks/Osteoporosis/`).
2. Mount Google Drive in the notebook:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
3. Copy the zip file to the local fast SSD runtime disk:
   ```python
   !cp "/content/drive/MyDrive/Osteoporosis/knee_osteoporosis_data.zip" ./
   !unzip -q knee_osteoporosis_data.zip -d ./data
   ```

---

## 3. Engineering Optimizations for Zero-Crash Execution

### 3.1 Automatic Mixed Precision (AMP)
By default, PyTorch uses Single Precision (FP32). Enabling Mixed Precision (FP16 math with FP32 master weights) cuts VRAM usage by **~50%** and increases T4 Tensor Core throughput by **2.5x**:
```python
from torch.amp import autocast, GradScaler

scaler = GradScaler('cuda')

with autocast('cuda'):
    outputs = model(inputs)
    loss = criterion(outputs, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 3.2 Sequential Model Training (Avoids Dual-Backbone VRAM Spikes)
Loading both ResNet-101 and DenseNet-201 simultaneously during backward passes with large batch sizes can push VRAM toward the 15 GB ceiling.
**Optimization strategy**:
1. Train **ResNet-101** first $\to$ save `best_resnet101.pth`.
2. Delete the model object from memory and force garbage collection:
   ```python
   del model_resnet
   import gc, torch
   gc.collect()
   torch.cuda.empty_cache()
   ```
3. Train **DenseNet-201** $\to$ save `best_densenet201.pth`.
4. Run validation and test predictions sequentially to extract scalar probabilities.
5. Pass the extracted 1D probabilities into the lightweight NumPy/PyTorch **Fuzzy Engine**, which consumes under **50 MB** of RAM and executes in **<1 second**.

### 3.3 Recommended Hyperparameters for Free Colab
- **Batch Size**: `16` or `32` (16 provides stable gradient descent on ~550 training images with negligible VRAM pressure).
- **Number of Workers**: `2` (`DataLoader(..., num_workers=2, pin_memory=True)`). Exceeding 2 workers on Colab Free Tier causes IPC overhead and random broken pipe errors.
- **Image Resolution**: `224x224` (Standard ImageNet resolution preserving trabecular details while optimizing computation).
- **Optimizer**: `AdamW(lr=1e-4, weight_decay=1e-2)` with `ReduceLROnPlateau(factor=0.5, patience=3)`.
- **Max Epochs**: `25` epochs with Early Stopping (`patience=7`). On T4 GPU with AMP, each epoch takes approximately **6 to 10 seconds**! Total training time for both models is under **5 minutes**.

---

## 4. Preventing Colab Disconnections

1. **Keep-Alive Script (Optional Browser Console Trick)**:
   If your browser enters background throttling:
   - Press `Ctrl + Shift + I` (or `F12`) to open Developer Tools in your browser.
   - Switch to the **Console** tab.
   - Paste the following snippet and press Enter:
     ```javascript
     function ConnectButton(){
         console.log("Connect Heartbeat sent"); 
         document.querySelector("#top-toolbar > colab-connect-button").shadowRoot.querySelector("#connect").click() 
     }
     setInterval(ConnectButton, 60000);
     ```
2. **Auto-Save Checkpoints to Drive**:
   Ensure all model weights (`.pth`) and test output metrics (`results.json`, `confusion_matrix.png`) are saved directly to Google Drive or downloaded locally at the end of the notebook.
