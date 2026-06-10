# Asset Harvester — Session Summary
**Date:** 10 June 2026  
**Last Updated:** ~06:45 UTC (12:15 IST)

---

## What We Did

### 1. Created GitHub Repo
- Repo name: `asset-harvester-project` (under AarushNegi)
- Structure:
```
asset-harvester-project/
├── inputs/sample_car/
├── outputs/
├── scripts/
│   ├── preprocess_image.py
│   └── generate_mask.py
├── setup/
│   └── download_checkpoints.py
├── requirements.txt
└── README.md
```

### 2. Set Up Kaggle Notebook
- Enabled internet access
- Verified phone number to unlock GPU (T4 16GB)
- Cloned both repos:
  - `https://github.com/AarushNegi/asset-harvester.git` → `asset-harvester`
  - `https://github.com/NVIDIA/asset-harvester.git` → `nvidia-asset-harvester`

### 3. Downloaded Checkpoints
- Downloaded to `./checkpoints/` on Kaggle
- ~8GB storage remaining after download

| File | Size |
|---|---|
| `AH_multiview_diffusion.safetensors` | 6.24 GB |
| `AH_tokengs_lifting.safetensors` | 1.21 GB |
| `AH_camera_estimator.safetensors` | 0.02 GB |
| `AH_object_seg_jit.pt` | 4.41 GB |

### 4. Dependency Issues Fixed (in order)
| Problem | Fix |
|---|---|
| `AutoModelForVision2Seq` not found | `pip install transformers==4.48.3` |
| `Dinov2WithRegistersConfig` not found | `pip install diffusers>=0.33.0,<0.37.0` |
| `open_clip` not found | `pip install open_clip_torch` |
| `guidance_embeds` unexpected keyword | Tried 0.31.0, 0.32.0, 0.32.2 — all wrong |
| Final fix | `pip install transformers==4.48.3 "diffusers>=0.33.0,<0.37.0" xformers accelerate` |
| TokenGS not found | `pip install -e nvidia-asset-harvester/` ✅ |

### 5. Correct Versions (from pyproject.toml)
```
transformers==4.48.3
diffusers>=0.33.0,<0.37.0
torch==2.10.0
open-clip-torch
xformers
accelerate
```

---

## Current Status
- VAE loaded ✅
- C-RADIO image encoder loaded ✅
- SparseViewDiT transformer loaded ✅
- Pipeline created ✅
- TokenGS package installed ✅ (via `pip install -e nvidia-asset-harvester/`)
- Inference running ⏳ — **awaiting result**

---

## Inference Command (correct — use this every time)
```bash
!cd nvidia-asset-harvester && python run_inference.py \
  --data_root data_samples/rectified_AV_objects \
  --output_dir ../outputs/ \
  --diffusion_checkpoint ../checkpoints/AH_multiview_diffusion.safetensors \
  --lifting_checkpoint ../checkpoints/AH_tokengs_lifting.safetensors \
  --ahc_checkpoint ../checkpoints/AH_camera_estimator.safetensors
```

## Full pip Install Command (run this if kernel restarts)
```bash
!pip install -q transformers==4.48.3 "diffusers>=0.33.0,<0.37.0" xformers accelerate open_clip_torch
!pip install -q -e nvidia-asset-harvester/
```

## Sample Data Location (on Kaggle)
```
nvidia-asset-harvester/data_samples/rectified_AV_objects/
├── consumer_vehicles/
│   ├── 382fee4aea8819ce/
│   │   ├── input_views/
│   │   │   ├── camera.json
│   │   │   ├── frame_00.jpeg
│   │   │   ├── frame_01.jpeg
│   │   │   ├── mask_00.png
│   │   │   └── mask_01.png
│   │   └── reserved_views/
│   ├── c3d2c4c995ce127c/
│   └── f32727777c346318/
├── VRU_pedestrians/
├── sample_paths.json
└── README.md
```

---

## Next Steps (Phase 1 remaining)
1. Confirm inference completes without error ⏳
2. Verify `gaussians.ply` generated in `outputs/`
3. Download `.ply` to local PC
4. Import into Blender via Gaussian Splat addon or SuperSplat → `.glb`

## Phase 2 (after Phase 1 works)
- Extract frames from dashcam video using `ffmpeg`
- Detect + crop objects using `YOLOv8`
- Feed crops into inference pipeline
- Batch process multiple objects

---

## Key Notes
- `--disable_image_guard` flag does NOT exist — image guard is off by default
- `--enable_image_guard` exists if needed
- TokenGS lives inside `asset_harvester/tokengs/` not a separate models/ folder
- C-RADIO model (~2.61GB) downloads automatically from HuggingFace on first run
- Kaggle T4 GPU = 16GB VRAM, enough for inference
- If kernel restarts: run the full pip install command above, then inference directly

---

## If You Get a Dedicated GPU (Local Setup)

### Recommended Hardware
| Component | Minimum | Recommended |
|---|---|---|
| GPU | RTX 3090 24GB | RTX 4090 24GB |
| RAM | 32GB | 64GB |
| Storage | 50GB SSD | 100GB SSD |
| OS | Windows + WSL2 | Ubuntu 22.04 |

### Steps to Run Locally (VS Code + Ubuntu/WSL2)

**Step 1 — Install CUDA Toolkit**
- Download CUDA 11.8 or 12.1 from nvidia.com/cuda
- Verify: `nvidia-smi` and `nvcc --version`

**Step 2 — Install Miniconda**
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

**Step 3 — Create environment**
```bash
conda create -n asset-harvester python=3.10
conda activate asset-harvester
```

**Step 4 — Clone repos**
```bash
git clone https://github.com/AarushNegi/asset-harvester.git
git clone https://github.com/NVIDIA/asset-harvester.git nvidia-asset-harvester
```

**Step 5 — Install dependencies**
```bash
pip install torch==2.0.1 torchvision --index-url https://download.pytorch.org/whl/cu118
pip install transformers==4.48.3 "diffusers>=0.33.0,<0.37.0"
pip install open_clip_torch xformers accelerate gsplat
pip install tyro kiui roma lpips scikit-image tensorboard decord
pip install -e nvidia-asset-harvester/
```

**Step 6 — Download checkpoints**
```bash
cd asset-harvester
python setup/download_checkpoints.py
```

**Step 7 — Run inference**
```bash
cd nvidia-asset-harvester
python run_inference.py \
  --data_root data_samples/rectified_AV_objects \
  --output_dir ../outputs/ \
  --diffusion_checkpoint ../checkpoints/AH_multiview_diffusion.safetensors \
  --lifting_checkpoint ../checkpoints/AH_tokengs_lifting.safetensors \
  --ahc_checkpoint ../checkpoints/AH_camera_estimator.safetensors
```

**Step 8 — Open output in Blender**
- Output: `outputs/gaussians.ply`
- Install Blender Gaussian Splat addon → import `.ply`
- Or use SuperSplat (browser) → export `.glb` → import into Blender

### No flags needed locally (unlike Kaggle)
- No `--offload_model_to_cpu` needed
- No `--precision fp16` needed
- No `--max_samples 1` needed
- Just run the base inference command directly

---

## Switching to a Different PC — Full Workflow

### What to do RIGHT NOW before switching

**On Kaggle — save your work:**

1. Save the notebook (File → Save)
2. Make sure `sample_paths.json` is updated (vehicles first)
3. Note your Kaggle notebook URL to reopen later

**Push everything from VS Code to GitHub:**
```bash
git add .
git commit -m "add all scripts and setup files"
git push origin main
```

---

### On the New PC — Full Setup in VS Code

**Step 1 — Install these first**
- VS Code: https://code.visualstudio.com
- Git: https://git-scm.com
- Python 3.10+: https://python.org
- CUDA Toolkit (if GPU available): https://developer.nvidia.com/cuda-downloads
- Blender (optional): https://blender.org

**Step 2 — Clone your repo**
```bash
git clone https://github.com/AarushNegi/asset-harvester.git
cd asset-harvester
```

**Step 3 — Clone NVIDIA repo**
```bash
git clone https://github.com/NVIDIA/asset-harvester.git nvidia-asset-harvester
```

**Step 4 — Install dependencies**
```bash
pip install transformers==4.48.3 "diffusers>=0.33.0,<0.37.0"
pip install open_clip_torch xformers accelerate gsplat
pip install tyro kiui roma lpips scikit-image tensorboard decord
pip install huggingface_hub opencv-python Pillow numpy
pip install -e nvidia-asset-harvester/
```

**Step 5 — Download checkpoints**
```bash
python setup/download_checkpoints.py
```
This downloads ~8GB to `./checkpoints/`

**Step 6 — Run inference**

If PC has good GPU (RTX 3090/4090):
```bash
cd nvidia-asset-harvester
python run_inference.py \
  --data_root data_samples/rectified_AV_objects \
  --output_dir ../outputs/ \
  --diffusion_checkpoint ../checkpoints/AH_multiview_diffusion.safetensors \
  --lifting_checkpoint ../checkpoints/AH_tokengs_lifting.safetensors \
  --ahc_checkpoint ../checkpoints/AH_camera_estimator.safetensors
```

If PC has limited GPU (like Kaggle T4):
```bash
cd nvidia-asset-harvester
PYTORCH_ALLOC_CONF=expandable_segments:True python run_inference.py \
  --data_root data_samples/rectified_AV_objects \
  --output_dir ../outputs/ \
  --diffusion_checkpoint ../checkpoints/AH_multiview_diffusion.safetensors \
  --lifting_checkpoint ../checkpoints/AH_tokengs_lifting.safetensors \
  --ahc_checkpoint ../checkpoints/AH_camera_estimator.safetensors \
  --precision fp16 \
  --offload_model_to_cpu \
  --max_samples 1
```

**Step 7 — Output location**
```
outputs/
└── consumer_vehicles/
    └── 382fee4aea8819ce/
        ├── gaussians.ply       ← 3D asset → open in Blender
        ├── multiview.mp4       ← preview of 16 generated views
        ├── multiview/          ← 16 individual generated view images
        ├── input/              ← original input images
        └── 3d_lifted/          ← 80 rendered views of final 3D asset
```

---

### Opening in Blender (on new PC)

**Option A — Gaussian Splat Addon (view only)**
1. Download addon ZIP: https://github.com/ReshotAI/gaussian-splatting-blender-addon
2. Blender → Edit → Preferences → Add-ons → Install → select ZIP
3. Enable the addon
4. File → Import → Gaussian Splatting (.ply) → select `gaussians.ply`

**Option B — SuperSplat (browser, no install)**
1. Go to: https://supersplat.playcanvas.com
2. Import → upload `gaussians.ply`
3. View in browser

**Option C — Convert to mesh for full Blender editing**
1. Open SuperSplat → import PLY
2. Export as `.glb`
3. Blender → File → Import → glTF 2.0 → select `.glb`
4. Now fully editable as a normal 3D mesh

---

### Kaggle Notebook — Reopen Instructions
- Go to kaggle.com → Your profile → Code → find your notebook
- All installed packages are GONE after session ends
- Re-run these cells in order on restart:

```bash
# Cell 1 — reinstall packages
!pip install -q transformers==4.48.3 "diffusers>=0.33.0,<0.37.0" open_clip_torch xformers accelerate gsplat tyro kiui roma lpips scikit-image decord
!pip install -q -e nvidia-asset-harvester/

# Cell 2 — run inference
!cd nvidia-asset-harvester && PYTORCH_ALLOC_CONF=expandable_segments:True python run_inference.py \
  --data_root data_samples/rectified_AV_objects \
  --output_dir ../outputs/ \
  --diffusion_checkpoint ../checkpoints/AH_multiview_diffusion.safetensors \
  --lifting_checkpoint ../checkpoints/AH_tokengs_lifting.safetensors \
  --ahc_checkpoint ../checkpoints/AH_camera_estimator.safetensors \
  --precision fp16 \
  --offload_model_to_cpu \
  --max_samples 1
```

Note: Checkpoints (8GB) are still saved in Kaggle storage — no need to redownload.

---

### Current Pending Task
- Run inference on `consumer_vehicles/382fee4aea8819ce` (vehicle sample)
- `sample_paths.json` already updated to put vehicles first
- Expected output: proper 3D car model in `gaussians.ply`