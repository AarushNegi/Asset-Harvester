# Asset Harvester Pipeline
> Extract 3D objects from dashcam video and import them into Blender

Built on top of [NVIDIA Asset Harvester](https://github.com/NVIDIA/asset-harvester) — a research pipeline that converts real-world object observations into simulation-ready 3D Gaussian Splat assets.

---

## What This Does

```
Dashcam video (.mp4)
        ↓
Extract frames (ffmpeg)
        ↓
Detect objects (YOLOv8)
        ↓
Format into Asset Harvester input
        ↓
Generate 3D Gaussian Splat (NVIDIA Asset Harvester)
        ↓
Export to Blender (.ply / .glb)
```

---

## Project Structure

```
asset-harvester/
├── inputs/
│   ├── frames/          ← extracted video frames
│   ├── crops/           ← YOLO detected object crops
│   └── samples/         ← formatted Asset Harvester input
├── outputs/             ← gaussians.ply, multiview.mp4 per object
├── checkpoints/         ← NVIDIA model weights (~8GB, not in repo)
├── nvidia-asset-harvester/  ← cloned NVIDIA repo (not in repo)
├── scripts/
│   ├── extract_frames.py       ← Step 1: extract frames from video
│   ├── detect_objects.py       ← Step 2: detect and crop objects
│   ├── build_sample_input.py   ← Step 3: format crops for inference
│   └── batch_inference.py      ← Step 4: run 3D generation
├── requirements.txt
└── README.md
```

---

## Requirements

### System
- Python 3.10+
- ffmpeg 8.x ([download](https://ffmpeg.org/download.html))
- GPU with 16GB+ VRAM for inference (or use Kaggle T4 — free)

### Python packages
```bash
pip install ultralytics opencv-python Pillow numpy
pip install torch torchvision
```

For inference (on GPU machine):
```bash
pip install transformers==4.48.3 "diffusers>=0.33.0,<0.37.0"
pip install open_clip_torch xformers accelerate gsplat
pip install tyro kiui roma lpips scikit-image decord
```

---

## Setup

### Step 1 — Clone this repo
```bash
git clone https://github.com/AarushNegi/asset-harvester-project.git
cd asset-harvester-project
```

### Step 2 — Clone NVIDIA repo
```bash
git clone https://github.com/NVIDIA/asset-harvester.git nvidia-asset-harvester
pip install -e nvidia-asset-harvester/
```

### Step 3 — Download model checkpoints (~8GB)
```python
from huggingface_hub import hf_hub_download
import os

os.makedirs("checkpoints", exist_ok=True)

for f in [
    "AH_multiview_diffusion.safetensors",
    "AH_tokengs_lifting.safetensors",
    "AH_camera_estimator.safetensors",
]:
    hf_hub_download(repo_id="nvidia/asset-harvester", filename=f, local_dir="checkpoints")
```

---

## Usage

### Step 1 — Extract frames from dashcam video
```bash
python scripts/extract_frames.py --video "path\to\video.mp4" --output inputs\frames --fps 2
```
Output: frames saved to `inputs/frames/`

---

### Step 2 — Detect and crop objects
```bash
python scripts/detect_objects.py --frames inputs\frames --output inputs\crops --conf 0.5
```
Detects: cars, trucks, motorcycles, buses, persons
Output: cropped objects saved to `inputs/crops/`

---

### Step 3 — Format crops for inference
```bash
python scripts/build_sample_input.py --crops inputs\crops --output inputs\samples
```
Output: formatted samples in `inputs/samples/` with:
- `frame_00.jpeg` — 512x512 resized crop
- `mask_00.png` — foreground mask
- `camera.json` — camera parameters

---

### Step 4 — Run 3D generation

**On a good GPU (RTX 3090/4090 — local):**
```bash
python scripts/batch_inference.py \
  --samples_dir inputs/samples \
  --output_dir outputs \
  --nvidia_repo nvidia-asset-harvester \
  --checkpoints checkpoints
```

**On Kaggle T4 (free):**
```bash
python scripts/batch_inference.py \
  --samples_dir inputs/samples \
  --output_dir outputs \
  --nvidia_repo nvidia-asset-harvester \
  --checkpoints checkpoints \
  --fp16 --offload --max_samples 5
```

Output per object in `outputs/<object_name>/`:
```
gaussians.ply     ← 3D Gaussian Splat asset
multiview.mp4     ← preview of 16 generated views
3d_lifted.mp4     ← 80 rendered views of final 3D asset
multiview/        ← 16 individual PNG views
input/            ← original input images
```

---

## Import into Blender

**Option A — Gaussian Splat Addon (view only)**
1. Download addon: [ReshotAI Gaussian Splatting Blender Addon](https://github.com/ReshotAI/gaussian-splatting-blender-addon)
2. Blender → Edit → Preferences → Add-ons → Install → select ZIP
3. File → Import → Gaussian Splatting (.ply) → select `gaussians.ply`

**Option B — SuperSplat (browser)**
1. Go to [supersplat.playcanvas.com](https://supersplat.playcanvas.com)
2. Import `gaussians.ply`
3. Press **F** to focus if black screen
4. Export as `.glb`

**Option C — Full mesh in Blender**
1. Open SuperSplat → import PLY → export `.glb`
2. Blender → File → Import → glTF 2.0 → select `.glb`
3. Fully editable as normal 3D mesh

---

## Running on Kaggle (No Local GPU)

1. Go to [kaggle.com](https://kaggle.com) → New Notebook
2. Enable GPU: Settings → Accelerator → GPU T4 x1
3. Run these cells:

```bash
# Cell 1 — clone NVIDIA repo
!git clone https://github.com/NVIDIA/asset-harvester.git /kaggle/working/nvidia-asset-harvester

# Cell 2 — install packages
!pip install -q transformers==4.48.3 "diffusers>=0.33.0,<0.37.0" open_clip_torch xformers accelerate gsplat tyro kiui roma lpips scikit-image decord
!pip install -q -e /kaggle/working/nvidia-asset-harvester/

# Cell 3 — download checkpoints
from huggingface_hub import hf_hub_download
import os
os.makedirs("/kaggle/working/checkpoints", exist_ok=True)
for f in ["AH_multiview_diffusion.safetensors", "AH_tokengs_lifting.safetensors", "AH_camera_estimator.safetensors"]:
    hf_hub_download(repo_id="nvidia/asset-harvester", filename=f, local_dir="/kaggle/working/checkpoints")

# Cell 4 — run inference on sample data
!python /kaggle/working/batch_inference.py \
  --samples_dir /kaggle/working/nvidia-asset-harvester/data_samples/rectified_AV_objects/consumer_vehicles \
  --output_dir /kaggle/working/outputs \
  --nvidia_repo /kaggle/working/nvidia-asset-harvester \
  --checkpoints /kaggle/working/checkpoints \
  --fp16 --offload --max_samples 1

# Cell 5 — download outputs immediately
!zip -r /kaggle/working/outputs.zip /kaggle/working/outputs/
```

Then download `outputs.zip` from the Kaggle right sidebar.

---

## Known Issues

| Issue | Cause | Fix |
|---|---|---|
| Black `multiview.mp4` | fp16 + offload on T4 degrades quality | Use RTX 4060+ without fp16/offload |
| Black screen in SuperSplat | Camera at wrong position | Press F to focus |
| Kaggle session wipe | `/kaggle/working/` is not persistent | Download outputs immediately after inference |
| Disk full on Kaggle | Checkpoints take 8GB of 16GB disk | Skip `AH_object_seg_jit.pt` (4.4GB), not needed if masks exist |

---

## Tech Stack

| Component | Technology |
|---|---|
| Frame extraction | ffmpeg 8.1.1 |
| Object detection | YOLOv8n (Ultralytics) |
| Image processing | OpenCV, Pillow |
| 3D generation | NVIDIA Asset Harvester (SparseViewDiT + TokenGS) |
| 3D format | 3D Gaussian Splatting (.ply) |
| Viewer | SuperSplat, Blender |
| Training compute | Kaggle T4 16GB / RTX 4060+ |

---

## Credits

- [NVIDIA Asset Harvester](https://github.com/NVIDIA/asset-harvester) — core 3D generation pipeline
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) — object detection
- [SuperSplat](https://supersplat.playcanvas.com) — Gaussian splat viewer
- [ReshotAI Blender Addon](https://github.com/ReshotAI/gaussian-splatting-blender-addon) — Blender import

---

## License

This project is for research and educational purposes.
NVIDIA Asset Harvester models are licensed under [NVIDIA Asset Harvester License](https://github.com/NVIDIA/asset-harvester/blob/main/LICENSE).