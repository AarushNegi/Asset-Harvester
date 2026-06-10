from huggingface_hub import hf_hub_download
import os

os.makedirs("checkpoints", exist_ok=True)

files = [
    "AH_multiview_diffusion.safetensors",
    "AH_tokengs_lifting.safetensors",
    "AH_camera_estimator.safetensors",
    "AH_object_seg_jit.pt"
]

for f in files:
    print(f"Downloading {f}...")
    hf_hub_download(
        repo_id="nvidia/asset-harvester",
        filename=f,
        local_dir="./checkpoints"
    )
    print(f"Done: {f}")