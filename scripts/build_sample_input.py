import os
import cv2
import json
import numpy as np
import argparse
from PIL import Image

def generate_mask(image):
    """
    Generate a simple foreground mask using GrabCut.
    White = object, Black = background
    """
    img = np.array(image)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    mask = np.zeros(img_bgr.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    h, w = img_bgr.shape[:2]
    margin = int(min(h, w) * 0.1)
    rect = (margin, margin, w - 2 * margin, h - 2 * margin)

    cv2.grabCut(img_bgr, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

    # Convert mask: 0,2 = background, 1,3 = foreground
    final_mask = np.where((mask == 2) | (mask == 0), 0, 255).astype(np.uint8)
    return final_mask

def default_camera_json():
    """
    Default camera parameters for a single input view.
    Asset Harvester uses these to estimate 3D pose.
    """
    return {
        "fl_x": 1000.0,
        "fl_y": 1000.0,
        "cx": 256.0,
        "cy": 256.0,
        "w": 512,
        "h": 512,
        "frames": [
            {
                "file_path": "frame_00.jpeg",
                "transform_matrix": [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 2.5],
                    [0.0, 0.0, 0.0, 1.0]
                ]
            }
        ]
    }

def build_sample_input(crops_dir, output_dir, size=512):
    """
    Format cropped images into Asset Harvester input structure.

    Args:
        crops_dir: directory containing cropped object images
        output_dir: output directory for formatted samples
        size: target image size (default 512x512)
    """
    os.makedirs(output_dir, exist_ok=True)

    crops = [f for f in os.listdir(crops_dir) if f.endswith(".jpeg")]
    print(f"Formatting {len(crops)} crops...")

    for crop_file in crops:
        crop_name = os.path.splitext(crop_file)[0]
        crop_path = os.path.join(crops_dir, crop_file)

        # Create sample directory
        sample_dir = os.path.join(output_dir, crop_name, "input_views")
        os.makedirs(sample_dir, exist_ok=True)

        # Load and resize image to 512x512
        img = Image.open(crop_path).convert("RGB")
        img_resized = img.resize((size, size), Image.LANCZOS)

        # Save resized frame
        frame_path = os.path.join(sample_dir, "frame_00.jpeg")
        img_resized.save(frame_path, quality=95)

        # Generate and save mask
        mask = generate_mask(img_resized)
        mask_path = os.path.join(sample_dir, "mask_00.png")
        cv2.imwrite(mask_path, mask)

        # Save camera.json
        camera_data = default_camera_json()
        camera_path = os.path.join(sample_dir, "camera.json")
        with open(camera_path, "w") as f:
            json.dump(camera_data, f, indent=2)

        print(f"  Built: {crop_name}/")

    print(f"\nDone! {len(crops)} samples saved to: {output_dir}")
    print("\nStructure:")
    print(f"  {output_dir}/")
    print(f"  └── <crop_name>/")
    print(f"      └── input_views/")
    print(f"          ├── frame_00.jpeg")
    print(f"          ├── mask_00.png")
    print(f"          └── camera.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Format crops into Asset Harvester input format")
    parser.add_argument("--crops", default="inputs/crops", help="Directory containing cropped images")
    parser.add_argument("--output", default="inputs/samples", help="Output directory for formatted samples")
    parser.add_argument("--size", type=int, default=512, help="Target image size")
    args = parser.parse_args()

    build_sample_input(args.crops, args.output, args.size)