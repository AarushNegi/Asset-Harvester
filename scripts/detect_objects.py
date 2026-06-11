import os
import argparse
import cv2
from ultralytics import YOLO

# Classes we care about from COCO dataset
TARGET_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    0: "person"
}

def detect_and_crop(frames_dir, output_dir, confidence=0.5):
    """
    Detect objects in frames and save cropped images.

    Args:
        frames_dir: directory containing extracted frames
        output_dir: directory to save cropped objects
        confidence: minimum confidence threshold (default 0.5)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load YOLOv8 model (downloads automatically on first run)
    model = YOLO("yolov8n.pt")

    frames = sorted([f for f in os.listdir(frames_dir) if f.endswith(".jpeg")])
    print(f"Processing {len(frames)} frames...")

    crop_count = 0

    for frame_file in frames:
        frame_path = os.path.join(frames_dir, frame_file)
        img = cv2.imread(frame_path)
        if img is None:
            continue

        results = model(frame_path, conf=confidence, verbose=False)

        for i, box in enumerate(results[0].boxes):
            class_id = int(box.cls[0])

            # Only process target classes
            if class_id not in TARGET_CLASSES:
                continue

            label = TARGET_CLASSES[class_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Add padding around crop
            pad = 20
            h, w = img.shape[:2]
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(w, x2 + pad)
            y2 = min(h, y2 + pad)

            crop = img[y1:y2, x1:x2]

            # Skip if crop is too small
            if crop.shape[0] < 64 or crop.shape[1] < 64:
                continue

            # Save crop
            frame_name = os.path.splitext(frame_file)[0]
            crop_name = f"{frame_name}_{label}_{i}.jpeg"
            crop_path = os.path.join(output_dir, crop_name)
            cv2.imwrite(crop_path, crop)
            crop_count += 1
            print(f"  Saved: {crop_name} ({label}, conf: {float(box.conf[0]):.2f})")

    print(f"\nDone! {crop_count} crops saved to: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect objects in frames and crop them")
    parser.add_argument("--frames", default="inputs/frames", help="Directory containing frames")
    parser.add_argument("--output", default="inputs/crops", help="Output directory for crops")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    args = parser.parse_args()

    detect_and_crop(args.frames, args.output, args.conf)