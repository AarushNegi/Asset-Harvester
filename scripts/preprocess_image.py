import cv2
import sys
from pathlib import Path

def preprocess(image_path, output_path, size=512):
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    min_dim = min(h, w)
    top = (h - min_dim) // 2
    left = (w - min_dim) // 2
    img = img[top:top+min_dim, left:left+min_dim]
    img = cv2.resize(img, (size, size))
    cv2.imwrite(output_path, img)
    print(f"Saved: {output_path}")

if __name__ == "__main__":
    preprocess(sys.argv[1], sys.argv[2])