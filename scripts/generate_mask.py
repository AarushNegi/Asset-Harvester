import torch
import cv2
import numpy as np
import sys

def generate_mask(image_path, output_path):
    img = cv2.imread(image_path)
    # placeholder — will use AH_object_seg_jit.pt on Kaggle
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    cv2.imwrite(output_path, mask)
    print(f"Mask saved: {output_path}")

if __name__ == "__main__":
    generate_mask(sys.argv[1], sys.argv[2])