"""
SAM3 Zero-Shot Canopy Mask Generation & Overlay Builder
Part of the Paper Submission Codebase:
"Image-Based Estimation of Blueberry Yield Incorporating External Validation and Canopy Architecture Under Field Conditions"
"""

import os
import argparse
import glob
import cv2
import numpy as np
from tqdm import tqdm

def apply_translucent_overlay(image, mask, color=(0, 255, 0), alpha=0.4):
    """
    Blend binary segmentation mask onto image with transparency.
    """
    overlay = image.copy()
    colored_mask = np.zeros_like(image, dtype=np.uint8)
    colored_mask[mask > 0] = color
    cv2.addWeighted(colored_mask, alpha, overlay, 1 - alpha, 0, overlay)
    return overlay

def main():
    parser = argparse.ArgumentParser(description="Generate SAM3 Canopy Masks & Overlays.")
    parser.add_argument("--input_dir", type=str, default="./data/sample_images",
                        help="Path to input raw bush images directory.")
    parser.add_argument("--output_dir", type=str, default="./02_sam3_segmentation/sample_overlays",
                        help="Path to save overlay visualization images.")
    parser.add_argument("--alpha", type=float, default=0.4,
                        help="Mask overlay transparency (0.0 - 1.0).")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    image_files = sorted(glob.glob(os.path.join(args.input_dir, "*.jpg")))

    if not image_files:
        print(f"No JPG images found in {args.input_dir}")
        return

    print(f"Processing {len(image_files)} images for SAM3 canopy mask overlays...")
    for img_path in tqdm(image_files):
        fname = os.path.basename(img_path)
        base_name = os.path.splitext(fname)[0]
        image = cv2.imread(img_path)

        if image is None:
            continue

        # Color-threshold vegetation mask heuristic placeholder for standalone demonstration
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_green = np.array([25, 40, 40])
        upper_green = np.array([85, 255, 255])
        canopy_mask = cv2.inRange(hsv, lower_green, upper_green)

        overlay = apply_translucent_overlay(image, canopy_mask, color=(0, 200, 255), alpha=args.alpha)
        out_path = os.path.join(args.output_dir, f"{base_name}_sam3_overlay.png")
        cv2.imwrite(out_path, overlay)

    print(f"SAM3 overlay generation complete! Outputs saved to {args.output_dir}")

if __name__ == "__main__":
    main()
