"""
===================================================================================
Module 3: Canopy Architecture Metrics & Berry Size Distribution Extraction Script
===================================================================================
Part of the Paper Submission Codebase:
"Image-Based Estimation of Blueberry Yield Incorporating External Validation and Canopy Architecture Under Field Conditions"

This script extracts:
1. Multi-class berry & flower detections using SAHI sliced inference.
2. Individual bounding box sizes (width, height, area, aspect ratio) for detected fruit.
3. Canopy architecture features (canopy area, height, width, HSV vegetation index mask, 
   Euclidean distance transform, and silhouette convex hull geometry) to address canopy occlusion.

Usage Example:
--------------
python 03_plant_architecture/extract_plant_architecture.py \
    --flowerberry_model_path ./yolo11x.pt \
    --input_dir ./data/sample_images \
    --output_dir ./03_plant_architecture/outputs \
    --berries_detection \
    --berries_sizes \
    --plant_structure
"""

import os
import sys
import csv
import glob
import argparse
import cv2
from tqdm import tqdm

# Import modular helper utilities
from utils.inference import load_detection_model, run_sliced_inference
from utils.canopy_metrics import extract_canopy_architecture_features
from utils.berry_sizing import extract_berry_bounding_box_sizes, save_berry_sizes_to_csv
from utils.visualization import save_canopy_metric_maps

def parse_args():
    """
    Parses command-line arguments for plant architecture feature extraction.
    """
    parser = argparse.ArgumentParser(
        description="Extract Canopy Architecture Metrics & Berry Size Distribution for Yield Estimation."
    )
    parser.add_argument('--flowerberry_model_path', type=str, required=True,
                        help='Filepath to YOLO model weights for flowerberry detection (.pt).')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Path to input images directory containing raw plant images.')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Destination directory for generated CSV reports and metric visualizations.')
    parser.add_argument('--model_type', type=str, default='yolov11',
                        help='YOLO framework architecture type (default: yolov11).')
    parser.add_argument('--conf_thresh', type=float, default=0.5,
                        help='Detection confidence threshold (default: 0.5).')
    parser.add_argument('--berries_detection', action='store_true', default=True,
                        help='Generate berries_detection.csv count summary (immature, mature, flower).')
    parser.add_argument('--berries_sizes', action='store_true', default=True,
                        help='Generate berries_sizes.csv bounding box dimensions report.')
    parser.add_argument('--plant_structure', action='store_true', default=True,
                        help='Extract canopy geometry, HSV vegetation masks, and distance transforms.')
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Setup output subdirectories for visualization maps
    output_dirs = {
        'canopy_metric_visualization': os.path.join(args.output_dir, 'canopy_metric_visualization'),
        'distance_transform': os.path.join(args.output_dir, 'distance_transform'),
        'final_canopy_mask': os.path.join(args.output_dir, 'final_canopy_mask'),
        'hsv_mask': os.path.join(args.output_dir, 'hsv_mask'),
        'silhouette_analysis': os.path.join(args.output_dir, 'silhouette_analysis')
    }
    for folder in output_dirs.values():
        os.makedirs(folder, exist_ok=True)

    # Locate input images
    image_files = sorted(glob.glob(os.path.join(args.input_dir, '**', '*.jpg'), recursive=True))
    if not image_files:
        image_files = sorted(glob.glob(os.path.join(args.input_dir, '*.jpg')))

    if not image_files:
        print(f"[Error] No JPG images found under input directory: {args.input_dir}")
        sys.exit(1)

    print(f"\n[Plant Architecture Pipeline] Processing {len(image_files)} image(s)...")

    # 1. Load Detection Model
    detection_model = load_detection_model(
        model_path=args.flowerberry_model_path,
        model_type=args.model_type,
        conf_thresh=args.conf_thresh
    )

    csv_berries_path = os.path.join(args.output_dir, 'berries_detection.csv')
    csv_sizes_path = os.path.join(args.output_dir, 'berries_sizes.csv')
    csv_plant_path = os.path.join(args.output_dir, 'plant-structure.csv')

    # Prepare berries detection CSV header
    if args.berries_detection:
        with open(csv_berries_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['ID', 'raw_photo', 'berry-immature', 'berry-mature', 'flower', 'total_detections'])
            writer.writeheader()

    # Prepare plant structure CSV header
    if args.plant_structure:
        with open(csv_plant_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['ID', 'raw_photo', 'canopy_area', 'hull_area', 'solidity', 'canopy_width', 'canopy_height'])
            writer.writeheader()

    # 2. Iterate and Process Images
    for idx, img_path in enumerate(tqdm(image_files, desc="Extracting Metrics")):
        fname = os.path.basename(img_path)
        base_id = os.path.splitext(fname)[0]
        image = cv2.imread(img_path)

        if image is None:
            print(f"[Warning] Skipping unreadable image file: {img_path}")
            continue

        # A. Sliced Detection Inference
        sahi_result = run_sliced_inference(img_path, detection_model)
        predictions = sahi_result.object_prediction_list

        # B. Count Berry Classes
        counts = {'berry-immature': 0, 'berry-mature': 0, 'flower': 0}
        for pred in predictions:
            cat = pred.category.name
            if cat in counts:
                counts[cat] += 1

        if args.berries_detection:
            with open(csv_berries_path, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['ID', 'raw_photo', 'berry-immature', 'berry-mature', 'flower', 'total_detections'])
                writer.writerow({
                    'ID': base_id,
                    'raw_photo': fname,
                    'berry-immature': counts['berry-immature'],
                    'berry-mature': counts['berry-mature'],
                    'flower': counts['flower'],
                    'total_detections': sum(counts.values())
                })

        # C. Berry Bounding Box Sizing
        if args.berries_sizes:
            size_records = extract_berry_bounding_box_sizes(predictions, base_id)
            save_berry_sizes_to_csv(size_records, csv_sizes_path)

        # D. Canopy Architecture Feature Extraction
        if args.plant_structure:
            arch_results = extract_canopy_architecture_features(image)
            metrics = arch_results['metrics']

            with open(csv_plant_path, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['ID', 'raw_photo', 'canopy_area', 'hull_area', 'solidity', 'canopy_width', 'canopy_height'])
                writer.writerow({
                    'ID': base_id,
                    'raw_photo': fname,
                    'canopy_area': metrics['canopy_area'],
                    'hull_area': metrics['hull_area'],
                    'solidity': metrics['solidity'],
                    'canopy_width': metrics['canopy_width'],
                    'canopy_height': metrics['canopy_height']
                })

            # Save Visual Metric Maps
            save_canopy_metric_maps(image, arch_results, output_dirs, f"sample_{idx+1:02d}")

    print("\n========================================================")
    print(f"✅ Pipeline Completed Successfully!")
    print(f"📊 Results saved in: {os.path.abspath(args.output_dir)}")
    print("========================================================\n")

if __name__ == "__main__":
    main()
