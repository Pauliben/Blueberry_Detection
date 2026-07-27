"""
Dataset Download, SAHI Slicing & COCO-to-YOLO Converter Script
Part of the Paper Submission Codebase:
"Image-Based Estimation of Blueberry Yield Incorporating External Validation and Canopy Architecture Under Field Conditions"
"""

import os
import shutil
import argparse
import yaml
from roboflow import Roboflow
from ultralytics.data.converter import convert_coco
from packages.FPTeam import FPTeam

def main():
    parser = argparse.ArgumentParser(description='Process and filter COCO annotations for blueberry detection.')
    parser.add_argument('--dataset_dir', type=str, default='fb', help='Base dataset directory name')
    parser.add_argument('--roboflow_version', type=int, default=2, help='Roboflow version number')
    parser.add_argument('--api_key', type=str, default="pPFezpLcSIHuJoKfXf3v", help='Roboflow API key')
    args = parser.parse_args()

    full_dataset_name = f"{args.dataset_dir}-{args.roboflow_version}"
    dataset_path = f"./datasets/flowerberry/{full_dataset_name}"

    if os.path.exists(dataset_path):
        print(f"Dataset already available at {dataset_path}. Skipping download.")
        return

    print(f"Downloading dataset from Roboflow ({full_dataset_name})...")
    rf = Roboflow(api_key=args.api_key)
    project = rf.workspace("smart-fruit-phenomics-team").project("fb-ilnk0")
    version = project.version(args.roboflow_version)
    dataset = version.download("coco")

    # Filter small annotations by area threshold
    print("Filtering annotations...")
    FPTeam.filter_annotations_by_area(full_dataset_name)

    # Slice COCO dataset into patches using SAHI
    print("Slicing images into tiled patches...")
    FPTeam.slice_coco_DS(f"{full_dataset_name}/valid/", f"./{full_dataset_name}_sliced/valid/images")
    FPTeam.slice_coco_DS(f"{full_dataset_name}/train/", f"./{full_dataset_name}_sliced/train/images")

    # Remove downloaded raw folder
    shutil.rmtree(f"./{full_dataset_name}")

    # Convert sliced COCO annotations into YOLO segment format
    print("Converting COCO labels to YOLO format...")
    convert_coco(labels_dir=f"./{full_dataset_name}_sliced/valid/images", save_dir=f"./{full_dataset_name}_yolo/valid", use_segments=True)
    convert_coco(labels_dir=f"./{full_dataset_name}_sliced/train/images", save_dir=f"./{full_dataset_name}_yolo/train", use_segments=True)

    # Rename and rearrange directory structure
    os.rename(f"./{full_dataset_name}_sliced", f"./{full_dataset_name}")
    shutil.move(f"./{full_dataset_name}_yolo/train/labels/_annotations.coco.json_coco/", f"./{full_dataset_name}/train/")
    shutil.move(f"./{full_dataset_name}_yolo/valid/labels/_annotations.coco.json_coco/", f"./{full_dataset_name}/valid/")

    os.rename(f"./{full_dataset_name}/train/_annotations.coco.json_coco", f"./{full_dataset_name}/train/labels")
    os.rename(f"./{full_dataset_name}/valid/_annotations.coco.json_coco", f"./{full_dataset_name}/valid/labels")
    shutil.rmtree(f"./{full_dataset_name}_yolo")

    # Generate data.yaml configuration file for Ultralytics YOLO
    yaml_content = {
        'names': ['berry-immature', 'berry-mature', 'flower'],
        'nc': 3,
        'train': 'train/images',
        'val': 'valid/images'
    }
    yaml_file_path = os.path.join(full_dataset_name, 'data.yaml')
    with open(yaml_file_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)

    os.makedirs("./datasets/flowerberry", exist_ok=True)
    shutil.move(f"./{full_dataset_name}", dataset_path)
    print(f"Dataset processing complete! Saved to: {dataset_path}")

if __name__ == "__main__":
    main()
