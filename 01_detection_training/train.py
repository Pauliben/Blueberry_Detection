"""
YOLO Model Training Script for Blueberry Detection
Part of the Paper Submission Codebase:
"Image-Based Estimation of Blueberry Yield Incorporating External Validation and Canopy Architecture Under Field Conditions"
"""

import argparse
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Train YOLO model for blueberry detection or leaf segmentation.")
    parser.add_argument("--train_type", type=str, required=True, choices=["flowerberries", "leafs"],
                        help="Type of training target: 'flowerberries' or 'leafs'.")
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to the dataset data.yaml file.")
    parser.add_argument("--model", type=str, default="yolo11x.pt",
                        help="Pretrained YOLO model path or name (default: yolo11x.pt).")
    parser.add_argument("--epochs", type=int, default=500,
                        help="Number of training epochs (default: 500).")
    parser.add_argument("--batch", type=int, default=8,
                        help="Batch size for training (default: 8).")
    parser.add_argument("--imgsz", type=int, default=400,
                        help="Input image resolution size (default: 400).")
    parser.add_argument("--device", type=str, default="0",
                        help="GPU device id (default: 0).")
    
    args = parser.parse_args()

    # Load pretrained model
    model = YOLO(args.model)

    if args.train_type == "flowerberries":
        print(f"Starting Blueberry Detection Training ({args.epochs} epochs, imgsz={args.imgsz})...")
        results = model.train(
            data=args.dataset_path,
            epochs=args.epochs,
            imgsz=args.imgsz,
            device=args.device,
            batch=args.batch
        )
    elif args.train_type == "leafs":
        print(f"Starting Leaf Segmentation Training ({args.epochs} epochs)...")
        results = model.train(
            data=args.dataset_path,
            epochs=300,
            imgsz=(480, 800),
            scale=0.5,
            device=args.device,
            batch=4,
            augment=True
        )

if __name__ == "__main__":
    main()
