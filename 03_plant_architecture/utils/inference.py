"""
SAHI Sliced Inference & Model Loading Module
Handles model initialization and high-resolution sliced prediction for small object detection.
"""

import torch
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

def load_detection_model(model_path: str, model_type: str = 'yolov11', conf_thresh: float = 0.5, device: str = None) -> AutoDetectionModel:
    """
    Loads pretrained YOLO detection model wrapped with SAHI AutoDetectionModel.

    Args:
        model_path (str): File path to weights (.pt file).
        model_type (str): Framework architecture type ('yolov11', 'yolov8', etc.).
        conf_thresh (float): Confidence threshold for detections.
        device (str): Execution device ('cuda:0' or 'cpu').

    Returns:
        AutoDetectionModel: Loaded SAHI detection model object.
    """
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print(f"[Inference] Loading {model_type} model from '{model_path}' on device '{device}'...")
    detection_model = AutoDetectionModel.from_pretrained(
        model_type=model_type,
        model_path=model_path,
        confidence_threshold=conf_thresh,
        device=device,
    )
    return detection_model

def run_sliced_inference(image_path: str, model: AutoDetectionModel, slice_h: int = 800, slice_w: int = 800, overlap_h: float = 0.2, overlap_w: float = 0.2):
    """
    Executes SAHI sliced prediction over high-resolution canopy images to detect small berries and flowers.

    Args:
        image_path (str): Path to input image file.
        model (AutoDetectionModel): SAHI model instance.
        slice_h (int): Height of image slices in pixels.
        slice_w (int): Width of image slices in pixels.
        overlap_h (float): Overlap ratio between vertical slices.
        overlap_w (float): Overlap ratio between horizontal slices.

    Returns:
        PredictionResult: SAHI prediction result containing bounding boxes, labels, and scores.
    """
    prediction = get_sliced_prediction(
        image_path,
        model,
        slice_height=slice_h,
        slice_width=slice_w,
        overlap_height_ratio=overlap_h,
        overlap_width_ratio=overlap_w,
        verbose=0
    )
    return prediction
