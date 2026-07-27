"""
Berry Bounding Box Size Distribution Extraction Module
Calculates physical pixel dimensions (width, height, area, aspect ratio) for detected berries to assist yield maturity modeling.
"""

import csv
import os

def extract_berry_bounding_box_sizes(predictions: list, image_id: str) -> list:
    """
    Extracts individual bounding box dimensions for all detected berries and flowers in an image.

    Args:
        predictions (list): List of SAHI ObjectPrediction objects.
        image_id (str): Identifier/filename base for the source image.

    Returns:
        list: List of dictionaries containing bounding box coordinates and size dimensions.
    """
    records = []
    for bbox_id, pred in enumerate(predictions):
        cat = pred.category.name
        if not pred.bbox:
            continue

        x1, y1, x2, y2 = map(float, pred.bbox.to_xyxy())
        width = x2 - x1
        height = y2 - y1
        area = width * height
        aspect_ratio = width / height if height > 0 else 0.0

        records.append({
            'image_id': image_id,
            'bbox_id': bbox_id,
            'category': cat,
            'x1': round(x1, 2),
            'y1': round(y1, 2),
            'x2': round(x2, 2),
            'y2': round(y2, 2),
            'width': round(width, 2),
            'height': round(height, 2),
            'area': round(area, 2),
            'aspect_ratio': round(aspect_ratio, 3),
            'confidence': round(float(pred.score.value), 4)
        })
    return records

def save_berry_sizes_to_csv(records: list, csv_path: str):
    """
    Appends berry size bounding box records to the output CSV file.

    Args:
        records (list): List of berry size dictionaries.
        csv_path (str): Filepath to output CSV.
    """
    if not records:
        return

    fieldnames = ['image_id', 'bbox_id', 'category', 'x1', 'y1', 'x2', 'y2', 'width', 'height', 'area', 'aspect_ratio', 'confidence']
    file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0

    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(records)
