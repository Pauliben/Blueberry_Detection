"""
Canopy Architecture Feature Extraction Module
Derives canopy area, height, width, HSV color vegetation masks, Euclidean distance transform,
and silhouette convex hull metrics to account for fruit occlusion under field conditions.
"""

import cv2
import numpy as np

def compute_hsv_vegetation_mask(image: np.ndarray, lower_hsv=(25, 40, 40), upper_hsv=(85, 255, 255)) -> np.ndarray:
    """
    Computes green vegetation mask in HSV color space to isolate plant canopy from soil/background.

    Args:
        image (np.ndarray): Input RGB/BGR image array.
        lower_hsv (tuple): Lower bound for HSV thresholding.
        upper_hsv (tuple): Upper bound for HSV thresholding.

    Returns:
        np.ndarray: Binary mask (255 for foliage, 0 for background).
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(lower_hsv), np.array(upper_hsv))
    
    # Morphological cleaning to remove isolated noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

def compute_distance_transform(binary_mask: np.ndarray) -> np.ndarray:
    """
    Computes Euclidean Distance Transform on the canopy binary mask to quantify canopy thickness.

    Args:
        binary_mask (np.ndarray): Binary canopy mask (uint8).

    Returns:
        np.ndarray: Normalized distance transform image (0 - 255 uint8 for visualization).
    """
    dist = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
    dist_norm = cv2.normalize(dist, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return dist_norm

def compute_silhouette_convex_hull(binary_mask: np.ndarray, original_image: np.ndarray) -> tuple:
    """
    Fits convex hull around the canopy contour to compute canopy silhouette, solidity, and bounding box geometry.

    Args:
        binary_mask (np.ndarray): Binary canopy mask.
        original_image (np.ndarray): Original image for visualization overlay.

    Returns:
        tuple: (hull_visualization_image, metrics_dict)
            - hull_visualization_image (np.ndarray): Image with convex hull drawn.
            - metrics_dict (dict): Dictionary of geometrical canopy properties.
    """
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull_img = original_image.copy()
    
    if not contours:
        return hull_img, {'canopy_area': 0, 'hull_area': 0, 'solidity': 0, 'canopy_width': 0, 'canopy_height': 0}

    # Find largest contour corresponding to the plant bush
    c = max(contours, key=cv2.contourArea)
    canopy_area = cv2.contourArea(c)
    hull = cv2.convexHull(c)
    hull_area = cv2.contourArea(hull)
    solidity = canopy_area / float(hull_area) if hull_area > 0 else 0

    x, y, w, h = cv2.boundingRect(c)

    # Draw contour and convex hull overlay
    cv2.drawContours(hull_img, [c], -1, (0, 255, 0), 2)
    cv2.drawContours(hull_img, [hull], -1, (0, 0, 255), 3)
    cv2.rectangle(hull_img, (x, y), (x + w, y + h), (255, 255, 0), 2)

    metrics = {
        'canopy_area': canopy_area,
        'hull_area': hull_area,
        'solidity': solidity,
        'canopy_width': w,
        'canopy_height': h
    }
    return hull_img, metrics

def extract_canopy_architecture_features(image: np.ndarray) -> dict:
    """
    Main pipeline function to extract all canopy architecture features from a single plant image.

    Args:
        image (np.ndarray): BGR image array.

    Returns:
        dict: Complete set of canopy masks, distance transform, hull overlay, and numerical metrics.
    """
    hsv_mask = compute_hsv_vegetation_mask(image)
    dist_map = compute_distance_transform(hsv_mask)
    hull_img, metrics = compute_silhouette_convex_hull(hsv_mask, image)

    return {
        'hsv_mask': hsv_mask,
        'distance_transform': dist_map,
        'silhouette_image': hull_img,
        'metrics': metrics
    }
