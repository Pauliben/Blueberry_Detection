"""
Visualization Generator & Mask Overlay Blending Module
Handles PyTorch-accelerated mask overlays and saves visual metric maps.
"""

import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F

def plot_translucent_segmented_mask_torch(file_names: list, images: list, results: list, mask_color=(255, 200, 100), alpha=0.5, use_gpu=True):
    """
    Accelerated PyTorch GPU mask blending for translucent detection visualization over image batches.

    Args:
        file_names (list): Output paths to save image files.
        images (list): List of input RGB numpy arrays.
        results (list): List of YOLO result objects.
        mask_color (tuple): RGB tuple for overlay color.
        alpha (float): Transparency factor (0.0 to 1.0).
        use_gpu (bool): Whether to leverage GPU if available.
    """
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    batch_size = len(images)
    blended_images = []

    for i in range(batch_size):
        img = images[i]
        result = results[i]
        img_tensor = torch.from_numpy(img).to(device).float() / 255.0
        mask = torch.zeros((img.shape[0], img.shape[1]), dtype=torch.float32, device=device)

        if result is not None and hasattr(result, 'masks') and result.masks is not None:
            for seg in result.masks.data:
                seg_resized = F.interpolate(
                    seg.unsqueeze(0).unsqueeze(0).float(),
                    size=(img.shape[0], img.shape[1]),
                    mode='nearest'
                ).squeeze()
                mask = torch.maximum(mask, seg_resized)

        mask_rgb = mask.unsqueeze(-1).expand(-1, -1, 3)
        mask_color_tensor = torch.tensor(mask_color, dtype=torch.float32, device=device) / 255.0
        overlay = mask_rgb * mask_color_tensor * alpha
        blended = img_tensor * (1 - mask_rgb * alpha) + overlay
        blended_np = (blended.cpu().numpy() * 255).astype(np.uint8)
        blended_images.append(blended_np)

    for i, file_name in enumerate(file_names):
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        cv2.imwrite(file_name, cv2.cvtColor(blended_images[i], cv2.COLOR_RGB2BGR))

def save_canopy_metric_maps(image: np.ndarray, architecture_results: dict, output_dirs: dict, sample_label: str):
    """
    Saves visual metric maps across the 5 plant architecture output subdirectories:
    - canopy_metric_visualization
    - distance_transform
    - final_canopy_mask
    - hsv_mask
    - silhouette_analysis

    Args:
        image (np.ndarray): Source BGR image.
        architecture_results (dict): Dictionary output from extract_canopy_architecture_features.
        output_dirs (dict): Dictionary mapping subdirectory key to destination folder path.
        sample_label (str): Clean sample image label (e.g., 'sample_01').
    """
    hsv_mask = architecture_results['hsv_mask']
    dist_map = architecture_results['distance_transform']
    sil_img = architecture_results['silhouette_image']
    metrics = architecture_results['metrics']

    # 1. Save HSV mask
    cv2.imwrite(os.path.join(output_dirs['hsv_mask'], f"{sample_label}_hsv_mask.jpg"), hsv_mask)

    # 2. Save Final Canopy Mask
    cv2.imwrite(os.path.join(output_dirs['final_canopy_mask'], f"{sample_label}_canopy_mask.jpg"), hsv_mask)

    # 3. Save Distance Transform Map with colormap
    dist_color = cv2.applyColorMap(dist_map, cv2.COLORMAP_JET)
    cv2.imwrite(os.path.join(output_dirs['distance_transform'], f"{sample_label}_distance_transform.jpg"), dist_color)

    # 4. Save Silhouette Analysis (Convex Hull & Bounding Box)
    cv2.imwrite(os.path.join(output_dirs['silhouette_analysis'], f"{sample_label}_silhouette_analysis.jpg"), sil_img)

    # 5. Save Canopy Metric Visualization (Annotated Combined View)
    combined_vis = image.copy()
    cv2.putText(combined_vis, f"Canopy Area: {int(metrics['canopy_area'])} px", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.putText(combined_vis, f"Solidity: {metrics['solidity']:.2f}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.putText(combined_vis, f"W: {metrics['canopy_width']} px, H: {metrics['canopy_height']} px", (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.imwrite(os.path.join(output_dirs['canopy_metric_visualization'], f"{sample_label}_canopy_metrics.jpg"), combined_vis)
