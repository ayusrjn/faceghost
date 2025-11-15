import cv2
import numpy as np


def gaussian_blur(
    img: np.ndarray, results: np.ndarray, KERNEL_SIZE: tuple
) -> np.ndarray:
    """
    Applies Gaussian Blur to the region of interest using the bounding boxes
    provided in the 'results' NumPy array.

    Args:
        img: The original image (H, W, 3) in BGR format.
        results: A NumPy array of detections, e.g., [[x1, y1, x2, y2, conf, class_id], ...].
        KERNEL_SIZE: The tuple (K, K) for the Gaussian Blur kernel (K must be odd).

    Returns:
        The image copy with blurred regions.
    """
    img_copy = img.copy()

    # If no results are passed, return the original image copy
    if results is None or results.size == 0:
        return img_copy

    H, W = img.shape[:2]

    # Iterate over each detected box in the NumPy array
    for box in results:
        # Assuming the first four elements are the scaled integer coordinates [x1, y1, x2, y2]
        x1, y1, x2, y2 = box[:4].astype(int)

        # Ensure coordinates are in correct order (for safety, though infer.py should handle this)
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)

        # Clip coordinates to be within image bounds
        x1 = np.clip(x1, 0, W)
        x2 = np.clip(x2, 0, W)
        y1 = np.clip(y1, 0, H)
        y2 = np.clip(y2, 0, H)

        # Extract the Region of Interest (ROI) from the image copy
        roi = img_copy[y1:y2, x1:x2]

        # Check if the ROI is valid (non-zero area)
        if roi.shape[0] > 0 and roi.shape[1] > 0:
            # Apply Gaussian Blur to the ROI
            blurred_roi = cv2.GaussianBlur(roi, KERNEL_SIZE, 0)

            # Place the blurred ROI back into the image copy
            img_copy[y1:y2, x1:x2] = blurred_roi

    return img_copy

