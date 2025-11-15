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

    if results is None or results.size == 0:
        return img_copy

    H, W = img.shape[:2]

    for box in results:
        
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

def gaussian_oval_blur(img: np.ndarray, results: np.ndarray, KERNEL_SIZE: tuple) -> np.ndarray:
    """
    Apply Gaussian blur inside an ellipse (oval) region for each bounding box.

    Expects `results` with boxes in format [x1, y1, x2, y2, ...].
    (If your model uses a different box format, adapt the unpacking.)
    KERNEL_SIZE is used as provided (assumed validated elsewhere).
    """
    img_copy = img.copy()

    if results is None or results.size == 0:
        return img_copy

    h, w = img.shape[:2]

    for box in results:

        x1, y1, x2, y2 = box[:4].astype(int)

        
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)

        x1 = int(np.clip(x1, 0, w))
        x2 = int(np.clip(x2, 0, w))
        y1 = int(np.clip(y1, 0, h))
        y2 = int(np.clip(y2, 0, h))

        roi_h = y2 - y1
        roi_w = x2 - x1
        if roi_h <= 0 or roi_w <= 0:
            continue

        roi = img_copy[y1:y2, x1:x2]

        
        axis_w = max(1, roi_w // 2)
        axis_h = max(1, roi_h // 2)

        
        blurred_roi_rect = cv2.GaussianBlur(roi, KERNEL_SIZE, 0)

        mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
        center = (roi_w // 2, roi_h // 2)
        axes = (axis_w, axis_h)
        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)

        mask_inv = cv2.bitwise_not(mask)

        roi_background = cv2.bitwise_and(roi, roi, mask=mask_inv)
        blurred_foreground = cv2.bitwise_and(blurred_roi_rect, blurred_roi_rect, mask=mask)

        final_blurred_roi = cv2.add(roi_background, blurred_foreground)
        img_copy[y1:y2, x1:x2] = final_blurred_roi

    return img_copy

# debug utils

# import cv2
# def gaussian_blur(img, results, KERNEL_SIZE):
#     img_copy = img.copy()
#     for r in results:
#         boxes = r.boxes
#         for box in boxes:
#             x1, y1, x2, y2 = box.xyxy[0].int().cpu().numpy()
#             roi = img[y1:y2,x1:x2]
#             if roi.shape[0] > 0 and roi.shape[1] >0:
#                 blurred_roi = cv2.GaussianBlur(roi, KERNEL_SIZE, 0)
#                 img_copy[y1:y2, x1:x2] = blurred_roi
#     return img_copy
