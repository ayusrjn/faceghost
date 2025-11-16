import cv2
import numpy as np


# implementation fo pixelation blur Mossaic effect
def pixelation_mossaic(
    img: np.ndarray, results: np.ndarray, pixelation_factor: int
) -> np.ndarray:
    """pixelation_mossaic implements the mossaic blur effect to give out the blurred region
    Arguments:
        img : Numpy array of colored representation of the image
        results: Results returned from ONNX system
        pixelation_factor : Blur factoring to divide the pixels into

    Output:
        img_copy : numpy array of colored representation of the image with blurred regions
    """
    h, w = img.shape[:2]
    img_copy = img.copy()

    if results is None or results.size == 0:
        return img_copy

    for box in results:

        x1, y1, x2, y2 = box[:4].astype(int)
    

        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)

        x1 = np.clip(x1, 0, w)
        x2 = np.clip(x2, 0, w)
        y1 = np.clip(y1, 0, h)
        y2 = np.clip(y2, 0, h)

        roi = img[y1:y2, x1:x2]

        if roi.shape[0] > 0 and roi.shape[1] > 0:
            roi_h, roi_w = roi.shape[:2]

            small_roi_h, small_roi_w = max(1, roi_h // pixelation_factor), max(
                1, roi_w // pixelation_factor
            )

            blurred_region = cv2.resize(
                roi, (small_roi_w, small_roi_h), interpolation=cv2.INTER_LINEAR
            )  # downscaling

            blurred_img_region = cv2.resize(
                blurred_region, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST
            )  # upscaling

            img_copy[y1:y2, x1:x2] = blurred_img_region

    return img_copy


def _get_img_shape(img):
    """ _get_img_shape returns the dimension of the image
        Arguments: img Takes in an image ndarray

        Output: returns a tuple with height and weight
    """
    h, w = img.shape[:2]
    return (h, w)


def median_blurring(img: np.ndarray, results: np.ndarray, kernel: int):
    """ median_blurring function implements the blurring by taking the median of the pixel and interpolating the surronding area
        Arguments: img -> np.ndarray containing the colored representation of the img
                    results -> np.ndarray containing the result returned for ONNX model
                    kernel -> the size of the kernel which will be used to interpolate the model\

        Outputs:

    """
    h, w = _get_img_shape(img)

    if kernel % 2 == 0:
        kernel += 1

    img_copy = img.copy()

    if results is None or results.size == 0:
        return img_copy

    for box in results:

        x1, y1, x2, y2 = box[:4].astype(int)

        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)

        x1 = np.clip(x1, 0, w)
        x2 = np.clip(x2, 0, w)
        y1 = np.clip(y1, 0, h)
        y2 = np.clip(y2, 0, h)

        roi = img_copy[y1:y2, x1:x2]

        roi_h, roi_w = y2 - y1, x2 - x1

        if roi_h > 0 and roi_w > 0:

            blurred_region = cv2.medianBlur(roi, kernel)

            mask = np.zeros((roi_h, roi_w, 1), dtype=np.uint8)

            center = roi_w // 2, roi_h // 2
            axis = roi_w // 2, roi_h // 2
            rotation = 0

            cv2.ellipse(mask, center, axis, rotation, 0, 360, 255, -1) #ellipse on the mask

            mask_inv = cv2.bitwise_not(mask) # Inversing the mask for the roi background
            roi_background = cv2.bitwise_and(roi, roi, mask=mask_inv) # preserving the roi square background
            blurred_foreground = cv2.bitwise_and(blurred_region, blurred_region, mask=mask) #blurring the face ellipse
            final_blurred_roi = cv2.add(roi_background, blurred_foreground)

        img_copy[y1:y2, x1:x2] = blurred_region

    return img_copy

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
