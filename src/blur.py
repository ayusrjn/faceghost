import cv2

def gaussian_blur(img, results, KERNEL_SIZE):
	""" 
		Applies Gaussian Blur to the region of interest 
		the boxes returned by the predicting model
	"""
    img_copy = img.copy()

    for r in results:

        boxes = r.boxes
        for box in boxes:

            x1, y1, x2, y2 = box.xyxy[0].int().cpu().numpy()
            roi = img[y1:y2, x1:x2]
            if roi.shape[0] > 0 and roi.shape[1] > 0:

                blurred_roi = cv2.GaussianBlur(roi, KERNEL_SIZE, 0)
                img_copy[y1:y2, x1:x2] = blurred_roi

    return img_copy
