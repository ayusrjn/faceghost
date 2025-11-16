import onnxruntime as ort
import numpy as np
import cv2
import importlib.resources as resources
import os

session = None
INPUT_NAME = None
OUTPUT_NAME = None

INPUT_SIZE = 640
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45

MODEL_PATH = 'model'
MODEL_FILE = 'v0-1.onnx'

def get_model_path():
    resource_path = resources.files('faceghost') / MODEL_PATH / MODEL_FILE
    return resource_path

def load_onnx_model():
    """Loads the ONNX model using its path."""
    
    global session, INPUT_NAME, OUTPUT_NAME 
    # -----------------------------
    
    model_path = get_model_path()
    
    
    if model_path.exists():
        
        try:
            
            session = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])
            INPUT_NAME = session.get_inputs()[0].name
            OUTPUT_NAME = session.get_outputs()[0].name
            
        except Exception as e:
            print(f"Error loading ONNX model: {e}")
            session = None
            INPUT_NAME = None
            OUTPUT_NAME = None
    else:
        print("Error: Model file does not exist at the resource path.")

load_onnx_model()


def preprocess(img: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Resizes, converts to RGB, normalizes, and reshapes for ONNX input."""
    
    # Save original dimensions for rescaling bounding boxes later
    original_h, original_w = img.shape[:2]

    # Resize image to model input size (640x640)
    img_resized = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
    
    # BGR to RGB conversion
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    
    # HWC to CHW (3, 640, 640)
    input_tensor = img_rgb.transpose((2, 0, 1))

    # Normalize (0-255 to 0.0-1.0) and convert to float32
    input_tensor = input_tensor.astype(np.float32) / 255.0

    # Add batch dimension (1, 3, 640, 640)
    input_tensor = np.expand_dims(input_tensor, 0)
    
    return input_tensor, original_w, original_h

def postprocess(output: np.ndarray, original_w: int, original_h: int) -> np.ndarray:
    """Filters, scales, and applies NMS to the raw model output."""
    
    # The output tensor shape is typically (1, 84, N), where 84 = 4 (bbox) + 1 (conf) + 79 (classes)
    # Transpose to (N, 84) to make it easier to work with.
    predictions = np.squeeze(output).T 

    # Filter by objectness confidence (max score across all classes)
    scores = np.max(predictions[:, 4:], axis=1)
    predictions = predictions[scores >= CONF_THRESHOLD, :]
    scores = scores[scores >= CONF_THRESHOLD]
    
    if len(predictions) == 0:
        return np.array([])

    # Get bounding boxes and class IDs
    boxes = predictions[:, :4]
    class_ids = np.argmax(predictions[:, 4:], axis=1)

    # Convert boxes from (x_center, y_center, width, height) to (x1, y1, x2, y2)
    boxes_xyxy = np.copy(boxes)
    boxes_xyxy[..., 0] = boxes[..., 0] - boxes[..., 2] / 2 # x1
    boxes_xyxy[..., 1] = boxes[..., 1] - boxes[..., 3] / 2 # y1
    boxes_xyxy[..., 2] = boxes[..., 0] + boxes[..., 2] / 2 # x2
    boxes_xyxy[..., 3] = boxes[..., 1] + boxes[..., 3] / 2 # y2

    # Rescale coordinates to original image dimensions
    ratio_w = original_w / INPUT_SIZE
    ratio_h = original_h / INPUT_SIZE
    boxes_xyxy[:, 0] *= ratio_w
    boxes_xyxy[:, 2] *= ratio_w
    boxes_xyxy[:, 1] *= ratio_h
    boxes_xyxy[:, 3] *= ratio_h
    
    # Clip coordinates to bounds (already done in blur.py, but good practice here too)
    boxes_xyxy = np.clip(boxes_xyxy, 0, [original_w, original_h, original_w, original_h])
    
    # Prepare boxes in (x, y, w, h) format required by cv2.dnn.NMSBoxes
    boxes_xywh = np.array([
        [x1, y1, x2 - x1, y2 - y1] for x1, y1, x2, y2 in boxes_xyxy
    ]).astype(np.float32)

    # Apply Non-Maximum Suppression (NMS)
    indices = cv2.dnn.NMSBoxes(
        boxes_xywh, 
        scores.tolist(), 
        CONF_THRESHOLD, 
        IOU_THRESHOLD
    )
    
    # Flatten the indices array if NMS found results
    if len(indices) == 0:
        return np.array([])

    final_indices = indices.flatten()
    
    # Combine final results into a single NumPy array: [x1, y1, x2, y2, score, class_id]
    final_boxes = boxes_xyxy[final_indices]
    final_scores = scores[final_indices]
    final_class_ids = class_ids[final_indices]
    
    # Stack the final results (N, 6)
    results_array = np.hstack((
        final_boxes, 
        final_scores[:, np.newaxis], 
        final_class_ids[:, np.newaxis]
    )).astype(np.float32)
    
    return results_array

def predict(img: np.ndarray) -> np.ndarray:
    """
    Runs the full ONNX inference pipeline (Pre-process, Infer, Post-process).
    Returns a NumPy array of detected objects: [x1, y1, x2, y2, conf, class_id].
    """
    if session is None:
        print("Model session not initialized. Check ONNX_MODEL_PATH.")
        return np.array([])
        
    # 1. Pre-process
    input_tensor, original_w, original_h = preprocess(img)

    # 2. Run Inference
    raw_output = session.run([OUTPUT_NAME], {INPUT_NAME: input_tensor})[0]

    # 3. Post-process
    final_detections = postprocess(raw_output, original_w, original_h)
    
    # Convert coordinates to integers for direct use in blur.py
    if final_detections.size > 0:
        final_detections[:, :4] = final_detections[:, :4].astype(int)
        
    return final_detections

# debug utils

# from ultralytics import YOLO


# model = YOLO("../models/v0-1.pt")


# def predict(img):

#     results = model.predict(source=img, save=False, conf=0.3, verbose=False)
#     return results
    