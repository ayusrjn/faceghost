import cv2
import argparse, shutil, os
from yolo_infer import predict
from blur import gaussian_blur, pixelation_mossaic as mossaic, median_blurring as median, gaussian_oval_blur as gaussian_oval

""" parsing the argument passed on cli"""

ap = argparse.ArgumentParser()
ap.add_argument("--img", required=True, help="Path to the input image file.")
ap.add_argument(
    "--kernel",
    type=int,
    default=39,
    required=False,
    help="Size of the Gaussian Blur Kernel (e.g., 79 for 79X79) Must be a positive odd integer.",
)
ap.add_argument("--vid", required=False, help="Specify the path for video file")
ap.add_argument("--cam", required=False, help="Specify The Cam Mount - list by face_anonymization cams -l")
args = ap.parse_args()

# ap.add_argument("--video", required=False, help="Path to the input video files")

# if (args.vid is Not None):
#     vid_path = args.vid_path
# if(args.cam is Not None):
#     cam_path = args.cam



img_path = args.img
kernel_val = args.kernel

if kernel_val % 2 == 0:  # Kernel has to be an odd number for Gaussian Blur
    kernel_val += 1
if kernel_val <= 0:
    kernel_val = 79

KERNEL_SIZE = (kernel_val, kernel_val)

img = cv2.imread(img_path, cv2.IMREAD_COLOR)

if img is None:
    print(f"Error: Could not open or find the image at '{img_path}")
    exit()


results = predict(img)

''' function can be used for benchmarking'''

# from utils_debug import save_viz
# save_viz(img, results, "debug_boxes.jpg")



blurred_frame = gaussian_oval(img, results, KERNEL_SIZE)

# blurred_frame = median(img, results, 29)

filename_base = os.path.splitext(os.path.basename(img_path))[0]
output_filename = f"{filename_base}_blurred.jpg"


cv2.imwrite(output_filename, blurred_frame)
print(f"Successfully blurred and saved image to : {output_filename}")
