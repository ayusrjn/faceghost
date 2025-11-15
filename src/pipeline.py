import cv2
import argparse, shutil, os
from yolo_infer import predict
from blur import gaussian_blur
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

args = ap.parse_args()


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


blurred_frame = gaussian_blur(img, results, KERNEL_SIZE)

filename_base = os.path.splitext(os.path.basename(img_path))[0]
output_filename = f"{filename_base}_blurred.jpg"


cv2.imwrite(output_filename, blurred_frame)
print(f"Successfully blurred and saved image to : {output_filename}")
