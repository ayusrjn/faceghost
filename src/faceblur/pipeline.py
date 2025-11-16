import cv2
import argparse, shutil, os
from .yolo_infer import predict
from tqdm import tqdm
import math

from .blur import (
    gaussian_blur,
    pixelation_mossaic as mossaic,
    median_blurring as median,
    gaussian_oval_blur as gaussian_oval
)

def parse_args():

    ''' Parsing the Argument passed through the Command Line 
        Group Any one of the Following is required --img --vid this is the path to the file
        --kernel : the kernel size of the blur
        --blur : The blur type of the blurring algorithm you want to use

        Outputs:
            The parsed arguments from the Command Line
    '''

    ap = argparse.ArgumentParser()

    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--img", help="Path to the imput image file")
    group.add_argument("--vid", help="Path to the video file")
    group.add_argument("--dir", help="Path to the Folder where the images are")

    ap.add_argument(
        "--kernel",
        type=int,
        default=39,
        required=False,
        help="Size of the Gaussian Blur Kernel (e.g., 79 for 79X79) Must be a positive odd integer.",
    )
    ap.add_argument(
        "--blur",
        type=str,
        default="gaussian",
        required=False,
        choices=["gaussian", "gaussian_sqr", "mossaic", "median"],
        help=''' Select Different Blur with this arguments
                     Current Supported are
                     gaussian : Oval Gaussian Blur
                     gaussian_sqr : Sqaure Gaussian Blur
                     mossaic : Pixelation Mossaic Blur
                     median : Median Blur
        '''
    )

    return ap.parse_args()

def make_kernel_odd(k):
    k = int(k)
    if k <= 0:
        k = 39
    if k % 2 == 0:
        k += 1
    return k

def select_blur_function(blur_name):
    """ Returns (func, expects_kernel_tuple:bool, expect_kernel_int:bool)
        expects_kernel_tuple: true if function expects a tuple kernel size (kx, ky)
        expects_kernel_int: true if function expects an integer kernel/strength
    """
    if blur_name == "gaussian":
        return gaussian_oval, True, False
    if blur_name == "gaussian_sqr":
        return gaussian_blur, True, False
    if blur_name == "mossaic":
        return mossaic, False, True
    if blur_name == "median":
        return median, False, True

    return gaussian_oval, True, False #Fallback

# img_path = args.img
# kernel_val = args.kernel
# blur = args.blur

def process_frame(frame, detect_results, blur_fn, kernel_tuple,kernel_int):
    """
        Apply blur function to a single frame using detect_result returned from predict(frame).
    """
    if kernel_tuple is not None:
        return blur_fn(frame, detect_results, kernel_tuple)
    else:
        return blur_fn(frame, detect_results, kernel_int)


def run_on_image(img_path, blur_name, kernel_val, output_path):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not open image :{img_path}")

    blur_fn, expects_tuple, expects_int = select_blur_function(blur_name)
    kernel_tuple = (kernel_val, kernel_val) if expects_tuple else None
    kernel_int = kernel_val if expects_int else None

    detect_results = predict(img)
    out = process_frame(img, detect_results,blur_fn,kernel_tuple, kernel_int)

    filename_base = os.path.splitext(os.path.basename(img_path))[0]


    if out is not None:
        directory_to_save = output_path + "/face_anonymized"
        os.makedirs(directory_to_save, exist_ok=True)
        save_path = directory_to_save + f"/{filename_base}_blurred.jpg"
        cv2.imwrite(str(save_path), out)

        print(f"Blurred Image at : {save_path}")

    else:
        cv2.imwrite(f"{filename_base}_blurred.jpg", out)
        print(f"Saved blurred image  :{filename_base}")

    

def run_on_dir(dir_path, blur_name, kernel_val):

    if not os.path.isdir(dir_path):
        raise FileNotFoundError(f"Could not the Directory :{dir_path}")

    for filename in tqdm(os.listdir(dir_path), desc="Processing Images"):
        if filename.lower().endswith(('.png', '.jpg', 'jpeg')):
            img_path = os.path.join(dir_path, filename)
            run_on_image(img_path, blur_name, kernel_val, dir_path)

    print(f"Successful Run in Directory : {dir_path}")

def run_on_video(video_path, blur_name, kernel_val):

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <=1:
        fps = 30.0

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    base = os.path.splitext(os.path.basename(video_path))[0]
    output_file = f"{base}_blurred.mp4"

    writer = cv2.VideoWriter(output_file,fourcc, fps, (w, h))
    if not writer.isOpened():
        cap.release()
        raise IOError("Failed to open output video writer")

    blur_fn, expects_tuple, expects_int = select_blur_function(blur_name)
    kernel_tuple = (kernel_val, kernel_val) if expects_tuple else None
    kernel_int = kernel_val if expects_int else None

    progress_bar = tqdm(total_frame if total_frame > 0 else None,
                        desc=f"Processing Video {video_path}",
                        unit=" frame", ncols=100)

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = predict(frame)
            processed_frame = process_frame(frame, results, blur_fn, kernel_tuple, kernel_int)

            writer.write(processed_frame)

            frame_idx += 1
            progress_bar.update(1)

            
    finally:
        progress_bar.close()

        cap.release()
        writer.release()
        print(f"Saved Blurred Video -> {output_file}")


if __name__ == "__main__":
    args = parse_args()
    kernel_val = make_kernel_odd(args.kernel)
    blur_choice = args.blur

    if args.img:
        run_on_image(args.img, blur_choice, kernel_val)
    if args.dir:
        run_on_dir(args.dir, blur_choice, kernel_val)
    if args.vid:
        run_on_video(args.vid, blur_choice, kernel_val)

