# faceghost - face anonymization for image & videos

faceghost is a lightweight Python package for automatically detecting and anonymizing faces in images and videos.

It uses a YOLO-based face detector and offers multiple blur and pixelation methods, making it ideal for datasets, research, and privacy-sensitive media.

| Original Frame | `faceghost` output (using default options)|
|--------|-------|
| ![Original Frame](https://raw.githubusercontent.com/ayusrjn/face-anonymizer/refs/heads/video-support/images/target2.jpg) | ![`faceghost` Output](https://raw.githubusercontent.com/ayusrjn/face-anonymizer/refs/heads/video-support/images/target2_blurred.jpg) |

## Features

- Detects faces using a YOLO-based detector (`yolo_infer.predict`) and applies blurring only to detected face regions.
- Multiple blur modes:
  - `gaussian` — oval Gaussian blur (default)
  - `gaussian_sqr` — square Gaussian blur
  - `mossaic` — pixelation / mosaic blur
  - `median` — median filter blur
- Works on a single image, a directory of images, or a full video.
- CLI-ready and simple Python API for embedding in other projects.
- Safe fallback handling (validates inputs, ensures kernel size is a positive odd integer).

## Installation
`faceghost` supports all commonly operating system like linux, windows, mac. It can be used both on CLI like bash, shell, powershell.

Intallation of `faceghost` can be done through `pip` 

`pip install faceghost`

## Quick CLI Usage

Process a single Image:

`faceghost --img /path/to/photo.jpg`

Process a directory of images (saves blurred images into /folder_specified/face_anonymized/):

`faceghost --dir /path/to/images --kernel 51 --blur mossaic`

Process a video (outputs basename_blurred.mp4 in the current working directory):

`faceghost --vid /path/to/video.mp4 --kernel 41 --blur gaussian_sqr`

#### CLI argument notes

- `--img` / `--vid` / `--dir` — supply one of these (mutually exclusive).

- `--kernel` — blur size (default `39`). The package enforces a positive odd kernel; if you pass an even number it will be rounded up to the next odd integer. For Gaussian-style blurs the kernel is used as a `(k,k)` tuple; for mosaic/median blurs it is used as a single integer magnitude.

- `--blur` — one of `gaussian`, `gaussian_sqr`, `mossaic`, `median`.

## Python API 

You can use the function directly in Python:
```python 
from faceghost import run_on_image, run_on_dir, run_on_video

# Single image
run_on_image("photo.jpg", blur_name="mossaic", kernel_val=51, output_path="outdir")

# Directory
run_on_dir("data/images", blur_name="gaussian", kernel_val=39)

# Video
run_on_video("input.mp4", blur_name="gaussian_sqr", kernel_val=41)
```

#### Important internal helpers (useful if you integrate the pipeline):

- `select_blur_function(blur_name)` — returns the blur function and whether it expects a tuple or int kernel.
- `process_frame(frame, detect_results, blur_fn, kernel_tuple, kernel_int)` — applies the selected blur to the provided frame using detection results.
- The detection step is performed by calling `predict(frame)` from the `yolo_infer` module; this function must return the detections in the format the blur functions expect.

