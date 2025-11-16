# faceblur - face anonymization for image & videos
![PyPI](https://img.shields.io/pypi/v/faceghost)
![Python Versions](https://img.shields.io/pypi/pyversions/faceghost)
![Status](https://img.shields.io/badge/status-active-brightgreen)
![Build](https://img.shields.io/badge/build-passing-success)
![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)
![Tests](https://img.shields.io/badge/tests-passing-brightgreen)
![Benchmark Verified](https://img.shields.io/badge/benchmark-verified-blue)



Faceblur is a lightweight Python package for automatically detecting and anonymizing faces in images and videos.

It uses a YOLO-based face detector and offers multiple blur and pixelation methods, making it ideal for datasets, research, and privacy-sensitive media.

| Original Frame | `faceblur` output (using default options)|
|--------|-------|
| ![Original Frame](https://raw.githubusercontent.com/ayusrjn/face-anonymizer/refs/heads/video-support/images/target2.jpg) | ![`faceblur` Output](https://raw.githubusercontent.com/ayusrjn/face-anonymizer/refs/heads/video-support/images/target2_blurred.jpg) |

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
`faceblur` supports all commonly operating system like linux, windows, mac. It can be used both on CLI like bash, shell, powershell.

Intallation of `faceblur` can be done through `pip` 

`pip install faceblur`

## Quick CLI Usage

Process a single Image:

`faceblur --img /path/to/photo.jpg`

Process a directory of images (saves blurred images into /folder_specified/face_anonymized/):

`faceblur --dir /path/to/images --kernel 51 --blur mossaic`

Process a video (outputs basename_blurred.mp4 in the current working directory):

`faceblur --vid /path/to/video.mp4 --kernel 41 --blur gaussian_sqr`

#### CLI argument notes

- `--img` / `--vid` / `--dir` — supply one of these (mutually exclusive).

- `--kernel` — blur size (default `39`). The package enforces a positive odd kernel; if you pass an even number it will be rounded up to the next odd integer. For Gaussian-style blurs the kernel is used as a `(k,k)` tuple; for mosaic/median blurs it is used as a single integer magnitude.

- `--blur` — one of `gaussian`, `gaussian_sqr`, `mossaic`, `median`.

## Python API 

You can use the function directly in Python:
```python 
from faceblur import run_on_image, run_on_dir, run_on_video

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
## Performance Benchmark

A speed comparison between **`faceghost`** (Package A) and the popular anonymizer **`deface`** (Package B), tested on 5 videos using identical conditions.

| Video | FaceGhost (A) — s | deface (B) — s | Speed-up (B / A) | % Time Saved |
|-------|-------------------:|---------------:|------------------:|--------------:|
| video1.mp4 | 11.39 | 65.39  | 5.74× | 82.6% |
| video2.mp4 | 19.81 | 129.53 | 6.54× | 84.7% |
| video3.mp4 | 20.60 | 139.10 | 6.75× | 85.2% |
| video4.mp4 | 10.90 | 76.03  | 6.98× | 85.7% |
| video5.mp4 | 35.32 | 242.30 | 6.86× | 85.4% |
| **TOTAL**  | **98.02** | **652.35** | **6.66×** | **85.0%** |

###  Summary
FaceGhost is **~6.7× faster** than deface, reducing end-to-end anonymization time by **~85%** across the test suite.

---

###  Benchmark Setup

**Hardware:**  
- Ubuntu Linux  
- 16 GB RAM  
- NVIDIA GTX 2050 (8 GB VRAM)

**Method:**  
- Each package run via CLI using the same video files  
- Timed using Python `time` module  
- Command templates used:
  - FaceGhost: `faceblur --vid {video}`
  - deface: `deface {video}`
- Five videos processed sequentially

---

