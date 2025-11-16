# src/faceghost/__init__.py

from importlib.metadata import PackageNotFoundError, version as _get_version

try:
    __version__ = _get_version("faceghost")   
except PackageNotFoundError:
    __version__ = "0.0.0"

from .pipeline import (
    run_on_image,
    run_on_dir,
    run_on_video,
)


def main():
    from .pipeline import parse_args, make_kernel_odd, run_on_image, run_on_dir, run_on_video

    args = parse_args()
    kernel_val = make_kernel_odd(args.kernel)
    blur_choice = args.blur

    if args.img:
        run_on_image(args.img, blur_choice, kernel_val, output_path=".")
    if args.dir:
        run_on_dir(args.dir, blur_choice, kernel_val)
    if args.vid:
        run_on_video(args.vid, blur_choice, kernel_val)

    
__all__ = [
    "run_on_image",
    "run_on_dir",
    "run_on_video",
    "main",
    "__version__",
]
