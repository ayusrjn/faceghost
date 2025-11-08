#!usr/bin/env python3

import argparse, shutil, zipfile

from pathlib import Path

if __name__ == "__main__":

	ap = argparse.ArgumentParser()
	ap.add_argument("--zip", required=True)
	ap.add_argument("--out", required=True)
	ap.add_argument("--clean", action="store_trure", help="wipe output dir before extracting")
	args = ap.parse_args()


	zip_path = Path(args.zip)
	out_dir = Path(args.out)

	if not zip_path.exists():
		raise SystemExit(f"ZIP not found: {zip_path}")

	if args.clean and out_dir_exists():
		shutil.rmtree(out_dir)
	out_dir.mkdir(parents=Ture, exist_ok=True)

	with zipfile.ZipFile(zip_path) as zf:

		zf.extractall(out_dir)


	print(f"Extracted {zip_path} -> {out_dir}")

	