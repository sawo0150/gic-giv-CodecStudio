import argparse
import json
import random
import shutil
import tarfile
import urllib.request
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_DIR / "data" / "raw_downloads"
OUT_DIR = PROJECT_DIR / "data" / "demo_depth" / "diode_10"
DIODE_VAL_URL = "http://diode-dataset.s3.amazonaws.com/val.tar.gz"


def download(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        print(f"Using cached archive: {path}")
        return path
    print(f"Downloading {url} -> {path}")
    urllib.request.urlretrieve(url, path)
    return path


def main():
    parser = argparse.ArgumentParser(description="Prepare 10 lightweight DIODE validation RGB-depth-mask samples.")
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--cleanup", action="store_true", help="Delete extracted validation folder after copying samples.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    archive = download(DIODE_VAL_URL, RAW_DIR / "diode_val.tar.gz")
    extract_dir = RAW_DIR / "diode_val"
    if not extract_dir.exists():
        print(f"Extracting {archive}")
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(extract_dir)

    rgb_files = sorted(extract_dir.rglob("*.png"))
    samples = []
    for rgb in rgb_files:
        stem = rgb.stem
        depth = rgb.with_name(stem + "_depth.npy")
        mask = rgb.with_name(stem + "_depth_mask.npy")
        if depth.exists() and mask.exists():
            samples.append((rgb, depth, mask))

    if not samples:
        raise RuntimeError("No DIODE RGB/depth/mask triples found after extraction.")

    random.seed(args.seed)
    selected = random.sample(samples, min(args.num_samples, len(samples)))

    for folder in ["rgb", "depth", "mask"]:
        (OUT_DIR / folder).mkdir(parents=True, exist_ok=True)

    metadata = {"source": "DIODE validation", "url": DIODE_VAL_URL, "num_samples": len(selected), "samples": []}
    for idx, (rgb, depth, mask) in enumerate(selected):
        name = f"{idx:03d}"
        rgb_out = OUT_DIR / "rgb" / f"{name}_rgb.png"
        depth_out = OUT_DIR / "depth" / f"{name}_depth.npy"
        mask_out = OUT_DIR / "mask" / f"{name}_mask.npy"
        shutil.copy2(rgb, rgb_out)
        shutil.copy2(depth, depth_out)
        shutil.copy2(mask, mask_out)
        metadata["samples"].append({"name": name, "source_rgb": str(rgb), "rgb": str(rgb_out), "depth": str(depth_out), "mask": str(mask_out)})

    (OUT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Prepared {len(selected)} DIODE samples in {OUT_DIR}")

    if args.cleanup:
        shutil.rmtree(extract_dir, ignore_errors=True)
        print(f"Removed extracted folder: {extract_dir}")


if __name__ == "__main__":
    main()
