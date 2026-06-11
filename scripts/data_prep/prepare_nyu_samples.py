import argparse
import json
import shutil
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


PROJECT_DIR = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_DIR / "data" / "demo_depth" / "nyu_10"
IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
DEPTH_EXTS = {".npy", ".png", ".tif", ".tiff"}
OFFICIAL_SAMPLE_URL = "https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2_web.jpg"


def find_pairs(source):
    source = Path(source)
    rgb_dir = source / "rgb"
    depth_dir = source / "depth"
    if rgb_dir.exists() and depth_dir.exists():
        rgbs = sorted([p for p in rgb_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS])
        depths = sorted([p for p in depth_dir.iterdir() if p.suffix.lower() in DEPTH_EXTS])
        return list(zip(rgbs, depths))

    rgbs = sorted([p for p in source.rglob("*") if p.suffix.lower() in IMAGE_EXTS and "depth" not in p.stem.lower()])
    pairs = []
    for rgb in rgbs:
        stem = rgb.stem.replace("_rgb", "")
        candidates = []
        for suffix in DEPTH_EXTS:
            candidates.extend([rgb.with_name(stem + "_depth" + suffix), rgb.with_name(stem + suffix)])
        depth = next((p for p in candidates if p.exists() and p != rgb), None)
        if depth:
            pairs.append((rgb, depth))
    return pairs


def make_fallback_samples(num_samples, width=384, height=288):
    (OUT_DIR / "rgb").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "depth").mkdir(parents=True, exist_ok=True)

    x = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    metadata = {
        "source": "procedural_fallback",
        "num_samples": int(num_samples),
        "note": "NYU-like indoor RGB-D fallback. Use real NYU samples when available for final quantitative claims.",
        "samples": [],
    }

    for idx in range(int(num_samples)):
        t = idx / max(1, num_samples - 1)
        rgb = np.zeros((height, width, 3), dtype=np.uint8)
        rgb[..., 0] = np.clip(95 + 45 * x + 12 * idx, 0, 255).astype(np.uint8)
        rgb[..., 1] = np.clip(105 + 55 * y, 0, 255).astype(np.uint8)
        rgb[..., 2] = np.clip(125 + 30 * (1.0 - x), 0, 255).astype(np.uint8)

        img = Image.fromarray(rgb)
        draw = ImageDraw.Draw(img)
        floor_y = int(height * (0.58 + 0.04 * np.sin(t * np.pi)))
        draw.polygon([(0, floor_y), (width, floor_y - 18), (width, height), (0, height)], fill=(132, 119, 98))
        draw.rectangle([int(width * 0.08), int(height * 0.18), int(width * 0.32), floor_y], fill=(82, 110, 145))
        draw.rectangle([int(width * 0.66), int(height * 0.24), int(width * 0.88), floor_y - 10], fill=(178, 101, 82))

        table_x0 = int(width * (0.33 + 0.03 * np.sin(t * np.pi * 2.0)))
        table_y0 = int(height * 0.49)
        draw.rectangle([table_x0, table_y0, table_x0 + 115, table_y0 + 28], fill=(96, 74, 55))
        draw.rectangle([table_x0 + 12, table_y0 + 28, table_x0 + 26, floor_y + 42], fill=(72, 57, 44))
        draw.rectangle([table_x0 + 92, table_y0 + 28, table_x0 + 106, floor_y + 42], fill=(72, 57, 44))

        obj_x = table_x0 + 56
        obj_y = table_y0 - 18
        draw.ellipse([obj_x - 18, obj_y - 18, obj_x + 18, obj_y + 18], fill=(232, 169, 57))

        depth = 0.18 + 0.64 * y + 0.08 * x
        depth = np.broadcast_to(depth, (height, width)).copy()
        yy, xx = np.ogrid[:height, :width]
        floor_mask = np.broadcast_to(yy >= floor_y - 12, (height, width))
        left_wall = (xx > int(width * 0.08)) & (xx < int(width * 0.32)) & (yy > int(height * 0.18)) & (yy < floor_y)
        right_wall = (xx > int(width * 0.66)) & (xx < int(width * 0.88)) & (yy > int(height * 0.24)) & (yy < floor_y)
        table_mask = (xx > table_x0) & (xx < table_x0 + 115) & (yy > table_y0) & (yy < table_y0 + 34)
        obj_mask = (xx - obj_x) ** 2 + (yy - obj_y) ** 2 < 20 ** 2
        floor_depth = np.broadcast_to(0.70 + 0.12 * y, (height, width))
        depth[floor_mask] = floor_depth[floor_mask]
        depth[left_wall] = 0.52
        depth[right_wall] = 0.47
        depth[table_mask] = 0.35
        depth[obj_mask] = 0.22

        name = f"{idx:03d}"
        rgb_out = OUT_DIR / "rgb" / f"{name}_rgb.png"
        depth_out = OUT_DIR / "depth" / f"{name}_depth.npy"
        img.save(rgb_out)
        np.save(depth_out, np.clip(depth, 0.0, 1.0).astype(np.float32))
        metadata["samples"].append({"name": name, "rgb": str(rgb_out), "depth": str(depth_out)})

    (OUT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Prepared {num_samples} NYU-like fallback samples in {OUT_DIR}")


def _depth_from_nyu_colorized(depth_rgb):
    arr = np.asarray(depth_rgb).astype(np.float32) / 255.0
    score = 0.70 * arr[..., 0] + 0.35 * arr[..., 1] - 0.30 * arr[..., 2]
    lo = float(np.percentile(score, 2))
    hi = float(np.percentile(score, 98))
    if hi <= lo:
        return np.zeros(score.shape, dtype=np.float32)
    return np.clip((score - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def prepare_official_web_samples(num_samples):
    (OUT_DIR / "rgb").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "depth").mkdir(parents=True, exist_ok=True)
    raw_dir = PROJECT_DIR / "data" / "raw_downloads"
    raw_dir.mkdir(parents=True, exist_ok=True)
    montage_path = raw_dir / "nyu_depth_v2_web.jpg"
    if not montage_path.exists():
        print(f"Downloading official NYU Depth V2 sample montage: {OFFICIAL_SAMPLE_URL}")
        urllib.request.urlretrieve(OFFICIAL_SAMPLE_URL, montage_path)

    montage = Image.open(montage_path).convert("RGB")
    width, height = montage.size
    cols = [round(width * i / 6) for i in range(7)]
    rows = [round(height * i / 5) for i in range(6)]
    samples = []
    for row in range(5):
        samples.append((row, 0, 1))
        samples.append((row, 3, 4))

    metadata = {
        "source": "nyu_depth_v2_official_web_montage",
        "source_url": OFFICIAL_SAMPLE_URL,
        "num_samples": min(int(num_samples), len(samples)),
        "note": (
            "Cropped from the official NYU Depth V2 sample montage. Depth is a colorized visualization "
            "converted to an approximate scalar map for presentation-only rendering."
        ),
        "samples": [],
    }

    for idx, (row, rgb_col, depth_col) in enumerate(samples[: int(num_samples)]):
        y0, y1 = rows[row], rows[row + 1]
        rgb = montage.crop((cols[rgb_col], y0, cols[rgb_col + 1], y1))
        depth_rgb = montage.crop((cols[depth_col], y0, cols[depth_col + 1], y1))
        depth = _depth_from_nyu_colorized(depth_rgb)

        name = f"{idx:03d}"
        rgb_out = OUT_DIR / "rgb" / f"{name}_rgb.png"
        depth_out = OUT_DIR / "depth" / f"{name}_depth.npy"
        rgb.save(rgb_out)
        np.save(depth_out, depth)
        metadata["samples"].append({"name": name, "rgb": str(rgb_out), "depth": str(depth_out)})

    (OUT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Prepared {len(metadata['samples'])} official NYU web samples in {OUT_DIR}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Prepare 10 NYU Depth V2 RGB-depth samples. This script does not download the huge TFDS dataset. "
            "Provide --source pointing to a small NYU subset or an existing folder with rgb/ and depth/."
        )
    )
    parser.add_argument("--source", help="Existing NYU sample folder with rgb/ and depth/ subfolders")
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--fallback", action="store_true", help="Create a small NYU-like procedural fallback sample set.")
    parser.add_argument("--official_web_samples", action="store_true", help="Crop real RGB-D examples from the official NYU Depth V2 sample montage.")
    args = parser.parse_args()

    if args.official_web_samples:
        prepare_official_web_samples(args.num_samples)
        return

    if args.fallback:
        make_fallback_samples(args.num_samples)
        return

    if not args.source:
        print("No --source provided.")
        print("Expected layout: /path/to/nyu_subset/rgb/*.png and /path/to/nyu_subset/depth/*.npy or *.png")
        print("Example: python scripts/data_prep/prepare_nyu_samples.py --source /path/to/nyu_subset --num_samples 10")
        print("For official real sample crops: python scripts/data_prep/prepare_nyu_samples.py --official_web_samples --num_samples 10")
        print("For presentation-only fallback: python scripts/data_prep/prepare_nyu_samples.py --fallback --num_samples 10")
        return

    pairs = find_pairs(args.source)[: args.num_samples]
    if not pairs:
        raise FileNotFoundError("No RGB-depth pairs found. Use rgb/ and depth/ folders or *_depth files.")

    (OUT_DIR / "rgb").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "depth").mkdir(parents=True, exist_ok=True)
    metadata = {"source": str(args.source), "num_samples": len(pairs), "samples": []}
    for idx, (rgb, depth) in enumerate(pairs):
        name = f"{idx:03d}"
        rgb_out = OUT_DIR / "rgb" / f"{name}_rgb{rgb.suffix.lower()}"
        depth_out = OUT_DIR / "depth" / f"{name}_depth{depth.suffix.lower()}"
        shutil.copy2(rgb, rgb_out)
        shutil.copy2(depth, depth_out)
        metadata["samples"].append({"name": name, "rgb": str(rgb_out), "depth": str(depth_out)})

    (OUT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Prepared {len(pairs)} NYU samples in {OUT_DIR}")


if __name__ == "__main__":
    main()
