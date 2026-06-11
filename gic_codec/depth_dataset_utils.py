import json
from pathlib import Path

import numpy as np
from PIL import Image


PROJECT_DIR = Path(__file__).resolve().parent.parent
DEMO_DEPTH_DIR = PROJECT_DIR / "data" / "demo_depth"
DEMO_ROBOT_DIR = PROJECT_DIR / "data" / "demo_robot_rgbd"


def _resize_rgb_depth(rgb_img, depth_img, max_size):
    width, height = rgb_img.size
    if max(width, height) <= max_size:
        return rgb_img, depth_img
    ratio = max_size / float(max(width, height))
    size = (max(1, int(width * ratio)), max(1, int(height * ratio)))
    return (
        rgb_img.resize(size, Image.Resampling.LANCZOS),
        depth_img.resize(size, Image.Resampling.BILINEAR),
    )


def _load_depth_array(depth_path):
    depth_path = Path(depth_path)
    if depth_path.suffix.lower() == ".npy":
        return np.load(depth_path).astype(np.float32)

    img = Image.open(depth_path)
    arr = np.asarray(img).astype(np.float32)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr


def _normalize_depth(depth, mask=None):
    depth = depth.astype(np.float32)
    valid = np.isfinite(depth) & (depth > 0)
    if mask is not None:
        valid &= mask > 0

    if np.count_nonzero(valid) < 16:
        finite = np.isfinite(depth)
        if np.count_nonzero(finite) < 16:
            return np.zeros_like(depth, dtype=np.float32)
        valid = finite

    values = depth[valid]
    lo = float(np.percentile(values, 2))
    hi = float(np.percentile(values, 98))
    if hi <= lo:
        hi = float(values.max())
        lo = float(values.min())
    if hi <= lo:
        return np.zeros_like(depth, dtype=np.float32)

    out = (depth - lo) / (hi - lo)
    out[~valid] = 1.0
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def load_depth_pair(rgb_path, depth_path, mask_path=None, max_size=512):
    """
    Load RGB and depth from PNG/JPEG/NPY sources with robust depth normalization.
    """
    rgb_img = Image.open(rgb_path).convert("RGB")
    depth = _load_depth_array(depth_path)
    mask = None
    if mask_path:
        mask = _load_depth_array(mask_path)
        if mask.shape != depth.shape:
            mask = np.asarray(Image.fromarray(mask.astype(np.float32)).resize(depth.shape[::-1], Image.Resampling.NEAREST))

    depth = _normalize_depth(depth, mask=mask)
    depth_img = Image.fromarray((depth * 255).astype(np.uint8))
    rgb_img, depth_img = _resize_rgb_depth(rgb_img, depth_img, int(max_size))

    rgb = np.asarray(rgb_img).astype(np.float32) / 255.0
    depth = np.asarray(depth_img).astype(np.float32) / 255.0
    if depth.shape[:2] != rgb.shape[:2]:
        depth_img = Image.fromarray((np.clip(depth, 0.0, 1.0) * 255).astype(np.uint8)).resize(
            (rgb.shape[1], rgb.shape[0]),
            Image.Resampling.BILINEAR,
        )
        depth = np.asarray(depth_img).astype(np.float32) / 255.0
    depth_vis = (np.clip(depth, 0.0, 1.0) * 255).astype(np.uint8)
    return rgb.astype(np.float32), depth.astype(np.float32), depth_vis


def list_demo_depth_samples(dataset_name):
    dataset_name = dataset_name.lower()
    if dataset_name not in {"diode", "nyu"}:
        raise ValueError(f"Unknown depth dataset: {dataset_name}")

    root = DEMO_DEPTH_DIR / ("diode_10" if dataset_name == "diode" else "nyu_10")
    rgb_dir = root / "rgb"
    depth_dir = root / "depth"
    mask_dir = root / "mask"
    if not rgb_dir.exists() or not depth_dir.exists():
        return []

    records = []
    for rgb_path in sorted(rgb_dir.glob("*")):
        if rgb_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        stem = rgb_path.stem.replace("_rgb", "")
        candidates = [
            depth_dir / f"{stem}_depth.npy",
            depth_dir / f"{stem}_depth.png",
            depth_dir / f"{stem}.npy",
            depth_dir / f"{stem}.png",
        ]
        depth_path = next((p for p in candidates if p.exists()), None)
        if depth_path is None:
            continue
        mask_candidates = [mask_dir / f"{stem}_mask.npy", mask_dir / f"{stem}.npy", mask_dir / f"{stem}_mask.png"]
        mask_path = next((p for p in mask_candidates if p.exists()), None)
        records.append(
            {
                "name": stem,
                "rgb_path": str(rgb_path),
                "depth_path": str(depth_path),
                "mask_path": str(mask_path) if mask_path else None,
            }
        )
    return records


def list_robot_rgbd_sequences(dataset_name):
    dataset_name = dataset_name.lower()
    if dataset_name not in {"maniskill", "rlbench"}:
        raise ValueError(f"Unknown robot dataset: {dataset_name}")

    root = DEMO_ROBOT_DIR / dataset_name
    rgb_dir = root / "rgb"
    depth_dir = root / "depth"
    metadata_path = root / "metadata.json"
    if not rgb_dir.exists() or not depth_dir.exists():
        return None

    rgb_paths = sorted([p for p in rgb_dir.glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
    depth_paths = sorted([p for p in depth_dir.glob("*") if p.suffix.lower() in {".png", ".npy", ".jpg", ".jpeg"}])
    frames = list(zip(rgb_paths, depth_paths))
    metadata = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
    return {"root": str(root), "frames": [(str(r), str(d)) for r, d in frames], "metadata": metadata}


def load_robot_rgbd_sequence(dataset_name, max_frames=24, max_size=384):
    seq = list_robot_rgbd_sequences(dataset_name)
    if not seq or not seq["frames"]:
        return []

    frames = []
    for rgb_path, depth_path in seq["frames"][: int(max_frames)]:
        rgb, depth, depth_vis = load_depth_pair(rgb_path, depth_path, max_size=max_size)
        frames.append((rgb, depth, depth_vis))
    return frames
