import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_DIR))

from PIL import Image

from gic_codec.depth_gaussian import load_rgb_depth, render_depth_gaussians, rgbd_to_depth_gaussians
from gic_codec.depth_giv import save_depth_giv


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def main():
    parser = argparse.ArgumentParser(description="Create a GIV-D (.givd) demo file from RGB-D frame folders.")
    parser.add_argument("--rgb_dir", required=True, help="Directory containing RGB frames")
    parser.add_argument("--depth_dir", required=True, help="Directory containing depth frames")
    parser.add_argument("--output", required=True, help="Output .givd path")
    parser.add_argument("--max_frames", type=int, default=8)
    parser.add_argument("--max_size", type=int, default=320)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--fps", type=float, default=8.0)
    args = parser.parse_args()

    rgb_paths = sorted([p for p in Path(args.rgb_dir).iterdir() if p.suffix.lower() in IMAGE_EXTS])
    depth_paths = sorted([p for p in Path(args.depth_dir).iterdir() if p.suffix.lower() in IMAGE_EXTS])
    pairs = list(zip(rgb_paths, depth_paths))[: args.max_frames]
    if not pairs:
        raise FileNotFoundError("No RGB-D frame pairs found.")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames = []
    previews = {}
    start = time.time()

    for idx, (rgb_path, depth_path) in enumerate(pairs, start=1):
        rgb, depth = load_rgb_depth(rgb_path, depth_path, max_size=args.max_size)
        gaussians = rgbd_to_depth_gaussians(rgb, depth, stride=args.stride, edge_weight=True)
        height, width = depth.shape
        preview = render_depth_gaussians(gaussians, height, width)
        frames.append({"gaussians": gaussians, "width": width, "height": height})
        previews[idx] = preview
        Image.fromarray(preview).save(output_path.with_name(f"{output_path.stem}_frame_{idx:06d}.png"))
        print(f"[{idx}/{len(pairs)}] {rgb_path.name} + {depth_path.name} -> {len(gaussians['xyz'])} gaussians")

    height = frames[0]["height"]
    width = frames[0]["width"]
    header = {
        "codec_name": "GIV-D",
        "version": "1.0.0",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "video_info": {
            "width": width,
            "height": height,
            "total_frames": len(frames),
            "fps": args.fps,
        },
        "encoding_settings": {
            "max_size": args.max_size,
            "stride": args.stride,
            "representation": "depth-aware-gaussian",
        },
        "note": "Depth-aware Gaussian Video Container demo",
    }
    metrics = {
        "total_frames": len(frames),
        "avg_points_per_frame": sum(len(f["gaussians"]["xyz"]) for f in frames) / len(frames),
        "encoding_time_sec": time.time() - start,
    }

    save_depth_giv(output_path, header, frames, previews=previews, metrics=metrics)
    output_path.with_name(output_path.stem + "_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
