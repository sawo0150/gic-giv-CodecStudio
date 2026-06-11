import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_DIR))

from PIL import Image

from gic_codec.depth_gaussian import (
    load_rgb_depth,
    render_depth_gaussians,
    rgbd_to_depth_gaussians,
    save_depth_gic,
    visualize_depth_colored_gaussians,
)


def main():
    parser = argparse.ArgumentParser(description="Create a Depth-GIC (.gicd) demo file from RGB-D input.")
    parser.add_argument("--rgb", required=True, help="Path to RGB image")
    parser.add_argument("--depth", required=True, help="Path to depth image")
    parser.add_argument("--output", required=True, help="Output .gicd path")
    parser.add_argument("--max_size", type=int, default=384)
    parser.add_argument("--stride", type=int, default=6)
    args = parser.parse_args()

    start = time.time()
    rgb, depth = load_rgb_depth(args.rgb, args.depth, max_size=args.max_size)
    gaussians = rgbd_to_depth_gaussians(rgb, depth, stride=args.stride, edge_weight=True)
    height, width = depth.shape
    render = render_depth_gaussians(gaussians, height, width)
    depth_vis = visualize_depth_colored_gaussians(gaussians, height, width)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = {
        "codec_name": "GIC-D",
        "version": "1.0.0",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "image_info": {"width": width, "height": height, "channels": 3, "has_depth": True},
        "encoding_settings": {
            "max_size": args.max_size,
            "stride": args.stride,
            "edge_weight": True,
            "representation": "depth-aware-gaussian",
        },
        "gaussian_info": {"num_points": int(len(gaussians["xyz"]))},
        "note": "Depth-aware Gaussian Image Container demo",
    }
    metrics = {"num_points": int(len(gaussians["xyz"])), "encoding_time_sec": time.time() - start}

    save_depth_gic(output_path, header, gaussians, preview=render, metrics=metrics)
    Image.fromarray(render).save(output_path.with_name(output_path.stem + "_render.png"))
    Image.fromarray(depth_vis).save(output_path.with_name(output_path.stem + "_depth_vis.png"))
    output_path.with_name(output_path.stem + "_metrics.json").write_text(json.dumps(metrics, indent=2))

    print(f"Saved {output_path}")
    print(f"Gaussians: {len(gaussians['xyz'])}")


if __name__ == "__main__":
    main()
