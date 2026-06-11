import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


PROJECT_DIR = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_DIR / "data" / "demo_robot_rgbd" / "rlbench"


def make_fallback_sequence(num_frames, width=360, height=240):
    (OUT_DIR / "rgb").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "depth").mkdir(parents=True, exist_ok=True)

    x = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]

    for idx in range(num_frames):
        t = idx / max(1, num_frames - 1)
        rgb = np.zeros((height, width, 3), dtype=np.uint8)
        rgb[..., 0] = 40
        rgb[..., 1] = (75 + 60 * y).astype(np.uint8)
        rgb[..., 2] = (90 + 70 * x).astype(np.uint8)

        img = Image.fromarray(rgb)
        draw = ImageDraw.Draw(img)
        shelf_y = int(height * 0.68)
        draw.rectangle([20, shelf_y, width - 20, height], fill=(120, 105, 90))
        target_x = int(width * 0.66)
        target_y = int(height * 0.57)
        draw.rectangle([target_x - 30, target_y - 28, target_x + 30, target_y + 28], fill=(65, 165, 235))

        hand_x = int(width * (0.25 + 0.32 * t))
        hand_y = int(height * (0.24 + 0.14 * t))
        draw.line([hand_x, 0, hand_x, hand_y], fill=(55, 55, 60), width=12)
        draw.polygon([(hand_x - 42, hand_y), (hand_x - 14, hand_y + 60), (hand_x - 4, hand_y + 54), (hand_x - 26, hand_y)], fill=(80, 80, 85))
        draw.polygon([(hand_x + 42, hand_y), (hand_x + 14, hand_y + 60), (hand_x + 4, hand_y + 54), (hand_x + 26, hand_y)], fill=(80, 80, 85))

        depth = 0.30 + 0.44 * y + 0.10 * x
        depth = np.broadcast_to(depth, (height, width)).copy()
        yy, xx = np.ogrid[:height, :width]
        target_mask = (np.abs(xx - target_x) < 32) & (np.abs(yy - target_y) < 30)
        hand_mask = (np.abs(xx - hand_x) < 48) & (yy < hand_y + 64)
        depth[target_mask] = 0.34
        depth[hand_mask] = 0.13

        Image.fromarray(np.asarray(img)).save(OUT_DIR / "rgb" / f"frame_{idx:04d}.png")
        np.save(OUT_DIR / "depth" / f"frame_{idx:04d}.npy", depth.astype(np.float32))


def main():
    parser = argparse.ArgumentParser(description="Prepare a lightweight RLBench eye-in-hand RGB-D demo sequence.")
    parser.add_argument("--num_frames", type=int, default=18)
    parser.add_argument("--fallback", action="store_true", help="Force procedural fallback sequence.")
    args = parser.parse_args()

    source = "procedural_fallback"
    note = "RLBench/CoppeliaSim is not downloaded. This fallback approximates an eye-in-hand manipulation view."
    if not args.fallback:
        try:
            import rlbench  # noqa: F401

            note = "RLBench is installed, but simulator setup is environment-specific. Generated fallback for reproducibility."
        except Exception:
            pass

    make_fallback_sequence(args.num_frames)
    metadata = {"source": source, "num_frames": args.num_frames, "camera": "eye_in_hand_like", "note": note}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Prepared RLBench-compatible fallback sequence in {OUT_DIR}")


if __name__ == "__main__":
    main()
