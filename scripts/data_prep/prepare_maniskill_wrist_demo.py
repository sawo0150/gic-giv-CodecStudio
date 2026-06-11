import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


PROJECT_DIR = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_DIR / "data" / "demo_robot_rgbd" / "maniskill"


def make_fallback_sequence(num_frames, width=360, height=240):
    (OUT_DIR / "rgb").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "depth").mkdir(parents=True, exist_ok=True)

    x = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]

    for idx in range(num_frames):
        t = idx / max(1, num_frames - 1)
        rgb = np.zeros((height, width, 3), dtype=np.uint8)
        rgb[..., 0] = (55 + 85 * x).astype(np.uint8)
        rgb[..., 1] = (65 + 75 * y).astype(np.uint8)
        rgb[..., 2] = 95

        img = Image.fromarray(rgb)
        draw = ImageDraw.Draw(img)
        table_y = int(height * 0.62)
        draw.rectangle([0, table_y, width, height], fill=(135, 125, 105))

        obj_x = int(width * (0.34 + 0.28 * t))
        obj_y = int(height * 0.58)
        draw.ellipse([obj_x - 22, obj_y - 22, obj_x + 22, obj_y + 22], fill=(235, 95, 45))

        gripper_x = int(width * (0.20 + 0.45 * t))
        gripper_y = int(height * (0.22 + 0.10 * np.sin(t * np.pi)))
        draw.line([gripper_x, 0, gripper_x, gripper_y], fill=(45, 45, 45), width=10)
        draw.rectangle([gripper_x - 32, gripper_y, gripper_x - 10, gripper_y + 62], fill=(70, 70, 75))
        draw.rectangle([gripper_x + 10, gripper_y, gripper_x + 32, gripper_y + 62], fill=(70, 70, 75))

        depth = 0.28 + 0.40 * y + 0.16 * x
        depth = np.broadcast_to(depth, (height, width)).copy()
        yy, xx = np.ogrid[:height, :width]
        obj_mask = (xx - obj_x) ** 2 + (yy - obj_y) ** 2 < 24 ** 2
        grip_mask = (np.abs(xx - gripper_x) < 36) & (yy < gripper_y + 65)
        depth[obj_mask] = 0.25
        depth[grip_mask] = 0.12

        Image.fromarray(np.asarray(img)).save(OUT_DIR / "rgb" / f"frame_{idx:04d}.png")
        np.save(OUT_DIR / "depth" / f"frame_{idx:04d}.npy", depth.astype(np.float32))


def main():
    parser = argparse.ArgumentParser(description="Prepare a lightweight ManiSkill wrist RGB-D demo sequence.")
    parser.add_argument("--num_frames", type=int, default=18)
    parser.add_argument("--fallback", action="store_true", help="Force procedural fallback sequence.")
    args = parser.parse_args()

    source = "procedural_fallback"
    note = "ManiSkill is not downloaded. This fallback approximates a wrist camera over a tabletop manipulation scene."

    if not args.fallback:
        try:
            import mani_skill  # noqa: F401

            note = "ManiSkill is installed, but automatic task replay is version-dependent. Generated fallback for reproducibility."
        except Exception:
            pass

    make_fallback_sequence(args.num_frames)
    metadata = {"source": source, "num_frames": args.num_frames, "camera": "wrist_like", "note": note}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Prepared ManiSkill-compatible fallback sequence in {OUT_DIR}")


if __name__ == "__main__":
    main()
