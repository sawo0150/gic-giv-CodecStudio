import argparse
import json
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


PROJECT_DIR = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_DIR / "data" / "demo_robot_rgbd" / "maniskill"
OFFICIAL_RGBD_URL = "https://maniskill.readthedocs.io/en/latest/_images/rgbd_vis.png"


def _download_official_rgbd_sample():
    raw_dir = PROJECT_DIR / "data" / "raw_downloads"
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / "maniskill_rgbd_vis.png"
    if path.exists():
        return path
    req = urllib.request.Request(OFFICIAL_RGBD_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as response:
        path.write_bytes(response.read())
    return path


def make_official_doc_sequence(num_frames, width=360, height=240):
    (OUT_DIR / "rgb").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "depth").mkdir(parents=True, exist_ok=True)

    sample_path = _download_official_rgbd_sample()
    montage = Image.open(sample_path).convert("RGB")
    panel_w = montage.width // 2
    rgb_panel = montage.crop((0, 0, panel_w, montage.height))
    depth_panel = montage.crop((panel_w, 0, montage.width, montage.height)).convert("L")

    # The official image is one RGB/depth observation pair. We create a short
    # presentation sequence with small crop shifts so the replay page has frames.
    crop_w = int(panel_w * 0.92)
    crop_h = int(montage.height * 0.92)
    for idx in range(int(num_frames)):
        t = idx / max(1, int(num_frames) - 1)
        dx = int((panel_w - crop_w) * (0.5 + 0.45 * np.sin(2 * np.pi * t)))
        dy = int((montage.height - crop_h) * (0.5 + 0.35 * np.sin(2 * np.pi * t + 0.8)))
        box = (dx, dy, dx + crop_w, dy + crop_h)

        rgb = rgb_panel.crop(box).resize((width, height), Image.Resampling.LANCZOS)
        depth_img = depth_panel.crop(box).resize((width, height), Image.Resampling.BILINEAR)
        depth = np.asarray(depth_img).astype(np.float32) / 255.0
        depth = 1.0 - depth
        depth = np.clip(depth, 0.0, 1.0).astype(np.float32)

        rgb.save(OUT_DIR / "rgb" / f"frame_{idx:04d}.png")
        np.save(OUT_DIR / "depth" / f"frame_{idx:04d}.npy", depth)


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
    parser.add_argument("--procedural_fallback", action="store_true", help="Force older procedural fallback sequence.")
    args = parser.parse_args()

    source = "maniskill_official_doc_rgbd_sample"
    note = (
        "Cropped from the official ManiSkill RGB+Depth texture visualization. "
        "A short sequence is synthesized with small crop shifts for presentation replay."
    )

    if args.fallback or args.procedural_fallback:
        source = "procedural_fallback"
        note = "ManiSkill is not downloaded. This fallback approximates a wrist camera over a tabletop manipulation scene."
        make_fallback_sequence(args.num_frames)
    else:
        try:
            make_official_doc_sequence(args.num_frames)
        except Exception:
            source = "procedural_fallback"
            note = "Could not download official ManiSkill RGB-D sample. Generated procedural fallback instead."
            make_fallback_sequence(args.num_frames)

    metadata = {
        "source": source,
        "source_url": OFFICIAL_RGBD_URL if source == "maniskill_official_doc_rgbd_sample" else None,
        "num_frames": args.num_frames,
        "camera": "wrist_like",
        "note": note,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Prepared ManiSkill-compatible RGB-D sequence in {OUT_DIR} (source: {source})")


if __name__ == "__main__":
    main()
