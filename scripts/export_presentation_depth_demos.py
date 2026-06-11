import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_DIR))

from gic_codec.depth_dataset_utils import list_demo_depth_samples, load_depth_pair, load_robot_rgbd_sequence
from gic_codec.depth_gaussian import render_depth_gaussians, rgbd_to_depth_gaussians, visualize_depth_colored_gaussians
from gic_codec.depth_trajectory import (
    render_depth_gic_comparison_video,
    render_depth_gic_figure8_video,
    render_givd_robot_comparison_video,
    render_givd_third_person_replay,
)


OUTPUT_DIR = PROJECT_DIR / "outputs" / "ppt_assets" / "depth_demo"


def _rgb_to_u8(rgb):
    return (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8) if rgb.dtype != np.uint8 else rgb


def _save_image(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        arr = _rgb_to_u8(arr)
    if arr.ndim == 2:
        Image.fromarray(arr).save(path)
    else:
        Image.fromarray(arr).convert("RGB").save(path)


def _make_strip(images, label, output_path, max_items=8):
    selected = images[:max_items]
    if not selected:
        return None

    thumbs = []
    target_h = 150
    for image in selected:
        arr = np.asarray(image)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=2)
        arr = _rgb_to_u8(arr)
        pil = Image.fromarray(arr).convert("RGB")
        ratio = target_h / pil.height
        pil = pil.resize((max(1, int(pil.width * ratio)), target_h), Image.Resampling.LANCZOS)
        thumbs.append(pil)

    width = sum(img.width for img in thumbs)
    strip = Image.new("RGB", (width, target_h + 32), "white")
    x = 0
    for img in thumbs:
        strip.paste(img, (x, 32))
        x += img.width
    draw = ImageDraw.Draw(strip)
    draw.rectangle([0, 0, width, 32], fill=(0, 0, 0))
    draw.text((10, 9), label, fill=(255, 255, 255))
    strip.save(output_path)
    return str(output_path)


def _find_first_depth_sample():
    for dataset_name in ["diode", "nyu"]:
        samples = list_demo_depth_samples(dataset_name)
        if samples:
            return dataset_name, samples[0]
    return None, None


def _find_first_robot_sequence():
    for dataset_name in ["maniskill", "rlbench"]:
        frames = load_robot_rgbd_sequence(dataset_name, max_frames=18, max_size=320)
        if frames:
            return dataset_name, frames
    return None, []


def export_depth_gic_assets(summary):
    dataset_name, sample = _find_first_depth_sample()
    if sample is None:
        summary["depth_gic"] = {
            "status": "missing",
            "message": "No DIODE/NYU sample found. Run scripts/data_prep/prepare_diode_samples.py or prepare_nyu_samples.py.",
        }
        return

    rgb, depth, depth_vis = load_depth_pair(sample["rgb_path"], sample["depth_path"], sample.get("mask_path"), max_size=384)
    h, w = depth.shape
    gaussians = rgbd_to_depth_gaussians(rgb, depth, stride=5, edge_weight=True)
    render = render_depth_gaussians(gaussians, h, w, point_scale=1.5)
    depth_colored = visualize_depth_colored_gaussians(gaussians, h, w)

    _save_image(OUTPUT_DIR / "depth_gic_original_rgb.png", rgb)
    _save_image(OUTPUT_DIR / "depth_gic_depth_map.png", depth_vis)
    _save_image(OUTPUT_DIR / "depth_gic_gaussian_render.png", render)
    _save_image(OUTPUT_DIR / "depth_gic_depth_colored.png", depth_colored)
    figure8 = render_depth_gic_figure8_video(
        gaussians,
        h,
        w,
        OUTPUT_DIR / "depth_gic_figure8_demo.mp4",
        num_frames=72,
        fps=18,
        depth_scale=1.8,
        point_scale=1.5,
    )
    comparison = render_depth_gic_comparison_video(
        rgb,
        depth_vis,
        gaussians,
        OUTPUT_DIR / "depth_gic_comparison_demo.mp4",
        num_frames=72,
        fps=18,
    )

    summary["depth_gic"] = {
        "status": "generated",
        "dataset": dataset_name,
        "sample": sample["name"],
        "num_points": int(len(gaussians["xyz"])),
        "outputs": {
            "original_rgb": str(OUTPUT_DIR / "depth_gic_original_rgb.png"),
            "depth_map": str(OUTPUT_DIR / "depth_gic_depth_map.png"),
            "gaussian_render": str(OUTPUT_DIR / "depth_gic_gaussian_render.png"),
            "depth_colored": str(OUTPUT_DIR / "depth_gic_depth_colored.png"),
            "figure8_video": figure8,
            "comparison_video": comparison,
        },
    }


def export_givd_assets(summary):
    dataset_name, frames = _find_first_robot_sequence()
    if not frames:
        summary["givd"] = {
            "status": "missing",
            "message": "No ManiSkill/RLBench robot sequence found. Run a script in scripts/data_prep/ first.",
        }
        return

    rgb_frames = [frame[0] for frame in frames]
    depth_vis_frames = [frame[2] for frame in frames]
    gaussians_per_frame = [rgbd_to_depth_gaussians(rgb, depth, stride=7, edge_weight=True) for rgb, depth, _ in frames]
    givd_frames = [{"gaussians": g, "width": int(frames[i][1].shape[1]), "height": int(frames[i][1].shape[0])} for i, g in enumerate(gaussians_per_frame)]

    strip_rgb = _make_strip(rgb_frames, "Original wrist RGB sequence", OUTPUT_DIR / "givd_robot_original_strip.png")
    depth_rgb = [np.stack([d] * 3, axis=2) for d in depth_vis_frames]
    strip_depth = _make_strip(depth_rgb, "Depth sequence", OUTPUT_DIR / "givd_robot_depth_strip.png")
    replay = render_givd_third_person_replay(
        frames,
        givd_frames,
        OUTPUT_DIR / "givd_third_person_replay.mp4",
        fps=12,
        depth_scale=1.8,
        point_scale=1.5,
    )
    comparison = render_givd_robot_comparison_video(
        rgb_frames,
        depth_vis_frames,
        gaussians_per_frame,
        OUTPUT_DIR / "givd_comparison_demo.mp4",
        fps=12,
    )

    summary["givd"] = {
        "status": "generated",
        "dataset": dataset_name,
        "frames": len(frames),
        "avg_points_per_frame": float(np.mean([len(g["xyz"]) for g in gaussians_per_frame])),
        "outputs": {
            "original_strip": strip_rgb,
            "depth_strip": strip_depth,
            "third_person_video": replay,
            "comparison_video": comparison,
        },
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {"output_dir": str(OUTPUT_DIR)}
    export_depth_gic_assets(summary)
    export_givd_assets(summary)

    summary_path = OUTPUT_DIR / "demo_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
