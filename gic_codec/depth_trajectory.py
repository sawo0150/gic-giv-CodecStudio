import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from gic_codec.depth_gaussian import render_depth_gaussians


def generate_figure8_camera_path(num_frames=72, amplitude_x=0.25, amplitude_y=0.12):
    path = []
    for i in range(int(num_frames)):
        t = (2.0 * math.pi * i) / max(1, int(num_frames))
        path.append(
            {
                "view_x": float(amplitude_x * math.sin(t)),
                "view_y": float((amplitude_y * math.sin(2.0 * t)) / 2.0),
                "look_at_x": 0.0,
                "look_at_y": 0.0,
                "depth_scale": 1.8,
                "point_scale": 1.5,
            }
        )
    return path


def _label_image(img, label):
    pil = Image.fromarray(np.asarray(img).astype(np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(pil)
    draw.rectangle([0, 0, pil.width, 28], fill=(0, 0, 0))
    draw.text((8, 7), label, fill=(255, 255, 255))
    return np.asarray(pil)


def _resize_to(img, size):
    return np.asarray(Image.fromarray(np.asarray(img).astype(np.uint8)).resize(size, Image.Resampling.LANCZOS))


def _save_video(frames, output_path, fps):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames = [np.asarray(frame).astype(np.uint8) for frame in frames]

    try:
        import imageio.v2 as imageio

        imageio.mimsave(output_path, frames, fps=fps)
        return str(output_path)
    except Exception:
        pass

    try:
        import cv2

        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, float(fps), (w, h))
        if writer.isOpened():
            for frame in frames:
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            writer.release()
            return str(output_path)
    except Exception:
        pass

    gif_path = output_path.with_suffix(".gif")
    pil_frames = [Image.fromarray(frame) for frame in frames]
    pil_frames[0].save(gif_path, save_all=True, append_images=pil_frames[1:], duration=int(1000 / fps), loop=0)
    return str(gif_path)


def render_depth_gic_figure8_video(
    gaussians,
    height,
    width,
    output_path,
    num_frames=72,
    fps=18,
    depth_scale=1.8,
    point_scale=1.5,
):
    frames = []
    for camera in generate_figure8_camera_path(num_frames=num_frames):
        frame = render_depth_gaussians(
            gaussians,
            height,
            width,
            view_x=camera["view_x"],
            view_y=camera["view_y"],
            depth_scale=depth_scale,
            point_scale=point_scale,
        )
        frames.append(frame)
    return _save_video(frames, output_path, fps)


def render_depth_gic_comparison_video(rgb, depth_vis, gaussians, output_path, num_frames=72, fps=18):
    h, w = depth_vis.shape[:2]
    rgb_u8 = (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8) if rgb.dtype != np.uint8 else rgb
    depth_rgb = np.stack([depth_vis] * 3, axis=2) if depth_vis.ndim == 2 else depth_vis
    cell_size = (w, h)
    frames = []
    for camera in generate_figure8_camera_path(num_frames=num_frames):
        render = render_depth_gaussians(
            gaussians,
            h,
            w,
            view_x=camera["view_x"],
            view_y=camera["view_y"],
            depth_scale=1.8,
            point_scale=1.5,
        )
        panels = [
            _label_image(_resize_to(rgb_u8, cell_size), "Original RGB"),
            _label_image(_resize_to(depth_rgb, cell_size), "Depth"),
            _label_image(render, "Depth-GIC Figure-8 Render"),
        ]
        frames.append(np.concatenate(panels, axis=1))
    return _save_video(frames, output_path, fps)


def render_givd_third_person_replay(rgbd_frames, givd_frames, output_path, fps=12, depth_scale=1.8, point_scale=1.5):
    frames = []
    for idx, frame in enumerate(givd_frames):
        depth = rgbd_frames[idx][1]
        h, w = depth.shape
        frames.append(
            render_depth_gaussians(
                frame["gaussians"] if "gaussians" in frame else frame,
                h,
                w,
                view_x=0.08,
                view_y=-0.28,
                depth_scale=depth_scale,
                point_scale=point_scale,
            )
        )
    return _save_video(frames, output_path, fps)


def render_givd_robot_comparison_video(rgb_frames, depth_vis_frames, gaussians_per_frame, output_path, fps=12):
    frames = []
    for rgb, depth_vis, gaussians in zip(rgb_frames, depth_vis_frames, gaussians_per_frame):
        rgb_u8 = (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8) if rgb.dtype != np.uint8 else rgb
        depth_rgb = np.stack([depth_vis] * 3, axis=2) if depth_vis.ndim == 2 else depth_vis
        h, w = depth_vis.shape[:2]
        render = render_depth_gaussians(
            gaussians,
            h,
            w,
            view_x=0.08,
            view_y=-0.28,
            depth_scale=1.8,
            point_scale=1.5,
        )
        panels = [
            _label_image(_resize_to(rgb_u8, (w, h)), "Original wrist RGB"),
            _label_image(_resize_to(depth_rgb, (w, h)), "Depth"),
            _label_image(render, "GIV-D Third-person Replay"),
        ]
        frames.append(np.concatenate(panels, axis=1))
    return _save_video(frames, output_path, fps)
