import io
import json
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


def _resize_pair(rgb_img, depth_img, max_size):
    width, height = rgb_img.size
    if max(width, height) <= max_size:
        return rgb_img, depth_img

    ratio = max_size / float(max(width, height))
    new_size = (max(1, int(width * ratio)), max(1, int(height * ratio)))
    rgb_img = rgb_img.resize(new_size, Image.Resampling.LANCZOS)
    depth_img = depth_img.resize(new_size, Image.Resampling.BILINEAR)
    return rgb_img, depth_img


def _normalize_depth(depth):
    depth = depth.astype(np.float32)
    finite = np.isfinite(depth)
    if not np.any(finite):
        return np.zeros_like(depth, dtype=np.float32)

    valid = depth[finite]
    d_min = float(valid.min())
    d_max = float(valid.max())
    if d_max <= d_min:
        return np.zeros_like(depth, dtype=np.float32)

    depth = (depth - d_min) / (d_max - d_min)
    depth[~finite] = 1.0
    return np.clip(depth, 0.0, 1.0).astype(np.float32)


def _depth_to_uint8(depth):
    return (np.clip(depth, 0.0, 1.0) * 255).astype(np.uint8)


def _edge_strength(rgb, depth):
    gray = rgb.mean(axis=2)
    gx = np.zeros_like(gray, dtype=np.float32)
    gy = np.zeros_like(gray, dtype=np.float32)
    gx[:, 1:-1] = np.abs(gray[:, 2:] - gray[:, :-2])
    gy[1:-1, :] = np.abs(gray[2:, :] - gray[:-2, :])

    dgx = np.zeros_like(depth, dtype=np.float32)
    dgy = np.zeros_like(depth, dtype=np.float32)
    dgx[:, 1:-1] = np.abs(depth[:, 2:] - depth[:, :-2])
    dgy[1:-1, :] = np.abs(depth[2:, :] - depth[:-2, :])

    edge = gx + gy + dgx + dgy
    max_val = float(edge.max())
    if max_val > 0:
        edge = edge / max_val
    return edge.astype(np.float32)


def load_rgb_depth(rgb_path, depth_path, max_size=512):
    """
    Load an RGB image and matching depth map as normalized numpy arrays.

    Returns:
        rgb: float32 array [H, W, 3] in [0, 1]
        depth: float32 array [H, W] in [0, 1]
    """
    rgb_img = Image.open(rgb_path).convert("RGB")
    depth_img = Image.open(depth_path)

    if depth_img.mode in ("I;16", "I", "F"):
        depth_arr = np.asarray(depth_img).astype(np.float32)
        depth_img = Image.fromarray(_normalize_depth(depth_arr))
    else:
        depth_img = depth_img.convert("L")

    rgb_img, depth_img = _resize_pair(rgb_img, depth_img, int(max_size))

    rgb = np.asarray(rgb_img).astype(np.float32) / 255.0
    depth = np.asarray(depth_img).astype(np.float32)
    if depth.max() > 1.0:
        depth = depth / 255.0
    depth = _normalize_depth(depth)
    return rgb.astype(np.float32), depth.astype(np.float32)


def rgbd_to_depth_gaussians(rgb, depth, stride=6, edge_weight=True):
    """
    Convert RGB-D pixels into a simple depth-aware Gaussian primitive set.
    """
    h, w = depth.shape
    stride = max(1, int(stride))
    ys = np.arange(stride // 2, h, stride, dtype=np.int32)
    xs = np.arange(stride // 2, w, stride, dtype=np.int32)
    if len(xs) == 0:
        xs = np.array([w // 2], dtype=np.int32)
    if len(ys) == 0:
        ys = np.array([h // 2], dtype=np.int32)

    grid_x, grid_y = np.meshgrid(xs, ys)
    px = grid_x.reshape(-1)
    py = grid_y.reshape(-1)

    x_norm = (px.astype(np.float32) / max(1, w - 1)) * 2.0 - 1.0
    y_norm = (py.astype(np.float32) / max(1, h - 1)) * 2.0 - 1.0
    z = depth[py, px].astype(np.float32)

    colors = rgb[py, px].astype(np.float32)
    base_scale = np.array([stride / max(1, w), stride / max(1, h)], dtype=np.float32)
    scaling = np.tile(base_scale[None, :], (len(px), 1))
    opacity = np.full((len(px), 1), 0.80, dtype=np.float32)

    if edge_weight:
        edges = _edge_strength(rgb, depth)[py, px].reshape(-1, 1)
        opacity = np.clip(opacity + 0.20 * edges, 0.15, 1.0)
        scaling = scaling * (1.0 - 0.35 * edges)
        scaling = np.maximum(scaling, base_scale * 0.45)

    xyz = np.stack([x_norm, y_norm, z], axis=1).astype(np.float32)
    depth_col = z.reshape(-1, 1).astype(np.float32)
    rotation = np.zeros((len(px), 1), dtype=np.float32)

    return {
        "xyz": xyz,
        "scaling": scaling.astype(np.float32),
        "rotation": rotation,
        "opacity": opacity.astype(np.float32),
        "features_dc": colors,
        "depth": depth_col,
    }


def _draw_blobs(gaussians, height, width, colors, view_x=0.0, view_y=0.0, depth_scale=1.0, point_scale=1.0):
    xyz = np.asarray(gaussians["xyz"], dtype=np.float32)
    scaling = np.asarray(gaussians["scaling"], dtype=np.float32)
    opacity = np.asarray(gaussians["opacity"], dtype=np.float32).reshape(-1)
    z = xyz[:, 2]

    canvas = np.ones((height, width, 3), dtype=np.float32)
    order = np.argsort(z)[::-1]  # normalized depth: larger values are farther away

    for idx in order:
        depth = float(z[idx])
        x_shifted = float(xyz[idx, 0] + view_x * (1.0 - depth) * depth_scale)
        y_shifted = float(xyz[idx, 1] + view_y * (1.0 - depth) * depth_scale)
        cx = int(round((x_shifted + 1.0) * 0.5 * (width - 1)))
        cy = int(round((y_shifted + 1.0) * 0.5 * (height - 1)))
        if cx < -20 or cx >= width + 20 or cy < -20 or cy >= height + 20:
            continue

        radius = int(max(2, round(max(scaling[idx, 0] * width, scaling[idx, 1] * height) * point_scale * 1.15)))
        draw_radius = max(radius + 1, int(radius * 2.0))
        x0 = max(0, cx - draw_radius)
        x1 = min(width, cx + draw_radius + 1)
        y0 = max(0, cy - draw_radius)
        y1 = min(height, cy + draw_radius + 1)
        if x0 >= x1 or y0 >= y1:
            continue

        yy, xx = np.ogrid[y0:y1, x0:x1]
        dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
        sigma = max(0.8, radius / 2.2)
        weight = np.exp(-dist2 / (2.0 * sigma * sigma)).astype(np.float32)
        weight[dist2 > draw_radius ** 2] = 0.0
        if float(weight.max()) <= 0.0:
            continue

        alpha = float(np.clip(opacity[idx], 0.0, 1.0)) * 0.82
        patch = canvas[y0:y1, x0:x1]
        color = colors[idx].reshape(1, 1, 3)
        alpha_map = (weight[..., None] * alpha).clip(0.0, 1.0)
        patch[:] = patch * (1.0 - alpha_map) + color * alpha_map

    img = Image.fromarray((np.clip(canvas, 0.0, 1.0) * 255).astype(np.uint8))
    img = img.filter(ImageFilter.GaussianBlur(radius=max(0.0, point_scale * 0.12)))
    return np.asarray(img).astype(np.uint8)


def render_depth_gaussians(gaussians, height, width, view_x=0.0, view_y=0.0, depth_scale=1.0, point_scale=1.5):
    colors = np.asarray(gaussians["features_dc"], dtype=np.float32)
    return _draw_blobs(
        gaussians,
        int(height),
        int(width),
        colors,
        view_x=float(view_x),
        view_y=float(view_y),
        depth_scale=float(depth_scale),
        point_scale=float(point_scale),
    )


def visualize_depth_colored_gaussians(gaussians, height, width):
    depth = np.asarray(gaussians["depth"], dtype=np.float32).reshape(-1)
    near = np.stack([1.0 - depth, 0.25 + 0.55 * (1.0 - np.abs(depth - 0.5) * 2.0), depth], axis=1)
    colors = np.clip(near, 0.0, 1.0).astype(np.float32)
    return _draw_blobs(gaussians, int(height), int(width), colors, point_scale=1.5)


def _preview_bytes(preview):
    if preview is None:
        return None
    if isinstance(preview, Image.Image):
        img = preview.convert("RGB")
    else:
        img = Image.fromarray(np.asarray(preview).astype(np.uint8)).convert("RGB")
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


def save_depth_gic(output_path, header, gaussians, preview=None, metrics=None):
    header = dict(header)
    header.setdefault("codec_name", "GIC-D")
    header.setdefault("version", "1.0.0")
    header.setdefault("note", "Depth-aware Gaussian Image Container demo")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        zipf.writestr("header.json", json.dumps(header, indent=4))
        npz_buffer = io.BytesIO()
        np.savez_compressed(npz_buffer, **gaussians)
        zipf.writestr("gaussians.npz", npz_buffer.getvalue())

        preview_data = _preview_bytes(preview)
        if preview_data is not None:
            zipf.writestr("preview.png", preview_data)
        if metrics is not None:
            zipf.writestr("metrics.json", json.dumps(metrics, indent=4))


def load_depth_gic(input_path):
    data = {"header": None, "gaussians": None, "preview": None, "metrics": None}
    with zipfile.ZipFile(input_path, "r") as zipf:
        names = zipf.namelist()
        if "header.json" in names:
            data["header"] = json.loads(zipf.read("header.json").decode("utf-8"))
        if "gaussians.npz" in names:
            npz_data = np.load(io.BytesIO(zipf.read("gaussians.npz")))
            data["gaussians"] = {key: npz_data[key] for key in npz_data.files}
        if "preview.png" in names:
            data["preview"] = Image.open(io.BytesIO(zipf.read("preview.png"))).convert("RGB")
        if "metrics.json" in names:
            data["metrics"] = json.loads(zipf.read("metrics.json").decode("utf-8"))
    return data
