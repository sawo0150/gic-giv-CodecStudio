import io
import os
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

PROJECT_DIR = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_DIR))

from gic_codec.depth_gaussian import load_rgb_depth, render_depth_gaussians, rgbd_to_depth_gaussians
from gic_codec.depth_giv import save_depth_giv


st.set_page_config(layout="wide", page_title="GIV-D Replay Demo")
st.title("GIV-D Demo: First-person RGB-D Sequence to Gaussian Replay")

st.write(
    "This demo converts a short RGB-D sequence into a frame-wise depth-aware Gaussian "
    "sequence and replays it with limited viewpoint changes."
)

rgb_files = st.file_uploader(
    "RGB frame images 업로드",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True,
    key="givd_rgb_frames",
)
depth_files = st.file_uploader(
    "Depth frame images 업로드",
    type=["png", "jpg", "jpeg", "tif", "tiff"],
    accept_multiple_files=True,
    key="givd_depth_frames",
)
zip_file = st.file_uploader("또는 rgb/ depth/ 폴더를 포함한 zip 업로드", type=["zip"], key="givd_zip")

opt1, opt2, opt3 = st.columns(3)
with opt1:
    max_frames = st.slider("max_frames", 1, 30, 8, 1)
with opt2:
    max_size = st.slider("max_size", 128, 512, 320, 32)
with opt3:
    stride = st.slider("stride", 4, 20, 8, 1)


def _write_upload(uploaded, folder):
    path = folder / uploaded.name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(uploaded.getbuffer())
    return path


def _load_pairs_from_uploads(tmpdir):
    rgb_dir = tmpdir / "rgb_uploads"
    depth_dir = tmpdir / "depth_uploads"
    rgb_paths = sorted([_write_upload(f, rgb_dir) for f in rgb_files], key=lambda p: p.name)
    depth_paths = sorted([_write_upload(f, depth_dir) for f in depth_files], key=lambda p: p.name)
    return list(zip(rgb_paths, depth_paths))[:max_frames]


def _load_pairs_from_zip(tmpdir):
    zip_path = tmpdir / "sequence.zip"
    zip_path.write_bytes(zip_file.getbuffer())
    extract_dir = tmpdir / "zip_sequence"
    with zipfile.ZipFile(zip_path, "r") as zipf:
        zipf.extractall(extract_dir)

    rgb_paths = sorted([p for p in extract_dir.rglob("*") if p.parent.name.lower() == "rgb" and p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
    depth_paths = sorted([p for p in extract_dir.rglob("*") if p.parent.name.lower() == "depth" and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}])
    return list(zip(rgb_paths, depth_paths))[:max_frames]


if "givd_demo" not in st.session_state:
    st.session_state.givd_demo = None

can_encode = zip_file is not None or (rgb_files and depth_files and len(rgb_files) == len(depth_files))
if st.button("Encode sequence to GIV-D demo", disabled=not can_encode):
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        try:
            pairs = _load_pairs_from_zip(tmpdir) if zip_file is not None else _load_pairs_from_uploads(tmpdir)
            if not pairs:
                st.warning("No RGB-D frame pairs found.")
                st.stop()

            frames = []
            previews = {}
            original_rgbs = []
            depth_maps = []
            start = time.time()

            progress = st.progress(0)
            status = st.empty()
            for idx, (rgb_path, depth_path) in enumerate(pairs, start=1):
                rgb, depth = load_rgb_depth(rgb_path, depth_path, max_size=max_size)
                gaussians = rgbd_to_depth_gaussians(rgb, depth, stride=stride, edge_weight=True)
                h, w = depth.shape
                preview = render_depth_gaussians(gaussians, h, w)

                frames.append({"gaussians": gaussians, "width": w, "height": h})
                previews[idx] = preview
                original_rgbs.append(rgb)
                depth_maps.append(depth)
                progress.progress(idx / len(pairs))
                status.write(f"Encoded frame {idx}/{len(pairs)} | Gaussians: `{len(gaussians['xyz']):,}`")

            h, w = depth_maps[0].shape
            header = {
                "codec_name": "GIV-D",
                "version": "1.0.0",
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "video_info": {"width": int(w), "height": int(h), "total_frames": len(frames), "fps": 8.0},
                "encoding_settings": {
                    "max_size": int(max_size),
                    "stride": int(stride),
                    "representation": "depth-aware-gaussian",
                },
                "note": "Depth-aware Gaussian Video Container demo",
            }
            metrics = {
                "total_frames": len(frames),
                "avg_points_per_frame": float(np.mean([len(f["gaussians"]["xyz"]) for f in frames])),
                "encoding_time_sec": float(time.time() - start),
            }

            out_path = tmpdir / "sequence.givd"
            save_depth_giv(out_path, header, frames, previews=previews, metrics=metrics)
            givd_bytes = out_path.read_bytes()

            st.session_state.givd_demo = {
                "frames": frames,
                "previews": previews,
                "original_rgbs": original_rgbs,
                "depth_maps": depth_maps,
                "header": header,
                "metrics": metrics,
                "givd_bytes": givd_bytes,
            }
        except Exception as exc:
            st.error(f"GIV-D encoding failed: {exc}")

demo = st.session_state.givd_demo
if demo is None:
    st.info("여러 RGB frame과 matching depth frame을 업로드하거나 rgb/depth 폴더가 있는 zip을 업로드하세요.")
else:
    st.markdown("### GIV-D Replay")
    total = len(demo["frames"])
    frame_idx = st.slider("Frame", 1, total, 1)

    vc1, vc2, vc3, vc4 = st.columns(4)
    with vc1:
        view_x = st.slider("view_x", -0.5, 0.5, 0.0, 0.01)
    with vc2:
        view_y = st.slider("view_y", -0.5, 0.5, 0.0, 0.01)
    with vc3:
        depth_scale = st.slider("depth_scale", 0.0, 5.0, 1.5, 0.1)
    with vc4:
        point_scale = st.slider("point_scale", 0.3, 3.0, 1.0, 0.1)

    idx = frame_idx - 1
    frame = demo["frames"][idx]
    gaussians = frame["gaussians"]
    h, w = demo["depth_maps"][idx].shape
    original_render = render_depth_gaussians(gaussians, h, w, point_scale=point_scale)
    shifted_render = render_depth_gaussians(
        gaussians,
        h,
        w,
        view_x=view_x,
        view_y=view_y,
        depth_scale=depth_scale,
        point_scale=point_scale,
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.image((demo["original_rgbs"][idx] * 255).astype(np.uint8), caption="Original RGB", use_column_width=True)
    col2.image((demo["depth_maps"][idx] * 255).astype(np.uint8), caption="Depth Map", use_column_width=True)
    col3.image(original_render, caption="Gaussian Replay", use_column_width=True)
    col4.image(shifted_render, caption="Viewpoint-shifted Replay", use_column_width=True)

    st.json(demo["metrics"])
    st.download_button(
        "Download .givd",
        data=demo["givd_bytes"],
        file_name="sequence.givd",
        mime="application/octet-stream",
    )

    if st.button("Play Gaussian Replay"):
        placeholder = st.empty()
        for i, frame in enumerate(demo["frames"]):
            depth = demo["depth_maps"][i]
            img = render_depth_gaussians(
                frame["gaussians"],
                depth.shape[0],
                depth.shape[1],
                view_x=view_x,
                view_y=view_y,
                depth_scale=depth_scale,
                point_scale=point_scale,
            )
            placeholder.image(img, caption=f"Frame {i + 1}/{total}", use_column_width=True)
            time.sleep(0.12)
