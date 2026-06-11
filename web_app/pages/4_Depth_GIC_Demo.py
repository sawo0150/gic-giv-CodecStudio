import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import streamlit as st

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_DIR))

from gic_codec.depth_dataset_utils import list_demo_depth_samples, load_depth_pair
from gic_codec.depth_gaussian import (
    load_rgb_depth,
    render_depth_gaussians,
    rgbd_to_depth_gaussians,
    save_depth_gic,
    visualize_depth_colored_gaussians,
)
from gic_codec.depth_trajectory import (
    render_depth_gic_comparison_video,
    render_depth_gic_figure8_video,
)


OUTPUT_DIR = PROJECT_DIR / "outputs" / "ppt_assets" / "depth_demo"


st.set_page_config(layout="wide", page_title="Depth-GIC Spatial Demo")
st.title("Depth-GIC Spatial Image Demo")
st.write(
    "RGB-D is a pixel-wise sensor record. Depth-GIC converts it into depth-aware Gaussian primitives. "
    "The figure-eight camera trajectory shows that the stored representation can be re-rendered with "
    "limited viewpoint changes."
)


def _save_upload(uploaded, suffix):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded.getbuffer())
    tmp.close()
    return tmp.name


def _depth_to_u8(depth):
    return (np.clip(depth, 0.0, 1.0) * 255).astype(np.uint8)


def _rgb_to_u8(rgb):
    return (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8) if rgb.dtype != np.uint8 else rgb


def _build_depth_gic(rgb, depth, depth_vis, *, max_size, stride, edge_weight, file_stem, render_videos):
    start = time.time()
    height, width = depth.shape
    gaussians = rgbd_to_depth_gaussians(rgb, depth, stride=stride, edge_weight=edge_weight)
    render = render_depth_gaussians(gaussians, height, width, point_scale=1.5)
    depth_colored = visualize_depth_colored_gaussians(gaussians, height, width)

    header = {
        "codec_name": "GIC-D",
        "version": "1.0.0",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "image_info": {"width": int(width), "height": int(height), "channels": 3, "has_depth": True},
        "encoding_settings": {
            "max_size": int(max_size),
            "stride": int(stride),
            "edge_weight": bool(edge_weight),
            "representation": "depth-aware-gaussian",
        },
        "gaussian_info": {"num_points": int(len(gaussians["xyz"]))},
        "note": "Depth-aware Gaussian Image Container demo",
    }
    metrics = {
        "num_points": int(len(gaussians["xyz"])),
        "encoding_time_sec": float(time.time() - start),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    gicd_path = OUTPUT_DIR / f"{file_stem}.gicd"
    save_depth_gic(gicd_path, header, gaussians, preview=render, metrics=metrics)

    figure8_path = None
    comparison_path = None
    if render_videos:
        figure8_path = render_depth_gic_figure8_video(
            gaussians,
            height,
            width,
            OUTPUT_DIR / "depth_gic_figure8_demo.mp4",
            num_frames=72,
            fps=18,
            depth_scale=1.8,
            point_scale=1.5,
        )
        comparison_path = render_depth_gic_comparison_video(
            rgb,
            depth_vis,
            gaussians,
            OUTPUT_DIR / "depth_gic_comparison_demo.mp4",
            num_frames=72,
            fps=18,
        )

    return {
        "rgb": rgb,
        "depth": depth,
        "depth_vis": depth_vis,
        "gaussians": gaussians,
        "render": render,
        "depth_colored": depth_colored,
        "header": header,
        "metrics": metrics,
        "gicd_path": str(gicd_path),
        "figure8_path": figure8_path,
        "comparison_path": comparison_path,
    }


def _download_button_for_file(label, path, file_name, mime):
    if path and Path(path).exists():
        st.download_button(label, data=Path(path).read_bytes(), file_name=file_name, mime=mime)


st.markdown("### Presentation-ready sample")
dataset_label = st.selectbox("Dataset", ["DIODE", "NYU Depth V2"])
dataset_key = "diode" if dataset_label == "DIODE" else "nyu"
samples = list_demo_depth_samples(dataset_key)

if not samples:
    if dataset_key == "diode":
        st.warning("DIODE samples are missing. Run `python scripts/data_prep/prepare_diode_samples.py --num_samples 10 --cleanup` first.")
    else:
        st.warning("NYU samples are missing. Run `python scripts/data_prep/prepare_nyu_samples.py --source /path/to/nyu --num_samples 10` first.")
else:
    sample_names = [f"{idx:02d} - {record['name']}" for idx, record in enumerate(samples)]
    sample_idx = st.selectbox("Sample", range(len(samples)), format_func=lambda idx: sample_names[idx])
    c1, c2, c3 = st.columns(3)
    with c1:
        max_size = st.slider("max_size", 128, 768, 384, 32, key="presentation_max_size")
    with c2:
        stride = st.slider("stride", 2, 20, 5, 1, key="presentation_stride")
    with c3:
        edge_weight = st.checkbox("edge_weight", value=True, key="presentation_edge_weight")

    if st.button("Generate Depth-GIC figure-eight video", type="primary"):
        try:
            record = samples[sample_idx]
            rgb, depth, depth_vis = load_depth_pair(
                record["rgb_path"],
                record["depth_path"],
                record.get("mask_path"),
                max_size=max_size,
            )
            with st.spinner("Encoding Depth-GIC and rendering figure-eight MP4..."):
                st.session_state.depth_gic_presentation = _build_depth_gic(
                    rgb,
                    depth,
                    depth_vis,
                    max_size=max_size,
                    stride=stride,
                    edge_weight=edge_weight,
                    file_stem=f"{dataset_key}_{record['name']}",
                    render_videos=True,
                )
        except Exception as exc:
            st.error(f"Depth-GIC presentation demo failed: {exc}")


presentation = st.session_state.get("depth_gic_presentation")
if presentation:
    st.markdown("### Generated Depth-GIC representation")
    p1, p2, p3, p4 = st.columns(4)
    p1.image(_rgb_to_u8(presentation["rgb"]), caption="Original RGB", width="stretch")
    p2.image(presentation["depth_vis"], caption="Depth Map", width="stretch")
    p3.image(presentation["render"], caption="Gaussian Render", width="stretch")
    p4.image(presentation["depth_colored"], caption="Depth-colored Gaussians", width="stretch")

    h, w = presentation["depth"].shape
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Image Size", f"{w}x{h}")
    m2.metric("Gaussian Count", f"{len(presentation['gaussians']['xyz']):,}")
    m3.metric("Stride", str(presentation["header"]["encoding_settings"]["stride"]))
    m4.metric(".gicd Size", f"{Path(presentation['gicd_path']).stat().st_size / 1024:.1f} KB")

    if presentation.get("comparison_path") and Path(presentation["comparison_path"]).exists():
        st.video(presentation["comparison_path"])

    d1, d2, d3 = st.columns(3)
    with d1:
        _download_button_for_file("Download .gicd", presentation["gicd_path"], Path(presentation["gicd_path"]).name, "application/octet-stream")
    with d2:
        _download_button_for_file("Download figure-eight MP4", presentation["figure8_path"], "depth_gic_figure8_demo.mp4", "video/mp4")
    with d3:
        _download_button_for_file("Download comparison MP4", presentation["comparison_path"], "depth_gic_comparison_demo.mp4", "video/mp4")

    with st.expander("Container metadata"):
        st.json(presentation["header"])


with st.expander("Interactive debug controls"):
    st.caption("Upload one RGB image and one depth map for quick inspection without preparing a dataset.")
    rgb_file = st.file_uploader("RGB image", type=["png", "jpg", "jpeg"], key="depth_gic_rgb_upload")
    depth_file = st.file_uploader("Depth image", type=["png", "jpg", "jpeg", "tif", "tiff"], key="depth_gic_depth_upload")

    c1, c2, c3 = st.columns(3)
    with c1:
        debug_max_size = st.slider("debug max_size", 128, 768, 384, 32)
    with c2:
        debug_stride = st.slider("debug stride", 2, 20, 6, 1)
    with c3:
        debug_edge_weight = st.checkbox("debug edge_weight", value=True)

    if st.button("Encode uploaded RGB-D to Depth-GIC", disabled=rgb_file is None or depth_file is None):
        rgb_path = depth_path = None
        try:
            rgb_path = _save_upload(rgb_file, Path(rgb_file.name).suffix or ".png")
            depth_path = _save_upload(depth_file, Path(depth_file.name).suffix or ".png")
            rgb, depth = load_rgb_depth(rgb_path, depth_path, max_size=debug_max_size)
            st.session_state.depth_gic_debug = _build_depth_gic(
                rgb,
                depth,
                _depth_to_u8(depth),
                max_size=debug_max_size,
                stride=debug_stride,
                edge_weight=debug_edge_weight,
                file_stem=Path(rgb_file.name).stem,
                render_videos=False,
            )
        except Exception as exc:
            st.error(f"Depth-GIC encoding failed: {exc}")
        finally:
            for path in [rgb_path, depth_path]:
                if path and os.path.exists(path):
                    os.remove(path)

    debug = st.session_state.get("depth_gic_debug")
    if debug:
        h, w = debug["depth"].shape
        vc1, vc2, vc3, vc4 = st.columns(4)
        with vc1:
            view_x = st.slider("view_x", -0.5, 0.5, 0.0, 0.01)
        with vc2:
            view_y = st.slider("view_y", -0.5, 0.5, 0.0, 0.01)
        with vc3:
            depth_scale = st.slider("depth_scale", 0.0, 5.0, 1.5, 0.1)
        with vc4:
            point_scale = st.slider("point_scale", 0.3, 3.0, 1.5, 0.1)

        shifted = render_depth_gaussians(
            debug["gaussians"],
            h,
            w,
            view_x=view_x,
            view_y=view_y,
            depth_scale=depth_scale,
            point_scale=point_scale,
        )
        q1, q2, q3, q4 = st.columns(4)
        q1.image(_rgb_to_u8(debug["rgb"]), caption="Original RGB", width="stretch")
        q2.image(_depth_to_u8(debug["depth"]), caption="Depth Map", width="stretch")
        q3.image(debug["render"], caption="Original View Rendering", width="stretch")
        q4.image(shifted, caption="Viewpoint-shifted Rendering", width="stretch")
        _download_button_for_file("Download debug .gicd", debug["gicd_path"], Path(debug["gicd_path"]).name, "application/octet-stream")
