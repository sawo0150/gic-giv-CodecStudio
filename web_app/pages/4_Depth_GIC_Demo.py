import io
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

PROJECT_DIR = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_DIR))

from gic_codec.depth_gaussian import (
    load_rgb_depth,
    rgbd_to_depth_gaussians,
    render_depth_gaussians,
    save_depth_gic,
    visualize_depth_colored_gaussians,
)


st.set_page_config(layout="wide", page_title="Depth-GIC Demo")
st.title("Depth-GIC Demo: RGB-D to Gaussian Representation")

st.write(
    "RGB-D stores pixel-wise color and depth. Depth-GIC converts it into depth-aware "
    "Gaussian primitives, so the same representation can be re-rendered with limited viewpoint changes."
)

rgb_file = st.file_uploader("RGB 이미지 업로드", type=["png", "jpg", "jpeg"], key="depth_gic_rgb")
depth_file = st.file_uploader("Depth 이미지 업로드", type=["png", "jpg", "jpeg", "tif", "tiff"], key="depth_gic_depth")

col_opt1, col_opt2, col_opt3 = st.columns(3)
with col_opt1:
    max_size = st.slider("max_size", min_value=128, max_value=768, value=384, step=32)
with col_opt2:
    stride = st.slider("stride", min_value=2, max_value=20, value=6, step=1)
with col_opt3:
    edge_weight = st.checkbox("edge_weight", value=True)


def _save_upload(uploaded, suffix):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded.getbuffer())
    tmp.close()
    return tmp.name


if "depth_gic_demo" not in st.session_state:
    st.session_state.depth_gic_demo = None

if st.button("Encode RGB-D to Depth-GIC", disabled=rgb_file is None or depth_file is None):
    rgb_path = depth_path = None
    try:
        rgb_path = _save_upload(rgb_file, Path(rgb_file.name).suffix or ".png")
        depth_path = _save_upload(depth_file, Path(depth_file.name).suffix or ".png")

        start = time.time()
        rgb, depth = load_rgb_depth(rgb_path, depth_path, max_size=max_size)
        gaussians = rgbd_to_depth_gaussians(rgb, depth, stride=stride, edge_weight=edge_weight)
        height, width = depth.shape
        render = render_depth_gaussians(gaussians, height, width)
        depth_vis = visualize_depth_colored_gaussians(gaussians, height, width)

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

        buffer_path = tempfile.NamedTemporaryFile(delete=False, suffix=".gicd").name
        save_depth_gic(buffer_path, header, gaussians, preview=render, metrics=metrics)
        with open(buffer_path, "rb") as f:
            gicd_bytes = f.read()

        st.session_state.depth_gic_demo = {
            "rgb": rgb,
            "depth": depth,
            "gaussians": gaussians,
            "render": render,
            "depth_vis": depth_vis,
            "header": header,
            "metrics": metrics,
            "gicd_bytes": gicd_bytes,
            "file_size": len(gicd_bytes),
            "file_name": f"{Path(rgb_file.name).stem}.gicd",
        }
        os.remove(buffer_path)
    except Exception as exc:
        st.error(f"Depth-GIC encoding failed: {exc}")
    finally:
        for path in [rgb_path, depth_path]:
            if path and os.path.exists(path):
                os.remove(path)

demo = st.session_state.depth_gic_demo
if demo is None:
    st.info("RGB 이미지와 depth map을 업로드한 뒤 Depth-GIC로 변환하세요.")
else:
    rgb_uint8 = (np.clip(demo["rgb"], 0.0, 1.0) * 255).astype(np.uint8)
    depth_uint8 = (np.clip(demo["depth"], 0.0, 1.0) * 255).astype(np.uint8)

    st.markdown("### Encoded RGB-D Gaussian Representation")
    col1, col2, col3, col4 = st.columns(4)
    col1.image(rgb_uint8, caption="Original RGB", use_column_width=True)
    col2.image(depth_uint8, caption="Depth Map", use_column_width=True)
    col3.image(demo["render"], caption="Gaussian Reconstruction", use_column_width=True)
    col4.image(demo["depth_vis"], caption="Depth-colored Gaussians", use_column_width=True)

    h, w = demo["depth"].shape
    meta1, meta2, meta3, meta4 = st.columns(4)
    meta1.metric("Image Size", f"{w}x{h}")
    meta2.metric("Gaussian Count", f"{len(demo['gaussians']['xyz']):,}")
    meta3.metric("Stride", str(demo["header"]["encoding_settings"]["stride"]))
    meta4.metric("Estimated .gicd Size", f"{demo['file_size'] / 1024:.1f} KB")

    st.json(demo["header"])
    st.download_button(
        "Download .gicd",
        data=demo["gicd_bytes"],
        file_name=demo["file_name"],
        mime="application/octet-stream",
    )

    st.markdown("### Viewpoint Slider Demo")
    vc1, vc2, vc3, vc4 = st.columns(4)
    with vc1:
        view_x = st.slider("view_x", -0.5, 0.5, 0.0, 0.01)
    with vc2:
        view_y = st.slider("view_y", -0.5, 0.5, 0.0, 0.01)
    with vc3:
        depth_scale = st.slider("depth_scale", 0.0, 5.0, 1.5, 0.1)
    with vc4:
        point_scale = st.slider("point_scale", 0.3, 3.0, 1.0, 0.1)

    shifted = render_depth_gaussians(
        demo["gaussians"],
        h,
        w,
        view_x=view_x,
        view_y=view_y,
        depth_scale=depth_scale,
        point_scale=point_scale,
    )

    col_orig, col_shift = st.columns(2)
    col_orig.image(
        render_depth_gaussians(demo["gaussians"], h, w, point_scale=point_scale),
        caption="Original View Rendering",
        use_column_width=True,
    )
    col_shift.image(shifted, caption="Viewpoint-shifted Rendering", use_column_width=True)
