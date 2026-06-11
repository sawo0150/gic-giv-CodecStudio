import os
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_DIR))

from gic_codec.depth_dataset_utils import load_robot_rgbd_sequence, list_robot_rgbd_sequences
from gic_codec.depth_gaussian import load_rgb_depth, render_depth_gaussians, rgbd_to_depth_gaussians
from gic_codec.depth_giv import save_depth_giv
from gic_codec.depth_trajectory import render_givd_robot_comparison_video, render_givd_third_person_replay


OUTPUT_DIR = PROJECT_DIR / "outputs" / "ppt_assets" / "depth_demo"


st.set_page_config(layout="wide", page_title="GIV-D Robot Replay")
st.title("GIV-D Robot Eye-in-Hand Replay Demo")
st.write(
    "Wrist RGB-D videos are fixed to the robot's first-person view. GIV-D stores each frame as "
    "depth-aware Gaussian primitives, enabling a pre-rendered third-person-style replay from a shifted "
    "back-and-up viewpoint."
)


def _rgb_to_u8(rgb):
    return (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8) if rgb.dtype != np.uint8 else rgb


def _depth_to_u8(depth):
    return (np.clip(depth, 0.0, 1.0) * 255).astype(np.uint8)


def _make_strip(images, label, max_items=8):
    if not images:
        return None
    selected = images[:max_items]
    thumbs = []
    target_h = 120
    for img in selected:
        arr = _rgb_to_u8(img)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=2)
        pil = Image.fromarray(arr).convert("RGB")
        ratio = target_h / pil.height
        pil = pil.resize((max(1, int(pil.width * ratio)), target_h), Image.Resampling.LANCZOS)
        thumbs.append(pil)

    width = sum(img.width for img in thumbs)
    strip = Image.new("RGB", (width, target_h + 28), "white")
    x = 0
    for img in thumbs:
        strip.paste(img, (x, 28))
        x += img.width
    draw = ImageDraw.Draw(strip)
    draw.rectangle([0, 0, width, 28], fill=(0, 0, 0))
    draw.text((8, 7), label, fill=(255, 255, 255))
    return np.asarray(strip)


def _save_upload(uploaded, folder):
    path = folder / uploaded.name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(uploaded.getbuffer())
    return path


def _load_pairs_from_zip(zip_file, tmpdir, max_frames):
    zip_path = tmpdir / "sequence.zip"
    zip_path.write_bytes(zip_file.getbuffer())
    extract_dir = tmpdir / "zip_sequence"
    with zipfile.ZipFile(zip_path, "r") as zipf:
        zipf.extractall(extract_dir)

    rgb_paths = sorted(
        [p for p in extract_dir.rglob("*") if p.parent.name.lower() == "rgb" and p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    )
    depth_paths = sorted(
        [
            p
            for p in extract_dir.rglob("*")
            if p.parent.name.lower() == "depth" and p.suffix.lower() in {".png", ".npy", ".jpg", ".jpeg", ".tif", ".tiff"}
        ]
    )
    return list(zip(rgb_paths, depth_paths))[:max_frames]


def _build_givd(rgbd_frames, *, dataset_name, max_size, stride, fps=12, render_videos=True):
    start = time.time()
    frames = []
    previews = {}
    rgb_frames = []
    depth_vis_frames = []
    gaussians_per_frame = []

    progress = st.progress(0)
    status = st.empty()
    for idx, (rgb, depth, depth_vis) in enumerate(rgbd_frames, start=1):
        gaussians = rgbd_to_depth_gaussians(rgb, depth, stride=stride, edge_weight=True)
        h, w = depth.shape
        preview = render_depth_gaussians(gaussians, h, w, point_scale=1.5)
        frames.append({"gaussians": gaussians, "width": w, "height": h})
        previews[idx] = preview
        rgb_frames.append(rgb)
        depth_vis_frames.append(depth_vis)
        gaussians_per_frame.append(gaussians)
        progress.progress(idx / len(rgbd_frames))
        status.write(f"Encoded frame {idx}/{len(rgbd_frames)} | Gaussians: `{len(gaussians['xyz']):,}`")

    h, w = rgbd_frames[0][1].shape
    header = {
        "codec_name": "GIV-D",
        "version": "1.0.0",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "video_info": {"width": int(w), "height": int(h), "total_frames": len(frames), "fps": float(fps)},
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

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    givd_path = OUTPUT_DIR / f"{dataset_name}_robot_replay.givd"
    save_depth_giv(givd_path, header, frames, previews=previews, metrics=metrics)

    replay_path = None
    comparison_path = None
    if render_videos:
        replay_path = render_givd_third_person_replay(
            rgbd_frames,
            frames,
            OUTPUT_DIR / "givd_third_person_replay.mp4",
            fps=fps,
            depth_scale=1.8,
            point_scale=1.5,
        )
        comparison_path = render_givd_robot_comparison_video(
            rgb_frames,
            depth_vis_frames,
            gaussians_per_frame,
            OUTPUT_DIR / "givd_comparison_demo.mp4",
            fps=fps,
        )

    return {
        "frames": frames,
        "rgbd_frames": rgbd_frames,
        "rgb_frames": rgb_frames,
        "depth_vis_frames": depth_vis_frames,
        "gaussians_per_frame": gaussians_per_frame,
        "header": header,
        "metrics": metrics,
        "givd_path": str(givd_path),
        "replay_path": replay_path,
        "comparison_path": comparison_path,
    }


def _download_button_for_file(label, path, file_name, mime):
    if path and Path(path).exists():
        st.download_button(label, data=Path(path).read_bytes(), file_name=file_name, mime=mime)


st.markdown("### Presentation-ready robot sequence")
dataset_label = st.selectbox("Dataset", ["ManiSkill wrist RGB-D", "RLBench eye-in-hand RGB-D"])
dataset_key = "maniskill" if dataset_label.startswith("ManiSkill") else "rlbench"
seq_meta = list_robot_rgbd_sequences(dataset_key)

c1, c2, c3 = st.columns(3)
with c1:
    max_frames = st.slider("max_frames", 1, 30, 18, 1, key="robot_max_frames")
with c2:
    max_size = st.slider("max_size", 128, 512, 320, 32, key="robot_max_size")
with c3:
    stride = st.slider("stride", 4, 20, 7, 1, key="robot_stride")

if not seq_meta or not seq_meta.get("frames"):
    script_name = "prepare_maniskill_wrist_demo.py" if dataset_key == "maniskill" else "prepare_rlbench_eye_in_hand_demo.py"
    st.warning(f"{dataset_label} samples are missing. Run `python scripts/data_prep/{script_name}` first.")
else:
    metadata = seq_meta.get("metadata", {})
    source = metadata.get("source", "unknown")
    st.caption(f"Found {len(seq_meta['frames'])} frames in `{seq_meta['root']}` | source: `{source}`")
    if source == "procedural_fallback":
        st.warning("This sequence is procedural fallback data and is not presentation-ready. Prefer the ManiSkill official-doc RGB-D sample or real generated robot RGB-D data.")
    elif source == "maniskill_official_doc_rgbd_sample":
        st.info("This sequence is derived from the official ManiSkill RGB+Depth texture visualization, with small crop shifts to create a short replay.")
    if st.button("Generate third-person GIV-D replay", type="primary"):
        try:
            rgbd_frames = load_robot_rgbd_sequence(dataset_key, max_frames=max_frames, max_size=max_size)
            if not rgbd_frames:
                st.warning("No RGB-D frames could be loaded.")
            else:
                with st.spinner("Encoding GIV-D and rendering third-person replay MP4..."):
                    st.session_state.givd_presentation = _build_givd(
                        rgbd_frames,
                        dataset_name=dataset_key,
                        max_size=max_size,
                        stride=stride,
                        fps=12,
                        render_videos=True,
                    )
        except Exception as exc:
            st.error(f"GIV-D presentation demo failed: {exc}")


presentation = st.session_state.get("givd_presentation")
if presentation:
    st.markdown("### Generated GIV-D replay")
    strip1, strip2 = st.columns(2)
    strip1.image(_make_strip(presentation["rgb_frames"], "Original wrist RGB sequence"), caption="Original RGB frame strip", width="stretch")
    depth_rgb_frames = [np.stack([d] * 3, axis=2) for d in presentation["depth_vis_frames"]]
    strip2.image(_make_strip(depth_rgb_frames, "Depth sequence"), caption="Depth frame strip", width="stretch")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Frames", str(presentation["metrics"]["total_frames"]))
    m2.metric("Avg. Points / Frame", f"{presentation['metrics']['avg_points_per_frame']:,.0f}")
    m3.metric("Stride", str(presentation["header"]["encoding_settings"]["stride"]))
    m4.metric(".givd Size", f"{Path(presentation['givd_path']).stat().st_size / 1024:.1f} KB")

    if presentation.get("comparison_path") and Path(presentation["comparison_path"]).exists():
        st.video(presentation["comparison_path"], format="video/mp4")

    d1, d2, d3 = st.columns(3)
    with d1:
        _download_button_for_file("Download .givd", presentation["givd_path"], Path(presentation["givd_path"]).name, "application/octet-stream")
    with d2:
        _download_button_for_file("Download replay MP4", presentation["replay_path"], "givd_third_person_replay.mp4", "video/mp4")
    with d3:
        _download_button_for_file("Download comparison MP4", presentation["comparison_path"], "givd_comparison_demo.mp4", "video/mp4")

    with st.expander("Container metadata"):
        st.json(presentation["header"])
        st.json(presentation["metrics"])


with st.expander("Interactive inspection"):
    st.caption("Upload a short RGB-D sequence for quick inspection, or inspect the last generated presentation sequence.")
    rgb_files = st.file_uploader(
        "RGB frame images",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
        key="givd_rgb_frames",
    )
    depth_files = st.file_uploader(
        "Depth frame images",
        type=["png", "jpg", "jpeg", "tif", "tiff"],
        accept_multiple_files=True,
        key="givd_depth_frames",
    )
    zip_file = st.file_uploader("or upload a zip containing rgb/ and depth/ folders", type=["zip"], key="givd_zip")

    can_encode = zip_file is not None or (rgb_files and depth_files and len(rgb_files) == len(depth_files))
    if st.button("Encode uploaded sequence to GIV-D demo", disabled=not can_encode):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            try:
                if zip_file is not None:
                    pairs = _load_pairs_from_zip(zip_file, tmpdir, max_frames=max_frames)
                else:
                    rgb_dir = tmpdir / "rgb_uploads"
                    depth_dir = tmpdir / "depth_uploads"
                    rgb_paths = sorted([_save_upload(f, rgb_dir) for f in rgb_files], key=lambda p: p.name)
                    depth_paths = sorted([_save_upload(f, depth_dir) for f in depth_files], key=lambda p: p.name)
                    pairs = list(zip(rgb_paths, depth_paths))[:max_frames]

                rgbd_frames = []
                for rgb_path, depth_path in pairs:
                    rgb, depth = load_rgb_depth(rgb_path, depth_path, max_size=max_size)
                    rgbd_frames.append((rgb, depth, _depth_to_u8(depth)))
                st.session_state.givd_debug = _build_givd(
                    rgbd_frames,
                    dataset_name="uploaded",
                    max_size=max_size,
                    stride=stride,
                    fps=12,
                    render_videos=False,
                )
            except Exception as exc:
                st.error(f"GIV-D encoding failed: {exc}")

    debug = st.session_state.get("givd_debug") or presentation
    if debug:
        total = len(debug["frames"])
        frame_idx = st.slider("Frame", 1, total, 1)
        vc1, vc2, vc3, vc4 = st.columns(4)
        with vc1:
            view_x = st.slider("view_x", -0.5, 0.5, 0.0, 0.01)
        with vc2:
            view_y = st.slider("view_y", -0.5, 0.5, -0.25, 0.01)
        with vc3:
            depth_scale = st.slider("depth_scale", 0.0, 5.0, 1.8, 0.1)
        with vc4:
            point_scale = st.slider("point_scale", 0.3, 3.0, 1.5, 0.1)

        idx = frame_idx - 1
        rgb, depth, depth_vis = debug["rgbd_frames"][idx]
        gaussians = debug["frames"][idx]["gaussians"]
        shifted = render_depth_gaussians(
            gaussians,
            depth.shape[0],
            depth.shape[1],
            view_x=view_x,
            view_y=view_y,
            depth_scale=depth_scale,
            point_scale=point_scale,
        )

        q1, q2, q3 = st.columns(3)
        q1.image(_rgb_to_u8(rgb), caption="Original wrist RGB", width="stretch")
        q2.image(depth_vis, caption="Depth Map", width="stretch")
        q3.image(shifted, caption="Viewpoint-shifted GIV-D Replay", width="stretch")

        if st.button("Play Gaussian Replay"):
            placeholder = st.empty()
            for i, frame in enumerate(debug["frames"]):
                _, depth_i, _ = debug["rgbd_frames"][i]
                img = render_depth_gaussians(
                    frame["gaussians"],
                    depth_i.shape[0],
                    depth_i.shape[1],
                    view_x=view_x,
                    view_y=view_y,
                    depth_scale=depth_scale,
                    point_scale=point_scale,
                )
                placeholder.image(img, caption=f"Frame {i + 1}/{total}", width="stretch")
                time.sleep(0.10)
