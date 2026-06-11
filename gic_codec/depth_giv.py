import io
import json
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image


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


def save_depth_giv(output_path, header, frames, previews=None, metrics=None):
    header = dict(header)
    header.setdefault("codec_name", "GIV-D")
    header.setdefault("version", "1.0.0")
    header.setdefault("note", "Depth-aware Gaussian Video Container demo")
    previews = previews or {}

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    index_frames = []
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        zipf.writestr("header.json", json.dumps(header, indent=4))

        for idx, frame in enumerate(frames, start=1):
            filename = f"frames/frame_{idx:06d}.npz"
            preview_name = f"previews/frame_{idx:06d}.png"
            gaussians = frame["gaussians"] if "gaussians" in frame else frame

            buffer = io.BytesIO()
            np.savez_compressed(buffer, **gaussians)
            zipf.writestr(filename, buffer.getvalue())

            if idx in previews:
                preview_data = _preview_bytes(previews[idx])
                if preview_data is not None:
                    zipf.writestr(preview_name, preview_data)

            index_frames.append(
                {
                    "frame_idx": idx,
                    "filename": filename,
                    "preview": preview_name if idx in previews else None,
                    "num_points": int(len(gaussians["xyz"])),
                    "width": int(frame.get("width", header["video_info"]["width"])),
                    "height": int(frame.get("height", header["video_info"]["height"])),
                }
            )

        zipf.writestr("index.json", json.dumps({"frames": index_frames}, indent=4))
        if metrics is not None:
            zipf.writestr("metrics.json", json.dumps(metrics, indent=4))


def load_depth_giv(input_path, load_frames=True):
    data = {"header": None, "index": None, "frames": {}, "previews": {}, "metrics": None}
    with zipfile.ZipFile(input_path, "r") as zipf:
        names = zipf.namelist()
        if "header.json" in names:
            data["header"] = json.loads(zipf.read("header.json").decode("utf-8"))
        if "index.json" in names:
            data["index"] = json.loads(zipf.read("index.json").decode("utf-8"))
        if "metrics.json" in names:
            data["metrics"] = json.loads(zipf.read("metrics.json").decode("utf-8"))

        for name in names:
            if name.startswith("previews/") and name.endswith(".png"):
                frame_idx = int(Path(name).stem.split("_")[-1])
                data["previews"][frame_idx] = Image.open(io.BytesIO(zipf.read(name))).convert("RGB")

        if load_frames:
            for name in names:
                if name.startswith("frames/") and name.endswith(".npz"):
                    frame_idx = int(Path(name).stem.split("_")[-1])
                    npz_data = np.load(io.BytesIO(zipf.read(name)))
                    data["frames"][frame_idx] = {key: npz_data[key] for key in npz_data.files}
    return data
