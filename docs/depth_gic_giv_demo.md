# Depth-GIC / GIV-D Demo

## What This Demo Is

This demo extends the existing GIC/GIV Codec Studio with depth-aware Gaussian containers:

- `.gicd`: a depth-aware Gaussian image container derived from one RGB-D image.
- `.givd`: a depth-aware Gaussian video container derived from a short RGB-D sequence.

The goal is not to prove better compression than JPEG/WebP or raw RGB-D. The goal is to show that RGB-D data can be converted into a primitive-wise renderable representation.

> RGB-D is pixel-wise sensor data, while GIC-D/GIV-D is a primitive-wise renderable representation derived from RGB-D.

## Difference Between RGB-D and Depth-GIC/GIV-D

| Representation | Data Unit | Main Use |
|---|---|---|
| RGB-D | Per-pixel RGB and depth | Sensor frame, dense image/depth map |
| GIC-D | Per-primitive color, 2D position, depth, scale, opacity | Renderable Gaussian-like image representation |
| GIV-D | Sequence of GIC-D-like frame primitives | RGB-D sequence replay with limited viewpoint shift |

## Demo 1: RGB-D to Depth-GIC

The Streamlit page `4_Depth_GIC_Demo.py` accepts one RGB image and one depth image. It samples RGB-D pixels on a regular grid and creates one Gaussian-like primitive per sampled pixel.

Each primitive stores:

- `xyz`: normalized x/y position and normalized depth z
- `scaling`: approximate 2D primitive size
- `rotation`: currently zero
- `opacity`: constant or edge-adjusted opacity
- `features_dc`: RGB color
- `depth`: explicit depth column

The output can be downloaded as `.gicd`, a zip container with:

```text
header.json
gaussians.npz
preview.png
metrics.json
```

## Demo 2: Viewpoint Slider

The same Streamlit page provides sliders for:

- `view_x`
- `view_y`
- `depth_scale`
- `point_scale`

Rendering applies a simple parallax rule:

```text
x_shifted = x + view_x * (1 - z) * depth_scale
y_shifted = y + view_y * (1 - z) * depth_scale
```

Nearby points shift more than far points. This is an approximate visual demonstration, not physically correct novel view synthesis.

## Demo 3: First-person RGB-D Sequence to GIV-D Replay

The Streamlit page `5_GIVD_Replay_Demo.py` accepts either:

- multiple RGB frames and matching depth frames, paired by sorted filename, or
- a zip file containing `rgb/` and `depth/` folders.

Each RGB-D frame is converted into depth-aware Gaussian primitives and saved in a `.givd` container:

```text
header.json
index.json
frames/frame_000001.npz
previews/frame_000001.png
metrics.json
```

The page provides a frame slider, viewpoint sliders, a simple play button, and a `.givd` download button.

## CLI Usage

Depth-GIC:

```bash
python scripts/demo_depth_gic.py \
  --rgb path/to/rgb.png \
  --depth path/to/depth.png \
  --output outputs/demo/sample.gicd \
  --max_size 384 \
  --stride 6
```

GIV-D:

```bash
python scripts/demo_depth_giv.py \
  --rgb_dir path/to/rgb \
  --depth_dir path/to/depth \
  --output outputs/demo/sample.givd \
  --max_frames 8 \
  --max_size 320 \
  --stride 8
```

## Limitations

- This is a proof-of-concept.
- Viewpoint change is limited and approximate.
- This is not full 3D reconstruction.
- This does not yet perform occlusion-complete novel view synthesis.
- Compression performance is not the goal of this demo.
- Geometry accuracy depends on depth quality.
- The renderer uses simple depth-sorted colored circles rather than a physically accurate splatting pipeline.

## Future Work

- Better camera model and depth unprojection.
- Occlusion-aware renderer.
- Temporal reuse between GIV-D frames.
- Primitive quantization and compact binary storage.
- Depth confidence estimation.
- Integration with learned monocular depth prediction for RGB-only inputs.
