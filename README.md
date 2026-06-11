# GIC/GIV Codec Studio

Instant-GI 기반 2D Gaussian Splatting을 이미지/동영상 코덱 형태로 확장한 기말 프로젝트입니다. 이미지는 `.gic`(Gaussian Image Codec), 동영상은 `.giv`(Gaussian Image Video) 컨테이너로 저장하고, Streamlit 웹앱에서 인코딩과 재생을 시연합니다.

## What This Project Builds

- `gic_codec/`: `.gic/.giv` 인코더, 디코더, 포맷 I/O, metrics, Auto Quality Analyzer
- `web_app/`: Streamlit 기반 Encoder / Player / Export Report Studio
- `experiments/`: 벤치마크 및 PPT용 asset export pipeline
- `configs/`: Low / Medium / High / Auto quality 설정
- `docs/`: 발표 자료 제작용 구현 설명 문서

## Format Overview

`.gic` is a zip container:

```text
sample.gic
├── header.json
├── gaussians.npz
├── preview.png
├── metrics.json
└── logs.json
```

`.giv` is a Gaussian video container. It supports independent per-frame initialization and optional previous-frame warm start:

```text
sample.giv
├── header.json
├── index.json
├── frames/frame_000001.npz
├── previews/frame_000001.png
└── metrics.json
```

## Repository Layout

```text
final_project/
├── Instant-GI/                 # upstream Instant-GI backend
├── gic_codec/                  # codec core package
├── web_app/                    # Streamlit GUI
├── experiments/                # benchmark and PPT asset scripts
├── scripts/                    # setup and run helpers
├── configs/                    # quality mode configs
├── docs/                       # project explanation for PPT
├── data/                       # small sample data only
├── outputs/                    # generated outputs, gitignored
└── pretrained/                 # checkpoint README only, model gitignored
```

## GitHub / Large File Policy

Large files are intentionally excluded by `.gitignore`:

- pretrained checkpoints: `*.pth`, `pretrained/*`, `Instant-GI/checkpoints/`
- generated codec outputs: `outputs/`, `*.gic`, `*.giv`
- large datasets: `data/div2k/`, downloaded zip archives
- local videos: `*.mp4`, `*.avi`, `*.mov`, `*.mkv`

The Instant-GI source is expected at `Instant-GI/`. For a clean GitHub repository, add it as a submodule or document the clone step:

```bash
git submodule add https://github.com/whoiszzj/Instant-GI.git Instant-GI
```

If `Instant-GI/` already exists locally as a normal clone, avoid committing large checkpoints under `Instant-GI/checkpoints/`.

## Setup

Recommended local environment on this machine:

```bash
conda activate 3dgs
python -m pip install -r requirements.txt
python -m pip install streamlit gdown
python scripts/setup_models.py
```

`scripts/setup_models.py` checks for `epoch_best_ks_3.pth` and mirrors it into:

```text
pretrained/epoch_best_ks_3.pth
Instant-GI/checkpoints/epoch_best_ks_3.pth
```

The full Torch/Instant-GI backend is used when available. If Torch or Instant-GI imports fail, the package can fall back to a NumPy demo backend that preserves the same `.gic/.giv` container structure.

## CLI Usage

Image to `.gic`:

```bash
python -m gic_codec.encoder \
  --input data/kodak/kodim06.png \
  --mode auto \
  --output outputs/gic/kodim06.gic
```

Decode `.gic`:

```bash
python -m gic_codec.decoder \
  --input outputs/gic/kodim06.gic \
  --output outputs/gic/kodim06_recon.png
```

Video or frame folder to `.giv`:

```bash
python -m gic_codec.encoder \
  --input data/davis/bear \
  --video \
  --mode auto \
  --max_frames 30 \
  --output outputs/giv/bear.giv
```

Warm-start each frame from the previous optimized Gaussian state:

```bash
python -m gic_codec.encoder \
  --input data/davis/bear \
  --video \
  --mode auto \
  --max_frames 30 \
  --video_init previous_frame \
  --output outputs/giv/bear_prev_init.giv
```

MP4 to `.giv` with explicit FPS and visible progress:

```bash
PYTHONUNBUFFERED=1 conda run --no-capture-output -n 3dgs python -m gic_codec.encoder \
  --input "/path/to/video.mp4" \
  --video \
  --mode auto \
  --init net \
  --iter 1000 \
  --max_frames 300 \
  --fps 30 \
  --video_init previous_frame \
  --output outputs/giv/video_300.giv
```

Decode `.giv`:

```bash
python -m gic_codec.decoder \
  --input outputs/giv/bear.giv \
  --output outputs/giv/bear_frames
```

## Streamlit App

Run locally:

```bash
conda activate 3dgs
bash scripts/run_streamlit.sh
```

Open:

```text
http://127.0.0.1:8501
```

Pages:

- `Encoder`: upload image/MP4 or use a frame folder path, then export `.gic/.giv`
- `Player`: upload `.gic/.giv`, inspect metadata, preview, and render frames
- `Export Report`: generate PPT charts and image assets

## PPT Assets

Generate presentation assets:

```bash
python experiments/export_ppt_assets.py
```

Generated files are written to `outputs/ppt_assets/`, including:

- `codec_grid_comparison.png`
- `restored_error_map.png`
- `video_frames_strip.png`
- `rd_curve_chart.png`
- `frame_psnr_timeline.png`

## Documentation

The detailed Korean explanation for PPT creation is here:

```text
docs/project_explanation_for_ppt.md
```

It includes architecture, format details, implementation notes, actual metrics found in `outputs/metrics`, slide-by-slide outline, and TODO items.
