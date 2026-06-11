# Depth-GIC / GIV-D Presentation Demo

## 핵심 메시지

Depth-GIC/GIV-D 데모는 기존 `.gic` / `.giv` 코덱을 대체하는 기능이 아니라, RGB-D 데이터를 Gaussian primitive representation으로 확장할 수 있음을 보여주는 발표용 proof-of-concept이다.

> RGB-D는 pixel-wise sensor data이고, GIC-D/GIV-D는 RGB-D로부터 파생된 primitive-wise renderable representation이다.

이 데모의 목표는 JPEG/WebP보다 압축률이 좋다는 것을 증명하는 것이 아니다. RGB, 위치, 깊이, scale, opacity를 가진 Gaussian-like primitive를 저장하면 같은 데이터를 제한적인 다른 시점에서 다시 렌더링할 수 있다는 점을 시각적으로 보여주는 것이 목표다.

## 저장소 구조

| 경로 | 역할 |
|---|---|
| `gic_codec/depth_gaussian.py` | RGB-D를 depth-aware Gaussian primitive로 변환하고 pseudo-3D 렌더링 |
| `gic_codec/depth_giv.py` | `.givd` zip container 저장/로드 |
| `gic_codec/depth_dataset_utils.py` | DIODE/NYU/robot RGB-D 샘플 로드 및 정규화 |
| `gic_codec/depth_trajectory.py` | figure-eight spatial image video와 robot third-person replay MP4 생성 |
| `scripts/data_prep/` | 작은 발표용 RGB-D 샘플 준비 스크립트 |
| `scripts/export_presentation_depth_demos.py` | PPT용 PNG/MP4 asset 일괄 생성 |
| `web_app/pages/4_Depth_GIC_Demo.py` | Depth-GIC spatial image presentation page |
| `web_app/pages/5_GIVD_Replay_Demo.py` | GIV-D robot replay presentation page |
| `outputs/ppt_assets/depth_demo/` | 생성된 발표용 이미지/비디오 저장 위치 |

## 데이터 준비 원칙

디스크 여유 공간을 고려해 대형 전체 데이터셋을 자동으로 받지 않는다.

| 데이터셋 | 처리 방식 |
|---|---|
| DIODE | validation archive만 사용하고 10개 샘플만 보존 |
| NYU Depth V2 | 공식 web sample montage crop 또는 기존 로컬 폴더에서 10개만 추출, huge TFDS 자동 다운로드 금지 |
| ManiSkill | full RLDS/TFDS 다운로드 금지, 공식 RGB+Depth texture 예시를 lightweight replay sample로 사용 |
| RLBench | prebuilt 대형 dataset 다운로드 금지, 설치되어 있지 않으면 fallback 또는 missing metadata 처리 |

## 데이터 준비 명령어

DIODE 10개 샘플:

```bash
python scripts/data_prep/prepare_diode_samples.py --num_samples 10 --cleanup
```

NYU Depth V2 10개 샘플:

```bash
python scripts/data_prep/prepare_nyu_samples.py --source /path/to/nyu --num_samples 10
```

실제 NYU subset이 없지만 Streamlit 발표 흐름을 바로 확인해야 하는 경우:

```bash
python scripts/data_prep/prepare_nyu_samples.py --official_web_samples --num_samples 10
```

이 모드는 NYU Depth V2 공식 페이지의 sample montage에서 실제 실내 RGB와 colorized depth 예시를 crop한다. 단, depth는 원본 meter 단위 depth가 아니라 colorized visualization을 scalar map으로 변환한 것이므로 정량 depth 평가에는 사용하지 않는다.

ManiSkill wrist RGB-D demo:

```bash
python scripts/data_prep/prepare_maniskill_wrist_demo.py
```

RLBench eye-in-hand RGB-D demo:

```bash
python scripts/data_prep/prepare_rlbench_eye_in_hand_demo.py
```

ManiSkill script는 기본적으로 공식 ManiSkill RGB+Depth texture visualization을 다운로드해 짧은 replay sequence로 변환한다. 설치 없이도 메인 발표 데모가 단순 도형이 아니라 robot/tabletop RGB-D 예시로 동작한다. 강제 procedural fallback은 `--fallback` 옵션을 사용할 때만 생성한다.

## Depth-GIC Spatial Image Demo

Streamlit page:

```text
web_app/pages/4_Depth_GIC_Demo.py
```

데이터 흐름:

```text
DIODE/NYU RGB-D sample
-> robust depth normalization
-> RGB-D grid sampling
-> depth-aware Gaussian primitives
-> .gicd container
-> figure-eight camera trajectory rendering
-> MP4 comparison video
```

생성되는 컨테이너:

```text
sample.gicd
├── header.json
├── gaussians.npz
├── preview.png
└── metrics.json
```

`gaussians.npz`의 주요 key:

| key | shape | 의미 |
|---|---:|---|
| `xyz` | `[N, 3]` | normalized x/y position and normalized depth z |
| `scaling` | `[N, 2]` | 2D Gaussian-like primitive size |
| `rotation` | `[N, 1]` | 현재 demo에서는 0 |
| `opacity` | `[N, 1]` | primitive alpha, edge 근처에서 보정 가능 |
| `features_dc` | `[N, 3]` | RGB color |
| `depth` | `[N, 1]` | explicit normalized depth |

## Figure-eight Camera Trajectory

구현 위치:

```text
gic_codec/depth_trajectory.py
```

핵심 함수:

```text
generate_figure8_camera_path()
render_depth_gic_figure8_video()
render_depth_gic_comparison_video()
```

trajectory는 다음과 같은 smooth figure-eight offset을 사용한다.

```text
view_x = amplitude_x * sin(t)
view_y = amplitude_y * sin(2t) / 2
```

renderer는 실제 3D unprojection이 아니라 normalized depth 기반 pseudo-parallax를 적용한다.

```text
x_shifted = x + view_x * (1 - z) * depth_scale
y_shifted = y + view_y * (1 - z) * depth_scale
```

near primitive가 far primitive보다 더 크게 움직이므로 제한적인 spatial image 느낌을 낼 수 있다.

## GIV-D Robot Replay Demo

Streamlit page:

```text
web_app/pages/5_GIVD_Replay_Demo.py
```

데이터 흐름:

```text
ManiSkill/RLBench wrist RGB-D sequence
-> frame-wise depth Gaussian conversion
-> .givd container
-> shifted back-and-up pseudo camera render
-> third-person-style replay MP4
```

생성되는 컨테이너:

```text
sample.givd
├── header.json
├── index.json
├── frames/frame_000001.npz
├── previews/frame_000001.png
└── metrics.json
```

Robot replay는 wrist/eye-in-hand 원본 시점과 비교해 `view_y`를 위/뒤쪽으로 이동한 pseudo camera를 사용한다. 정확한 3D geometry를 복원하는 기능은 아니지만, 같은 primitive sequence를 다른 시점에서 재렌더링한다는 발표 메시지를 만들 수 있다.

## Streamlit 사용법

앱 실행:

```bash
streamlit run web_app/app.py
```

Depth-GIC demo:

1. `Depth-GIC Spatial Image Demo` 페이지 열기
2. DIODE 또는 NYU Depth V2 선택
3. 샘플 선택
4. `Generate Depth-GIC figure-eight video` 클릭
5. RGB, depth, Gaussian render, depth-colored render, MP4 comparison video 확인
6. `.gicd`, figure-eight MP4, comparison MP4 다운로드

GIV-D demo:

1. `GIV-D Robot Eye-in-Hand Replay Demo` 페이지 열기
2. ManiSkill 또는 RLBench 선택
3. `Generate third-person GIV-D replay` 클릭
4. RGB frame strip, depth frame strip, MP4 replay 확인
5. `.givd`, replay MP4, comparison MP4 다운로드

수동 업로드와 viewpoint slider는 발표 메인 흐름에서 제외하고, 각 페이지의 expander 안에 debug/inspection 기능으로 남겨두었다.

## PPT Asset Export

일괄 생성 명령어:

```bash
python scripts/export_presentation_depth_demos.py
```

출력 위치:

```text
outputs/ppt_assets/depth_demo/
```

생성 파일:

| 파일 | 내용 | PPT 사용 위치 |
|---|---|---|
| `depth_gic_original_rgb.png` | RGB 입력 샘플 | RGB-D 입력 설명 |
| `depth_gic_depth_map.png` | depth map visualization | depth signal 설명 |
| `depth_gic_gaussian_render.png` | Depth-GIC primitive render | representation 결과 |
| `depth_gic_depth_colored.png` | depth-colored primitive visualization | primitive가 depth를 포함함을 설명 |
| `depth_gic_figure8_demo.mp4` | figure-eight camera trajectory | spatial image demo |
| `depth_gic_comparison_demo.mp4` | RGB/depth/render side-by-side | 발표 시연 영상 |
| `givd_robot_original_strip.png` | robot wrist RGB sequence strip | first-person input 설명 |
| `givd_robot_depth_strip.png` | robot depth sequence strip | RGB-D sequence 설명 |
| `givd_third_person_replay.mp4` | shifted viewpoint replay | GIV-D 핵심 시연 |
| `givd_comparison_demo.mp4` | RGB/depth/replay side-by-side | 발표 시연 영상 |
| `demo_summary.json` | 생성 asset과 샘플 정보 | 결과 확인 및 재현성 |

## 한계

- proof-of-concept demo이다.
- pseudo-3D parallax만 구현한다.
- full 3D reconstruction이 아니다.
- occlusion-complete novel view synthesis를 수행하지 않는다.
- compression benchmark가 목적이 아니다.
- geometry 품질은 depth map 품질에 크게 의존한다.
- current renderer는 depth-sorted soft blob renderer이며, Instant-GI의 differentiable rasterizer를 사용하는 것은 아니다.

## 향후 개선

- camera intrinsics 기반 depth unprojection
- z-buffer와 hole filling을 포함한 occlusion-aware renderer
- temporal primitive reuse for GIV-D
- keyframe/delta compression
- RGB-only input을 위한 monocular depth estimator 연동
- `.gicd` / `.givd` binary compact format
- Web backend 분리 및 GPU 렌더링 최적화
