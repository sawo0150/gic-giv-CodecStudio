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

## 최종 반영 사항 요약

초기 데모는 사각형, 원, 단색 물체로 구성된 procedural fallback을 사용했기 때문에 발표용 데이터로 부적절했다. 현재 버전은 메인 데모 데이터를 실제/공식 샘플 기반으로 교체했다.

| 항목 | 이전 상태 | 현재 상태 |
|---|---|---|
| Depth-GIC 이미지 샘플 | procedural NYU-like 실내 도형 | NYU Depth V2 공식 웹 sample montage에서 crop한 실제 실내 RGB-D 예시 |
| GIV-D robot 샘플 | procedural gripper/tabletop 도형 | ManiSkill 공식 RGB+Depth texture visualization 기반 robot/tabletop RGB-D 예시 |
| RLBench 샘플 | procedural fallback 포함 | synthetic 샘플 제거, 실제 데이터가 없으면 missing 안내 |
| MP4 저장 방식 | browser 호환이 보장되지 않는 MP4 | H.264 `avc1` + `yuv420p`로 저장 |
| Streamlit video 표시 | MIME 미지정 | `st.video(..., format="video/mp4")`로 명시 |
| NYU depth shape | RGB와 depth 폭이 1px 어긋날 수 있음 | depth crop을 RGB 크기로 resize 후 저장 |

현재 Git commit 기준 주요 변경:

| commit | 내용 |
|---|---|
| `ec8c4eb` | presentation-ready depth Gaussian demo 구조 추가 |
| `7b36f01` | 실제/공식 샘플 출처 기반으로 Depth-GIC/GIV-D 데이터 교체, MP4 H.264 저장 |
| `1514195` | NYU 공식 montage crop의 depth dimension mismatch 수정 |

## 데이터 준비 원칙

디스크 여유 공간을 고려해 대형 전체 데이터셋을 자동으로 받지 않는다.

| 데이터셋 | 처리 방식 |
|---|---|
| DIODE | validation archive만 사용하고 10개 샘플만 보존 |
| NYU Depth V2 | 공식 web sample montage crop 또는 기존 로컬 폴더에서 10개만 추출, huge TFDS 자동 다운로드 금지 |
| ManiSkill | full RLDS/TFDS 다운로드 금지, 공식 RGB+Depth texture 예시를 lightweight replay sample로 사용 |
| RLBench | prebuilt 대형 dataset 다운로드 금지, 설치되어 있지 않으면 fallback 또는 missing metadata 처리 |

## 현재 포함된 발표용 샘플 데이터

현재 저장소에는 바로 Streamlit에서 실행 가능한 작은 샘플만 포함되어 있다.

| 경로 | 데이터 성격 | 설명 |
|---|---|---|
| `data/demo_depth/nyu_10/rgb/` | 실제 NYU 공식 예시 crop | NYU Depth V2 공식 웹 montage에서 crop한 RGB 실내 장면 10개 |
| `data/demo_depth/nyu_10/depth/` | approximate depth map | 공식 montage의 colorized depth visualization을 scalar depth로 변환한 `.npy` 10개 |
| `data/demo_depth/nyu_10/metadata.json` | 출처 기록 | `source: nyu_depth_v2_official_web_montage` |
| `data/demo_robot_rgbd/maniskill/rgb/` | ManiSkill 공식 예시 기반 sequence | 공식 ManiSkill RGB+Depth texture visualization에서 crop shift로 만든 18 frame RGB |
| `data/demo_robot_rgbd/maniskill/depth/` | ManiSkill 공식 예시 기반 depth sequence | 같은 crop shift로 만든 18 frame depth `.npy` |
| `data/demo_robot_rgbd/maniskill/metadata.json` | 출처 기록 | `source: maniskill_official_doc_rgbd_sample` |

중요한 제한:

- NYU 샘플의 depth는 meter 단위 raw depth가 아니라 공식 웹 이미지에 보이는 colorized depth를 scalar map으로 변환한 것이다.
- ManiSkill 샘플은 실제 공식 RGB-D visualization에서 가져온 한 장의 예시를 crop shift로 sequence화한 것이다.
- 따라서 현재 샘플은 발표용 시각화에는 적합하지만, 정량 depth metric이나 로봇 policy benchmark에는 사용하면 안 된다.

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

NYU 공식 web sample mode에서 처리하는 일:

1. `https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2_web.jpg`를 `data/raw_downloads/`에 다운로드한다.
2. montage의 RGB column과 depth visualization column을 crop한다.
3. depth crop 크기를 RGB crop 크기와 동일하게 resize한다.
4. colorized depth를 approximate scalar depth로 변환한다.
5. `data/demo_depth/nyu_10/rgb/`와 `data/demo_depth/nyu_10/depth/`에 10개 pair를 저장한다.

이 과정에서 발견된 실제 버그:

| 문제 | 원인 | 수정 |
|---|---|---|
| `operands could not be broadcast together with shapes (200,278) (200,279)` | NYU montage의 RGB crop과 depth crop 폭이 1px 달랐음 | `prepare_nyu_samples.py`에서 depth crop을 RGB crop 크기로 resize 후 `.npy` 저장 |

ManiSkill wrist RGB-D demo:

```bash
python scripts/data_prep/prepare_maniskill_wrist_demo.py
```

RLBench eye-in-hand RGB-D demo:

```bash
python scripts/data_prep/prepare_rlbench_eye_in_hand_demo.py
```

ManiSkill script는 기본적으로 공식 ManiSkill RGB+Depth texture visualization을 다운로드해 짧은 replay sequence로 변환한다. 설치 없이도 메인 발표 데모가 단순 도형이 아니라 robot/tabletop RGB-D 예시로 동작한다. 강제 procedural fallback은 `--fallback` 옵션을 사용할 때만 생성한다.

ManiSkill sample mode에서 처리하는 일:

1. `https://maniskill.readthedocs.io/en/latest/_images/rgbd_vis.png`를 다운로드한다.
2. 왼쪽 RGB panel과 오른쪽 depth panel을 분리한다.
3. 작은 crop shift를 적용해 18 frame sequence를 만든다.
4. RGB는 `.png`, depth는 `.npy`로 저장한다.
5. metadata에 공식 문서 이미지 기반 샘플임을 기록한다.

RLBench는 현재 저장소에 synthetic fallback을 포함하지 않는다. 실제 RLBench/CoppeliaSim 환경이 준비되어 있지 않으면 Streamlit에서 준비 안내를 보여주는 방식으로 처리한다.

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

Streamlit UI 구성:

| UI 요소 | 역할 |
|---|---|
| Dataset selectbox | DIODE 또는 NYU Depth V2 선택. 현재 기본은 준비된 NYU sample |
| Sample selectbox | `data/demo_depth/<dataset>/`의 sample pair 선택 |
| `max_size` slider | RGB/depth를 최대 해상도 기준으로 resize |
| `stride` slider | grid sampling 간격. 값이 작을수록 Gaussian 수 증가 |
| `edge_weight` checkbox | RGB/depth edge 부근에서 opacity/scale 보정 |
| `Generate Depth-GIC figure-eight video` | `.gicd` 저장, preview render, figure-eight MP4 생성 |
| download buttons | `.gicd`, figure-eight MP4, comparison MP4 다운로드 |

생성 후 표시되는 panel:

| panel | 설명 |
|---|---|
| Original RGB | 입력 RGB image |
| Depth Map | normalized/converted depth visualization |
| Gaussian Render | depth-aware Gaussian primitive를 원래 시점에서 렌더링 |
| Depth-colored Gaussians | primitive depth를 색으로 시각화 |
| MP4 comparison | RGB/depth/render를 나란히 보여주는 figure-eight video |

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

Streamlit UI 구성:

| UI 요소 | 역할 |
|---|---|
| Dataset selectbox | ManiSkill wrist RGB-D 또는 RLBench eye-in-hand RGB-D 선택 |
| `max_frames` slider | 사용할 frame 수 제한 |
| `max_size` slider | RGB/depth frame resize 기준 |
| `stride` slider | frame별 Gaussian sampling 간격 |
| `Generate third-person GIV-D replay` | `.givd` 저장, third-person-style replay MP4 생성 |
| frame strip | 원본 RGB sequence와 depth sequence를 빠르게 확인 |
| MP4 comparison | 원본 wrist RGB, depth, GIV-D replay를 나란히 표시 |

현재 권장 시연:

1. Dataset은 `ManiSkill wrist RGB-D`를 선택한다.
2. `Generate third-person GIV-D replay`를 누른다.
3. 표시되는 comparison MP4를 발표에서 재생한다.
4. RLBench는 실제 데이터가 없으면 선택하지 않는다.

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

현재 생성 확인된 asset 예:

| 파일 | 현재 소스 |
|---|---|
| `depth_gic_original_rgb.png` | NYU official web montage crop |
| `depth_gic_depth_map.png` | NYU colorized depth에서 변환한 approximate depth |
| `depth_gic_comparison_demo.mp4` | H.264 MP4, RGB/depth/Depth-GIC render side-by-side |
| `givd_robot_original_strip.png` | ManiSkill official RGB-D visualization 기반 sequence |
| `givd_comparison_demo.mp4` | H.264 MP4, wrist RGB/depth/GIV-D replay side-by-side |

## MP4 재생 호환성 수정

Streamlit에서 `No Video with supported format and MIME type found` 오류가 발생할 수 있었다. 원인은 MP4 파일이 확장자는 `.mp4`이지만 브라우저가 안정적으로 지원하는 codec/pixel format이 아닐 수 있었기 때문이다.

수정 내용:

| 파일 | 수정 내용 |
|---|---|
| `gic_codec/depth_trajectory.py` | `_save_video()`에서 `imageio-ffmpeg`를 사용해 `codec="libx264"`, `pixelformat="yuv420p"`로 저장 |
| `requirements.txt` | `imageio-ffmpeg` 추가 |
| `web_app/pages/4_Depth_GIC_Demo.py` | `st.video(path, format="video/mp4")`로 MIME 명시 |
| `web_app/pages/5_GIVD_Replay_Demo.py` | `st.video(path, format="video/mp4")`로 MIME 명시 |

검증 결과:

```text
depth_gic_comparison_demo.mp4
Video: h264 (High) (avc1), yuv420p, 18 fps

givd_comparison_demo.mp4
Video: h264 (High) (avc1), yuv420p, 12 fps
```

따라서 현재 생성되는 MP4는 일반 브라우저와 Streamlit `st.video`에서 재생 가능한 형식이다.

## 문제 해결 기록

| 날짜/단계 | 문제 | 처리 |
|---|---|---|
| 초기 depth demo | 샘플이 단순 도형이라 발표용으로 부적절 | NYU 공식 실내 RGB-D 예시 crop으로 교체 |
| 초기 robot demo | gripper/tabletop procedural scene이 너무 toy-like | ManiSkill 공식 RGB+Depth texture visualization 기반으로 교체 |
| GIV-D RLBench | 실제 RLBench가 없는데 synthetic fallback이 메인처럼 보임 | RLBench synthetic sample 제거, 실제 데이터 없으면 안내 |
| Streamlit video | `No Video with supported format and MIME type found` | H.264/yuv420p MP4 저장 및 MIME 명시 |
| NYU sample 001 | RGB/depth shape mismatch로 broadcast error | depth crop resize 및 `.npy` 재생성 |

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
