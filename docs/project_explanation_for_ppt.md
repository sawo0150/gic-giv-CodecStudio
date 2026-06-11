# Instant-GI 기반 GIC/GIV Codec Studio 프로젝트 설명서

## 목차

1. 프로젝트 개요
2. 전체 프로젝트 구조
3. 시스템 전체 아키텍처
4. `.gic` 포맷 구현 상세
5. `.giv` 포맷 구현 상세
6. Encoder 구현 상세
7. Decoder / Player 구현 상세
8. Auto Quality Mode 구현 상세
9. Instant-GI 및 pretrained model 연동 방식
10. Streamlit 웹앱 구현 상세
11. PPT Asset Export Pipeline 구현 상세
12. 실험 및 평가 결과 정리
13. 구현 과정 설명
14. 구현 결과 요약
15. PPT 슬라이드별 참고 내용
16. PPT 제작 시 우선적으로 사용할 자료 목록

---

## 1. 프로젝트 개요

### 프로젝트 주제

> Instant-GI 기반 커스텀 Gaussian 이미지/동영상 인코더·디코더 시스템

이 프로젝트는 Instant-GI의 2D Gaussian Splatting 기반 이미지 표현 능력을 단순 시각화가 아니라 **파일 컨테이너 기반 코덱 시스템**으로 확장한 구현이다. 이미지는 `.gic` 파일로, 동영상은 `.giv` 파일로 저장되며, Streamlit 웹앱에서 인코딩과 디코딩/재생을 수행한다.

### PPT에 바로 넣을 수 있는 설명 문장

> 본 프로젝트는 이미지를 2D Gaussian parameter 집합으로 변환하고, 이를 커스텀 컨테이너 파일로 저장한 뒤 다시 렌더링하는 Gaussian codec-like system이다.

> 기존 이미지 파일을 직접 압축하는 대신, 이미지의 구조를 Gaussian primitive들의 위치, 크기, 회전, 투명도, 색상 파라미터로 표현하여 `.gic`와 `.giv` 포맷으로 저장한다.

---

## 2. 전체 프로젝트 구조

### 주요 디렉터리 구조

| 경로 | 역할 | 발표에서 설명할 포인트 |
|---|---|---|
| `Instant-GI/` | 원본 Instant-GI 코드와 렌더러/학습 모듈 | 기존 연구 코드를 백엔드로 사용하고, 새 코드는 codec wrapper 역할을 수행 |
| `gic_codec/` | 커스텀 코덱 핵심 Python 패키지 | Encoder, Decoder, format I/O, metrics, analyzer, Instant-GI wrapper 구현 |
| `web_app/` | Streamlit 기반 GUI | 비개발자도 `.gic/.giv` 인코딩과 재생을 시연 가능 |
| `web_app/pages/` | Streamlit multi-page 앱 | Encoder, Player, Export Report 페이지로 기능 분리 |
| `experiments/` | 벤치마크 및 PPT asset 생성 스크립트 | 발표용 그래프와 비교 이미지를 자동 생성 |
| `scripts/` | 모델 준비 및 실행 보조 스크립트 | pretrained model setup, Streamlit 실행 스크립트 |
| `configs/` | 품질 모드 설정 YAML | Low/Medium/High/Auto 품질 정책을 별도 파일로 문서화 |
| `pretrained/` | Instant-GI pretrained checkpoint 저장 위치 | `epoch_best_ks_3.pth` 모델 파일 관리 |
| `data/` | 샘플 이미지/비디오 프레임 데이터 | Kodak, DIV2K, DAVIS 샘플 입력 데이터 |
| `outputs/gic/` | 생성된 `.gic`와 복원 이미지 | 이미지 코덱 결과 확인 |
| `outputs/metrics/` | 실험 결과 `.gic/.giv`와 CSV | 실제 평가 수치 확인 |
| `outputs/ppt_assets/` | 발표용 이미지/그래프 | PPT에 직접 넣을 수 있는 자료 |
| `docs/` | 설명 문서 | PPT 제작용 구현 설명 문서 저장 |

### 주요 Python 파일 역할

| 파일 | 주요 클래스/함수 | 역할 |
|---|---|---|
| `gic_codec/encoder.py` | `GICEncoder.encode_image`, `GICEncoder.encode_video` | 이미지/동영상을 `.gic/.giv`로 인코딩 |
| `gic_codec/decoder.py` | `GICDecoder.decode_image`, `GICDecoder.decode_video` | `.gic/.giv`를 다시 이미지/프레임으로 복원 |
| `gic_codec/gic_format.py` | `GICFormat.save/load` | `.gic` zip container 저장/로드 |
| `gic_codec/giv_format.py` | `GIVFormat.save/load` | `.giv` zip container 저장/로드 |
| `gic_codec/analyzer.py` | `ImageComplexityAnalyzer.calculate_complexity` | Auto Quality Mode를 위한 복잡도 계산 |
| `gic_codec/metrics.py` | `CodecMetrics` | PSNR, SSIM, BPP, 압축률 계산 |
| `gic_codec/instant_gi_wrapper.py` | `InstantGIWrapper` | Instant-GI 모델 호출, Gaussian fitting/rendering 추상화 |
| `web_app/app.py` | Streamlit main page | 시스템 상태와 페이지 소개 |
| `web_app/pages/1_Encoder.py` | Streamlit Encoder page | 업로드/경로 입력 후 인코더 호출 |
| `web_app/pages/2_Player.py` | Streamlit Player page | `.gic/.giv` 업로드 후 metadata 표시 및 렌더링 |
| `web_app/pages/3_Export_Report.py` | Streamlit Export page | PPT asset 생성 및 zip 다운로드 |
| `experiments/export_ppt_assets.py` | `export_charts` | 발표용 차트/그리드/오차맵 생성 |
| `scripts/setup_models.py` | `setup_models` | pretrained checkpoint 확인/다운로드/복사 |
| `scripts/run_streamlit.sh` | shell script | 로컬 Streamlit 실행 |

### CLI와 Streamlit 앱의 관계

CLI와 Streamlit은 같은 backend 클래스를 공유한다. CLI는 `python -m gic_codec.encoder`와 `python -m gic_codec.decoder`로 직접 실행하고, Streamlit은 내부에서 `GICEncoder`, `GICDecoder`, `GICFormat`, `GIVFormat`을 직접 import하여 호출한다. 현재 웹앱은 subprocess로 CLI를 호출하지 않는다.

---

## 3. 시스템 전체 아키텍처

### 전체 데이터 흐름

```text
Input Image / Video
→ Complexity Analyzer
→ Quality Mode Selector
→ Instant-GI Encoder
→ Gaussian Parameters
→ .gic / .giv Container
→ Decoder / Player
→ Reconstructed Image / Video
→ Metrics / PPT Assets
```

### 구성 요소별 역할

| 구성 요소 | 입력 | 출력 | 역할 |
|---|---|---|---|
| Complexity Analyzer | RGB image 또는 frame | score, edge density, color variance, Laplacian variance, 추천 mode | 이미지 복잡도 기반 품질 모드 결정 |
| Quality Mode Selector | 사용자 mode 또는 Auto 결과 | Low/Medium/High, target Gaussian count | 인코딩 품질 정책 결정 |
| InstantGIWrapper | image tensor, init points | Gaussian parameter dict | Instant-GI fitting 및 rendering 추상화 |
| GICFormat | header, Gaussian dict, preview, metrics | `.gic` zip file | 이미지용 컨테이너 저장 |
| GIVFormat | video header, index, frame Gaussian dicts | `.giv` zip file | 동영상용 컨테이너 저장 |
| GICDecoder | `.gic` | reconstructed image | 저장된 Gaussian parameter 렌더링 |
| Streamlit Player | `.gic/.giv` upload | metadata, preview, rendered output | 평가자용 GUI 재생기 |
| Export Pipeline | 샘플 이미지와 mock/default chart data | `outputs/ppt_assets/*.png` | 발표 자료 생성 |

### `.gic`와 `.giv`가 필요한 이유

`.gic`와 `.giv`는 단순 이미지 파일이 아니라 Gaussian parameter와 metadata를 함께 보관하는 **코덱 컨테이너**이다. 이미지 복원에 필요한 `xyz`, `scaling`, `rotation`, `opacity`, `features_dc`와 인코딩 설정, 품질 지표, preview를 함께 저장하므로, 파일 자체가 실험 재현 단위가 된다.

### PPT용 아키텍처 설명 문장

> 시스템은 입력 이미지를 먼저 복잡도 분석기로 평가하고, 그 결과에 따라 Gaussian 수와 품질 모드를 결정한다. 이후 Instant-GI backend가 이미지를 Gaussian parameter로 fitting하고, 결과 parameter는 `.gic` 또는 `.giv` zip container에 metadata와 함께 저장된다.

---

## 4. `.gic` 포맷 구현 상세

### 저장 방식

`.gic`는 `zipfile.ZipFile(..., ZIP_DEFLATED)`로 저장되는 zip 기반 컨테이너이다. 구현 파일은 `gic_codec/gic_format.py`이며, 저장은 `GICFormat.save`, 로드는 `GICFormat.load`가 담당한다.

### 내부 파일 구조

| 내부 파일 | 역할 | 포함 정보 |
|---|---|---|
| `header.json` | 코덱 metadata | codec name, version, image size, original format, encoding settings, Gaussian count |
| `gaussians.npz` | Gaussian parameter 배열 | `xyz`, `scaling`, `rotation`, `opacity`, `features_dc` |
| `preview.png` | 빠른 미리보기 | full decode 없이 보여줄 128x128 PNG |
| `metrics.json` | 품질/압축 지표 | file size, original size, compression ratio, bpp, PSNR, SSIM, encode/decode time |
| `logs.json` | fitting log | iteration별 loss/PSNR 기록 |

### 실제 `header.json` 예시

아래 예시는 `outputs/gic/kodim06_low_torch_test.gic`에서 확인한 실제 header이다.

```json
{
  "codec_name": "GIC",
  "version": "1.0.0",
  "created_at": "2026-06-09T05:06:00Z",
  "image_info": {
    "width": 512,
    "height": 336,
    "channels": 3,
    "original_format": "PNG"
  },
  "encoding_settings": {
    "quality_mode": "low",
    "decided_mode": "Low",
    "init_method": "net",
    "backend": "instant_gi",
    "iterations": 1,
    "lr": 0.001
  },
  "gaussian_info": {
    "num_points": 17723
  }
}
```

### `gaussians.npz` key

`outputs/gic/kodim06_low_torch_test.gic` 기준으로 다음 key와 shape가 확인되었다.

| Key | Shape 예시 | 의미 |
|---|---:|---|
| `xyz` | `(17723, 2)` | 2D Gaussian 중심 좌표 |
| `scaling` | `(17723, 2)` | Gaussian x/y scale |
| `rotation` | `(17723, 1)` | Gaussian 회전 파라미터 |
| `opacity` | `(17723, 1)` | Gaussian 불투명도 |
| `features_dc` | `(17723, 3)` | RGB 색상 feature |

NumPy fallback backend로 생성된 파일에는 추가로 `fallback_shape`가 들어갈 수 있다. 이는 Torch/Instant-GI가 없는 환경에서 저해상도 grid 기반 복원을 수행하기 위한 보조 metadata이다.

### `.gic` 생성 과정

```text
[Image Encoder Flow]
1. 입력 이미지 로드
2. wrapper.image_to_tensor()로 tensor 변환 및 필요 시 downsample
3. ImageComplexityAnalyzer로 complexity score 계산
4. quality_mode가 auto이면 recommended_mode와 target_gaussians 사용
5. InstantGIWrapper.initialize_gaussians()로 초기 Gaussian 생성
6. InstantGIWrapper.fit()으로 Gaussian fitting
7. InstantGIWrapper.render()로 복원 이미지 생성
8. PSNR, SSIM, BPP, 압축률 계산
9. preview.png 생성
10. GICFormat.save()로 header/gaussians/preview/log 저장
11. 실제 파일 크기 측정 후 metrics를 채워 다시 저장
```

### `.gic` 디코딩 과정

```text
1. GICFormat.load()로 zip container 읽기
2. header.json에서 width/height 확인
3. gaussians.npz에서 Gaussian parameter dict 로드
4. InstantGIWrapper.render() 호출
5. PIL Image로 output_path에 PNG 저장
```

### 파일 크기 계산 방식

`encoder.py`는 먼저 metrics 없이 `.gic`를 저장하여 실제 container 크기를 `os.path.getsize(output_path)`로 측정한다. 이후 `CodecMetrics.calculate_compression_ratio(original_size, compressed_size)`와 `CodecMetrics.calculate_bpp(compressed_size, w, h)`를 계산하고, metrics를 포함하여 다시 저장한다.

### 장점과 한계

| 구분 | 내용 |
|---|---|
| 장점 | zip 기반이라 구현이 단순하고, JSON/NPZ/PNG를 분리해 디버깅과 발표 설명이 쉽다. |
| 장점 | header와 metrics가 파일 안에 있어 결과 재현성과 설명력이 높다. |
| 한계 | 진짜 binary codec bitstream은 아니며 zip+npz container 방식이다. |
| 한계 | 현재 Gaussian parameter quantization이나 entropy coding은 구현되어 있지 않다. |
| 한계 | Instant-GI fitting iteration이 적으면 PSNR이 낮게 나올 수 있다. |

---

## 5. `.giv` 포맷 구현 상세

### 저장 방식

`.giv`는 `.gic`와 동일하게 zip 기반 container이다. 구현 파일은 `gic_codec/giv_format.py`이며, `GIVFormat.save`와 `GIVFormat.load`가 담당한다.

### 내부 파일 구조

| 내부 파일/폴더 | 역할 | 포함 정보 |
|---|---|---|
| `header.json` | 비디오 metadata | width, height, fps, total frames, duration, average Gaussian count |
| `index.json` | 프레임 index | 각 frame의 npz path, preview path, quality mode, Gaussian count, PSNR, SSIM |
| `frames/frame_000001.npz` | 프레임별 Gaussian parameter | `xyz`, `scaling`, `rotation`, `opacity`, `features_dc` |
| `previews/frame_000001.png` | 프레임별 썸네일 | 128x128 preview PNG |
| `metrics.json` | 전체 평균 지표 | file size, avg bpp, avg PSNR, avg SSIM, total encoding time |
| `logs.json` | 로그 | 현재 `encode_video`에서는 `logs=None`으로 저장하므로 기본 생성되지 않음 |

### 실제 `header.json` 예시

`outputs/giv_bear_low_demo.giv` 기준:

```json
{
  "codec_name": "GIV",
  "version": "1.0.0",
  "created_at": "2026-06-09T05:02:18Z",
  "video_info": {
    "width": 768,
    "height": 512,
    "fps": 24.0,
    "total_frames": 2,
    "duration_sec": 0.08333333333333333
  },
  "encoding_settings": {
    "quality_mode": "low",
    "init_method": "net",
    "backend": "numpy_fallback",
    "iterations": 5,
    "lr": 0.001
  },
  "average_gaussian_info": {
    "avg_points_per_frame": 9963
  }
}
```

기존 `outputs/metrics/motocross_auto.giv`는 최근 추가된 `encoding_settings` 필드가 없는 상태로 생성되어 있다. 문서와 발표에서는 “현재 코드에서는 저장됨, 기존 생성 파일 일부에는 없음”으로 구분해야 한다.

### 실제 `index.json` 예시

`outputs/giv_bear_low_demo.giv` 기준:

```json
{
  "frames": [
    {
      "frame_idx": 1,
      "filename": "frames/frame_000001.npz",
      "preview": "previews/frame_000001.png",
      "quality_mode": "Low",
      "num_points": 9963,
      "psnr": 22.650893139552988,
      "ssim": 0.5015831904347413
    },
    {
      "frame_idx": 2,
      "filename": "frames/frame_000002.npz",
      "preview": "previews/frame_000002.png",
      "quality_mode": "Low",
      "num_points": 9963,
      "psnr": 25.124399716388297,
      "ssim": 0.5922779939219095
    }
  ]
}
```

### Frame-wise encoding 구조

현재 `.giv`는 motion estimation이나 inter-frame residual compression을 사용하지 않는다. 각 프레임을 독립 이미지처럼 Gaussian fitting하고, 결과를 `frames/frame_xxxxxx.npz`로 저장한다.

PPT 설명:

> 첫 구현에서는 동영상 압축의 복잡도를 낮추기 위해 frame-wise independent encoding을 선택했다. 이는 구현 안정성이 높고 seek가 단순하지만, 프레임 간 중복을 제거하지 못한다는 한계가 있다.

### 향후 확장 가능성

| 향후 기능 | 설명 |
|---|---|
| Keyframe/Delta 구조 | I-frame은 전체 Gaussian, P-frame은 이전 프레임 대비 변화량 저장 |
| Motion compensation | Gaussian center의 이동 벡터 또는 optical flow 기반 보정 |
| Delta color/opacity | 색상과 opacity 변화만 저장하여 bitrate 감소 |
| Frame cache | Player에서 매 프레임 전체 `.npz` 로드를 줄이기 위한 cache |

---

## 6. Encoder 구현 상세

### 이미지 Encoder

구현 위치: `gic_codec/encoder.py`, `GICEncoder.encode_image`.

```text
[Image Encoder Flow]
1. 입력 이미지 로드
2. complexity score 계산
3. quality mode 결정
4. Instant-GI encoding 실행
5. Gaussian parameter 추출
6. .gic container 생성
7. preview 및 metrics 저장
```

### 동영상 Encoder

구현 위치: `gic_codec/encoder.py`, `GICEncoder.encode_video`.

```text
[Video Encoder Flow]
1. 입력 비디오 로드 또는 frame directory 탐색
2. MP4이면 OpenCV VideoCapture로 frame extraction
3. directory이면 png/jpg/jpeg 파일 정렬
4. 각 프레임에 대해 complexity score 계산
5. 각 프레임 quality mode 결정
6. frame-wise Gaussian encoding
7. frame별 preview 및 metrics 저장
8. .giv container 생성
```

### 입력 처리

| 입력 타입 | 처리 방식 |
|---|---|
| 단일 이미지 | PIL로 RGB 변환 |
| 프레임 폴더 | `*.png`, `*.jpg`, `*.jpeg` 정렬 후 순차 처리 |
| MP4/AVI | OpenCV `cv2.VideoCapture`로 frame 추출 후 BGR→RGB 변환 |

### 품질 모드 처리

| 입력 mode | 처리 |
|---|---|
| `low` | `decided_mode = Low`, target Gaussian 10000 |
| `medium` | `decided_mode = Medium`, target Gaussian 25000 |
| `high` | `decided_mode = High`, target Gaussian 50000 |
| `auto` | Analyzer의 `recommended_mode`, `target_gaussians` 사용 |

주의: Instant-GI `net` 초기화는 target Gaussian 수를 직접 강제하기보다 InitNet 출력에 의해 실제 point 수가 결정될 수 있다. 예를 들어 `kodim06_low_torch_test.gic`는 Low mode이지만 실제 `num_points`가 17723으로 저장되었다.

### Gaussian parameter 저장 방식

`InstantGIWrapper.fit()`은 `GaussianImage_RS.state_dict()`에서 `_xyz`, `_scaling`, `_rotation`, `_opacity`, `_features_dc`를 추출하고, 저장용 key인 `xyz`, `scaling`, `rotation`, `opacity`, `features_dc`로 mapping한다.

### CLI 예시

```bash
python -m gic_codec.encoder --input data/kodak/kodim06.png --mode auto --output outputs/gic/kodim06.gic
python -m gic_codec.encoder --input data/davis/bear --video --mode auto --max_frames 3 --output outputs/bear.giv
```

### Streamlit Encoder page 호출 방식

`web_app/pages/1_Encoder.py`는 `GICEncoder()`를 직접 생성하고, 버튼 클릭 시 `encode_image()` 또는 `encode_video()`를 호출한다. 임시 입력 파일과 임시 출력 디렉터리를 만들고, 결과 `.gic/.giv` bytes를 `st.download_button`으로 제공한다.

---

## 7. Decoder / Player 구현 상세

### `.gic` decode 흐름

구현 위치: `gic_codec/decoder.py`, `GICDecoder.decode_image`.

```text
1. GICFormat.load()로 header, gaussians, preview, metrics 로드
2. header.image_info에서 height/width 확인
3. InstantGIWrapper.render()에 Gaussian dict 전달
4. 복원 RGB numpy array 생성
5. output_path가 있으면 PNG 저장
```

### `.giv` decode 흐름

구현 위치: `gic_codec/decoder.py`, `GICDecoder.decode_video`.

```text
1. GIVFormat.load(load_frames=True)로 전체 frame Gaussian 로드
2. index.json의 frame 순서대로 반복
3. 각 frame의 Gaussian dict를 render()
4. output_dir이 있으면 frame_000001.png 형식으로 저장
```

### Preview와 실제 decode의 차이

| 구분 | Preview | Full Decode |
|---|---|---|
| 입력 | `preview.png` 또는 `previews/*.png` | `gaussians.npz` 또는 `frames/*.npz` |
| 속도 | 매우 빠름 | Gaussian rendering 필요 |
| 품질 | 128x128 썸네일 | 원래 header resolution |
| 목적 | 빠른 파일 확인 | 실제 복원 결과 확인 |

### Streamlit Player 구현

`web_app/pages/2_Player.py`는 `.gic` 또는 `.giv` 업로드를 받는다.

| 모드 | 구현 |
|---|---|
| `.gic` | header/metrics/preview 표시 후 `Decode & Render` 버튼으로 복원 |
| `.giv` | header/metrics/thumbnail grid 표시, frame slider로 특정 frame 렌더링 |
| Play 기능 | `Play Video` 버튼 클릭 시 전체 frame을 순차 render하고 `time.sleep(1/fps)`로 재생 시뮬레이션 |

### Encoder와 Player 차이

| 모듈 | 입력 | 출력 | 역할 |
|---|---|---|---|
| Encoder | PNG/JPG/MP4/frame folder | `.gic` 또는 `.giv` | Gaussian parameter 생성 및 컨테이너 저장 |
| Decoder | `.gic` 또는 `.giv` | 복원 PNG/frame sequence | Gaussian parameter를 이미지로 렌더링 |
| Player | 업로드된 `.gic/.giv` | metadata, preview, rendered image/frame | 평가자용 시각적 재생 인터페이스 |

### CLI 예시

```bash
python -m gic_codec.decoder --input outputs/gic/kodim06.gic --output outputs/gic/kodim06_recon.png
python -m gic_codec.decoder --input outputs/bear.giv --output outputs/bear_frames
```

---

## 8. Auto Quality Mode 구현 상세

### 구현 위치

`gic_codec/analyzer.py`, `ImageComplexityAnalyzer.calculate_complexity`.

### Complexity metric

| Metric | 계산 방식 | 의미 |
|---|---|---|
| Edge Density | grayscale 변환 후 `cv2.Canny(gray, 50, 150)`, edge pixel 비율 계산 | 경계선과 구조가 많은 이미지인지 판단 |
| Color Variance | RGB 각 채널 분산의 평균 | 색상 변화가 큰 이미지인지 판단 |
| Laplacian Variance | `cv2.Laplacian(gray, cv2.CV_64F).var()` | 질감과 고주파 디테일 정도 판단 |

### 최종 score 계산식

```text
C = 50.0 * edge_density
  + 0.0001 * color_variance
  + 0.0005 * laplacian_variance
```

### Mode threshold

| Complexity Score 범위 | 선택 모드 | Target Gaussian | 설명 |
|---:|---|---:|---|
| `C < 1.5` | Low | 10000 | 단순 이미지 |
| `1.5 <= C < 3.5` | Medium | 25000 | 일반 이미지 |
| `C >= 3.5` | High | 50000 | 복잡한 질감/경계 이미지 |

### Auto mode 적용 방식

이미지 인코딩에서는 전체 이미지 1장에 대해 score를 계산하고, `header.encoding_settings.quality_mode`와 `decided_mode`에 기록한다.

동영상 인코딩에서는 각 프레임마다 score를 계산하고, `index.json`의 각 frame entry에 `quality_mode`, `num_points`, `psnr`, `ssim`을 저장한다. 현재 `index.json`에는 raw complexity score 자체는 저장하지 않는다.

### PPT용 짧은 설명 문장

> Auto Quality Mode는 Canny edge density, color variance, Laplacian variance를 결합해 이미지 복잡도를 계산하고, 복잡도가 높을수록 더 많은 Gaussian primitive를 배정하는 적응형 품질 제어 방식이다.

---

## 9. Instant-GI 및 pretrained model 연동 방식

### 원본 repo 위치

Instant-GI 원본 코드는 프로젝트 루트의 `Instant-GI/`에 위치한다.

### import 방식

`gic_codec/instant_gi_wrapper.py`는 다음 방식으로 Instant-GI를 import한다.

```python
INSTANT_GI_DIR = PROJECT_DIR / "Instant-GI"
sys.path.append(str(INSTANT_GI_DIR))
from generalizable_model.init_net import InitNet
from quard_image import QuardImage
from gaussianimage_rs import GaussianImage_RS
```

현재 구현은 subprocess 호출이 아니라 Python import 기반 wrapper 방식이다.

### pretrained model 경로

| 경로 | 역할 |
|---|---|
| `pretrained/epoch_best_ks_3.pth` | `InstantGIWrapper._load_init_net()` 기본 checkpoint |
| `Instant-GI/checkpoints/epoch_best_ks_3.pth` | 원본 Instant-GI 위치와 호환되는 checkpoint copy |

현재 실제 파일은 두 위치 모두에 존재한다.

### setup script

`scripts/setup_models.py`는 다음 순서로 동작한다.

1. `pretrained/`, `Instant-GI/checkpoints/` 생성
2. 기존 checkpoint 후보 경로 검색
3. 발견되면 두 target path에 복사
4. 없으면 `gdown.download_folder()`로 Google Drive folder 다운로드 시도

### backend 선택

`InstantGIWrapper`는 Instant-GI import 성공 시 `backend = "instant_gi"`를 사용한다. 실패하면 `backend = "numpy_fallback"`으로 전환한다. 최근 검증에서는 `3dgs` conda 환경에서 `torch 2.1.2+cu118`, CUDA, Instant-GI import가 성공했고 `backend = instant_gi`, `device = cuda:0`가 확인되었다.

### Streamlit 표시

`web_app/app.py`는 checkpoint 존재 여부와 `HAS_INSTANT_GI` 값을 검사한다.

| 상태 | 화면 표시 |
|---|---|
| checkpoint 있고 Instant-GI import 가능 | Instant-GI backend 준비 완료 |
| checkpoint는 있으나 Torch/Instant-GI import 불가 | NumPy fallback backend 안내 |
| checkpoint 없음 | setup script 실행 안내 |

### 발표 문장

> 새로 만든 codec layer는 Instant-GI를 외부 프로그램처럼 실행하지 않고, Python wrapper를 통해 InitNet과 GaussianImage_RS를 직접 호출한다. 따라서 Gaussian fitting 결과를 바로 `.npz` parameter로 추출해 커스텀 컨테이너에 저장할 수 있다.

---

## 10. Streamlit 웹앱 구현 상세

### 앱 이름과 목적

앱 이름은 `GIC/GIV Codec Studio`이다. 목적은 비교 대시보드가 아니라 **실제 인코딩과 디코딩/재생을 시연하는 codec studio**이다.

### Page 구조

| Page | 핵심 기능 | 입력 | 출력 |
|---|---|---|---|
| `app.py` | 시스템 상태 검사, 페이지 소개 | 없음 | backend/model 상태 안내 |
| `1_Encoder.py` | 이미지/비디오 인코딩 | 이미지 업로드, MP4 업로드, frame folder path | `.gic` 또는 `.giv` 다운로드 |
| `2_Player.py` | `.gic/.giv` 재생 | `.gic` 또는 `.giv` 업로드 | metadata, preview, rendered image/frame |
| `3_Export_Report.py` | PPT asset 생성 | 버튼 클릭 | chart preview, `ppt_assets.zip` 다운로드 |

### Backend 호출 방식

웹앱은 CLI subprocess가 아니라 Python 객체를 직접 호출한다.

| Page | 호출 backend |
|---|---|
| Encoder | `GICEncoder.encode_image`, `GICEncoder.encode_video` |
| Player | `GICFormat.load`, `GIVFormat.load`, `GICDecoder.decode_image`, `decoder.wrapper.render` |
| Export Report | `experiments.export_ppt_assets.export_charts` |

### File upload/download 처리

업로드 파일은 `tempfile.NamedTemporaryFile`로 임시 저장된다. 생성된 `.gic/.giv`는 임시 output path에서 bytes로 읽고 `st.download_button`으로 제공된다.

### Progress 처리

현재 구현은 실제 progress bar가 아니라 `st.spinner()` 중심이다. Iteration별 progress bar는 TODO로 남아 있다.

### Session state/cache 사용 여부

현재 코드 기준으로 `st.session_state`나 `st.cache_data`/`st.cache_resource`는 사용하지 않는다. Player에서 `.giv` frame slider를 움직일 때는 `GIVFormat.load(..., load_frames=True)`를 다시 호출하는 구조다.

### 비교 기능을 과하게 넣지 않은 이유

웹앱은 코덱 도구로서 인코딩/재생에 집중한다. JPEG/WebP 비교와 발표용 차트는 `experiments/export_ppt_assets.py` 및 CSV 결과로 분리했다.

---

## 11. PPT Asset Export Pipeline 구현 상세

### 구현 위치

`experiments/export_ppt_assets.py`, 핵심 함수 `export_charts(output_dir)`.

### 목적

PPT에 바로 넣을 수 있는 고해상도 chart와 비교 이미지를 `outputs/ppt_assets/`에 생성한다.

### 생성 파일 목록

| Export 파일 | 내용 | PPT에서 사용할 위치 |
|---|---|---|
| `rd_curve_chart.png` | JPEG/WebP/GIC mock/default RD scatter plot | Image Results, R-D Curve 슬라이드 |
| `rd_curve.png` | `rd_curve_chart.png`와 같은 내용의 호환 파일명 | Export Report page preview |
| `encoding_time_bar.png` | codec별 encoding time log-scale bar chart | 인코딩 복잡도 설명 |
| `decoding_time_bar.png` | codec별 decoding time chart | 실시간 렌더링/재생 설명 |
| `comp_ratio_ssim.png` | 압축률과 SSIM 이중축 chart | 압축률 vs 품질 trade-off |
| `frame_psnr_timeline.png` | frame-wise PSNR timeline | Video Results 슬라이드 |
| `video_frame_psnr_timeline.png` | timeline 호환 파일명 | Export Report page preview |
| `codec_grid_comparison.png` | Original/JPEG/WebP/GIC Low/Medium/High/Auto grid | 정성 비교 슬라이드 |
| `restored_error_map.png` | 복원 이미지와 error heatmap | 오차 분석 슬라이드 |
| `video_frames_strip.png` | DAVIS frame sequence strip | 동영상 입력/재생 시연 |
| `metrics_summary.json` | 요약 JSON | 발표 자료 수치 메모 |

### 구현 방식

| 항목 | 실제 구현 |
|---|---|
| 이미지 비교 grid | Kodak `kodim06.png` 또는 Instant-GI teaser를 resize하고 JPEG/WebP/GIC-like resize variant를 2x4 grid로 배치 |
| Difference map | PIL `ImageChops.difference`, grayscale, `ImageOps.colorize`로 heatmap 생성 |
| Video strip | `data/davis/bear` 또는 `data/davis/motocross` frame을 가로 strip으로 배치 |
| RD curve | 현재 코드 내부 default/mock 배열 사용 |
| Frame-wise plot | `np.random.seed(7)`로 생성한 synthetic PSNR timeline 사용 |
| Zoomed crop | 현재 별도 zoomed crop 파일은 구현되지 않음 |
| Gaussian center overlay | 현재 별도 overlay 파일은 구현되지 않음 |
| Metrics summary table | `metrics_summary.json` 생성, CSV table 이미지화는 현재 미구현 |

주의: export pipeline의 chart 수치 일부는 `outputs/metrics/*.csv`를 읽는 것이 아니라 코드 내부 default/mock data를 사용한다. 발표에서는 “PPT asset generator의 예시 차트”와 “실제 benchmark CSV 결과”를 구분해야 한다.

---

## 12. 실험 및 평가 결과 정리

### 데이터셋/샘플

| 데이터 | 확인된 경로 | 현재 사용 |
|---|---|---|
| Kodak | `data/kodak/kodim06.png`, `kodim17.png`, `kodim20.png`, `kodim22.png` | CLI torch test 및 PPT asset source |
| DAVIS bear | `data/davis/bear/frame_*.png` | `.giv` demo, video strip |
| DAVIS motocross | `data/davis/motocross/frame_*.png` | `outputs/metrics/motocross_*.giv` |
| Instant-GI assets | `Instant-GI/assets/*.png` | image benchmark CSV |

### 실제 이미지 결과: `outputs/metrics/image_results_table.csv`

| Image | Method | File Size KB | Compression Ratio | BPP | PSNR | SSIM | Encode Time s | Decode Time s |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| pipeline | JPEG | 1940.02 | 2.03 | 0.3860 | 39.76 | 0.9672 | 0.0827 | 0.2934 |
| pipeline | WebP | 614.60 | 6.42 | 0.1223 | 39.98 | 0.9688 | 1.7913 | 0.6353 |
| pipeline | GIC Low | 403.61 | 9.78 | 26.9072 | 13.99 | 0.4666 | 3.1945 | 0.0007 |
| pipeline | GIC Medium | 403.58 | 9.78 | 26.9050 | 13.99 | 0.4666 | 0.1543 | 0.0008 |
| pipeline | GIC High | 403.62 | 9.78 | 26.9083 | 13.99 | 0.4666 | 0.1539 | 0.0007 |
| pipeline | GIC Auto | 403.58 | 9.78 | 26.9055 | 13.99 | 0.4666 | 0.1552 | 0.0008 |
| teaser | JPEG | 1714.53 | 2.93 | 0.7974 | 32.24 | 0.9161 | 0.0371 | 0.0637 |
| teaser | WebP | 1130.80 | 4.44 | 0.5259 | 32.70 | 0.9175 | 0.9507 | 0.2131 |
| teaser | GIC Low | 778.66 | 6.45 | 25.9552 | 11.98 | 0.5845 | 0.3073 | 0.0010 |
| teaser | GIC Medium | 778.67 | 6.45 | 25.9555 | 11.98 | 0.5845 | 0.3016 | 0.0010 |
| teaser | GIC High | 778.63 | 6.45 | 25.9544 | 11.98 | 0.5845 | 0.3050 | 0.0009 |
| teaser | GIC Auto | 778.66 | 6.45 | 25.9553 | 11.98 | 0.5845 | 0.3037 | 0.0009 |

해석 주의: 위 GIC 결과는 iteration 10 기반의 빠른 실험 결과다. PSNR은 JPEG/WebP보다 낮지만 decode time은 매우 짧게 측정되었다. 실제 품질 비교를 위해서는 iteration 수를 늘린 재실험이 필요하다.

### 실제 Torch backend 단일 `.gic` 결과

`outputs/gic/kodim06_low_torch_test.gic`:

| Method | Backend | File Size | Compression Ratio | BPP | PSNR | SSIM | Encode Time | Decode Time | Gaussian Count |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GIC Low | instant_gi | 539,757 bytes | 1.15x | 25.1003 | 12.01 | 0.5801 | 3.3625s | 0.0009s | 17,723 |

### 실제 동영상 결과: `outputs/metrics/video_results_table.csv`

| Video | Method | Size MB | Compression Ratio | Avg BPP | Avg PSNR | Avg SSIM | Total Encode Time |
|---|---|---:|---:|---:|---:|---:|---:|
| motocross | GIV Low | 1.5761 | 1.1738 | 11.2076 | 14.1017 | 0.6659 | 0.6160 |
| motocross | GIV Medium | 1.5760 | 1.1738 | 11.2073 | 14.1017 | 0.6659 | 0.6442 |
| motocross | GIV High | 1.5761 | 1.1738 | 11.2076 | 14.1017 | 0.6659 | 0.6164 |
| motocross | GIV Auto | 1.5761 | 1.1738 | 11.2075 | 14.1017 | 0.6659 | 0.6107 |

### 실제 `.giv` frame index 예시

`outputs/metrics/motocross_auto.giv`:

| Frame | Mode | Gaussian Count | PSNR | SSIM |
|---:|---|---:|---:|---:|
| 1 | High | 18,775 | 15.7161 | 0.6884 |
| 2 | High | 17,723 | 13.2945 | 0.6546 |
| 3 | High | 17,723 | 13.2945 | 0.6546 |

### TODO로 남은 평가 항목

| 항목 | 상태 |
|---|---|
| 충분한 iteration 기반 GIC 품질 실험 | TODO |
| Kodak 24장 전체 평균 결과 | TODO |
| DAVIS 2개 clip 이상 장기 frame-wise 결과 | TODO |
| complexity score를 frame별 CSV에 저장 | TODO |
| JPEG/WebP/GIC의 공정한 bitrate 기준 R-D curve | TODO |
| `.giv` frame별 실제 frame size와 encode time | TODO |

---

## 13. 구현 과정 설명

| 단계 | 구현한 내용 | 어려웠던 점 | 해결 방식 | 발표에서 강조할 점 |
|---|---|---|---|---|
| 1. Instant-GI 실행 환경 구성 | `Instant-GI/`를 프로젝트 내부에 두고 wrapper에서 import | Python/Torch/CUDA 환경 의존성 | `3dgs` conda 환경에서 Torch 2.1.2+cu118 확인 | 연구 코드와 새 codec layer를 결합 |
| 2. pretrained model 준비 | `scripts/setup_models.py`로 checkpoint 확인/복사/다운로드 | 대용량 1.2GB 모델 파일 관리 | `pretrained/`와 `Instant-GI/checkpoints/` 양쪽 경로 지원 | 재현 가능한 모델 setup |
| 3. Gaussian parameter 추출 구조 분석 | `GaussianImage_RS.state_dict()`에서 핵심 tensor 추출 | 내부 key가 `_xyz` 형태 | 저장용 key로 mapping | Gaussian primitive를 파일 포맷화 |
| 4. `.gic` 포맷 설계 | zip 안에 JSON/NPZ/PNG 저장 | binary format 설계 부담 | container-based codec으로 구현 | 구현 가능성과 설명력 확보 |
| 5. 이미지 encoder 구현 | `GICEncoder.encode_image` | 품질 모드와 metrics 측정 순서 | 임시 저장 후 파일 크기 측정, metrics 포함 재저장 | 실제 파일 생성까지 완료 |
| 6. 이미지 decoder/player 구현 | `GICDecoder.decode_image`, Streamlit Player | preview와 full decode 분리 | preview 먼저 표시, 버튼으로 render | 평가자에게 즉시 시연 가능 |
| 7. Auto Quality Mode 구현 | edge/color/laplacian score | 복잡도 기준 설계 | threshold 기반 Low/Medium/High mapping | 입력 적응형 Gaussian 수 제어 |
| 8. `.giv` 포맷 설계 | header/index/frames/previews 구조 | 동영상 inter-frame compression 복잡도 | frame-wise independent encoding | 확장 가능한 video container |
| 9. 동영상 encoder 구현 | folder/MP4 입력 지원 | MP4 frame 추출과 frame별 metadata | OpenCV VideoCapture와 index.json | frame seek 가능한 구조 |
| 10. `.giv` player 구현 | slider, thumbnail, play simulation | 큰 파일 로딩 비효율 | 현재는 전체 load, 향후 cache 개선 | Player 중심 도구 구현 |
| 11. Streamlit 앱 통합 | 3-page app | Torch 환경 차이 | backend 상태 표시와 fallback 안내 | 로컬 데모 가능 |
| 12. PPT asset export | chart/grid/error map/strip 생성 | 실제 CSV와 mock chart 분리 필요 | 생성 파일명 명확화 | 발표 자료 자동화 |
| 13. 실험 및 평가 수행 | image/video CSV와 sample container 생성 | iteration이 낮아 품질 제한 | 현재 수치와 TODO 구분 | 솔직한 한계와 개선 방향 제시 |

---

## 14. 구현 결과 요약

### 최종 구현 기능

| 기능 | 상태 |
|---|---|
| `.gic` zip container 저장/로드 | 완료 |
| `.giv` frame-wise zip container 저장/로드 | 완료 |
| 이미지 CLI encoder/decoder | 완료 |
| 비디오/frame folder CLI encoder/decoder | 완료 |
| Instant-GI backend wrapper | 완료 |
| NumPy fallback backend | 제한적 완료 |
| Auto Quality Mode | 완료 |
| Streamlit Encoder page | 완료 |
| Streamlit Player page | 완료 |
| Streamlit Export Report page | 완료 |
| PPT asset generation | 완료 |
| 실제 장시간 고품질 benchmark | TODO |

### 장점

- Instant-GI를 코덱 컨테이너 시스템으로 확장했다.
- `.gic/.giv` 내부 구조가 명확해 발표 설명과 디버깅이 쉽다.
- Streamlit GUI로 인코딩/재생 시연이 가능하다.
- Auto Quality Mode로 입력 복잡도 기반 적응형 인코딩 흐름을 구현했다.
- PPT asset export pipeline으로 발표 자료 제작까지 연결했다.

### 한계

- 현재 `.gic/.giv`는 zip+npz container이며, 실제 상용 codec 수준의 bitstream은 아니다.
- Gaussian parameter quantization, entropy coding은 구현되어 있지 않다.
- `.giv`는 frame-wise 방식이라 프레임 간 중복을 제거하지 못한다.
- Player는 `.giv` frame을 매번 전체 로드하는 구조라 큰 파일에는 비효율적이다.
- 일부 chart는 실제 CSV 기반이 아니라 mock/default data 기반이다.
- 낮은 iteration 실험에서는 PSNR이 낮게 나온다.

### 향후 확장 방향

| 확장 방향 | 설명 |
|---|---|
| `.gic` binary format 변환 | JSON/NPZ zip 대신 compact binary layout 설계 |
| `.giv` keyframe/delta compression | I-frame/P-frame 구조로 frame 간 중복 제거 |
| Motion compensation | Gaussian center motion vector 저장 |
| 정교한 Auto Quality Mode | saliency, texture, semantic region 기반 adaptive allocation |
| GPU 최적화 | batch frame encoding, cached renderer, mixed precision |
| 웹 배포 backend 분리 | Streamlit UI와 Torch inference worker 분리 |

---

## 15. PPT 슬라이드별 참고 내용

| Slide | 제목 | 핵심 메시지 | 넣을 그림/표 | 발표 대본 느낌 문장 |
|---:|---|---|---|---|
| 1 | Title | Instant-GI 기반 Gaussian codec studio | 프로젝트 제목 | “저희 프로젝트는 이미지를 Gaussian parameter로 변환해 저장하고 재생하는 커스텀 코덱 시스템입니다.” |
| 2 | Motivation | 이미지 표현 모델을 파일 포맷으로 확장 | 기존 codec vs Gaussian 표현 다이어그램 | “단순 모델 실행이 아니라, 저장 가능한 코덱 파일로 만드는 것이 목표였습니다.” |
| 3 | Background: Instant-GI | 2D Gaussian Splatting 기반 이미지 표현 | Instant-GI assets/pipeline 이미지 | “Instant-GI는 이미지를 다수의 2D Gaussian primitive로 표현하고 빠르게 렌더링합니다.” |
| 4 | Project Goal | `.gic`, `.giv`, Streamlit Studio | 산출물 목록 표 | “핵심 산출물은 파일 포맷, Python 패키지, 웹 GUI, PPT export pipeline입니다.” |
| 5 | System Architecture | Analyzer→Encoder→Container→Player | 아키텍처 flow | “입력은 복잡도 분석을 거쳐 품질 모드가 결정되고, Gaussian parameter로 저장됩니다.” |
| 6 | `.gic` Format | 이미지용 Gaussian container | `.gic` 내부 파일 표 | “`.gic`는 header, Gaussian npz, preview, metrics를 포함하는 zip container입니다.” |
| 7 | `.giv` Format | frame-wise video container | `.giv` 구조 표 | “초기 버전은 각 프레임을 독립 Gaussian image로 저장합니다.” |
| 8 | Encoder Implementation | 이미지/비디오 encoding 흐름 | Encoder flow block | “Encoder는 fitting 결과를 파일 크기와 품질 지표와 함께 저장합니다.” |
| 9 | Player Implementation | metadata+preview+full decode | Player screenshot 또는 표 | “Player는 preview로 빠르게 확인하고, 버튼 또는 slider로 실제 Gaussian rendering을 수행합니다.” |
| 10 | Auto Quality Mode | 복잡도 기반 품질 제어 | metric/threshold 표 | “edge, color variance, Laplacian variance를 결합해 자동으로 Gaussian 수를 결정합니다.” |
| 11 | Streamlit Demo | 로컬 codec studio | 웹앱 화면 캡처 | “웹앱은 비교 대시보드가 아니라 실제 인코딩과 재생을 수행하는 도구입니다.” |
| 12 | Experiment Setup | 데이터셋과 지표 | dataset/metric 표 | “평가는 file size, compression ratio, PSNR, SSIM, encode/decode time으로 정리했습니다.” |
| 13 | Image Results | JPEG/WebP/GIC 비교 | image_results_table, `codec_grid_comparison.png` | “현재 낮은 iteration에서는 품질이 낮지만 컨테이너와 decode pipeline은 정상 작동합니다.” |
| 14 | Video Results | GIV frame-wise 결과 | `video_frames_strip.png`, frame table | “동영상은 frame-wise 방식으로 구현되어 seek와 구현 안정성이 높습니다.” |
| 15 | PPT Asset Export | 발표 자료 자동 생성 | `outputs/ppt_assets` 목록 | “실험 결과와 발표 이미지를 자동으로 추출하는 pipeline까지 포함했습니다.” |
| 16 | Limitations | 현재 한계 | 한계 표 | “현재는 zip+npz container이고 motion compensation은 아직 구현 전입니다.” |
| 17 | Future Work | keyframe/delta, binary, GPU 최적화 | roadmap | “다음 단계는 Gaussian parameter의 delta compression과 binary bitstream 설계입니다.” |
| 18 | Conclusion | Gaussian codec 가능성 검증 | 최종 요약 | “Instant-GI를 기반으로 인코딩, 저장, 재생, 발표 자료 생성까지 연결한 end-to-end prototype을 구현했습니다.” |

---

## 16. PPT 제작 시 우선적으로 사용할 자료 목록

### 우선 사용 이미지/그래프

| 우선순위 | 파일 | 사용 슬라이드 |
|---:|---|---|
| 1 | `outputs/ppt_assets/codec_grid_comparison.png` | Image Results |
| 2 | `outputs/ppt_assets/restored_error_map.png` | Error Map / Limitations |
| 3 | `outputs/ppt_assets/video_frames_strip.png` | Video Demo |
| 4 | `outputs/ppt_assets/rd_curve_chart.png` | R-D Curve |
| 5 | `outputs/ppt_assets/frame_psnr_timeline.png` | Video Timeline |
| 6 | `outputs/ppt_assets/encoding_time_bar.png` | Encoding Complexity |
| 7 | `outputs/ppt_assets/decoding_time_bar.png` | Decoding Speed |
| 8 | `outputs/ppt_assets/comp_ratio_ssim.png` | Compression Ratio vs SSIM |

### 우선 사용 표

| 표 | 위치 |
|---|---|
| 프로젝트 구조 표 | 이 문서 2장 |
| `.gic` 내부 파일 표 | 이 문서 4장 |
| `.giv` 내부 파일 표 | 이 문서 5장 |
| Auto Quality metric/threshold 표 | 이 문서 8장 |
| 실제 image results 표 | 이 문서 12장 |
| 실제 video results 표 | 이 문서 12장 |
| 슬라이드별 구성표 | 이 문서 15장 |

### 발표에서 반드시 구분할 것

| 구분 | 설명 |
|---|---|
| 실제 구현 | `.gic/.giv` container, encoder/decoder, Streamlit, analyzer, export pipeline |
| 실제 확인 수치 | `outputs/metrics/*.csv`, 생성된 `.gic/.giv`의 `metrics.json` |
| 설계 의도/향후 개선 | keyframe/delta compression, binary bitstream, motion compensation |
| 예시 chart | `export_ppt_assets.py` 내부 mock/default data 기반 chart |

### TODO 항목 요약

- 고 iteration 기반 GIC 품질 재실험
- Kodak 24장 전체 benchmark
- DAVIS clip 장기 frame-wise benchmark
- frame별 complexity score 저장
- 실제 CSV 기반 RD curve 자동 생성
- Gaussian center overlay, zoomed crop asset 생성
- `.giv` player frame cache 및 큰 파일 최적화

