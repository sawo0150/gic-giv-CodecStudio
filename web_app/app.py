import os
import streamlit as st
from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).parent.parent
sys.path.append(str(PROJECT_DIR))

from gic_codec.instant_gi_wrapper import HAS_INSTANT_GI, INSTANT_GI_IMPORT_ERROR

# Streamlit config
st.set_page_config(
    page_title="GIC/GIV Codec Studio",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar Welcome
st.sidebar.title("⚡ GIC/GIV Codec Studio")
st.sidebar.markdown("""
2D Gaussian Splatting (Instant-GI) 기술을 응용한 이미지/동영상 코덱 시스템입니다.
""")

st.title("⚡ GIC/GIV Codec Studio")
st.subheader("Implicit Neural Representation 기반 커스텀 이미지/동영상 코덱 시스템")

# System Pretrained Model Check
PRETRAINED_MODEL_PATH = PROJECT_DIR / "pretrained" / "epoch_best_ks_3.pth"
INSTANT_GI_MODEL_PATH = PROJECT_DIR / "Instant-GI" / "checkpoints" / "epoch_best_ks_3.pth"

st.markdown("---")

st.markdown("### 🔍 시스템 환경 검사")
if PRETRAINED_MODEL_PATH.exists() and INSTANT_GI_MODEL_PATH.exists() and HAS_INSTANT_GI:
    st.success("✅ **Instant-GI backend 준비 완료.** 체크포인트와 Torch 기반 래스터라이저를 사용할 수 있습니다.")
    st.info(f"모델 경로: `{PRETRAINED_MODEL_PATH}`")
elif PRETRAINED_MODEL_PATH.exists() and INSTANT_GI_MODEL_PATH.exists():
    st.warning("⚠️ **체크포인트는 있지만 Torch/Instant-GI 런타임이 없어 NumPy fallback backend로 실행됩니다.**")
    st.info(f"fallback 모드는 동일한 `.gic/.giv` 컨테이너 규격으로 빠른 데모 인코딩/재생을 수행합니다. 원인: `{type(INSTANT_GI_IMPORT_ERROR).__name__}: {INSTANT_GI_IMPORT_ERROR}`")
else:
    st.warning("⚠️ **Pretrained 모델 체크포인트가 누락되었습니다!**")
    st.error("Instant-GI Net Init을 사용하려면 모델 파일이 필요합니다. 모델 없이도 fallback backend 데모는 실행할 수 있습니다.")
    st.markdown("""
    **해결 방법:**
    터미널을 열고 아래 셋업 스크립트를 실행해 모델 체크포인트를 다운로드해 주세요.
    ```bash
    python scripts/setup_models.py
    ```
    """)

st.markdown("### 📂 코덱 스튜디오 핵심 기능")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### 📷 1_Encoder")
    st.write("이미지 또는 비디오 프레임을 로드하여 `.gic` (이미지), `.giv` (동영상) 커스텀 가우시안 코덱 형식으로 인코딩(압축)합니다.")
    st.write("복잡도 분석기(Complexity Analyzer) 기반의 Auto Mode를 지원합니다.")

with col2:
    st.markdown("#### 🎥 2_Player")
    st.write("제작된 `.gic` / `.giv` 컨테이너 파일을 로드하여 헤더 정보를 파싱하고, 가우시안 래스터라이제이션 기법으로 실시간 복원 렌더링 및 비디오 재생을 수행합니다.")

with col3:
    st.markdown("#### 📊 3_Export_Report")
    st.write("타 코덱(JPEG, WebP)과의 비교 벤치마크 실험 결과를 기반으로 발표 자료(PPT)용 오차 맵, R-D Curve, 프레임별 PSNR 차트 등의 시각화 자료를 자동 패키징 다운로드합니다.")
