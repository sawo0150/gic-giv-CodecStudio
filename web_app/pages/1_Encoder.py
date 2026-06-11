import os
import tempfile
import shutil
from pathlib import Path
import streamlit as st
from PIL import Image
import numpy as np

# Add parent dir to sys.path to enable imports
import sys
PROJECT_DIR = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_DIR))

from gic_codec.analyzer import ImageComplexityAnalyzer
from gic_codec.encoder import GICEncoder
from gic_codec.instant_gi_wrapper import HAS_INSTANT_GI, INSTANT_GI_IMPORT_ERROR

st.set_page_config(layout="wide", page_title="GIC/GIV Codec Studio - Encoder")

st.sidebar.title("⚡ GIC/GIV Encoder")
st.sidebar.markdown("이미지/비디오 데이터를 가우시안 코덱 형식으로 인코딩합니다.")

st.title("📷 GIC/GIV Encoder Studio")

# Check pretrained weights
PRETRAINED_MODEL_PATH = PROJECT_DIR / "pretrained" / "epoch_best_ks_3.pth"
if not HAS_INSTANT_GI:
    st.warning(f"Instant-GI 런타임을 불러오지 못해 NumPy fallback backend로 실행합니다: {type(INSTANT_GI_IMPORT_ERROR).__name__}: {INSTANT_GI_IMPORT_ERROR}")
elif not PRETRAINED_MODEL_PATH.exists():
    st.warning("Pretrained 모델 체크포인트가 없습니다. `net` 초기화 대신 fallback/random 계열 실행을 권장합니다.")

# Select input type
input_type = st.radio("입력 소스 선택", ["단일 이미지 업로드 (GIC)", "동영상(MP4) 업로드 (GIV)", "비디오 프레임 폴더 경로 입력 (GIV)"])

# Common settings
st.markdown("### ⚙️ 코덱 및 학습 설정")
col_s1, col_s2, col_s3 = st.columns(3)
with col_s1:
    quality_mode = st.selectbox("품질 제어 모드 (Quality Mode)", ["Auto", "Low", "Medium", "High"])
with col_s2:
    init_method = st.selectbox("초기 가우시안 할당 기법", ["net", "quard", "random"])
with col_s3:
    iterations = st.number_input("가우시안 피팅 에폭 (Iterations)", min_value=10, max_value=10000, value=1000, step=100)

encoder = GICEncoder()

if input_type == "단일 이미지 업로드 (GIC)":
    uploaded_file = st.file_uploader("이미지 파일 선택 (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        # Load and display input image
        original_img = Image.open(uploaded_file).convert("RGB")
        original_np = np.array(original_img)
        
        col_img1, col_img2 = st.columns(2)
        with col_img1:
            st.image(original_img, caption="입력 원본 이미지", use_column_width=True)
            
        # Analyze Complexity
        analyzer = ImageComplexityAnalyzer()
        analysis = analyzer.calculate_complexity(original_np)
        
        with col_img2:
            st.markdown("#### 🔍 이미지 복잡도 분석 결과 (Complexity Report)")
            st.metric("종합 복잡도 스코어 (Complexity Score)", f"{analysis['score']:.3f}")
            st.write(f"- Edge Density: `{analysis['edge_density']:.4f}`")
            st.write(f"- Color Variance: `{analysis['color_variance']:.1f}`")
            st.write(f"- Laplacian Variance: `{analysis['laplacian_variance']:.1f}`")
            st.info(f"💡 **추천 품질 모드:** `{analysis['recommended_mode']}` (가우시안 개수: `{analysis['target_gaussians']:,}개` 할당)")
            
            # Show edge map
            st.image(analysis["edge_map"], caption="에지 디텍션 맵 (Edge Map)", use_column_width=True)

        if st.button("🚀 Encode to .gic"):
            with st.spinner("Instant-GI를 사용해 가우시안 2D Representation 최적화 피팅 진행 중..."):
                # Save input to temp file
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_in:
                    original_img.save(temp_in.name)
                    temp_in_path = temp_in.name
                
                # Output path
                temp_out_dir = tempfile.mkdtemp()
                temp_out_path = os.path.join(temp_out_dir, "output.gic")
                
                # Run encoder
                try:
                    metrics = encoder.encode_image(
                        image_path=temp_in_path,
                        output_path=temp_out_path,
                        quality_mode=quality_mode,
                        init_method=init_method,
                        iterations=iterations
                    )
                    
                    st.success("🎉 **인코딩 완료!** 커스텀 가우시안 압축 파일(.gic)이 성공적으로 빌드되었습니다.")
                    
                    # Display performance metrics
                    st.markdown("#### 📊 압축 성능 지표")
                    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                    col_m1.metric("PSNR 화질", f"{metrics['psnr']:.2f} dB")
                    col_m2.metric("SSIM 구조유사도", f"{metrics['ssim']:.4f}")
                    col_m3.metric("압축률 (Original/Compressed)", f"{metrics['compression_ratio']:.2f}x")
                    col_m4.metric("인코딩 소요 시간", f"{metrics['encoding_time_sec']:.2f}초")
                    
                    # Read gic bytes for download
                    with open(temp_out_path, "rb") as f:
                        gic_bytes = f.read()
                        
                    st.download_button(
                        label="💾 .gic 압축 파일 다운로드",
                        data=gic_bytes,
                        file_name=f"{Path(uploaded_file.name).stem}.gic",
                        mime="application/octet-stream"
                    )
                except Exception as e:
                    st.error(f"인코딩 에러 발생: {e}")
                finally:
                    # Cleanup temp
                    if os.path.exists(temp_in_path):
                        os.remove(temp_in_path)
                    shutil.rmtree(temp_out_dir, ignore_errors=True)

elif input_type == "동영상(MP4) 업로드 (GIV)":
    uploaded_video = st.file_uploader("동영상 파일 선택 (MP4)", type=["mp4"])
    max_frames = st.number_input("인코딩할 최대 프레임 수 제한 (시연 단축용)", min_value=1, max_value=300, value=30)
    
    if uploaded_video is not None:
        st.video(uploaded_video)
        
        if st.button("🚀 Encode to .giv"):
            with st.spinner("비디오 프레임을 추출하고 각 프레임별 가우시안 인코딩 피팅 수행 중..."):
                # Save video to temp
                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_vid:
                    temp_vid.write(uploaded_video.read())
                    temp_vid_path = temp_vid.name
                
                # Output path
                temp_out_dir = tempfile.mkdtemp()
                temp_out_path = os.path.join(temp_out_dir, "output.giv")
                
                try:
                    # Run video encoder
                    metrics = encoder.encode_video(
                        input_path=temp_vid_path,
                        output_path=temp_out_path,
                        quality_mode=quality_mode,
                        init_method=init_method,
                        iterations=iterations,
                        max_frames=int(max_frames)
                    )
                    
                    st.success("🎉 **비디오 인코딩 완료!** 커스텀 가우시안 동영상 아카이브(.giv)가 생성되었습니다.")
                    st.markdown("#### 📊 비디오 평균 압축 성능 지표")
                    col_vm1, col_vm2, col_vm3 = st.columns(3)
                    col_vm1.metric("평균 PSNR", f"{metrics['avg_psnr']:.2f} dB")
                    col_vm2.metric("평균 SSIM", f"{metrics['avg_ssim']:.4f}")
                    col_vm3.metric("평균 BPP (Bits Per Pixel)", f"{metrics['avg_bpp']:.4f}")
                    
                    with open(temp_out_path, "rb") as f:
                        giv_bytes = f.read()
                        
                    st.download_button(
                        label="💾 .giv 비디오 파일 다운로드",
                        data=giv_bytes,
                        file_name=f"{Path(uploaded_video.name).stem}.giv",
                        mime="application/octet-stream"
                    )
                except Exception as e:
                    st.error(f"비디오 인코딩 에러 발생: {e}")
                finally:
                    if os.path.exists(temp_vid_path):
                        os.remove(temp_vid_path)
                    shutil.rmtree(temp_out_dir, ignore_errors=True)

else:
    dir_path = st.text_input("서버 로컬 프레임 디렉터리 경로 입력 (예: data/davis/bear/)", "data/davis/bear/")
    max_frames = st.number_input("인코딩할 최대 프레임 수 제한 (시연 단축용)", min_value=1, max_value=300, value=30, key="dir_max_frames")
    
    if st.button("🚀 Encode Folder to .giv"):
        if not os.path.exists(dir_path):
            st.error(f"디렉터리 경로가 존재하지 않습니다: {dir_path}")
        else:
            with st.spinner("디렉터리 내 프레임들 순차 인코딩 수행 중..."):
                temp_out_dir = tempfile.mkdtemp()
                temp_out_path = os.path.join(temp_out_dir, "output.giv")
                
                try:
                    metrics = encoder.encode_video(
                        input_path=dir_path,
                        output_path=temp_out_path,
                        quality_mode=quality_mode,
                        init_method=init_method,
                        iterations=iterations,
                        max_frames=int(max_frames)
                    )
                    
                    st.success("🎉 **폴더 인코딩 완료!** .giv 아카이브 생성 성공.")
                    st.markdown("#### 📊 비디오 평균 압축 성능 지표")
                    col_vm1, col_vm2, col_vm3 = st.columns(3)
                    col_vm1.metric("평균 PSNR", f"{metrics['avg_psnr']:.2f} dB")
                    col_vm2.metric("평균 SSIM", f"{metrics['avg_ssim']:.4f}")
                    col_vm3.metric("평균 BPP", f"{metrics['avg_bpp']:.4f}")
                    
                    with open(temp_out_path, "rb") as f:
                        giv_bytes = f.read()
                        
                    st.download_button(
                        label="💾 .giv 비디오 파일 다운로드",
                        data=giv_bytes,
                        file_name=f"{Path(dir_path).name}.giv",
                        mime="application/octet-stream"
                    )
                except Exception as e:
                    st.error(f"비디오 인코딩 에러 발생: {e}")
                finally:
                    shutil.rmtree(temp_out_dir, ignore_errors=True)
