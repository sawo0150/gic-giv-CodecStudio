import os
import tempfile
import time
import shutil
from pathlib import Path
import streamlit as st
from PIL import Image
import numpy as np

# Add parent dir to sys.path to enable imports
import sys
PROJECT_DIR = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_DIR))

from gic_codec.decoder import GICDecoder
from gic_codec.giv_format import GIVFormat
from gic_codec.gic_format import GICFormat

st.set_page_config(layout="wide", page_title="GIC/GIV Codec Studio - Player")

st.sidebar.title("🎥 GIC/GIV Player")
st.sidebar.markdown(".gic 및 .giv 확장자 파일을 로드하여 2D Gaussian Splatting으로 실시간 디코딩합니다.")

st.title("🎥 GIC/GIV Codec Player")

uploaded_codec_file = st.file_uploader("압축 코덱 파일 선택 (.gic, .giv)", type=["gic", "giv"])

decoder = GICDecoder()

if uploaded_codec_file is not None:
    filename = uploaded_codec_file.name
    suffix = Path(filename).suffix.lower()
    
    # Write to a temporary file
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
        temp_file.write(uploaded_codec_file.read())
        temp_file_path = temp_file.name
        
    try:
        if suffix == ".gic":
            st.markdown(f"### 📷 Image Player - `{filename}`")
            
            # Quick preview from file container before full decode
            raw_data = GICFormat.load(temp_file_path)
            
            col_left, col_right = st.columns([1, 2])
            
            with col_left:
                st.markdown("#### 📄 Header Metadata")
                st.json(raw_data["header"])
                
                if raw_data["metrics"] is not None:
                    st.markdown("#### 📊 인코딩 품질 메트릭")
                    st.json(raw_data["metrics"])
                    
                if raw_data["preview"] is not None:
                    st.image(raw_data["preview"], caption="저용량 썸네일 미리보기 (preview.png)", width="stretch")
                    
            with col_right:
                st.markdown("#### 🖼️ 실시간 가우시안 래스터라이제이션 복원 (Full Decode)")
                if st.button("🚀 Decode & Render"):
                    with st.spinner("가우시안 파라미터 로딩 및 GPU 래스터라이제이션 렌더링 중..."):
                        dec_result = decoder.decode_image(temp_file_path)
                        st.image(dec_result["image"], caption=f"복원 이미지 (디코딩 소요 시간: {dec_result['decoding_time']:.4f}초)", width="stretch")
                        st.success(f"성공적으로 디코딩되었습니다. 해상도: {dec_result['image'].shape[1]}x{dec_result['image'].shape[0]}")
                        
        elif suffix == ".giv":
            st.markdown(f"### 🎥 Video Player - `{filename}`")
            
            # Load GIV metadata only
            raw_data = GIVFormat.load(temp_file_path, load_frames=False)
            header = raw_data["header"]
            index = raw_data["index"]
            metrics = raw_data["metrics"]
            
            col_vleft, col_vright = st.columns([1, 2])
            
            with col_vleft:
                st.markdown("#### 📄 Video Header")
                st.json(header)
                if metrics is not None:
                    st.markdown("#### 📊 비디오 평균 지표")
                    st.json(metrics)
                    
                # Show thumbnails grid
                st.markdown("#### 🖼️ 프레임 미리보기 갤러리")
                thumbnails = list(raw_data["previews"].values())
                if thumbnails:
                    st.image(thumbnails[:6], width=80, caption=[f"F_{i+1}" for i in range(len(thumbnails[:6]))])

            with col_vright:
                st.markdown("#### 🎞️ 비디오 재생 & 개별 프레임 디코딩")
                total_frames = header["video_info"]["total_frames"]
                
                # Active slider for frame seek
                frame_idx = st.slider("프레임 번호 선택 (Seek)", min_value=1, max_value=total_frames, value=1)
                
                # Load frame parameter on demand
                # GIVFormat.load of specific frame to prevent full loading in player for giant files
                with st.spinner(f"프레임 {frame_idx} 가우시안 파라미터 압축해제 및 렌더링 중..."):
                    # We reload the full file with frames, but only for the chosen frame index (optimised)
                    data_full = GIVFormat.load(temp_file_path, load_frames=True)
                    frame_gaussians = data_full["frames"].get(frame_idx)
                    h, w = header["video_info"]["height"], header["video_info"]["width"]
                    
                    if frame_gaussians is not None:
                        frame_img, dec_t = decoder.wrapper.render(frame_gaussians, h, w)
                        
                        st.image(frame_img, caption=f"Frame {frame_idx}/{total_frames} (렌더링 소요시간: {dec_t:.4f}초)", width="stretch")
                        
                        # Show current frame parameters
                        frame_info = index["frames"][frame_idx - 1]
                        col_fi1, col_fi2, col_fi3, col_fi4 = st.columns(4)
                        col_fi1.write(f"품질 모드: `{frame_info['quality_mode']}`")
                        col_fi2.write(f"가우시안 수: `{frame_info['num_points']:,}개`")
                        col_fi3.write(f"화질 (PSNR): `{frame_info['psnr']:.2f} dB`")
                        col_fi4.write(f"SSIM: `{frame_info.get('ssim', 0):.4f}`")
                        st.caption(
                            f"Init: `{frame_info.get('init_source', 'unknown')}` | "
                            f"Complexity: `{frame_info.get('complexity_score', 0):.3f}` | "
                            f"Encode: `{frame_info.get('encoding_time_sec', 0):.3f}s` | "
                            f"Decode: `{frame_info.get('decoding_time_sec', dec_t):.4f}s`"
                        )
                    else:
                        st.error("프레임 데이터를 찾을 수 없습니다.")

                # Realtime Play Loop Simulation
                play_col, fps_col, _ = st.columns([1, 1, 2])
                with play_col:
                    play_clicked = st.button("▶️ Play Video")
                with fps_col:
                    play_fps = st.slider("재생 FPS", min_value=1, max_value=24, value=8)

                if play_clicked:
                    st.info("비디오 순차 디코딩 및 시뮬레이션 재생 시작...")
                    canvas = st.empty()
                    status_txt = st.empty()
                    
                    # Store rendered frames to run smoothly
                    data_full = GIVFormat.load(temp_file_path, load_frames=True)
                    
                    for i in range(1, total_frames + 1):
                        fg = data_full["frames"].get(i)
                        if fg is not None:
                            img, _ = decoder.wrapper.render(fg, h, w)
                            canvas.image(img, caption=f"재생 중: Frame {i}/{total_frames}", width="stretch")
                            frame_info = index['frames'][i-1]
                            status_txt.text(
                                f"Frame {i} - Init: {frame_info.get('init_source', 'unknown')} | "
                                f"Gaussians: {frame_info['num_points']} | "
                                f"PSNR: {frame_info['psnr']:.2f}dB | "
                                f"SSIM: {frame_info.get('ssim', 0):.4f}"
                            )
                            time.sleep(1.0 / play_fps)
                    st.success("재생이 종료되었습니다.")

    except Exception as e:
        st.error(f"디코딩 에러 발생: {e}")
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
