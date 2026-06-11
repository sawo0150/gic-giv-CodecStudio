import os
import io
import shutil
import zipfile
from pathlib import Path
import streamlit as st

# Add parent dir to sys.path to enable imports
import sys
PROJECT_DIR = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_DIR))

from experiments.export_ppt_assets import export_charts

st.set_page_config(layout="wide", page_title="GIC/GIV Codec Studio - Export Report")

st.sidebar.title("📊 Export Report")
st.sidebar.markdown("실험 성능 지표와 PPT용 시각화 차트를 일괄 생성하여 다운로드합니다.")

st.title("📊 PPT Assets & Report Generator")
st.write("알고리즘 비교 실험 결과를 바탕으로 학술용 플롯과 차트를 한 번에 생성해 발표 자료(PPT) 슬라이드에 즉시 활용할 수 있도록 돕습니다.")

st.markdown("---")

output_assets_dir = PROJECT_DIR / "outputs" / "ppt_assets"
zip_output_path = PROJECT_DIR / "outputs" / "ppt_assets.zip"

if st.button("🚀 Generate PPT Charts"):
    with st.spinner("Matplotlib 시각화 파이프라인 가동 및 차트 생성 중..."):
        try:
            # Run the exporter
            export_charts(str(output_assets_dir))
            
            # Pack folder to zip
            with zipfile.ZipFile(zip_output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, dirs, files in os.walk(output_assets_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        # Archive relative to output_assets_dir
                        zipf.write(file_path, os.path.relpath(file_path, output_assets_dir))
                        
            st.success("🎉 **PPT 차트 및 리포트 자산 패키징 성공!** 아래에서 미리보기하고 ZIP 파일로 일괄 다운로드할 수 있습니다.")
            
        except Exception as e:
            st.error(f"차트 생성 중 오류 발생: {e}")

# If charts already generated or just successfully finished, show preview
if output_assets_dir.exists():
    st.markdown("### 🖼️ 생성된 차트 미리보기")
    
    col_c1, col_c2 = st.columns(2)
    
    with col_c1:
        rd_curve_path = output_assets_dir / "rd_curve.png"
        if rd_curve_path.exists():
            st.image(str(rd_curve_path), caption="Rate-Distortion Performance Curve (용량 대 화질)", width="stretch")
            
        enc_bar_path = output_assets_dir / "encoding_time_bar.png"
        if enc_bar_path.exists():
            st.image(str(enc_bar_path), caption="Encoding Time / Complexity 비교 (로그 스케일)", width="stretch")
            
    with col_c2:
        comp_ssim_path = output_assets_dir / "comp_ratio_ssim.png"
        if comp_ssim_path.exists():
            st.image(str(comp_ssim_path), caption="압축률 대 구조유사도 (SSIM) 이중축 비교", width="stretch")
            
        dec_bar_path = output_assets_dir / "decoding_time_bar.png"
        if dec_bar_path.exists():
            st.image(str(dec_bar_path), caption="Decoding Speed / FPS 성능 (가우시안 래스터라이제이션)", width="stretch")

    # Frame timeline chart if exists
    timeline_path = output_assets_dir / "video_frame_psnr_timeline.png"
    if timeline_path.exists():
        st.markdown("#### 🎥 동영상 프레임 화질 추이")
        st.image(str(timeline_path), caption="프레임 복잡도 변동에 따른 PSNR 복원 안정성", width="stretch")

    # Download button for packaged zip
    if zip_output_path.exists():
        with open(zip_output_path, "rb") as f:
            zip_bytes = f.read()
            
        st.markdown("### 📥 ZIP 다운로드")
        st.download_button(
            label="💾 ppt_assets.zip 일괄 다운로드",
            data=zip_bytes,
            file_name="ppt_assets.zip",
            mime="application/zip"
        )
else:
    st.info("💡 'Generate PPT Charts' 버튼을 누르면 비교 플롯 시각화가 이 탭에 로드됩니다.")
