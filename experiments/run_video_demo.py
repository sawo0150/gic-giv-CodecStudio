import os
import time
import pandas as pd
from pathlib import Path

# Add parent dir to sys.path to enable imports
import sys
PROJECT_DIR = Path(__file__).parent.parent
sys.path.append(str(PROJECT_DIR))

from gic_codec.encoder import GICEncoder

def run_video_demo(video_dirs, output_dir, iterations=1000, max_frames=5, video_init_mode="independent"):
    os.makedirs(output_dir, exist_ok=True)
    encoder = GICEncoder()
    
    results = []
    
    for idx, video_dir in enumerate(video_dirs):
        video_dir = Path(video_dir)
        video_name = video_dir.name
        print(f"\n[{idx+1}/{len(video_dirs)}] Benchmarking Video: {video_name}")
        
        # Test GIV on Low, Med, High, Auto modes
        modes = ["low", "medium", "high", "auto"]
        for mode in modes:
            giv_file = Path(output_dir) / f"{video_name}_{mode}.giv"
            try:
                metrics = encoder.encode_video(
                    input_path=str(video_dir),
                    output_path=str(giv_file),
                    quality_mode=mode,
                    iterations=iterations,
                    max_frames=max_frames,
                    video_init_mode=video_init_mode,
                )
                
                results.append({
                    "Video": video_name,
                    "Codec": f"GIV ({mode.capitalize()})",
                    "Size_MB": metrics["file_size_bytes"] / (1024.0 * 1024.0),
                    "Ratio": metrics["compression_ratio"],
                    "Avg_BPP": metrics["avg_bpp"],
                    "Avg_PSNR": metrics["avg_psnr"],
                    "Avg_SSIM": metrics["avg_ssim"],
                    "Total_Enc_Time": metrics["total_encoding_time_sec"],
                    "Avg_Enc_Time": metrics.get("avg_encoding_time_sec"),
                    "Avg_Dec_Time": metrics.get("avg_decoding_time_sec"),
                    "Video_Init_Mode": video_init_mode,
                })
            except Exception as e:
                print(f"Error evaluating GIV {mode} for video {video_name}: {e}")

    # Export CSV
    df = pd.DataFrame(results)
    csv_path = Path(output_dir) / "video_results_table.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nVideo benchmark results saved to: {csv_path}")
    
    # Calculate and print averages
    print("\n--- Video Average Metrics ---")
    print(df)
    
if __name__ == "__main__":
    # Look for DAVIS dataset directories
    davis_dir = PROJECT_DIR / "data" / "davis"
    video_folders = []
    
    if davis_dir.exists():
        video_folders = [f for f in davis_dir.iterdir() if f.is_dir()]
        
    # If no folders found, look at workspace files or create mock frame structure
    if not video_folders:
        print("DAVIS dataset directory not found. Creating a mock frame folder structure for verification...")
        mock_video_dir = PROJECT_DIR / "data" / "davis" / "mock_clip"
        os.makedirs(mock_video_dir, exist_ok=True)
        from PIL import Image
        for i in range(1, 4):
            dummy_img = Image.new('RGB', (256, 256), color=(i * 50, 100, 150))
            dummy_img.save(mock_video_dir / f"frame_{i:06d}.png")
        video_folders = [mock_video_dir]

    run_video_demo(video_folders[:2], './outputs/metrics', iterations=10, max_frames=3)
