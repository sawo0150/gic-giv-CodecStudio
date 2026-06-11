import os
import time
import pandas as pd
from pathlib import Path
from PIL import Image
import numpy as np

# Add parent dir to sys.path to enable imports
import sys
PROJECT_DIR = Path(__file__).parent.parent
sys.path.append(str(PROJECT_DIR))

from gic_codec.encoder import GICEncoder
from gic_codec.decoder import GICDecoder
from gic_codec.metrics import CodecMetrics

def run_benchmark(image_paths, output_dir, iterations=1000):
    os.makedirs(output_dir, exist_ok=True)
    encoder = GICEncoder()
    decoder = GICDecoder()
    
    results = []
    
    for idx, img_path in enumerate(image_paths):
        img_path = Path(img_path)
        img_name = img_path.stem
        print(f"\n[{idx+1}/{len(image_paths)}] Benchmarking: {img_name}")
        
        # 1. Original stats
        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img)
        h, w, c = img_np.shape
        orig_size = os.path.getsize(img_path)
        
        # We test: JPEG(Q=75), WebP(Q=75), GIC(Low), GIC(Med), GIC(High), GIC(Auto)
        # 2. JPEG
        temp_jpg = Path(output_dir) / f"{img_name}_temp.jpg"
        start_t = time.time()
        img.save(temp_jpg, format="JPEG", quality=75)
        enc_t = time.time() - start_t
        jpg_size = os.path.getsize(temp_jpg)
        
        start_t = time.time()
        jpg_recon = Image.open(temp_jpg)
        jpg_recon_np = np.array(jpg_recon)
        dec_t = time.time() - start_t
        
        jpg_psnr = CodecMetrics.calculate_psnr(img_np, jpg_recon_np)
        jpg_ssim = CodecMetrics.calculate_ssim(img_np, jpg_recon_np)
        os.remove(temp_jpg)
        
        results.append({
            "Image": img_name, "Codec": "JPEG", "Size_KB": jpg_size / 1024.0,
            "Ratio": CodecMetrics.calculate_compression_ratio(orig_size, jpg_size),
            "BPP": CodecMetrics.calculate_bpp(jpg_size, w, h), "PSNR": jpg_psnr, "SSIM": jpg_ssim,
            "Enc_Time": enc_t, "Dec_Time": dec_t
        })

        # 3. WebP
        temp_webp = Path(output_dir) / f"{img_name}_temp.webp"
        start_t = time.time()
        img.save(temp_webp, format="WebP", quality=75)
        enc_t = time.time() - start_t
        webp_size = os.path.getsize(temp_webp)
        
        start_t = time.time()
        webp_recon = Image.open(temp_webp)
        webp_recon_np = np.array(webp_recon)
        dec_t = time.time() - start_t
        
        webp_psnr = CodecMetrics.calculate_psnr(img_np, webp_recon_np)
        webp_ssim = CodecMetrics.calculate_ssim(img_np, webp_recon_np)
        os.remove(temp_webp)
        
        results.append({
            "Image": img_name, "Codec": "WebP", "Size_KB": webp_size / 1024.0,
            "Ratio": CodecMetrics.calculate_compression_ratio(orig_size, webp_size),
            "BPP": CodecMetrics.calculate_bpp(webp_size, w, h), "PSNR": webp_psnr, "SSIM": webp_ssim,
            "Enc_Time": enc_t, "Dec_Time": dec_t
        })

        # 4. GIC Modes
        modes = ["low", "medium", "high", "auto"]
        for mode in modes:
            gic_file = Path(output_dir) / f"{img_name}_{mode}.gic"
            try:
                metrics = encoder.encode_image(
                    image_path=str(img_path),
                    output_path=str(gic_file),
                    quality_mode=mode,
                    iterations=iterations
                )
                
                results.append({
                    "Image": img_name, "Codec": f"GIC ({mode.capitalize()})",
                    "Size_KB": metrics["file_size_bytes"] / 1024.0,
                    "Ratio": metrics["compression_ratio"],
                    "BPP": metrics["bpp"],
                    "PSNR": metrics["psnr"],
                    "SSIM": metrics["ssim"],
                    "Enc_Time": metrics["encoding_time_sec"],
                    "Dec_Time": metrics["decoding_time_sec"]
                })
            except Exception as e:
                print(f"Error evaluating GIC {mode} for {img_name}: {e}")

    # Export CSV
    df = pd.DataFrame(results)
    csv_path = Path(output_dir) / "image_results_table.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nBenchmark results saved to: {csv_path}")
    
    # Calculate and print averages
    print("\n--- Average Metrics ---")
    avg_df = df.groupby("Codec").mean(numeric_only=True)
    print(avg_df[["Size_KB", "PSNR", "SSIM", "Enc_Time", "Dec_Time"]])
    
if __name__ == "__main__":
    # Test on a small set of images from assets
    assets_dir = PROJECT_DIR / "Instant-GI" / "assets"
    images = list(assets_dir.glob("*.png"))
    
    if not images:
        # Create dummy assets for testing if empty
        os.makedirs(assets_dir, exist_ok=True)
        dummy_img = Image.new('RGB', (256, 256), color='red')
        dummy_img_path = assets_dir / "0829x2.png"
        dummy_img.save(dummy_img_path)
        images = [dummy_img_path]

    run_benchmark(images[:2], './outputs/metrics', iterations=10)
