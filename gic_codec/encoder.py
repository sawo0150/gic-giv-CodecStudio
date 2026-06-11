import os
import io
import time
import glob
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

from gic_codec.analyzer import ImageComplexityAnalyzer
from gic_codec.instant_gi_wrapper import InstantGIWrapper
from gic_codec.gic_format import GICFormat
from gic_codec.giv_format import GIVFormat
from gic_codec.metrics import CodecMetrics

class GICEncoder:
    def __init__(self, device=None):
        self.analyzer = ImageComplexityAnalyzer()
        self.wrapper = InstantGIWrapper(device=device)

    def encode_image(self, image_path, output_path, quality_mode="auto", init_method="net", iterations=2000, lr=0.001):
        """
        Encode an image into a .gic compressed file.
        """
        print(f"Encoding image: {image_path} -> {output_path} (Mode: {quality_mode})")
        start_total = time.time()

        # 1. Load and Setup tensor (will auto-downsample inside wrapper if too large)
        original_img = Image.open(image_path).convert("RGB")
        original_np = np.array(original_img)
        image_tensor = self.wrapper.image_to_tensor(original_np)
        if len(image_tensor.shape) == 4:
            h, w = image_tensor.shape[2], image_tensor.shape[3]
            original_np = (image_tensor[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        else:
            h, w = image_tensor.shape[:2]
            original_np = image_tensor.astype(np.uint8)
        
        original_size = os.path.getsize(image_path)

        # 2. Analyze complexity (on potentially downsampled image)
        analysis = self.analyzer.calculate_complexity(original_np)
        
        # Determine parameters based on quality_mode
        if quality_mode.lower() == "auto":
            decided_mode = analysis["recommended_mode"]
            num_points = analysis["target_gaussians"]
        else:
            decided_mode = quality_mode.capitalize()
            if decided_mode == "Low":
                num_points = 10000
            elif decided_mode == "Medium":
                num_points = 25000
            elif decided_mode == "High":
                num_points = 50000
            else:
                raise ValueError(f"Unknown quality mode: {quality_mode}")

        # 3. Setup initialization
        init_points, init_time = self.wrapper.initialize_gaussians(
            image_tensor, method=init_method, image_path=image_path, target_gaussians=num_points
        )

        # 4. Optimize (fitting) Gaussians
        gaussians_dict, fit_time, logs = self.wrapper.fit(
            original_np, init_points, iterations=iterations, lr=lr
        )

        # 5. Render reconstructed image for metrics computation
        reconstructed_np, decode_time = self.wrapper.render(gaussians_dict, h, w)

        # 6. Calculate Metrics
        psnr = CodecMetrics.calculate_psnr(original_np, reconstructed_np)
        ssim = CodecMetrics.calculate_ssim(original_np, reconstructed_np)
        
        # 7. Create temporary GIC file to measure actual compressed file size
        c = 3
        temp_header = {
            "codec_name": "GIC",
            "version": "1.0.0",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "image_info": {
                "width": w,
                "height": h,
                "channels": c,
                "original_format": Path(image_path).suffix[1:].upper()
            },
            "encoding_settings": {
                "quality_mode": quality_mode,
                "decided_mode": decided_mode,
                "init_method": init_method,
                "backend": self.wrapper.backend,
                "iterations": iterations,
                "lr": lr
            },
            "gaussian_info": {
                "num_points": len(init_points)
            }
        }

        # Create quick low-res preview
        preview_img = Image.fromarray(reconstructed_np).resize((128, 128))
        preview_buffer = io.BytesIO()
        preview_img.save(preview_buffer, format="PNG")
        preview_bytes = preview_buffer.getvalue()

        # Save temporarily
        os.makedirs(Path(output_path).parent, exist_ok=True)
        GICFormat.save(
            filepath=output_path,
            header=temp_header,
            gaussians=gaussians_dict,
            preview_bytes=preview_bytes,
            metrics=None,
            logs=logs
        )

        compressed_size = os.path.getsize(output_path)
        comp_ratio = CodecMetrics.calculate_compression_ratio(original_size, compressed_size)
        bpp = CodecMetrics.calculate_bpp(compressed_size, w, h)

        # 8. Complete Metrics
        metrics = {
            "file_size_bytes": compressed_size,
            "original_file_size_bytes": original_size,
            "compression_ratio": float(comp_ratio),
            "bpp": float(bpp),
            "psnr": float(psnr),
            "ssim": float(ssim),
            "encoding_time_sec": float(fit_time + init_time),
            "decoding_time_sec": float(decode_time)
        }

        # Resave GIC with metrics populated
        GICFormat.save(
            filepath=output_path,
            header=temp_header,
            gaussians=gaussians_dict,
            preview_bytes=preview_bytes,
            metrics=metrics,
            logs=logs
        )

        return metrics

    def encode_video(
        self,
        input_path,
        output_path,
        quality_mode="auto",
        init_method="net",
        iterations=1000,
        lr=0.001,
        fps=None,
        max_frames=None,
        video_init_mode="independent",
        progress_callback=None,
    ):
        """
        Encode a video (directory of images or MP4 file) into a .giv compressed file.
        """
        print(f"Encoding video: {input_path} -> {output_path} (Mode: {quality_mode}, Init: {video_init_mode})", flush=True)
        start_total = time.time()
        video_init_mode = video_init_mode.lower()
        if video_init_mode not in {"independent", "previous_frame"}:
            raise ValueError(f"Unknown video_init_mode: {video_init_mode}")

        # 1. Gather frame paths/tensors
        frames_np = []
        original_sizes = []
        
        input_path = Path(input_path)
        if input_path.is_dir():
            # Search for sorted images
            exts = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']
            frame_files = []
            for ext in exts:
                frame_files.extend(glob.glob(str(input_path / ext)))
            frame_files = sorted(frame_files)
            
            if not frame_files:
                raise FileNotFoundError(f"No image frames found in directory: {input_path}")
                
            if max_frames is not None:
                frame_files = frame_files[:max_frames]
                
            for fp in frame_files:
                img = Image.open(fp).convert("RGB")
                frames_np.append(np.array(img))
                original_sizes.append(os.path.getsize(fp))
        else:
            # MP4/AVI file
            cap = cv2.VideoCapture(str(input_path))
            if not cap.isOpened():
                raise FileNotFoundError(f"Could not open video file: {input_path}")

            source_fps = cap.get(cv2.CAP_PROP_FPS)
            if fps is None and source_fps and source_fps > 0:
                fps = float(source_fps)
            
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                # BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames_np.append(frame_rgb)
                frame_count += 1
                if max_frames is not None and frame_count >= max_frames:
                    break
            cap.release()
            original_sizes = [os.path.getsize(input_path) // len(frames_np)] * len(frames_np) # estimate

        if not frames_np:
            raise ValueError(f"No frames extracted from input: {input_path}")

        if fps is None:
            fps = 24.0

        h, w, c = frames_np[0].shape
        total_frames = len(frames_np)
        print(f"Extracted {total_frames} frames of size {w}x{h} at {fps:.3f} fps.", flush=True)

        frames_dict = {}
        previews_dict = {}
        index_frames = []
        total_fit_time = 0
        total_decode_time = 0
        total_psnr = 0
        total_ssim = 0
        total_points = 0
        previous_gaussians = None

        # 2. Loop over each frame and encode it
        for idx, original_np in enumerate(frames_np):
            frame_idx = idx + 1
            print(f"Processing Frame {frame_idx}/{total_frames}...", flush=True)
            
            # Analyze complexity
            analysis = self.analyzer.calculate_complexity(original_np)
            
            if quality_mode.lower() == "auto":
                decided_mode = analysis["recommended_mode"]
                num_points = analysis["target_gaussians"]
            else:
                decided_mode = quality_mode.capitalize()
                if decided_mode == "Low":
                    num_points = 10000
                elif decided_mode == "Medium":
                    num_points = 25000
                elif decided_mode == "High":
                    num_points = 50000
                else:
                    raise ValueError(f"Unknown quality mode: {quality_mode}")

            # Fit frame
            image_tensor = self.wrapper.image_to_tensor(original_np)
            if video_init_mode == "previous_frame" and previous_gaussians is not None:
                init_start = time.time()
                init_points = self.wrapper.gaussians_to_init_points(previous_gaussians)
                init_time = time.time() - init_start
                init_source = "previous_frame"
            else:
                init_points, init_time = self.wrapper.initialize_gaussians(
                    image_tensor, method=init_method, target_gaussians=num_points
                )
                init_source = init_method
            
            gaussians_dict, fit_time, _ = self.wrapper.fit(
                original_np, init_points, iterations=iterations, lr=lr
            )
            frame_encoding_time = fit_time + init_time
            total_fit_time += frame_encoding_time
            
            # Render for metrics
            reconstructed_np, frame_decode_time = self.wrapper.render(gaussians_dict, h, w)
            total_decode_time += frame_decode_time
            psnr = CodecMetrics.calculate_psnr(original_np, reconstructed_np)
            ssim = CodecMetrics.calculate_ssim(original_np, reconstructed_np)
            
            total_psnr += psnr
            total_ssim += ssim
            total_points += len(init_points)
            
            # Save frame parameters
            frames_dict[frame_idx] = gaussians_dict
            previous_gaussians = gaussians_dict
            
            # Generate preview image
            preview_img = Image.fromarray(reconstructed_np).resize((128, 128))
            preview_buffer = io.BytesIO()
            preview_img.save(preview_buffer, format="PNG")
            previews_dict[frame_idx] = preview_buffer.getvalue()
            
            # Update index entry
            index_frames.append({
                "frame_idx": frame_idx,
                "filename": f"frames/frame_{frame_idx:06d}.npz",
                "preview": f"previews/frame_{frame_idx:06d}.png",
                "quality_mode": decided_mode,
                "complexity_score": float(analysis["score"]),
                "init_source": init_source,
                "num_points": len(init_points),
                "psnr": float(psnr),
                "ssim": float(ssim),
                "init_time_sec": float(init_time),
                "fit_time_sec": float(fit_time),
                "encoding_time_sec": float(frame_encoding_time),
                "decoding_time_sec": float(frame_decode_time),
            })

            if progress_callback is not None:
                progress_callback(frame_idx, total_frames, index_frames[-1])

        # 3. Create metadata header and index
        avg_points = total_points / total_frames
        avg_psnr = total_psnr / total_frames
        avg_ssim = total_ssim / total_frames
        
        header = {
            "codec_name": "GIV",
            "version": "1.0.0",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "video_info": {
                "width": w,
                "height": h,
                "fps": fps,
                "total_frames": total_frames,
                "duration_sec": float(total_frames / fps)
            },
            "encoding_settings": {
                "quality_mode": quality_mode,
                "init_method": init_method,
                "video_init_mode": video_init_mode,
                "backend": self.wrapper.backend,
                "iterations": iterations,
                "lr": lr
            },
            "average_gaussian_info": {
                "avg_points_per_frame": int(avg_points)
            }
        }
        
        index = {
            "frames": index_frames
        }

        # 4. Save initially to measure size
        os.makedirs(Path(output_path).parent, exist_ok=True)
        GIVFormat.save(
            filepath=output_path,
            header=header,
            index=index,
            frames_dict=frames_dict,
            previews_dict=previews_dict,
            metrics=None,
            logs=None
        )

        compressed_size = os.path.getsize(output_path)
        original_size = sum(original_sizes)
        comp_ratio = CodecMetrics.calculate_compression_ratio(original_size, compressed_size)
        bpp = CodecMetrics.calculate_bpp(compressed_size // total_frames, w, h)

        metrics = {
            "file_size_bytes": compressed_size,
            "original_file_size_bytes": original_size,
            "compression_ratio": float(comp_ratio),
            "avg_bpp": float(bpp),
            "avg_psnr": float(avg_psnr),
            "avg_ssim": float(avg_ssim),
            "total_encoding_time_sec": float(total_fit_time),
            "avg_encoding_time_sec": float(total_fit_time / total_frames),
            "total_decoding_time_sec": float(total_decode_time),
            "avg_decoding_time_sec": float(total_decode_time / total_frames),
        }

        # Resave GIV with metrics populated
        GIVFormat.save(
            filepath=output_path,
            header=header,
            index=index,
            frames_dict=frames_dict,
            previews_dict=previews_dict,
            metrics=metrics,
            logs=None
        )

        print(f"Video encoding successful. Output: {output_path} ({compressed_size / (1024 * 1024):.2f} MB), Avg PSNR: {avg_psnr:.2f} dB, Avg SSIM: {avg_ssim:.4f}", flush=True)
        return metrics

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="GIC/GIV Codec Encoder CLI")
    parser.add_argument("--input", type=str, required=True, help="Path to input image or video frame directory")
    parser.add_argument("--output", type=str, required=True, help="Path to save compressed (.gic or .giv) file")
    parser.add_argument("--mode", type=str, default="auto", choices=["auto", "low", "medium", "high"], help="Quality Mode")
    parser.add_argument("--init", type=str, default="net", choices=["net", "quard", "random"], help="Gaussian Initialization Method")
    parser.add_argument("--iter", type=int, default=1000, help="Number of fitting iterations")
    parser.add_argument("--video", action="store_true", help="Flag to indicate video encoding")
    parser.add_argument("--max_frames", type=int, default=None, help="Max frames limit for video encoding")
    parser.add_argument("--fps", type=float, default=None, help="Override output GIV FPS. MP4 input FPS is used automatically when omitted.")
    parser.add_argument(
        "--video_init",
        type=str,
        default="independent",
        choices=["independent", "previous_frame"],
        help="Video initialization strategy. previous_frame warm-starts each frame from the previous optimized Gaussians.",
    )
    
    args = parser.parse_args()
    
    encoder = GICEncoder()
    if args.video or Path(args.input).is_dir() or Path(args.input).suffix.lower() == ".mp4":
        encoder.encode_video(
            input_path=args.input,
            output_path=args.output,
            quality_mode=args.mode,
            init_method=args.init,
            iterations=args.iter,
            fps=args.fps,
            max_frames=args.max_frames,
            video_init_mode=args.video_init,
        )
    else:
        encoder.encode_image(
            image_path=args.input,
            output_path=args.output,
            quality_mode=args.mode,
            init_method=args.init,
            iterations=args.iter
        )
