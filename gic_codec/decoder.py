import os
from pathlib import Path
from PIL import Image
import numpy as np

from gic_codec.giv_format import GIVFormat
from gic_codec.gic_format import GICFormat
from gic_codec.instant_gi_wrapper import InstantGIWrapper

class GICDecoder:
    def __init__(self, device=None):
        self.wrapper = InstantGIWrapper(device=device)

    def decode_image(self, gic_path, output_path=None):
        """
        Decodes a .gic compressed file back into an RGB image.
        """
        # 1. Parse .gic archive
        data = GICFormat.load(gic_path)
        if data["header"] is None or data["gaussians"] is None:
            raise ValueError(f"Invalid or corrupted .gic file: {gic_path}")

        header = data["header"]
        gaussians = data["gaussians"]
        metrics = data["metrics"]
        
        h = header["image_info"]["height"]
        w = header["image_info"]["width"]

        # 2. Render Gaussian Splatting back to RGB image
        reconstructed_np, decode_time = self.wrapper.render(gaussians, h, w)

        # 3. Save to disk if requested
        if output_path is not None:
            os.makedirs(Path(output_path).parent, exist_ok=True)
            recon_img = Image.fromarray(reconstructed_np)
            recon_img.save(output_path)
            print(f"Decoded image saved to: {output_path}")

        return {
            "image": reconstructed_np,
            "header": header,
            "metrics": metrics,
            "decoding_time": decode_time
        }

    def decode_video(self, giv_path, output_dir=None):
        """
        Decodes a .giv compressed file back into a sequence of RGB images.
        """
        # 1. Parse .giv archive
        data = GIVFormat.load(giv_path, load_frames=True)
        if data["header"] is None or data["index"] is None:
            raise ValueError(f"Invalid or corrupted .giv file: {giv_path}")

        header = data["header"]
        index = data["index"]
        frames = data["frames"]
        metrics = data["metrics"]
        
        h = header["video_info"]["height"]
        w = header["video_info"]["width"]
        total_frames = header["video_info"]["total_frames"]

        decoded_frames = []
        total_decode_time = 0

        # 2. Decode each frame sequentially
        for frame_entry in index["frames"]:
            frame_idx = frame_entry["frame_idx"]
            print(f"Decoding frame {frame_idx}/{total_frames}...")
            
            gaussians = frames.get(frame_idx)
            if gaussians is None:
                raise ValueError(f"Missing Gaussian parameter data for frame: {frame_idx}")
                
            # Render frame
            reconstructed_np, decode_time = self.wrapper.render(gaussians, h, w)
            total_decode_time += decode_time
            decoded_frames.append(reconstructed_np)
            
            # Save frame if requested
            if output_dir is not None:
                out_path = Path(output_dir) / f"frame_{frame_idx:06d}.png"
                os.makedirs(out_path.parent, exist_ok=True)
                recon_img = Image.fromarray(reconstructed_np)
                recon_img.save(out_path)

        if output_dir is not None:
            print(f"Decoded video frames saved to: {output_dir}")

        return {
            "frames": decoded_frames,
            "header": header,
            "index": index,
            "metrics": metrics,
            "total_decoding_time_sec": total_decode_time
        }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="GIC/GIV Codec Decoder CLI")
    parser.add_argument("--input", type=str, required=True, help="Path to compressed (.gic or .giv) file")
    parser.add_argument("--output", type=str, required=True, help="Path to save decoded image or frame directory")
    
    args = parser.parse_args()
    
    decoder = GICDecoder()
    suffix = Path(args.input).suffix.lower()
    if suffix == ".gic":
        decoder.decode_image(args.input, args.output)
    elif suffix == ".giv":
        decoder.decode_video(args.input, args.output)
    else:
        print(f"Unknown extension: {suffix}. Must be .gic or .giv")

