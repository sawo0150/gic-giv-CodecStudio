import math
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric

class CodecMetrics:
    @staticmethod
    def calculate_psnr(original_np, reconstructed_np):
        """
        Calculate PSNR between original and reconstructed numpy arrays (RGB/Grayscale).
        """
        # Ensure values are float in [0, 255]
        orig = original_np.astype(np.float64)
        recon = reconstructed_np.astype(np.float64)
        
        mse = np.mean((orig - recon) ** 2)
        if mse == 0:
            return float('inf')
        
        max_pixel = 255.0
        psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
        return psnr

    @staticmethod
    def calculate_ssim(original_np, reconstructed_np):
        """
        Calculate SSIM between original and reconstructed numpy arrays (RGB).
        """
        # skimage ssim requires channel_axis for multichannel images in newer versions
        if len(original_np.shape) == 3:
            # check the channel axis
            channel_axis = 2 if original_np.shape[2] in [3, 4] else 0
            ssim = ssim_metric(original_np, reconstructed_np, channel_axis=channel_axis, data_range=255)
        else:
            ssim = ssim_metric(original_np, reconstructed_np, data_range=255)
        return float(ssim)

    @staticmethod
    def calculate_bpp(file_size_bytes, width, height):
        """
        Calculate Bits Per Pixel (BPP).
        """
        total_bits = file_size_bytes * 8
        total_pixels = width * height
        return float(total_bits / total_pixels)

    @staticmethod
    def calculate_compression_ratio(original_size_bytes, compressed_size_bytes):
        """
        Calculate compression ratio (Original / Compressed).
        """
        if compressed_size_bytes == 0:
            return float('inf')
        return float(original_size_bytes / compressed_size_bytes)
