import os
import cv2
import numpy as np

class ImageComplexityAnalyzer:
    def __init__(self, w_edge=50.0, w_var=1e-4, w_lap=5e-4):
        self.w_edge = w_edge
        self.w_var = w_var
        self.w_lap = w_lap

    def calculate_complexity(self, image_path_or_np):
        """
        Calculate complexity score of an image based on Color Variance, Edge Density, and Laplacian Variance.
        Args:
            image_path_or_np: str path to image or numpy array (RGB)
        Returns:
            dict containing metrics, recommended quality mode, and target gaussians.
        """
        if isinstance(image_path_or_np, (str, bytes, os.PathLike)):
            # Read image in BGR, convert to RGB
            image_bgr = cv2.imread(str(image_path_or_np))
            if image_bgr is None:
                raise FileNotFoundError(f"Could not load image: {image_path_or_np}")
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image_path_or_np

        # Ensure image is uint8
        if image_rgb.dtype != np.uint8:
            image_rgb = (image_rgb * 255).astype(np.uint8)

        # 1. Grayscale conversion
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape

        # 2. Color Variance
        color_var = np.mean([np.var(image_rgb[:, :, c]) for c in range(3)])

        # 3. Edge Density (Canny)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (h * w)

        # 4. Laplacian Variance (Texture sharpness)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        # 5. Composite Complexity Score
        score = (self.w_edge * edge_density) + (self.w_var * color_var) + (self.w_lap * laplacian_var)

        # 6. Mode Decision based on thresholds
        # Low Mode: score < 1.5, target gaussians = 10,000
        # Medium Mode: 1.5 <= score < 3.5, target gaussians = 25,000
        # High Mode: score >= 3.5, target gaussians = 50,000
        if score < 1.5:
            recommended_mode = "Low"
            target_gaussians = 10000
        elif score < 3.5:
            recommended_mode = "Medium"
            target_gaussians = 25000
        else:
            recommended_mode = "High"
            target_gaussians = 50000

        return {
            "score": float(score),
            "edge_density": float(edge_density),
            "color_variance": float(color_var),
            "laplacian_variance": float(laplacian_var),
            "recommended_mode": recommended_mode,
            "target_gaussians": target_gaussians,
            "edge_map": edges
        }
