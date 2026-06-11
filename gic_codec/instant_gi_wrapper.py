import os
import sys
import time
import math
from pathlib import Path
import numpy as np
from PIL import Image

# Add Instant-GI path to sys.path to enable loading submodules
PROJECT_DIR = Path(__file__).parent.parent
INSTANT_GI_DIR = PROJECT_DIR / "Instant-GI"
sys.path.append(str(INSTANT_GI_DIR))

try:
    import torch
    import torchvision.transforms as transforms
    from generalizable_model.init_net import InitNet
    from quard_image import QuardImage
    from gaussianimage_rs import GaussianImage_RS
    HAS_INSTANT_GI = True
    INSTANT_GI_IMPORT_ERROR = None
except Exception as exc:
    torch = None
    transforms = None
    InitNet = None
    QuardImage = None
    GaussianImage_RS = None
    HAS_INSTANT_GI = False
    INSTANT_GI_IMPORT_ERROR = exc

class InstantGIWrapper:
    def __init__(self, device=None):
        self.device = device if device is not None else (
            torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if HAS_INSTANT_GI else "cpu"
        )
        self.init_net = None
        self.backend = "instant_gi" if HAS_INSTANT_GI else "numpy_fallback"

    def _load_init_net(self, checkpoint_path=None):
        if self.init_net is not None:
            return
            
        if checkpoint_path is None:
            checkpoint_path = PROJECT_DIR / "pretrained" / "epoch_best_ks_3.pth"
            
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"InitNet weights not found at: {checkpoint_path}. Run setup_models.py first.")
            
        print(f"Loading InitNet weights from {checkpoint_path}...")
        self.init_net = InitNet(kernel_size=3).to(self.device)
        self.init_net.load_state_dict(torch.load(checkpoint_path, map_location=self.device)["model"])
        self.init_net.eval()

    def image_to_tensor(self, image_input, max_size=512):
        """
        Converts file path or numpy array (RGB) to Torch tensor [1, C, H, W] with optional downsampling.
        """
        if isinstance(image_input, (str, Path)):
            img = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, np.ndarray):
            img = Image.fromarray(image_input)
        else:
            if not HAS_INSTANT_GI:
                return image_input
            return image_input.to(self.device)

        # Downsample if image is larger than max_size to prevent CUDA OOM
        w, h = img.size
        if max(w, h) > max_size:
            ratio = max_size / max(w, h)
            new_w = int(w * ratio)
            new_h = int(h * ratio)
            # Ensure divisibility by 16 for BLOCK grid
            new_w = (new_w // 16) * 16
            new_h = (new_h // 16) * 16
            new_w = max(16, new_w)
            new_h = max(16, new_h)
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            print(f"Downsampled image from {w}x{h} to {new_w}x{new_h} to avoid CUDA OOM.")

        if not HAS_INSTANT_GI:
            return np.array(img)

        transform = transforms.ToTensor()
        return transform(img).unsqueeze(0).to(self.device)

    def _fallback_image_np(self, image_input, max_size=512):
        if isinstance(image_input, (str, Path)):
            img = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, np.ndarray):
            img = Image.fromarray(image_input.astype(np.uint8)).convert("RGB")
        else:
            arr = np.asarray(image_input)
            if arr.ndim == 4:
                arr = arr[0]
            if arr.ndim == 3 and arr.shape[0] in (1, 3):
                arr = np.moveaxis(arr, 0, -1)
            img = Image.fromarray((arr * 255).clip(0, 255).astype(np.uint8)).convert("RGB")

        w, h = img.size
        if max(w, h) > max_size:
            ratio = max_size / max(w, h)
            img = img.resize((max(1, int(w * ratio)), max(1, int(h * ratio))), Image.Resampling.LANCZOS)
        return np.array(img)

    def _make_fallback_points(self, h, w, target_gaussians):
        target = max(64, min(int(target_gaussians), h * w))
        scale = math.sqrt(target / float(h * w))
        low_w = max(8, int(round(w * scale)))
        low_h = max(8, int(round(h * scale)))
        while low_w * low_h > target and (low_w > 8 or low_h > 8):
            if low_w >= low_h and low_w > 8:
                low_w -= 1
            elif low_h > 8:
                low_h -= 1
            else:
                break
        xs = np.linspace(-1.0, 1.0, low_w, dtype=np.float32)
        ys = np.linspace(-1.0, 1.0, low_h, dtype=np.float32)
        grid_x, grid_y = np.meshgrid(xs, ys)
        xy = np.stack([grid_x.reshape(-1), grid_y.reshape(-1)], axis=1)
        scale_arr = np.full((xy.shape[0], 2), [1.0 / low_w, 1.0 / low_h], dtype=np.float32)
        rotation = np.zeros((xy.shape[0], 1), dtype=np.float32)
        color = np.zeros((xy.shape[0], 3), dtype=np.float32)
        return np.concatenate([xy, scale_arr, rotation, color], axis=1), low_h, low_w

    def initialize_gaussians(self, image_tensor, method="net", image_path=None, target_gaussians=20000, checkpoint_path=None):
        """
        Runs the initialization phase (Network PPM, Quadtree, or Random).
        Returns:
            numpy array: init_points [N, 12] (contains: xy, scale, rotation, color)
            float: elapsed time
        """
        start_time = time.time()

        if not HAS_INSTANT_GI or method == "fallback":
            if hasattr(image_tensor, "shape") and len(image_tensor.shape) == 4:
                h, w = image_tensor.shape[2], image_tensor.shape[3]
            else:
                arr = np.asarray(image_tensor)
                h, w = arr.shape[:2]
            init_points, _, _ = self._make_fallback_points(h, w, target_gaussians)
            return init_points, time.time() - start_time
        
        if method == "net":
            self._load_init_net(checkpoint_path)
            with torch.no_grad():
                xy, scale, rotation, color, _ = self.init_net(image_tensor, get_gaussians=True)
            xy = xy.cpu().numpy()
            scale = scale.cpu().numpy()
            rotation = rotation.cpu().numpy()
            color = color.cpu().numpy()
            init_points = np.concatenate([xy, scale, rotation, color], axis=1)
            
        elif method == "quard":
            if image_path is None:
                raise ValueError("Quardtree initialization requires a physical image_path.")
            quard_image = QuardImage(str(image_path))
            init_points, _ = quard_image.split()
            
        elif method == "random":
            # Just sample target_gaussians randomly
            h, w = image_tensor.shape[2], image_tensor.shape[3]
            xy = np.random.uniform(-1, 1, (target_gaussians, 2))
            scale = np.random.uniform(-5, -2, (target_gaussians, 2))
            rotation = np.random.uniform(0, 360, (target_gaussians, 1)) / 360.0
            color = np.random.uniform(0, 1, (target_gaussians, 3))
            init_points = np.concatenate([xy, scale, rotation, color], axis=1)
            
        else:
            raise ValueError(f"Unknown initialization method: {method}")

        elapsed_time = time.time() - start_time
        return init_points, elapsed_time

    def fit(self, image_input, init_points, iterations=1000, lr=0.001):
        """
        Optimizes (fits) the 2D Gaussians to match the target image.
        Returns:
            dict: final gaussian parameters as numpy arrays.
            float: encoding time.
            dict: log history (loss and psnr curves).
        """
        start_time = time.time()
        gt_tensor = self.image_to_tensor(image_input)
        if not HAS_INSTANT_GI:
            original_np = self._fallback_image_np(image_input)
            h, w = original_np.shape[:2]
            _, low_h, low_w = self._make_fallback_points(h, w, len(init_points))
            low_img = Image.fromarray(original_np).resize((low_w, low_h), Image.Resampling.LANCZOS)
            colors = np.asarray(low_img).reshape(-1, 3).astype(np.float32) / 255.0
            xs = np.linspace(-1.0, 1.0, low_w, dtype=np.float32)
            ys = np.linspace(-1.0, 1.0, low_h, dtype=np.float32)
            grid_x, grid_y = np.meshgrid(xs, ys)
            xyz = np.stack([grid_x.reshape(-1), grid_y.reshape(-1)], axis=1).astype(np.float32)
            gaussians = {
                "xyz": xyz,
                "scaling": np.full((xyz.shape[0], 2), [1.0 / low_w, 1.0 / low_h], dtype=np.float32),
                "rotation": np.zeros((xyz.shape[0], 1), dtype=np.float32),
                "opacity": np.ones((xyz.shape[0], 1), dtype=np.float32),
                "features_dc": colors,
                "fallback_shape": np.array([low_h, low_w], dtype=np.int32),
            }
            recon = np.asarray(low_img.resize((w, h), Image.Resampling.BICUBIC))
            mse = float(np.mean((original_np.astype(np.float32) - recon.astype(np.float32)) ** 2))
            psnr = float("inf") if mse == 0 else 20.0 * math.log10(255.0 / math.sqrt(mse))
            logs = [{"iteration": 1, "loss": mse, "psnr": psnr, "backend": "numpy_fallback"}]
            return gaussians, time.time() - start_time, logs

        h, w = gt_tensor.shape[2], gt_tensor.shape[3]
        num_points = len(init_points)

        # Build standard GaussianImage_RS model
        gaussian_model = GaussianImage_RS(
            loss_type="L2", opt_type="adan", num_points=num_points,
            H=h, W=w, BLOCK_H=16, BLOCK_W=16,
            device=self.device, lr=lr, quantize=False,
            init_points=init_points, gt_image=gt_tensor
        ).to(self.device)

        gaussian_model.train()
        start_time = time.time()
        
        log_history = []
        for iteration in range(1, iterations + 1):
            loss, psnr, _ = gaussian_model.train_iter(gt_tensor)
            if iteration % 200 == 0 or iteration == iterations:
                log_history.append({"iteration": iteration, "loss": float(loss.item()), "psnr": float(psnr)})

        encoding_time = time.time() - start_time

        # Extract optimized parameters
        state_dict = gaussian_model.state_dict()
        # Convert state_dict tensors to numpy
        gaussians_np = {k: v.cpu().numpy() for k, v in state_dict.items() if k in [
            '_xyz', '_scaling', '_rotation', '_opacity', '_features_dc'
        ]}
        
        # Clean keys (remove leading underscore if any, or map standard keys)
        # Standard keys: xyz, scaling, rotation, opacity, features_dc
        standard_gaussians = {
            "xyz": gaussians_np.get("_xyz"),
            "scaling": gaussians_np.get("_scaling"),
            "rotation": gaussians_np.get("_rotation"),
            "opacity": gaussians_np.get("_opacity"),
            "features_dc": gaussians_np.get("_features_dc")
        }

        return standard_gaussians, encoding_time, log_history

    def render(self, gaussians_dict, h, w):
        """
        Decodes/renders the RGB image from Gaussian parameters.
        Returns:
            numpy array: reconstructed RGB image [H, W, 3] (uint8)
            float: rendering/decoding time
        """
        num_points = len(gaussians_dict["xyz"])

        if "fallback_shape" in gaussians_dict or not HAS_INSTANT_GI:
            start_time = time.time()
            if "fallback_shape" not in gaussians_dict:
                raise RuntimeError("Instant-GI backend is unavailable and this file has no fallback_shape metadata.")
            low_h, low_w = [int(v) for v in np.asarray(gaussians_dict["fallback_shape"]).tolist()]
            colors = np.asarray(gaussians_dict["features_dc"], dtype=np.float32)
            low_img = (colors.reshape(low_h, low_w, 3) * 255).clip(0, 255).astype(np.uint8)
            reconstructed_np = np.asarray(Image.fromarray(low_img).resize((w, h), Image.Resampling.BICUBIC))
            return reconstructed_np, time.time() - start_time
        
        # Build state dict for loading
        state_dict = {
            "_xyz": torch.tensor(gaussians_dict["xyz"]).to(self.device),
            "_scaling": torch.tensor(gaussians_dict["scaling"]).to(self.device),
            "_rotation": torch.tensor(gaussians_dict["rotation"]).to(self.device),
            "_opacity": torch.tensor(gaussians_dict["opacity"]).to(self.device),
            "_features_dc": torch.tensor(gaussians_dict["features_dc"]).to(self.device)
        }

        # Create model instance
        gaussian_model = GaussianImage_RS(
            loss_type="L2", opt_type="adan", num_points=num_points,
            H=h, W=w, BLOCK_H=16, BLOCK_W=16,
            device=self.device, lr=0.001, quantize=False,
            init_points=None, gt_image=None
        ).to(self.device)
        
        # Load state with strict=False to bypass missing buffer keys like 'bound'
        gaussian_model.load_state_dict(state_dict, strict=False)
        gaussian_model.eval()

        start_time = time.time()
        with torch.no_grad():
            out = gaussian_model()
            render_tensor = out["render"][0] # [C, H, W]
        decoding_time = time.time() - start_time

        # Convert to numpy uint8 RGB [H, W, C]
        reconstructed_np = (render_tensor.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        return reconstructed_np, decoding_time
