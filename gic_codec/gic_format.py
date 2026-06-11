import json
import zipfile
import io
import numpy as np
from PIL import Image

class GICFormat:
    @staticmethod
    def save(filepath, header, gaussians, preview_bytes=None, metrics=None, logs=None):
        """
        Saves Gaussian parameters and metadata into a .gic zip container.
        Args:
            filepath: Target .gic file path.
            header: Dict containing metadata.
            gaussians: Dict containing numpy arrays (xyz, scaling, rotation, opacity, features_dc).
            preview_bytes: Bytes containing PNG preview image.
            metrics: Dict containing PSNR, SSIM, sizes, etc.
            logs: Dict containing training logs.
        """
        with zipfile.ZipFile(filepath, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # 1. Save header.json
            zipf.writestr('header.json', json.dumps(header, indent=4))
            
            # 2. Save gaussians.npz
            npz_buffer = io.BytesIO()
            np.savez_compressed(npz_buffer, **gaussians)
            zipf.writestr('gaussians.npz', npz_buffer.getvalue())
            
            # 3. Save preview.png (if provided)
            if preview_bytes is not None:
                zipf.writestr('preview.png', preview_bytes)
                
            # 4. Save metrics.json (if provided)
            if metrics is not None:
                zipf.writestr('metrics.json', json.dumps(metrics, indent=4))
                
            # 5. Save logs.json (if provided)
            if logs is not None:
                zipf.writestr('logs.json', json.dumps(logs, indent=4))

    @staticmethod
    def load(filepath):
        """
        Loads and parses a .gic zip container.
        Args:
            filepath: Source .gic file path.
        Returns:
            Dict containing header, gaussians, preview, metrics, logs.
        """
        data = {
            "header": None,
            "gaussians": None,
            "preview": None,
            "metrics": None,
            "logs": None
        }
        with zipfile.ZipFile(filepath, 'r') as zipf:
            # 1. Load header.json
            if 'header.json' in zipf.namelist():
                data["header"] = json.loads(zipf.read('header.json').decode('utf-8'))
                
            # 2. Load gaussians.npz
            if 'gaussians.npz' in zipf.namelist():
                npz_data = np.load(io.BytesIO(zipf.read('gaussians.npz')))
                data["gaussians"] = {key: npz_data[key] for key in npz_data.files}
                
            # 3. Load preview.png
            if 'preview.png' in zipf.namelist():
                data["preview"] = Image.open(io.BytesIO(zipf.read('preview.png')))
                
            # 4. Load metrics.json
            if 'metrics.json' in zipf.namelist():
                data["metrics"] = json.loads(zipf.read('metrics.json').decode('utf-8'))
                
            # 5. Load logs.json
            if 'logs.json' in zipf.namelist():
                data["logs"] = json.loads(zipf.read('logs.json').decode('utf-8'))
                
        return data
