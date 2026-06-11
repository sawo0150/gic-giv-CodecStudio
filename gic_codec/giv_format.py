import json
import zipfile
import io
import numpy as np
from PIL import Image

class GIVFormat:
    @staticmethod
    def save(filepath, header, index, frames_dict, previews_dict=None, metrics=None, logs=None):
        """
        Saves Gaussian video frames and metadata into a .giv zip container.
        Args:
            filepath: Target .giv file path.
            header: Dict containing video level metadata.
            index: Dict containing frame timeline information and quality metrics.
            frames_dict: Dict mapping frame index (int) to Gaussian params dict.
            previews_dict: Dict mapping frame index (int) to PNG bytes.
            metrics: Dict containing video average metrics.
            logs: Dict containing logs.
        """
        with zipfile.ZipFile(filepath, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # 1. Save header.json
            zipf.writestr('header.json', json.dumps(header, indent=4))
            
            # 2. Save index.json
            zipf.writestr('index.json', json.dumps(index, indent=4))
            
            # 3. Save frames/frame_XXXXXX.npz
            for frame_idx, gaussians in frames_dict.items():
                filename = f"frames/frame_{frame_idx:06d}.npz"
                npz_buffer = io.BytesIO()
                np.savez_compressed(npz_buffer, **gaussians)
                zipf.writestr(filename, npz_buffer.getvalue())
                
            # 4. Save previews/frame_XXXXXX.png
            if previews_dict is not None:
                for frame_idx, preview_bytes in previews_dict.items():
                    filename = f"previews/frame_{frame_idx:06d}.png"
                    zipf.writestr(filename, preview_bytes)
                    
            # 5. Save metrics.json (if provided)
            if metrics is not None:
                zipf.writestr('metrics.json', json.dumps(metrics, indent=4))
                
            # 6. Save logs.json (if provided)
            if logs is not None:
                zipf.writestr('logs.json', json.dumps(logs, indent=4))

    @staticmethod
    def load(filepath, load_frames=True):
        """
        Loads and parses a .giv zip container.
        Args:
            filepath: Source .giv file path.
            load_frames: If True, loads all frame .npz files into memory. Set False for metadata query.
        Returns:
            Dict containing header, index, frames, previews, metrics, logs.
        """
        data = {
            "header": None,
            "index": None,
            "frames": {},
            "previews": {},
            "metrics": None,
            "logs": None
        }
        with zipfile.ZipFile(filepath, 'r') as zipf:
            namelist = zipf.namelist()
            
            # 1. Load header.json
            if 'header.json' in namelist:
                data["header"] = json.loads(zipf.read('header.json').decode('utf-8'))
                
            # 2. Load index.json
            if 'index.json' in namelist:
                data["index"] = json.loads(zipf.read('index.json').decode('utf-8'))
                
            # 3. Load metrics.json
            if 'metrics.json' in namelist:
                data["metrics"] = json.loads(zipf.read('metrics.json').decode('utf-8'))
                
            # 4. Load logs.json
            if 'logs.json' in namelist:
                data["logs"] = json.loads(zipf.read('logs.json').decode('utf-8'))

            # 5. Load previews
            for name in namelist:
                if name.startswith('previews/') and name.endswith('.png'):
                    # Extract frame index
                    filename = name.split('/')[-1]
                    frame_idx = int(filename.split('_')[-1].split('.')[0])
                    data["previews"][frame_idx] = Image.open(io.BytesIO(zipf.read(name)))
            
            # 6. Load frames if requested
            if load_frames:
                for name in namelist:
                    if name.startswith('frames/') and name.endswith('.npz'):
                        filename = name.split('/')[-1]
                        frame_idx = int(filename.split('_')[-1].split('.')[0])
                        npz_data = np.load(io.BytesIO(zipf.read(name)))
                        data["frames"][frame_idx] = {key: npz_data[key] for key in npz_data.files}
                        
        return data
