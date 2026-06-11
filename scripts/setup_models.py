import os
import shutil
from pathlib import Path

# Paths
PROJECT_DIR = Path(__file__).parent.parent
PRETRAINED_DIR = PROJECT_DIR / "pretrained"
CHECKPOINTS_DIR = PROJECT_DIR / "Instant-GI" / "checkpoints"
MODEL_FILENAME = "epoch_best_ks_3.pth"

def setup_models():
    os.makedirs(PRETRAINED_DIR, exist_ok=True)
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    
    target_path = PRETRAINED_DIR / MODEL_FILENAME
    instant_gi_target_path = CHECKPOINTS_DIR / MODEL_FILENAME
    
    # Check if model is already in checkpoints or pretrained
    checkpoints_source = PROJECT_DIR / "checkpoints" / MODEL_FILENAME
    nested_source = PROJECT_DIR / "checkpoints" / "checkpoints" / MODEL_FILENAME
    
    # 1. Look for existing downloaded model and copy/symlink to target
    found_source = None
    if target_path.exists():
        found_source = target_path
    elif instant_gi_target_path.exists():
        found_source = instant_gi_target_path
    elif checkpoints_source.exists():
        found_source = checkpoints_source
    elif nested_source.exists():
        found_source = nested_source
        
    if found_source:
        print(f"Found existing pretrained model at: {found_source}")
        # Ensure it exists in both 'pretrained/' and 'Instant-GI/checkpoints/'
        if not target_path.exists():
            print(f"Copying model to: {target_path}")
            shutil.copy2(found_source, target_path)
        if not instant_gi_target_path.exists():
            print(f"Copying model to: {instant_gi_target_path}")
            shutil.copy2(found_source, instant_gi_target_path)
        print("Setup completed using existing downloaded file.")
        return True

    # 2. If not found, download using gdown
    print("Pretrained model not found. Attempting download via gdown...")
    try:
        import gdown
        os.makedirs(PRETRAINED_DIR, exist_ok=True)
        # Download epoch_best_ks_3.pth
        file_id = "1lpgi6hq5oJjiPOT28wUWYm2oqZpq_Dcb" # Folder containing the files
        print(f"Downloading from Google Drive Folder ID: {file_id}")
        
        # Download the folder contents to PRETRAINED_DIR
        gdown.download_folder(id=file_id, output=str(PRETRAINED_DIR), quiet=False, use_cookies=False)
        
        # If files were placed inside a nested checkpoints directory by gdown
        nested_download = PRETRAINED_DIR / "checkpoints" / MODEL_FILENAME
        if nested_download.exists() and not target_path.exists():
            shutil.move(str(nested_download), str(target_path))
            
        if target_path.exists():
            print(f"Copying downloaded model to Instant-GI checkpoints: {instant_gi_target_path}")
            shutil.copy2(target_path, instant_gi_target_path)
            print("Download and setup completed successfully.")
            return True
        else:
            print("Download completed but target model file was not found.")
            return False
            
    except ImportError:
        print("Error: 'gdown' package is not installed. Please install it using 'pip install gdown'.")
        return False
    except Exception as e:
        print(f"An error occurred during model setup: {e}")
        return False

if __name__ == "__main__":
    setup_models()
