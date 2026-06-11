import os
import zipfile
import urllib.request
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"

def download_file(url, target_path):
    print(f"Downloading {url} to {target_path}...")
    os.makedirs(target_path.parent, exist_ok=True)
    
    # Custom User-Agent to bypass potential bot blocks
    req = urllib.request.Request(
        url, 
        headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    )
    with urllib.request.urlopen(req) as response, open(target_path, 'wb') as out_file:
        data = response.read()
        out_file.write(data)
    print("Download completed.")

def setup_kodak():
    kodak_dir = DATA_DIR / "kodak"
    os.makedirs(kodak_dir, exist_ok=True)
    print("\n--- Setting up Kodak Dataset ---")
    
    # Download 24 images
    for i in range(1, 25):
        filename = f"kodim{i:02d}.png"
        file_path = kodak_dir / filename
        if file_path.exists():
            continue
        url = f"http://r0k.us/graphics/kodak/kodak/{filename}"
        try:
            download_file(url, file_path)
        except Exception as e:
            print(f"Failed to download {filename}: {e}")
            # Try alternate source (e.g. from sherylmehta's repo or kaggle mirrors if needed)
            alt_url = f"https://raw.githubusercontent.com/afw/AFW-Dataset/master/Kodak/{filename}"
            try:
                print(f"Trying alternative source for {filename}...")
                download_file(alt_url, file_path)
            except Exception as ae:
                print(f"Alternative download also failed: {ae}")

def setup_div2k():
    div2k_dir = DATA_DIR / "div2k"
    os.makedirs(div2k_dir, exist_ok=True)
    print("\n--- Setting up DIV2K Validation Dataset (Subset) ---")
    
    zip_path = DATA_DIR / "DIV2K_valid_HR.zip"
    if not zip_path.exists():
        url = "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip"
        try:
            download_file(url, zip_path)
        except Exception as e:
            print(f"Failed to download DIV2K zip: {e}")
            return
            
    # Unzip Validation images
    print("Unzipping DIV2K validation HR...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(DATA_DIR)
        
    # Reorganise: move DIV2K_valid_HR contents to data/div2k/
    extracted_dir = DATA_DIR / "DIV2K_valid_HR"
    if extracted_dir.exists():
        for f in extracted_dir.glob("*.png"):
            shutil_dest = div2k_dir / f.name
            if not shutil_dest.exists():
                f.rename(shutil_dest)
        # Cleanup
        try:
            os.rmdir(extracted_dir)
        except:
            pass
            
    if zip_path.exists():
        os.remove(zip_path)
    print("DIV2K dataset setup done.")

def setup_davis():
    davis_dir = DATA_DIR / "davis"
    os.makedirs(davis_dir, exist_ok=True)
    print("\n--- Setting up DAVIS 2016 Dataset (Subset) ---")
    
    zip_path = DATA_DIR / "DAVIS-data.zip"
    if not zip_path.exists():
        # Direct ETHZ download link for DAVIS 2016 480p
        url = "https://graphics.ethz.ch/Downloads/Data/Davis/DAVIS-data.zip"
        try:
            download_file(url, zip_path)
        except Exception as e:
            print(f"Failed to download DAVIS zip: {e}")
            return
            
    # Unzip
    print("Unzipping DAVIS dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # We only want JPEGImages/480p/bear and JPEGImages/480p/motocross to save disk space
        # Extract selective folders
        for file_info in zip_ref.infolist():
            if "JPEGImages/480p/bear" in file_info.filename or "JPEGImages/480p/motocross" in file_info.filename:
                zip_ref.extract(file_info, DATA_DIR)
                
    # Reorganise folders: move to data/davis/bear and data/davis/motocross
    extracted_bear = DATA_DIR / "DAVIS" / "JPEGImages" / "480p" / "bear"
    extracted_moto = DATA_DIR / "DAVIS" / "JPEGImages" / "480p" / "motocross"
    
    if extracted_bear.exists():
        shutil_bear = davis_dir / "bear"
        if shutil_bear.exists():
            import shutil
            shutil.rmtree(shutil_bear)
        extracted_bear.rename(shutil_bear)
        
    if extracted_moto.exists():
        shutil_moto = davis_dir / "motocross"
        if shutil_moto.exists():
            import shutil
            shutil.rmtree(shutil_moto)
        extracted_moto.rename(shutil_moto)
        
    # Cleanup nested extracted folders
    import shutil
    shutil.rmtree(DATA_DIR / "DAVIS", ignore_errors=True)
    if zip_path.exists():
        os.remove(zip_path)
    print("DAVIS dataset setup done.")

if __name__ == "__main__":
    setup_kodak()
    setup_div2k()
    setup_davis()
