"""
VisDrone Download Helper

Downloads VisDrone-DET dataset splits.

Note: These are large files. Total ~2-3GB.
If you have slow internet, download manually from:
https://github.com/VisDrone/VisDrone-Dataset
"""

import os
import subprocess
from pathlib import Path


# Google Drive file IDs for VisDrone-DET
# These are from the official VisDrone repository
VISDRONE_URLS = {
    'train': 'https://drive.google.com/uc?id=1a2oHjcEcwXP8oUF95qiwrqzACb2YlUhn',
    'val': 'https://drive.google.com/uc?id=1bxK5zgLn0_L8x276eKkuYA_FzwCIjb59',
    'test-dev': 'https://drive.google.com/uc?id=1PFdW_VFSCfZ_sTSZAGjQdifF_Xd5mf0V'
}

# Alternative: direct links if gdown doesn't work
# Check the VisDrone GitHub for updated links


def download_with_gdown(file_id: str, output_path: str):
    """Download from Google Drive using gdown."""
    try:
        import gdown
    except ImportError:
        print("Installing gdown...")
        subprocess.run(['pip', 'install', 'gdown'], check=True)
        import gdown
    
    gdown.download(file_id, output_path, quiet=False)


def extract_zip(zip_path: str, extract_to: str):
    """Extract a zip file."""
    import zipfile
    
    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted to {extract_to}")


def main():
    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("VisDrone-DET Dataset Download")
    print("=" * 60)
    print()
    print("This will download ~2-3GB of data.")
    print("Press Ctrl+C to cancel.")
    print()
    
    for split_name, url in VISDRONE_URLS.items():
        zip_path = output_dir / f"VisDrone2019-DET-{split_name}.zip"
        
        if zip_path.exists():
            print(f"[SKIP] {split_name} already downloaded")
            continue
        
        print(f"\n[DOWNLOAD] {split_name}...")
        print(f"  URL: {url}")
        print(f"  Output: {zip_path}")
        
        try:
            download_with_gdown(url, str(zip_path))
        except Exception as e:
            print(f"  [ERROR] Download failed: {e}")
            print(f"  Please download manually from the VisDrone GitHub")
            continue
        
        # Extract
        try:
            extract_zip(str(zip_path), str(output_dir))
        except Exception as e:
            print(f"  [ERROR] Extraction failed: {e}")
    
    print("\n" + "=" * 60)
    print("Download complete!")
    print()
    print("Next step: Run data preparation")
    print("  python src/data/prepare.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
