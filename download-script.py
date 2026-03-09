#!/usr/bin/env python3
"""
Simple image downloader for cat/dog training data
"""

from duckduckgo_search import DDGS
from pathlib import Path
import requests
import time

# CONFIGURATION
NUM_IMAGES = 30  # How many images per class
BASE_FOLDER = Path("./my_cats_dogs") / "train"

def download_images(keyword, folder, count):
    """Download images from DuckDuckGo."""
    folder.mkdir(parents=True, exist_ok=True)
    print(f"\nDownloading {count} {keyword} images...")
    
    downloaded = 0
    with DDGS() as ddgs:
        for result in ddgs.images(f"{keyword} photo", max_results=count * 2):
            if downloaded >= count:
                break
            
            url = result.get("image")
            if not url:
                continue
            
            try:
                img_data = requests.get(url, timeout=10).content
                filename = folder / f"{keyword}_{downloaded+1}.jpg"
                filename.write_bytes(img_data)
                downloaded += 1
                print(f"  [{downloaded}/{count}] Downloaded")
            except:
                pass
            
            time.sleep(5.5)  # Avoid rate limiting
    
    print(f"✓ Done: {downloaded} images saved to {folder}")

if __name__ == "__main__":
    # Create folders
    (BASE_FOLDER / "cat").mkdir(parents=True, exist_ok=True)
    (BASE_FOLDER / "dog").mkdir(parents=True, exist_ok=True)
    
    # Download
    download_images("cat", BASE_FOLDER / "cat", NUM_IMAGES)
    download_images("dog", BASE_FOLDER / "dog", NUM_IMAGES)
    
    print("\n✅ Ready! Run your training script now.")