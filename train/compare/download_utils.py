#!/usr/bin/env python3
"""
download_utils.py - Download management utilities
"""

import subprocess
from pathlib import Path

def download_file(index, output_dir, url_template):
    """Download a single bag file"""
    index = index % 2148
    url = url_template.format(index)
    bag_filename = f"action_value-{index:05d}-of-02148_data.bag"
    output_path = output_dir / bag_filename

    # Avoid re-downloading if it already exists
    if output_path.exists():
        #print(f"File {bag_filename} already exists. Skipping download.")
        return output_path

    #print(f"Downloading {bag_filename} from {url} ...")
    result = subprocess.run(["wget", "-O", str(output_path), url], capture_output=True)

    if result.returncode == 0 and output_path.exists():
        return output_path
    else:
        # <<< FIX: Added detailed error reporting for failed downloads.
        # This makes it clear *why* a download failed instead of failing silently.
        print(f"--- WGET FAILED (code: {result.returncode}) ---")
        print(f"Failed to download index {index} from URL: {url}")
        # stderr is bytes, so we decode it for printing.
        print(f"Error message: {result.stderr.decode().strip()}")
        print("---------------------------------")
        # Clean up partially downloaded file if it exists
        if output_path.exists():
            output_path.unlink()
        return None

def get_file_size_gb(filepath):
    """Get file size in GB"""
    return filepath.stat().st_size / (1024**3)

def get_total_size_gb(directory):
    """Get total size of all .bag files in directory in GB"""
    total = 0
    for f in directory.glob("*.bag"):
        total += f.stat().st_size
    return total / (1024**3)
