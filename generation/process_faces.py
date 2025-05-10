#!/usr/bin/env python3

import os
import glob
import subprocess
from pathlib import Path

def process_folder(folder_path, output_dir):
    # Get the folder name for output filename
    folder_name = os.path.basename(folder_path)
    
    # Find source and target images
    source_image = glob.glob(os.path.join(folder_path, "source*.png"))[0]
    target_image = glob.glob(os.path.join(folder_path, "target*.png"))[0]
    
    # Set up output path
    output_path = os.path.join(output_dir, f"{folder_name}.png")
    
    # Run facefusion command with headless-run subcommand
    cmd = [
        "python", "facefusion.py",
        "headless-run",
        "--source", source_image,
        "--target", target_image,
        "--output-path", output_path,
        "--face-swapper-model", "inswapper_128",
        "--face-enhancer-model", "gfpgan_1.4",
        "--face-enhancer-blend", "100"
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"Successfully processed {folder_name}")
    except subprocess.CalledProcessError as e:
        print(f"Error processing {folder_name}: {str(e)}")

def main():
    input_dir = "/home/lzz/src/output_frames"
    output_dir = "/home/lzz/src/output"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each folder
    for folder_path in glob.glob(os.path.join(input_dir, "*")):
        if os.path.isdir(folder_path):
            try:
                print(f"Processing folder: {folder_path}")
                process_folder(folder_path, output_dir)
            except Exception as e:
                print(f"Error processing {folder_path}: {str(e)}")

if __name__ == "__main__":
    main() 