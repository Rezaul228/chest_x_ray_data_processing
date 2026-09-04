#!/usr/bin/env python3
"""
Check ALBEF Format Creation Progress

This script checks the progress of the ALBEF format creation process.
Run this when you reconnect to see the current status.
"""

import os
import json
import glob

def check_progress():
    """Check the progress of ALBEF format creation"""
    
    output_dir = "/home/abedin/Developments/chest_x_ray_data_processing/mimic_raw_image_text"
    
    print("=== ALBEF Format Creation Progress Check ===\n")
    
    # Check if progress file exists
    progress_file = os.path.join(output_dir, 'progress.txt')
    if os.path.exists(progress_file):
        print("📊 Current Progress:")
        print("-" * 50)
        with open(progress_file, 'r') as f:
            print(f.read())
    else:
        print("❌ No progress file found. Process may not have started yet.")
    
    print("\n📁 Directory Status:")
    print("-" * 50)
    
    # Check if output directory exists
    if os.path.exists(output_dir):
        print(f"✅ Output directory exists: {output_dir}")
        
        # Check image directories
        for split in ['train', 'val', 'test']:
            image_dir = os.path.join(output_dir, 'images', split)
            if os.path.exists(image_dir):
                image_count = len(glob.glob(os.path.join(image_dir, '*.jpg')))
                print(f"  📸 {split}/: {image_count:,} images")
            else:
                print(f"  ❌ {split}/: Directory not found")
        
        # Check JSON files
        print("\n📄 JSON Files:")
        for split in ['train', 'val', 'test']:
            json_file = os.path.join(output_dir, f'{split}.json')
            if os.path.exists(json_file):
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                    print(f"  ✅ {split}.json: {len(data):,} entries")
                except:
                    print(f"  ⚠️  {split}.json: File exists but may be corrupted")
            else:
                print(f"  ❌ {split}.json: Not found")
    else:
        print(f"❌ Output directory not found: {output_dir}")
    
    print("\n🔍 Process Status:")
    print("-" * 50)
    
    # Check if process is still running
    import subprocess
    try:
        result = subprocess.run(['pgrep', '-f', 'create_albef_format_from_raw.py'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            pids = result.stdout.strip().split('\n')
            print(f"✅ Process is running (PID: {', '.join(pids)})")
        else:
            print("❌ Process is not running")
    except:
        print("⚠️  Could not check process status")
    
    print("\n📋 Quick Commands:")
    print("-" * 50)
    print("To check progress again: python check_albef_progress.py")
    print("To view log file: tail -f albef_creation.log")
    print("To check disk space: df -h")
    print("To kill process if needed: pkill -f create_albef_format_from_raw.py")

if __name__ == "__main__":
    check_progress() 