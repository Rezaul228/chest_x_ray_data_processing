#!/usr/bin/env python3
"""
MIMIC-CXR Raw Data Analysis Utility

This script analyzes the structure of the MIMIC-CXR raw data and provides
detailed information about the dataset organization.
"""

import os
import glob
from pathlib import Path
from collections import Counter, defaultdict
import json
from datetime import datetime

def analyze_mimic_raw_data(raw_data_path):
    """Analyze the MIMIC-CXR raw data structure"""
    print("=== MIMIC-CXR Raw Data Analysis ===")
    print(f"Raw data path: {raw_data_path}")
    print(f"Analysis timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if not os.path.exists(raw_data_path):
        print(f"❌ Raw data path does not exist: {raw_data_path}")
        return
    
    # Analyze directory structure
    images_path = os.path.join(raw_data_path, "images")
    if not os.path.exists(images_path):
        print(f"❌ Images directory not found: {images_path}")
        return
    
    print(f"\n📁 Directory Structure Analysis:")
    
    # Count patient directories
    patient_dirs = [d for d in os.listdir(images_path) if d.startswith('p')]
    print(f"  Patient directories: {len(patient_dirs)}")
    
    # Sample patient directories
    sample_patients = patient_dirs[:5]
    print(f"  Sample patient directories: {sample_patients}")
    
    # Analyze study structure
    study_count = 0
    image_count = 0
    patient_study_mapping = defaultdict(list)
    
    print(f"\n🔍 Detailed Structure Analysis:")
    
    for patient_dir in patient_dirs[:10]:  # Analyze first 10 patients for speed
        patient_path = os.path.join(images_path, patient_dir)
        if os.path.isdir(patient_path):
            studies = [d for d in os.listdir(patient_path) if d.startswith('s')]
            study_count += len(studies)
            patient_study_mapping[patient_dir] = studies
            
            for study_dir in studies[:3]:  # Sample first 3 studies per patient
                study_path = os.path.join(patient_path, study_dir)
                if os.path.isdir(study_path):
                    images = glob.glob(os.path.join(study_path, "*.jpg"))
                    image_count += len(images)
    
    print(f"  Total studies analyzed: {study_count}")
    print(f"  Total images analyzed: {image_count}")
    
    # Analyze file patterns
    print(f"\n📊 File Pattern Analysis:")
    
    # Check for different image formats
    image_extensions = Counter()
    for patient_dir in patient_dirs[:5]:
        patient_path = os.path.join(images_path, patient_dir)
        for study_dir in os.listdir(patient_path)[:3]:
            study_path = os.path.join(patient_path, study_dir)
            if os.path.isdir(study_path):
                for file in os.listdir(study_path):
                    ext = os.path.splitext(file)[1].lower()
                    image_extensions[ext] += 1
    
    print(f"  Image file extensions: {dict(image_extensions)}")
    
    # Analyze file naming patterns
    print(f"\n🏷️ File Naming Pattern Analysis:")
    
    sample_files = []
    for patient_dir in patient_dirs[:3]:
        patient_path = os.path.join(images_path, patient_dir)
        for study_dir in os.listdir(patient_path)[:2]:
            study_path = os.path.join(patient_path, study_dir)
            if os.path.isdir(study_path):
                files = os.listdir(study_path)
                sample_files.extend(files[:2])
    
    print(f"  Sample file names:")
    for i, filename in enumerate(sample_files[:5]):
        print(f"    {i+1}: {filename}")
    
    # Check for reports
    print(f"\n📄 Report Analysis:")
    
    # Look for reports in various locations
    possible_report_paths = [
        os.path.join(raw_data_path, "reports"),
        os.path.join(raw_data_path, "files", "reports"),
        os.path.join(raw_data_path, "texts"),
        os.path.join(raw_data_path, "annotations")
    ]
    
    report_found = False
    for report_path in possible_report_paths:
        if os.path.exists(report_path):
            print(f"  ✅ Reports found at: {report_path}")
            report_found = True
            # Analyze report structure
            report_files = glob.glob(os.path.join(report_path, "**/*.txt"), recursive=True)
            print(f"    Report files found: {len(report_files)}")
            if report_files:
                print(f"    Sample report files:")
                for i, report_file in enumerate(report_files[:3]):
                    print(f"      {i+1}: {os.path.basename(report_file)}")
            break
    
    if not report_found:
        print(f"  ❌ No reports directory found in standard locations")
        print(f"  🔍 Searching for text files...")
        
        # Search for text files
        text_files = glob.glob(os.path.join(raw_data_path, "**/*.txt"), recursive=True)
        if text_files:
            print(f"    Text files found: {len(text_files)}")
            print(f"    Sample text files:")
            for i, text_file in enumerate(text_files[:3]):
                print(f"      {i+1}: {os.path.basename(text_file)}")
        else:
            print(f"    No text files found")
    
    # Generate summary
    print(f"\n📋 Summary:")
    print(f"  Raw data path: {raw_data_path}")
    print(f"  Patient directories: {len(patient_dirs)}")
    print(f"  Sample patients analyzed: {min(10, len(patient_dirs))}")
    print(f"  Studies analyzed: {study_count}")
    print(f"  Images analyzed: {image_count}")
    print(f"  Reports found: {'Yes' if report_found else 'No'}")
    
    # Save analysis results
    analysis_results = {
        'raw_data_path': raw_data_path,
        'patient_directories': len(patient_dirs),
        'sample_patients': sample_patients,
        'studies_analyzed': study_count,
        'images_analyzed': image_count,
        'image_extensions': dict(image_extensions),
        'sample_files': sample_files[:10],
        'reports_found': report_found,
        'analysis_timestamp': datetime.now().isoformat()
    }
    
    output_file = f"mimic_raw_data_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(analysis_results, f, indent=2)
    
    print(f"\n✅ Analysis results saved to: {output_file}")
    
    return analysis_results

def create_data_structure_summary():
    """Create a summary of the MIMIC-CXR data structure"""
    print(f"\n📖 MIMIC-CXR Data Structure Summary:")
    print(f"  Structure: /images/p{'{patient_id}'}/s{'{study_id}'}/{'{image_file}'}.jpg")
    print(f"  Example: /images/p10/p10999395/s59802033/2e9ecc5c-5c8da30e-3589f536-3809cd0f-df6631f4.jpg")
    print(f"  Format: Patient ID → Study ID → Image File")
    print(f"  Images: JPEG format, various sizes")
    print(f"  Naming: UUID-based filenames")

if __name__ == "__main__":
    raw_data_path = "/home/abedin/Developments/mimic_cxr-raw-data/mimic-cxr"
    
    # Analyze the raw data
    results = analyze_mimic_raw_data(raw_data_path)
    
    # Create summary
    create_data_structure_summary() 