#!/usr/bin/env python3
"""
Analyze missing samples in MIMIC-CXR processing
"""

import pandas as pd
import os
from tqdm import tqdm

def analyze_missing_samples():
    """Analyze why some studies are missing from processing"""
    
    # Paths
    raw_data_dir = "/home/abedin/Developments/mimic_cxr-raw-data/mimic-cxr"
    metadata_path = os.path.join(raw_data_dir, "organized_data/metadata/processed_metadata.csv")
    images_dir = os.path.join(raw_data_dir, "organized_data/images")
    reports_dir = os.path.join(raw_data_dir, "organized_data/reports")
    
    print("=== Analyzing Missing Samples ===")
    print(f"Metadata: {metadata_path}")
    print(f"Images directory: {images_dir}")
    print(f"Reports directory: {reports_dir}")
    
    # Load metadata
    print("\nLoading metadata...")
    metadata_df = pd.read_csv(metadata_path)
    print(f"Total studies in metadata: {len(metadata_df)}")
    
    # Check file existence
    print("\n=== Checking File Existence ===")
    missing_images = 0
    missing_reports = 0
    missing_both = 0
    valid_studies = 0
    
    for _, row in tqdm(metadata_df.iterrows(), desc="Checking files"):
        study_id = row['study_id']
        
        image_path = os.path.join(images_dir, f"{study_id}.jpg")
        report_path = os.path.join(reports_dir, f"{study_id}.txt")
        
        image_exists = os.path.exists(image_path)
        report_exists = os.path.exists(report_path)
        
        if not image_exists and not report_exists:
            missing_both += 1
        elif not image_exists:
            missing_images += 1
        elif not report_exists:
            missing_reports += 1
        else:
            valid_studies += 1
    
    print(f"\nFile existence results:")
    print(f"✅ Valid studies (both files exist): {valid_studies}")
    print(f"❌ Missing images only: {missing_images}")
    print(f"❌ Missing reports only: {missing_reports}")
    print(f"❌ Missing both files: {missing_both}")
    print(f"📊 Total: {valid_studies + missing_images + missing_reports + missing_both}")
    
    # Check text content for valid studies
    print("\n=== Checking Text Content ===")
    no_findings = 0
    no_impression = 0
    no_both = 0
    valid_text = 0
    
    # Sample first 1000 valid studies for text analysis
    valid_count = 0
    for _, row in tqdm(metadata_df.iterrows(), desc="Checking text content"):
        if valid_count >= 1000:
            break
            
        study_id = row['study_id']
        image_path = os.path.join(images_dir, f"{study_id}.jpg")
        report_path = os.path.join(reports_dir, f"{study_id}.txt")
        
        if os.path.exists(image_path) and os.path.exists(report_path):
            valid_count += 1
            
            try:
                with open(report_path, 'r') as f:
                    content = f.read()
                
                # Check for findings and impressions
                has_findings = 'findings:' in content.lower() or 'finding:' in content.lower()
                has_impression = 'impression:' in content.lower() or 'impression:' in content.lower()
                
                if not has_findings and not has_impression:
                    no_both += 1
                elif not has_findings:
                    no_findings += 1
                elif not has_impression:
                    no_impression += 1
                else:
                    valid_text += 1
                    
            except Exception as e:
                print(f"Error reading report for study {study_id}: {e}")
    
    print(f"\nText content analysis (sample of {valid_count} studies):")
    print(f"✅ Valid text (both findings and impression): {valid_text}")
    print(f"❌ No findings: {no_findings}")
    print(f"❌ No impression: {no_impression}")
    print(f"❌ No both: {no_both}")
    
    # Check split distribution
    print(f"\n=== Split Distribution ===")
    split_counts = metadata_df['official_split'].value_counts()
    print(split_counts)
    
    # Summary
    print(f"\n=== SUMMARY ===")
    print(f"Expected valid studies: ~{valid_studies}")
    print(f"Expected processed studies: ~{valid_text * (len(metadata_df) / 1000)}")
    print(f"Actual processed studies: 117,307")
    print(f"Difference: {valid_text * (len(metadata_df) / 1000) - 117307:.0f}")

if __name__ == "__main__":
    analyze_missing_samples() 