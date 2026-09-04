#!/usr/bin/env python3
"""
Fix Validation Split - Process only the missing validation data
"""

import os
import json
import shutil
import pandas as pd
import re
from tqdm import tqdm

def extract_text_from_report(report_path):
    """Extract findings and impression from MIMIC-CXR report file"""
    try:
        with open(report_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        findings = ""
        impression = ""
        
        # Extract findings section
        findings_match = re.search(r'FINDINGS:(.*?)(?:IMPRESSION:|$)', content, re.DOTALL | re.IGNORECASE)
        if findings_match:
            findings = findings_match.group(1).strip()
        
        # Extract impression section
        impression_match = re.search(r'IMPRESSION:(.*?)$', content, re.DOTALL | re.IGNORECASE)
        if impression_match:
            impression = impression_match.group(1).strip()
        
        return findings, impression
    except Exception as e:
        print(f"Error reading report {report_path}: {e}")
        return "", ""

def process_validation_split():
    """Process only the validation split"""
    
    # Paths
    original_csv_path = "/home/abedin/Developments/mimic_cxr-raw-data/mimic-cxr/organized_data/metadata/processed_metadata_hybrid.csv"
    original_images_dir = "/home/abedin/Developments/mimic_cxr-raw-data/mimic-cxr/organized_data/images"
    original_reports_dir = "/home/abedin/Developments/mimic_cxr-raw-data/mimic-cxr/organized_data/reports"
    output_dir = "/home/abedin/Developments/chest_x_ray_data_processing/mimic_raw_image_text"
    
    print("=== Processing Missing Validation Split ===\n")
    
    # Load original CSV
    print("Loading original CSV...")
    original_df = pd.read_csv(original_csv_path)
    
    # Get validation studies
    validation_studies = original_df[original_df['hybrid_split'] == 'validate']
    print(f"Found {len(validation_studies):,} validation studies")
    
    # Create val directory if it doesn't exist
    val_image_dir = os.path.join(output_dir, 'images', 'val')
    os.makedirs(val_image_dir, exist_ok=True)
    
    # Process validation studies
    val_entries = []
    image_counter = 0
    
    print("Processing validation studies...")
    for idx, (_, row) in enumerate(tqdm(validation_studies.iterrows(), total=len(validation_studies), desc="Processing validation")):
        study_id = str(row['study_id'])
        image_file = row['image_file']
        report_file = row['report_file']
        
        # Construct paths
        original_image_path = os.path.join(original_images_dir, image_file)
        original_report_path = os.path.join(original_reports_dir, report_file)
        
        # Check if files exist
        if not os.path.exists(original_image_path):
            print(f"    Warning: Image file not found: {original_image_path}")
            continue
        if not os.path.exists(original_report_path):
            print(f"    Warning: Report file not found: {original_report_path}")
            continue
        
        # Copy image with ALBEF naming
        albef_image_name = f"val_image_{image_counter:06d}.jpg"
        albef_image_path = os.path.join(val_image_dir, albef_image_name)
        
        try:
            shutil.copy2(original_image_path, albef_image_path)
        except Exception as e:
            print(f"    Error copying image {original_image_path}: {e}")
            continue
        
        # Extract text from report
        findings, impression = extract_text_from_report(original_report_path)
        
        # Skip if no findings
        if not findings.strip():
            print(f"    Warning: No findings found for study {study_id}")
            continue
        
        # Create entry for JSON
        entry = {
            "image": f"val/{albef_image_name}",
            "findings": findings.strip(),
            "impression": impression.strip() if impression.strip() else "",
            "image_id": image_counter,
            "study_id": study_id
        }
        
        val_entries.append(entry)
        image_counter += 1
    
    # Save validation JSON
    val_json_path = os.path.join(output_dir, 'val.json')
    with open(val_json_path, 'w') as f:
        json.dump(val_entries, f, indent=2)
    
    print(f"\n✅ Validation split processed successfully!")
    print(f"   Images copied: {image_counter:,}")
    print(f"   JSON entries: {len(val_entries):,}")
    print(f"   Output file: {val_json_path}")
    
    # Update progress file
    progress_file = os.path.join(output_dir, 'progress.txt')
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            progress_content = f.read()
        
        # Update the validation count
        progress_content = progress_content.replace("validate: 0 samples", f"validate: {len(val_entries):,} samples")
        
        with open(progress_file, 'w') as f:
            f.write(progress_content)
        
        print(f"   Progress file updated: {progress_file}")

if __name__ == "__main__":
    process_validation_split() 