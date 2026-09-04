#!/usr/bin/env python3
"""
Create ALBEF Format from Raw MIMIC-CXR Data

This script creates ALBEF format by:
1. Using split information from processed metadata
2. Copying original images with ALBEF naming convention
3. Getting original text reports from raw files
4. Creating JSON files in ALBEF format with separate findings and impression

ALBEF Format:
- mimic_raw_image_text/
  ├── images/
  │   ├── train/ (train_image_000000.jpg, train_image_000001.jpg, ...)
  │   ├── val/ (val_image_000000.jpg, val_image_000001.jpg, ...)
  │   └── test/ (test_image_000000.jpg, test_image_000001.jpg, ...)
  ├── train.json
  ├── val.json
  └── test.json
"""

import os
import json
import pickle
import shutil
import pandas as pd
import argparse
from tqdm import tqdm
import re
import time
from datetime import datetime

def extract_text_from_report(report_path):
    """Extract findings and impression from MIMIC-CXR report file with improved logic"""
    try:
        with open(report_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        findings = ""
        impression = ""
        
        # First try to find explicit FINDINGS section
        findings_match = re.search(r'FINDINGS:(.*?)(?:IMPRESSION:|$)', content, re.DOTALL | re.IGNORECASE)
        if findings_match:
            findings = findings_match.group(1).strip()
        
        # If no explicit findings section, try to extract from the main body
        if not findings:
            # Look for common patterns in the report body
            # Skip header sections like "FINAL REPORT", "REASON FOR EXAMINATION", etc.
            lines = content.split('\n')
            findings_lines = []
            
            for line in lines:
                line = line.strip()
                
                # Skip header sections
                if any(header in line.upper() for header in ['FINAL REPORT', 'REASON FOR EXAMINATION', 'INDICATION:', 'EXAMINATION:', 'COMPARISON:', 'TECHNIQUE:']):
                    continue
                
                # Start collecting findings after these sections
                if line and not line.startswith('___') and not line.startswith('//'):
                    findings_lines.append(line)
            
            # Join the findings lines
            if findings_lines:
                findings = ' '.join(findings_lines)
                # Clean up the findings
                findings = re.sub(r'\s+', ' ', findings).strip()
        
        # Extract impression section
        impression_match = re.search(r'IMPRESSION:(.*?)$', content, re.DOTALL | re.IGNORECASE)
        if impression_match:
            impression = impression_match.group(1).strip()
        
        return findings, impression
    except Exception as e:
        print(f"Error reading report {report_path}: {e}")
        return "", ""

def save_progress(output_dir, split, processed_count, total_count, start_time):
    """Save progress information to a file"""
    progress_file = os.path.join(output_dir, 'progress.txt')
    elapsed_time = time.time() - start_time
    
    with open(progress_file, 'w') as f:
        f.write(f"ALBEF Format Creation Progress\n")
        f.write(f"=============================\n")
        f.write(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Elapsed Time: {elapsed_time/3600:.2f} hours\n")
        f.write(f"Current Split: {split}\n")
        f.write(f"Processed: {processed_count:,}\n")
        f.write(f"Total Expected: {total_count:,}\n")
        f.write(f"Progress: {processed_count/total_count*100:.2f}%\n")
        f.write(f"Status: Running...\n")

def create_albef_format_from_raw(processed_metadata_path, original_csv_path, 
                                original_images_dir, original_reports_dir, 
                                output_dir, max_samples=None):
    """Create ALBEF format from raw MIMIC-CXR data"""
    
    print("=== Creating ALBEF Format from Raw MIMIC-CXR Data ===\n")
    start_time = time.time()
    
    # Load processed metadata to get split information
    print("Loading processed metadata...")
    with open(processed_metadata_path, 'rb') as f:
        processed_metadata = pickle.load(f)
    
    # Load original CSV
    print("Loading original CSV...")
    original_df = pd.read_csv(original_csv_path)
    print(f"Original CSV contains {len(original_df):,} studies")
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images', 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images', 'val'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images', 'test'), exist_ok=True)
    
    # Create mapping from study_id to split
    print("Creating study ID to split mapping...")
    study_to_split = {}
    for _, row in original_df.iterrows():
        study_id = str(row['study_id'])
        hybrid_split = row['hybrid_split']
        study_to_split[study_id] = hybrid_split
    
    # Count studies in each split
    split_counts = original_df['hybrid_split'].value_counts()
    print("Original data split distribution:")
    for split, count in split_counts.items():
        print(f"  {split}: {count:,} studies")
    
    # Calculate total expected samples
    total_expected = split_counts.sum()
    if max_samples:
        total_expected = min(total_expected, max_samples * 3)  # 3 splits
    
    # Process each split
    # Map CSV split names to ALBEF format names
    split_mapping = {
        'train': 'train',
        'validate': 'val',  # ALBEF uses 'val' not 'validate'
        'test': 'test'
    }
    splits = ['train', 'validate', 'test']
    split_data = {}
    total_processed = 0
    
    for csv_split in splits:
        albef_split = split_mapping[csv_split]
        print(f"\nProcessing {csv_split} split (ALBEF: {albef_split})...")
        
        # Get studies for this split
        split_studies = original_df[original_df['hybrid_split'] == csv_split]
        
        if max_samples:
            split_studies = split_studies.head(max_samples)
        
        print(f"  Found {len(split_studies):,} studies for {csv_split} split")
        
        split_entries = []
        image_counter = 0
        
        # Process each study
        for idx, (_, row) in enumerate(tqdm(split_studies.iterrows(), total=len(split_studies), desc=f"Processing {csv_split}")):
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
            
            # Copy image with ALBEF naming (use albef_split for directory and filename)
            albef_image_name = f"{albef_split}_image_{image_counter:06d}.jpg"
            albef_image_path = os.path.join(output_dir, 'images', albef_split, albef_image_name)
            
            try:
                shutil.copy2(original_image_path, albef_image_path)
            except Exception as e:
                print(f"    Error copying image {original_image_path}: {e}")
                continue
            
            # Extract text from report
            findings, impression = extract_text_from_report(original_report_path)
            
            # Include if we have either findings OR impression (more flexible, like MIMIC data loader)
            if not findings.strip() and not impression.strip():
                print(f"    Warning: No findings or impression found for study {study_id}")
                continue
            
            # Create entry for JSON with separate findings and impression (use albef_split for image path)
            # Handle cases where one field might be empty (like MIMIC data loader)
            entry = {
                "image": f"{albef_split}/{albef_image_name}",
                "findings": findings.strip() if findings.strip() else "",  # Findings field (can be empty)
                "impression": impression.strip() if impression.strip() else "",  # Impression field (can be empty)
                "image_id": image_counter,
                "study_id": study_id
            }
            
            split_entries.append(entry)
            image_counter += 1
            
            # Update progress every 100 samples
            if (idx + 1) % 100 == 0:
                total_processed += 1
                save_progress(output_dir, csv_split, total_processed, total_expected, start_time)
        
        split_data[albef_split] = split_entries  # Store with ALBEF split name
        total_processed += len(split_entries)
        print(f"  {csv_split} -> {albef_split}: {len(split_entries)} samples processed")
        
        # Save progress after each split
        save_progress(output_dir, csv_split, total_processed, total_expected, start_time)
    
    # Save JSON files
    print("\nSaving JSON files...")
    
    # Use ALBEF split names for JSON files
    albef_splits = ['train', 'val', 'test']
    for albef_split in albef_splits:
        if albef_split in split_data and split_data[albef_split]:
            json_path = os.path.join(output_dir, f'{albef_split}.json')
            with open(json_path, 'w') as f:
                json.dump(split_data[albef_split], f, indent=2)
            print(f"  {albef_split}.json: {len(split_data[albef_split])} entries")
    
    # Print summary
    print("\n=== ALBEF Format Creation Summary ===")
    total_samples = sum(len(split_data.get(albef_split, [])) for albef_split in albef_splits)
    print(f"Total samples created: {total_samples:,}")
    
    for albef_split in albef_splits:
        count = len(split_data.get(albef_split, []))
        print(f"{albef_split}: {count:,} samples")
    
    # Save final progress
    total_time = time.time() - start_time
    with open(os.path.join(output_dir, 'progress.txt'), 'w') as f:
        f.write(f"ALBEF Format Creation Progress\n")
        f.write(f"=============================\n")
        f.write(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Time: {total_time/3600:.2f} hours\n")
        f.write(f"Total Processed: {total_samples:,}\n")
        f.write(f"Status: COMPLETED ✓\n")
        f.write(f"\nSplit Summary:\n")
        for albef_split in albef_splits:
            count = len(split_data.get(albef_split, []))
            f.write(f"  {albef_split}: {count:,} samples\n")
    
    print(f"\nOutput directory: {output_dir}")
    print("ALBEF format creation completed! ✓")
    
    return split_data

def main():
    parser = argparse.ArgumentParser(description="Create ALBEF format from raw MIMIC-CXR data")
    parser.add_argument("--processed_metadata", type=str, 
                       default="/home/abedin/Developments/chest_x_ray_data_processing/all_processed_data/mimic_shards_hybrid_full_ori/metadata.pkl",
                       help="Path to processed metadata.pkl")
    parser.add_argument("--original_csv", type=str,
                       default="/home/abedin/Developments/mimic_cxr-raw-data/mimic-cxr/organized_data/metadata/processed_metadata_hybrid.csv",
                       help="Path to original metadata CSV")
    parser.add_argument("--original_images_dir", type=str,
                       default="/home/abedin/Developments/mimic_cxr-raw-data/mimic-cxr/organized_data/images",
                       help="Directory containing original image files")
    parser.add_argument("--original_reports_dir", type=str,
                       default="/home/abedin/Developments/mimic_cxr-raw-data/mimic-cxr/organized_data/reports",
                       help="Directory containing original report files")
    parser.add_argument("--output_dir", type=str, 
                       default="/home/abedin/Developments/chest_x_ray_data_processing/mimic_raw_image_text",
                       help="Output directory for ALBEF format")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum number of samples per split (for testing)")
    
    args = parser.parse_args()
    
    # Validate paths
    if not os.path.exists(args.processed_metadata):
        print(f"Error: Processed metadata not found: {args.processed_metadata}")
        return
    
    if not os.path.exists(args.original_csv):
        print(f"Error: Original CSV not found: {args.original_csv}")
        return
    
    if not os.path.exists(args.original_images_dir):
        print(f"Error: Original images directory not found: {args.original_images_dir}")
        return
    
    if not os.path.exists(args.original_reports_dir):
        print(f"Error: Original reports directory not found: {args.original_reports_dir}")
        return
    
    # Create ALBEF format
    split_data = create_albef_format_from_raw(
        args.processed_metadata,
        args.original_csv,
        args.original_images_dir,
        args.original_reports_dir,
        args.output_dir,
        args.max_samples
    )

if __name__ == "__main__":
    main() 