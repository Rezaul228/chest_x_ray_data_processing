#!/usr/bin/env python3
"""
Check for Data Leakage in ALBEF Format

This script checks if any data appears in multiple splits, which would cause data leakage.
It examines:
1. Study IDs across splits
2. Image files across splits  
3. Text content similarity across splits
4. Patient IDs (if available) across splits
"""

import os
import json
import hashlib
from collections import defaultdict
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse

def load_json_data(json_path):
    """Load JSON data and return as list of dictionaries"""
    with open(json_path, 'r') as f:
        return json.load(f)

def check_study_id_leakage(split_data):
    """Check if any study IDs appear in multiple splits"""
    print("🔍 Checking Study ID Leakage...")
    
    study_id_splits = defaultdict(list)
    
    for split_name, data in split_data.items():
        for entry in data:
            study_id = entry.get('study_id', 'unknown')
            study_id_splits[study_id].append(split_name)
    
    # Find study IDs that appear in multiple splits
    leaked_study_ids = {study_id: splits for study_id, splits in study_id_splits.items() 
                       if len(splits) > 1}
    
    if leaked_study_ids:
        print(f"❌ DATA LEAKAGE DETECTED!")
        print(f"   {len(leaked_study_ids)} study IDs appear in multiple splits:")
        for study_id, splits in list(leaked_study_ids.items())[:10]:  # Show first 10
            print(f"     Study {study_id}: {splits}")
        if len(leaked_study_ids) > 10:
            print(f"     ... and {len(leaked_study_ids) - 10} more")
        return True
    else:
        print("✅ No study ID leakage detected")
        return False

def check_image_file_leakage(split_data):
    """Check if any image files appear in multiple splits"""
    print("\n🔍 Checking Image File Leakage...")
    
    image_splits = defaultdict(list)
    
    for split_name, data in split_data.items():
        for entry in data:
            image_path = entry.get('image', 'unknown')
            image_splits[image_path].append(split_name)
    
    # Find images that appear in multiple splits
    leaked_images = {image: splits for image, splits in image_splits.items() 
                    if len(splits) > 1}
    
    if leaked_images:
        print(f"❌ IMAGE LEAKAGE DETECTED!")
        print(f"   {len(leaked_images)} image files appear in multiple splits:")
        for image, splits in list(leaked_images.items())[:10]:  # Show first 10
            print(f"     {image}: {splits}")
        if len(leaked_images) > 10:
            print(f"     ... and {len(leaked_images) - 10} more")
        return True
    else:
        print("✅ No image file leakage detected")
        return False

def check_text_content_leakage(split_data, similarity_threshold=0.9):
    """Check for similar text content across splits"""
    print("\n🔍 Checking Text Content Leakage...")
    
    # Create text hashes for each split
    split_text_hashes = {}
    
    for split_name, data in split_data.items():
        text_hashes = set()
        for entry in data:
            findings = entry.get('findings', '').strip()
            impression = entry.get('impression', '').strip()
            
            # Create hash of combined text
            combined_text = f"{findings} {impression}".strip()
            if combined_text:
                text_hash = hashlib.md5(combined_text.encode()).hexdigest()
                text_hashes.add(text_hash)
        
        split_text_hashes[split_name] = text_hashes
    
    # Check for overlapping text hashes
    all_splits = list(split_text_hashes.keys())
    leaked_texts = []
    
    for i, split1 in enumerate(all_splits):
        for split2 in all_splits[i+1:]:
            overlap = split_text_hashes[split1] & split_text_hashes[split2]
            if overlap:
                leaked_texts.append((split1, split2, len(overlap)))
    
    if leaked_texts:
        print(f"❌ TEXT CONTENT LEAKAGE DETECTED!")
        print(f"   Text content appears in multiple splits:")
        for split1, split2, count in leaked_texts:
            print(f"     {split1} ↔ {split2}: {count} identical texts")
        return True
    else:
        print("✅ No text content leakage detected")
        return False

def check_original_csv_integrity(original_csv_path, split_data):
    """Check if the original CSV split assignments are respected"""
    print("\n🔍 Checking Original CSV Split Integrity...")
    
    # Load original CSV
    df = pd.read_csv(original_csv_path)
    print(f"   Original CSV has {len(df):,} studies")
    
    # Create mapping from study_id to original split
    original_splits = {}
    for _, row in df.iterrows():
        study_id = str(row['study_id'])
        original_split = row['hybrid_split']
        original_splits[study_id] = original_split
    
    # Check if our ALBEF splits respect the original assignments
    split_mapping = {'train': 'train', 'validate': 'val', 'test': 'test'}
    violations = []
    
    for albef_split, data in split_data.items():
        for entry in data:
            study_id = entry.get('study_id', 'unknown')
            if study_id in original_splits:
                original_split = original_splits[study_id]
                expected_albef_split = split_mapping.get(original_split, original_split)
                
                if albef_split != expected_albef_split:
                    violations.append((study_id, original_split, albef_split))
    
    if violations:
        print(f"❌ SPLIT ASSIGNMENT VIOLATIONS DETECTED!")
        print(f"   {len(violations)} studies are in wrong splits:")
        for study_id, original, albef in violations[:10]:
            print(f"     Study {study_id}: {original} → {albef}")
        if len(violations) > 10:
            print(f"     ... and {len(violations) - 10} more")
        return True
    else:
        print("✅ All split assignments respect original CSV")
        return False

def check_patient_level_leakage(original_csv_path, split_data):
    """Check for patient-level leakage (if patient_id is available)"""
    print("\n🔍 Checking Patient-Level Leakage...")
    
    # Load original CSV to get patient_id information
    df = pd.read_csv(original_csv_path)
    
    # Check if patient_id column exists
    if 'patient_id' not in df.columns:
        print("   ⚠️  No patient_id column found in CSV - skipping patient-level check")
        return False
    
    # Create mapping from study_id to patient_id
    study_to_patient = {}
    for _, row in df.iterrows():
        study_id = str(row['study_id'])
        patient_id = str(row['patient_id'])
        study_to_patient[study_id] = patient_id
    
    # Check for patient-level leakage
    patient_splits = defaultdict(list)
    
    for split_name, data in split_data.items():
        for entry in data:
            study_id = entry.get('study_id', 'unknown')
            if study_id in study_to_patient:
                patient_id = study_to_patient[study_id]
                patient_splits[patient_id].append(split_name)
    
    # Find patients that appear in multiple splits
    leaked_patients = {patient_id: splits for patient_id, splits in patient_splits.items() 
                      if len(splits) > 1}
    
    if leaked_patients:
        print(f"❌ PATIENT-LEVEL LEAKAGE DETECTED!")
        print(f"   {len(leaked_patients)} patients appear in multiple splits:")
        for patient_id, splits in list(leaked_patients.items())[:10]:
            print(f"     Patient {patient_id}: {splits}")
        if len(leaked_patients) > 10:
            print(f"     ... and {len(leaked_patients) - 10} more")
        return True
    else:
        print("✅ No patient-level leakage detected")
        return False

def main():
    parser = argparse.ArgumentParser(description="Check for data leakage in ALBEF format")
    parser.add_argument("--albef_dir", type=str, 
                       default="/home/abedin/Developments/chest_x_ray_data_processing/mimic_raw_image_text",
                       help="Directory containing ALBEF format data")
    parser.add_argument("--original_csv", type=str,
                       default="/home/abedin/Developments/mimic_cxr-raw-data/mimic-cxr/organized_data/metadata/processed_metadata_hybrid.csv",
                       help="Path to original metadata CSV")
    
    args = parser.parse_args()
    
    print("=== Data Leakage Check for ALBEF Format ===\n")
    
    # Load all split data
    split_data = {}
    splits = ['train', 'val', 'test']
    
    for split in splits:
        json_path = os.path.join(args.albef_dir, f'{split}.json')
        if os.path.exists(json_path):
            split_data[split] = load_json_data(json_path)
            print(f"📊 Loaded {split}: {len(split_data[split])} samples")
        else:
            print(f"⚠️  {split}.json not found")
    
    if not split_data:
        print("❌ No split data found!")
        return
    
    # Perform all leakage checks
    leakage_detected = False
    
    # 1. Study ID leakage
    if check_study_id_leakage(split_data):
        leakage_detected = True
    
    # 2. Image file leakage
    if check_image_file_leakage(split_data):
        leakage_detected = True
    
    # 3. Text content leakage
    if check_text_content_leakage(split_data):
        leakage_detected = True
    
    # 4. Original CSV integrity
    if os.path.exists(args.original_csv):
        if check_original_csv_integrity(args.original_csv, split_data):
            leakage_detected = True
    
    # 5. Patient-level leakage
    if os.path.exists(args.original_csv):
        if check_patient_level_leakage(args.original_csv, split_data):
            leakage_detected = True
    
    # Summary
    print("\n" + "="*60)
    if leakage_detected:
        print("❌ DATA LEAKAGE DETECTED - DO NOT USE FOR TRAINING!")
        print("   Please fix the leakage issues before proceeding.")
    else:
        print("✅ NO DATA LEAKAGE DETECTED - SAFE FOR TRAINING!")
        print("   The dataset is properly split and ready for use.")
    print("="*60)

if __name__ == "__main__":
    main() 