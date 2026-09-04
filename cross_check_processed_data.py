#!/usr/bin/env python3
"""
Cross-check Processed Data with Original Raw Data

This script verifies the integrity of processed MIMIC-CXR data by cross-checking
study IDs, image files, and report files between processed shards and original raw data.

Features:
- Loads processed shard data and original metadata
- Verifies study ID consistency across splits
- Checks image and report file existence
- Validates data counts and distributions
- Provides detailed cross-checking report
"""

import os
import pickle
import pandas as pd
import numpy as np
import glob
from collections import Counter
import argparse

def load_processed_metadata(metadata_path):
    """Load processed metadata from pickle file"""
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    return metadata

def load_original_metadata(csv_path):
    """Load original metadata from CSV file"""
    df = pd.read_csv(csv_path)
    return df

def extract_study_ids_from_shards(shard_dir):
    """Extract all study IDs from shard files in a directory"""
    study_ids = []
    shard_files = sorted(glob.glob(os.path.join(shard_dir, '*.pkl')))
    
    print(f"Loading study IDs from {len(shard_files)} shard files in {shard_dir}...")
    
    for shard_file in shard_files:
        with open(shard_file, 'rb') as f:
            shard_data = pickle.load(f)
        study_ids.extend(shard_data['study_ids'].tolist())
    
    return study_ids

def cross_check_data_integrity(processed_dir, original_csv_path, original_reports_dir, original_images_dir):
    """Cross-check processed data with original raw data"""
    
    print("=== MIMIC-CXR Data Cross-Checking Report ===\n")
    
    # Load metadata
    metadata_path = os.path.join(processed_dir, 'metadata.pkl')
    processed_metadata = load_processed_metadata(metadata_path)
    original_df = load_original_metadata(original_csv_path)
    
    print("1. METADATA ANALYSIS")
    print("-" * 50)
    print(f"Processed metadata keys: {list(processed_metadata.keys())}")
    print(f"Original CSV columns: {list(original_df.columns)}")
    print(f"Original total studies: {len(original_df):,}")
    print(f"Processed vocabulary size: {processed_metadata['vocab_size']:,}")
    print(f"Processed shard size: {processed_metadata['shard_size']}")
    print(f"Processed image size: {processed_metadata['image_size']}")
    print(f"Processed max sequence length: {processed_metadata['max_sequence_length']}")
    
    # Extract study IDs from processed shards
    train_dir = os.path.join(processed_dir, 'train')
    val_dir = os.path.join(processed_dir, 'val')
    test_dir = os.path.join(processed_dir, 'test')
    
    print("\n2. STUDY ID EXTRACTION FROM PROCESSED SHARDS")
    print("-" * 50)
    
    train_study_ids = extract_study_ids_from_shards(train_dir)
    val_study_ids = extract_study_ids_from_shards(val_dir)
    test_study_ids = extract_study_ids_from_shards(test_dir)
    
    print(f"Train studies in shards: {len(train_study_ids):,}")
    print(f"Validation studies in shards: {len(val_study_ids):,}")
    print(f"Test studies in shards: {len(test_study_ids):,}")
    print(f"Total studies in shards: {len(train_study_ids) + len(val_study_ids) + len(test_study_ids):,}")
    
    # Convert to sets for comparison
    processed_study_ids = set(train_study_ids + val_study_ids + test_study_ids)
    original_study_ids = set(original_df['study_id'].astype(str))
    
    print("\n3. STUDY ID CONSISTENCY CHECK")
    print("-" * 50)
    
    # Check for missing studies in processed data
    missing_in_processed = original_study_ids - processed_study_ids
    extra_in_processed = processed_study_ids - original_study_ids
    
    print(f"Studies in original but missing in processed: {len(missing_in_processed):,}")
    print(f"Studies in processed but not in original: {len(extra_in_processed):,}")
    print(f"Study ID consistency: {len(processed_study_ids & original_study_ids):,} / {len(original_study_ids):,} ({len(processed_study_ids & original_study_ids)/len(original_study_ids)*100:.2f}%)")
    
    if missing_in_processed:
        print(f"Sample missing study IDs: {list(missing_in_processed)[:10]}")
    
    if extra_in_processed:
        print(f"Sample extra study IDs: {list(extra_in_processed)[:10]}")
    
    # Check split consistency
    print("\n4. SPLIT CONSISTENCY CHECK")
    print("-" * 50)
    
    # Create mapping from original data
    original_split_map = {}
    for _, row in original_df.iterrows():
        study_id = str(row['study_id'])
        hybrid_split = row['hybrid_split']
        original_split_map[study_id] = hybrid_split
    
    # Check split consistency
    train_consistency = sum(1 for sid in train_study_ids if original_split_map.get(sid) == 'train')
    val_consistency = sum(1 for sid in val_study_ids if original_split_map.get(sid) == 'validate')
    test_consistency = sum(1 for sid in test_study_ids if original_split_map.get(sid) == 'test')
    
    print(f"Train split consistency: {train_consistency:,} / {len(train_study_ids):,} ({train_consistency/len(train_study_ids)*100:.2f}%)")
    print(f"Validation split consistency: {val_consistency:,} / {len(val_study_ids):,} ({val_consistency/len(val_study_ids)*100:.2f}%)")
    print(f"Test split consistency: {test_consistency:,} / {len(test_study_ids):,} ({test_consistency/len(test_study_ids)*100:.2f}%)")
    
    # Check file existence for a sample of studies
    print("\n5. FILE EXISTENCE CHECK (Sample)")
    print("-" * 50)
    
    sample_study_ids = list(processed_study_ids)[:20]
    missing_files = []
    
    for study_id in sample_study_ids:
        # Find corresponding row in original data
        original_row = original_df[original_df['study_id'].astype(str) == study_id]
        if len(original_row) > 0:
            row = original_row.iloc[0]
            image_file = row['image_file']
            report_file = row['report_file']
            
            image_path = os.path.join(original_images_dir, image_file)
            report_path = os.path.join(original_reports_dir, report_file)
            
            if not os.path.exists(image_path):
                missing_files.append(f"Image: {image_path}")
            if not os.path.exists(report_path):
                missing_files.append(f"Report: {report_path}")
    
    print(f"Checked {len(sample_study_ids)} sample studies")
    if missing_files:
        print(f"Missing files found: {len(missing_files)}")
        for missing in missing_files[:5]:
            print(f"  - {missing}")
    else:
        print("All sample files exist ✓")
    
    # Data distribution comparison
    print("\n6. DATA DISTRIBUTION COMPARISON")
    print("-" * 50)
    
    print("Original data distribution:")
    print(original_df['hybrid_split'].value_counts())
    
    print("\nProcessed data distribution:")
    processed_dist = {
        'train': len(train_study_ids),
        'validate': len(val_study_ids),
        'test': len(test_study_ids)
    }
    for split, count in processed_dist.items():
        print(f"{split}: {count:,}")
    
    # Calculate percentages
    total_processed = sum(processed_dist.values())
    print(f"\nProcessed percentages:")
    for split, count in processed_dist.items():
        print(f"{split}: {count/total_processed*100:.2f}%")
    
    # Summary
    print("\n7. SUMMARY")
    print("-" * 50)
    
    consistency_score = len(processed_study_ids & original_study_ids) / len(original_study_ids) * 100
    split_accuracy = (train_consistency + val_consistency + test_consistency) / total_processed * 100
    
    print(f"Overall data consistency: {consistency_score:.2f}%")
    print(f"Split accuracy: {split_accuracy:.2f}%")
    
    if consistency_score > 99 and split_accuracy > 99:
        print("✓ Data integrity check PASSED")
    else:
        print("⚠ Data integrity check FAILED - Review required")
    
    return {
        'consistency_score': consistency_score,
        'split_accuracy': split_accuracy,
        'total_processed': total_processed,
        'total_original': len(original_df),
        'missing_in_processed': len(missing_in_processed),
        'extra_in_processed': len(extra_in_processed)
    }

def main():
    parser = argparse.ArgumentParser(description="Cross-check processed MIMIC-CXR data with original raw data")
    parser.add_argument("--processed_dir", type=str, 
                       default="/home/abedin/Developments/chest_x_ray_data_processing/all_processed_data/mimic_shards_hybrid_full_ori",
                       help="Directory containing processed shards")
    parser.add_argument("--original_csv", type=str,
                       default="/home/abedin/Developments/mimic_cxr-raw-data/mimic-cxr/organized_data/metadata/processed_metadata_hybrid.csv",
                       help="Path to original metadata CSV")
    parser.add_argument("--original_reports_dir", type=str,
                       default="/home/abedin/Developments/mimic_cxr-raw-data/mimic-cxr/organized_data/reports",
                       help="Directory containing original report files")
    parser.add_argument("--original_images_dir", type=str,
                       default="/home/abedin/Developments/mimic_cxr-raw-data/mimic-cxr/organized_data/images",
                       help="Directory containing original image files")
    
    args = parser.parse_args()
    
    # Validate paths
    if not os.path.exists(args.processed_dir):
        print(f"Error: Processed directory not found: {args.processed_dir}")
        return
    
    if not os.path.exists(args.original_csv):
        print(f"Error: Original CSV not found: {args.original_csv}")
        return
    
    # Run cross-check
    results = cross_check_data_integrity(
        args.processed_dir,
        args.original_csv,
        args.original_reports_dir,
        args.original_images_dir
    )
    
    print(f"\nCross-check completed. Results saved to: cross_check_results.txt")
    
    # Save results to file
    with open('cross_check_results.txt', 'w') as f:
        f.write("MIMIC-CXR Data Cross-Check Results\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Overall data consistency: {results['consistency_score']:.2f}%\n")
        f.write(f"Split accuracy: {results['split_accuracy']:.2f}%\n")
        f.write(f"Total processed studies: {results['total_processed']:,}\n")
        f.write(f"Total original studies: {results['total_original']:,}\n")
        f.write(f"Missing in processed: {results['missing_in_processed']:,}\n")
        f.write(f"Extra in processed: {results['extra_in_processed']:,}\n")

if __name__ == "__main__":
    main() 