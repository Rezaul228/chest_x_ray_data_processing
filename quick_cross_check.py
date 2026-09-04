#!/usr/bin/env python3
"""
Quick Cross-check Processed Data with Original Raw Data (Sample-based)

This script verifies the integrity of processed MIMIC-CXR data by cross-checking
a sample of study IDs between processed shards and original raw data.

Features:
- Loads processed shard data and original metadata
- Tests with a small sample for quick verification
- Verifies study ID consistency and split accuracy
- Provides quick integrity check report
"""

import os
import pickle
import pandas as pd
import numpy as np
import glob
import random
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

def extract_sample_study_ids_from_shards(shard_dir, sample_size=100):
    """Extract a sample of study IDs from shard files in a directory"""
    study_ids = []
    shard_files = sorted(glob.glob(os.path.join(shard_dir, '*.pkl')))
    
    print(f"Loading sample study IDs from {len(shard_files)} shard files in {shard_dir}...")
    
    # Take a random sample of shards
    sample_shards = random.sample(shard_files, min(5, len(shard_files)))
    
    for shard_file in sample_shards:
        with open(shard_file, 'rb') as f:
            shard_data = pickle.load(f)
        study_ids.extend(shard_data['study_ids'].tolist())
    
    # Return a sample of study IDs
    return random.sample(study_ids, min(sample_size, len(study_ids)))

def quick_cross_check(processed_dir, original_csv_path, sample_size=100):
    """Quick cross-check processed data with original raw data using samples"""
    
    print("=== Quick MIMIC-CXR Data Cross-Checking Report ===\n")
    
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
    
    # Extract sample study IDs from processed shards
    train_dir = os.path.join(processed_dir, 'train')
    val_dir = os.path.join(processed_dir, 'val')
    test_dir = os.path.join(processed_dir, 'test')
    
    print(f"\n2. SAMPLE STUDY ID EXTRACTION (Sample size: {sample_size})")
    print("-" * 50)
    
    train_sample_ids = extract_sample_study_ids_from_shards(train_dir, sample_size//3)
    val_sample_ids = extract_sample_study_ids_from_shards(val_dir, sample_size//3)
    test_sample_ids = extract_sample_study_ids_from_shards(test_dir, sample_size//3)
    
    print(f"Train sample studies: {len(train_sample_ids)}")
    print(f"Validation sample studies: {len(val_sample_ids)}")
    print(f"Test sample studies: {len(test_sample_ids)}")
    print(f"Total sample studies: {len(train_sample_ids) + len(val_sample_ids) + len(test_sample_ids)}")
    
    # Convert to sets for comparison
    processed_sample_ids = set(train_sample_ids + val_sample_ids + test_sample_ids)
    original_study_ids = set(original_df['study_id'].astype(str))
    
    print("\n3. SAMPLE STUDY ID CONSISTENCY CHECK")
    print("-" * 50)
    
    # Check for missing studies in processed data
    missing_in_processed = processed_sample_ids - original_study_ids
    found_in_original = processed_sample_ids & original_study_ids
    
    print(f"Sample studies found in original: {len(found_in_original)}")
    print(f"Sample studies missing in original: {len(missing_in_processed)}")
    print(f"Sample consistency: {len(found_in_original)} / {len(processed_sample_ids)} ({len(found_in_original)/len(processed_sample_ids)*100:.2f}%)")
    
    if missing_in_processed:
        print(f"Sample missing study IDs: {list(missing_in_processed)[:5]}")
    
    # Check split consistency for sample
    print("\n4. SAMPLE SPLIT CONSISTENCY CHECK")
    print("-" * 50)
    
    # Create mapping from original data
    original_split_map = {}
    for _, row in original_df.iterrows():
        study_id = str(row['study_id'])
        hybrid_split = row['hybrid_split']
        original_split_map[study_id] = hybrid_split
    
    # Check split consistency for sample
    train_consistency = sum(1 for sid in train_sample_ids if original_split_map.get(sid) == 'train')
    val_consistency = sum(1 for sid in val_sample_ids if original_split_map.get(sid) == 'validate')
    test_consistency = sum(1 for sid in test_sample_ids if original_split_map.get(sid) == 'test')
    
    print(f"Train sample consistency: {train_consistency} / {len(train_sample_ids)} ({train_consistency/len(train_sample_ids)*100:.2f}%)")
    print(f"Validation sample consistency: {val_consistency} / {len(val_sample_ids)} ({val_consistency/len(val_sample_ids)*100:.2f}%)")
    print(f"Test sample consistency: {test_consistency} / {len(test_sample_ids)} ({test_consistency/len(test_sample_ids)*100:.2f}%)")
    
    # Check file existence for sample studies
    print("\n5. SAMPLE FILE EXISTENCE CHECK")
    print("-" * 50)
    
    missing_files = []
    checked_count = 0
    
    for study_id in list(processed_sample_ids)[:20]:  # Check first 20
        # Find corresponding row in original data
        original_row = original_df[original_df['study_id'].astype(str) == study_id]
        if len(original_row) > 0:
            row = original_row.iloc[0]
            image_file = row['image_file']
            report_file = row['report_file']
            
            # Check if files exist (just check names, not full paths)
            if pd.isna(image_file) or pd.isna(report_file):
                missing_files.append(f"Study {study_id}: Missing file names")
            else:
                checked_count += 1
    
    print(f"Checked {checked_count} sample studies")
    if missing_files:
        print(f"Missing files found: {len(missing_files)}")
        for missing in missing_files[:3]:
            print(f"  - {missing}")
    else:
        print("All sample files have valid names ✓")
    
    # Data distribution comparison (sample)
    print("\n6. SAMPLE DATA DISTRIBUTION COMPARISON")
    print("-" * 50)
    
    print("Original data distribution (full dataset):")
    print(original_df['hybrid_split'].value_counts())
    
    print("\nProcessed sample distribution:")
    processed_dist = {
        'train': len(train_sample_ids),
        'validate': len(val_sample_ids),
        'test': len(test_sample_ids)
    }
    for split, count in processed_dist.items():
        print(f"{split}: {count}")
    
    # Calculate percentages
    total_sample = sum(processed_dist.values())
    print(f"\nSample percentages:")
    for split, count in processed_dist.items():
        print(f"{split}: {count/total_sample*100:.2f}%")
    
    # Summary
    print("\n7. QUICK SUMMARY")
    print("-" * 50)
    
    consistency_score = len(found_in_original) / len(processed_sample_ids) * 100
    split_accuracy = (train_consistency + val_consistency + test_consistency) / total_sample * 100
    
    print(f"Sample data consistency: {consistency_score:.2f}%")
    print(f"Sample split accuracy: {split_accuracy:.2f}%")
    
    if consistency_score > 95 and split_accuracy > 95:
        print("✓ Quick data integrity check PASSED")
    else:
        print("⚠ Quick data integrity check FAILED - Full check recommended")
    
    return {
        'consistency_score': consistency_score,
        'split_accuracy': split_accuracy,
        'total_sample': total_sample,
        'total_original': len(original_df),
        'missing_in_original': len(missing_in_processed)
    }

def main():
    parser = argparse.ArgumentParser(description="Quick cross-check processed MIMIC-CXR data with original raw data")
    parser.add_argument("--processed_dir", type=str, 
                       default="/home/abedin/Developments/chest_x_ray_data_processing/all_processed_data/mimic_shards_hybrid_full_ori",
                       help="Directory containing processed shards")
    parser.add_argument("--original_csv", type=str,
                       default="/home/abedin/Developments/mimic_cxr-raw-data/mimic-cxr/organized_data/metadata/processed_metadata_hybrid.csv",
                       help="Path to original metadata CSV")
    parser.add_argument("--sample_size", type=int, default=100,
                       help="Number of sample study IDs to check")
    
    args = parser.parse_args()
    
    # Validate paths
    if not os.path.exists(args.processed_dir):
        print(f"Error: Processed directory not found: {args.processed_dir}")
        return
    
    if not os.path.exists(args.original_csv):
        print(f"Error: Original CSV not found: {args.original_csv}")
        return
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Run quick cross-check
    results = quick_cross_check(
        args.processed_dir,
        args.original_csv,
        args.sample_size
    )
    
    print(f"\nQuick cross-check completed. Results saved to: quick_cross_check_results.txt")
    
    # Save results to file
    with open('quick_cross_check_results.txt', 'w') as f:
        f.write("Quick MIMIC-CXR Data Cross-Check Results\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Sample data consistency: {results['consistency_score']:.2f}%\n")
        f.write(f"Sample split accuracy: {results['split_accuracy']:.2f}%\n")
        f.write(f"Total sample studies: {results['total_sample']}\n")
        f.write(f"Total original studies: {results['total_original']:,}\n")
        f.write(f"Missing in original: {results['missing_in_original']}\n")

if __name__ == "__main__":
    main() 