#!/usr/bin/env python3
"""
Demonstration of Cross-Checking Processed Data with Original Raw Data
"""

import pickle
import pandas as pd
import glob
import os

def demo_cross_check():
    print("=== Cross-Checking Demonstration ===\n")
    
    # Load a sample shard
    shard_file = glob.glob('/home/abedin/Developments/chest_x_ray_data_processing/all_processed_data/mimic_shards_hybrid_full_ori/train/*.pkl')[0]
    shard_data = pickle.load(open(shard_file, 'rb'))
    sample_study_ids = shard_data['study_ids'][:5]
    
    print("Sample study IDs from processed shards:")
    print(sample_study_ids)
    print()
    
    # Load original CSV
    df = pd.read_csv('/home/abedin/Developments/mimic_cxr-raw-data/mimic-cxr/organized_data/metadata/processed_metadata_hybrid.csv')
    
    print("Checking these study IDs in original CSV:")
    print("-" * 80)
    
    for study_id in sample_study_ids:
        row = df[df['study_id'].astype(str) == study_id]
        if len(row) > 0:
            r = row.iloc[0]
            print(f"Study {study_id}: ✓ FOUND")
            print(f"  - Split: {r['hybrid_split']}")
            print(f"  - Image file: {r['image_file']}")
            print(f"  - Report file: {r['report_file']}")
            print(f"  - Subject ID: {r['subject_id']}")
        else:
            print(f"Study {study_id}: ✗ NOT FOUND")
        print()
    
    # Show metadata information
    print("Metadata Information:")
    print("-" * 80)
    metadata_path = '/home/abedin/Developments/chest_x_ray_data_processing/all_processed_data/mimic_shards_hybrid_full_ori/metadata.pkl'
    metadata = pickle.load(open(metadata_path, 'rb'))
    
    print(f"Total shards: {metadata['total_shards']}")
    print(f"Train shards: {metadata['num_train_shards']}")
    print(f"Validation shards: {metadata['num_val_shards']}")
    print(f"Test shards: {metadata['num_test_shards']}")
    print(f"Shard size: {metadata['shard_size']}")
    print(f"Vocabulary size: {metadata['vocab_size']}")
    
    # Calculate total studies
    total_studies = (metadata['num_train_shards'] + metadata['num_val_shards'] + metadata['num_test_shards']) * metadata['shard_size']
    print(f"Estimated total studies: {total_studies:,}")
    print(f"Original CSV studies: {len(df):,}")

if __name__ == "__main__":
    demo_cross_check() 