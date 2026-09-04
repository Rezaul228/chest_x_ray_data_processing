#!/usr/bin/env python3
"""
Quick analysis of the data discrepancy
"""

import pandas as pd
import os
import json

def quick_analysis():
    """Quick analysis of why data counts don't match"""
    
    # Load CSV
    csv_path = "/home/abedin/Developments/mimic_cxr-raw-data/mimic-cxr/organized_data/metadata/processed_metadata_hybrid.csv"
    df = pd.read_csv(csv_path)
    
    print("=== Quick Analysis ===\n")
    print(f"Original CSV total: {len(df):,}")
    print("Original split distribution:")
    print(df['hybrid_split'].value_counts())
    print()
    
    # Check current ALBEF output
    train_json = "/home/abedin/Developments/chest_x_ray_data_processing/mimic_raw_image_text/train.json"
    test_json = "/home/abedin/Developments/chest_x_ray_data_processing/mimic_raw_image_text/test.json"
    
    if os.path.exists(train_json):
        with open(train_json, 'r') as f:
            train_data = json.load(f)
        print(f"Current train.json: {len(train_data):,} entries")
    
    if os.path.exists(test_json):
        with open(test_json, 'r') as f:
            test_data = json.load(f)
        print(f"Current test.json: {len(test_data):,} entries")
    
    # Check a few sample studies for missing files
    print("\n=== Sample File Check ===")
    missing_count = 0
    for idx, row in df.head(20).iterrows():
        study_id = row['study_id']
        image_file = row['image_file']
        report_file = row['report_file']
        
        img_path = f"/home/abedin/Developments/mimic_cxr-raw-data/mimic-cxr/organized_data/images/{image_file}"
        report_path = f"/home/abedin/Developments/mimic_cxr-raw-data/mimic-cxr/organized_data/reports/{report_file}"
        
        img_exists = os.path.exists(img_path)
        report_exists = os.path.exists(report_path)
        
        if not img_exists or not report_exists:
            missing_count += 1
            print(f"Study {study_id}: img={img_exists}, report={report_exists}")
    
    print(f"Missing files in first 20 studies: {missing_count}")
    
    # Check if this is a filtering issue
    print("\n=== Expected vs Actual ===")
    expected_train = len(df[df['hybrid_split'] == 'train'])
    expected_test = len(df[df['hybrid_split'] == 'test'])
    expected_val = len(df[df['hybrid_split'] == 'validate'])
    
    print(f"Train: Expected {expected_train:,}, Got {len(train_data):,}, Missing {expected_train - len(train_data):,}")
    print(f"Test:  Expected {expected_test:,}, Got {len(test_data):,}, Missing {expected_test - len(test_data):,}")
    print(f"Val:   Expected {expected_val:,}, Got 0, Missing {expected_val}")

if __name__ == "__main__":
    quick_analysis() 