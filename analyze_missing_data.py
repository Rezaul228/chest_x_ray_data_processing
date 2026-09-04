#!/usr/bin/env python3
"""
Analyze why some studies are missing from ALBEF output
"""

import pandas as pd
import os
import json

def analyze_missing_data():
    """Analyze why some studies are missing from ALBEF output"""
    
    # Paths
    csv_path = "/home/abedin/Developments/mimic_cxr-raw-data/mimic-cxr/organized_data/metadata/processed_metadata_hybrid.csv"
    images_dir = "/home/abedin/Developments/mimic_cxr-raw-data/mimic-cxr/organized_data/images"
    reports_dir = "/home/abedin/Developments/mimic_cxr-raw-data/mimic-cxr/organized_data/reports"
    
    print("=== Analyzing Missing Data ===\n")
    
    # Load CSV
    df = pd.read_csv(csv_path)
    print(f"Original CSV total studies: {len(df):,}")
    print("Original split distribution:")
    print(df['hybrid_split'].value_counts())
    print()
    
    # Check for missing files
    missing_image = 0
    missing_report = 0
    missing_both = 0
    empty_findings = 0
    
    for idx, row in df.iterrows():
        study_id = row['study_id']
        image_file = row['image_file']
        report_file = row['report_file']
        
        # Check if files exist
        img_path = os.path.join(images_dir, image_file)
        report_path = os.path.join(reports_dir, report_file)
        
        img_exists = os.path.exists(img_path)
        report_exists = os.path.exists(report_path)
        
        if not img_exists:
            missing_image += 1
        if not report_exists:
            missing_report += 1
        if not img_exists and not report_exists:
            missing_both += 1
            
        # Check for empty findings (this would cause filtering)
        if report_exists:
            try:
                with open(report_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check if findings section is empty
                import re
                findings_match = re.search(r'FINDINGS:(.*?)(?:IMPRESSION:|$)', content, re.DOTALL | re.IGNORECASE)
                if findings_match:
                    findings = findings_match.group(1).strip()
                    if not findings:
                        empty_findings += 1
                else:
                    empty_findings += 1
            except:
                empty_findings += 1
    
    print("=== File Availability Analysis ===")
    print(f"Missing image files: {missing_image:,}")
    print(f"Missing report files: {missing_report:,}")
    print(f"Missing both files: {missing_both:,}")
    print(f"Studies with empty findings: {empty_findings:,}")
    print()
    
    # Check current ALBEF output
    train_json = "/home/abedin/Developments/chest_x_ray_data_processing/mimic_raw_image_text/train.json"
    test_json = "/home/abedin/Developments/chest_x_ray_data_processing/mimic_raw_image_text/test.json"
    
    if os.path.exists(train_json):
        with open(train_json, 'r') as f:
            train_data = json.load(f)
        print(f"Current train.json entries: {len(train_data):,}")
    
    if os.path.exists(test_json):
        with open(test_json, 'r') as f:
            test_data = json.load(f)
        print(f"Current test.json entries: {len(test_data):,}")
    
    # Calculate expected vs actual
    expected_train = len(df[df['hybrid_split'] == 'train'])
    expected_test = len(df[df['hybrid_split'] == 'test'])
    expected_val = len(df[df['hybrid_split'] == 'validate'])
    
    print("\n=== Expected vs Actual ===")
    print(f"Train: Expected {expected_train:,}, Got {len(train_data):,}, Missing {expected_train - len(train_data):,}")
    print(f"Test:  Expected {expected_test:,}, Got {len(test_data):,}, Missing {expected_test - len(test_data):,}")
    print(f"Val:   Expected {expected_val:,}, Got 0, Missing {expected_val}")
    
    # Sample some missing studies
    print("\n=== Sample Missing Studies ===")
    processed_study_ids = set()
    if os.path.exists(train_json):
        with open(train_json, 'r') as f:
            train_data = json.load(f)
        processed_study_ids.update([entry['study_id'] for entry in train_data])
    
    if os.path.exists(test_json):
        with open(test_json, 'r') as f:
            test_data = json.load(f)
        processed_study_ids.update([entry['study_id'] for entry in test_data])
    
    missing_studies = []
    for idx, row in df.iterrows():
        if str(row['study_id']) not in processed_study_ids:
            missing_studies.append(row)
            if len(missing_studies) >= 5:
                break
    
    for study in missing_studies:
        print(f"Study {study['study_id']} ({study['hybrid_split']}):")
        img_path = os.path.join(images_dir, study['image_file'])
        report_path = os.path.join(reports_dir, study['report_file'])
        print(f"  Image exists: {os.path.exists(img_path)}")
        print(f"  Report exists: {os.path.exists(report_path)}")

if __name__ == "__main__":
    analyze_missing_data() 