#!/usr/bin/env python3
"""
Comprehensive analysis of MIMIC-CXR report content
"""

import pandas as pd
import os
import re
from tqdm import tqdm

def analyze_report_content():
    """Analyze all report content and cross-check with raw data"""
    
    # Paths
    raw_data_dir = "/home/abedin/Developments/mimic_cxr-raw-data/mimic-cxr"
    metadata_path = os.path.join(raw_data_dir, "organized_data/metadata/processed_metadata.csv")
    reports_dir = os.path.join(raw_data_dir, "organized_data/reports")
    original_reports_dir = os.path.join(raw_data_dir, "mimic-cxr-reports/files")
    
    print("=== Comprehensive Report Content Analysis ===")
    print(f"Metadata: {metadata_path}")
    print(f"Organized reports: {reports_dir}")
    print(f"Original reports: {original_reports_dir}")
    
    # Load metadata
    print("\nLoading metadata...")
    metadata_df = pd.read_csv(metadata_path)
    print(f"Total studies in metadata: {len(metadata_df)}")
    
    # Analyze organized reports
    print("\n=== Analyzing Organized Reports ===")
    stats = {
        'total': 0,
        'has_findings': 0,
        'has_impression': 0,
        'has_both': 0,
        'has_neither': 0,
        'missing_files': 0,
        'error_files': 0
    }
    
    findings_patterns = [
        r'findings?:?\s*\n',
        r'finding:?\s*\n',
        r'findings?:?\s*[A-Z]',
        r'finding:?\s*[A-Z]'
    ]
    
    impression_patterns = [
        r'impression:?\s*\n',
        r'impression:?\s*[A-Z]',
        r'impression:?\s*[0-9]'
    ]
    
    for _, row in tqdm(metadata_df.iterrows(), desc="Analyzing organized reports"):
        study_id = row['study_id']
        report_path = os.path.join(reports_dir, f"{study_id}.txt")
        
        stats['total'] += 1
        
        if not os.path.exists(report_path):
            stats['missing_files'] += 1
            continue
            
        try:
            with open(report_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read().lower()
            
            # Check for findings
            has_findings = any(re.search(pattern, content) for pattern in findings_patterns)
            
            # Check for impression
            has_impression = any(re.search(pattern, content) for pattern in impression_patterns)
            
            if has_findings and has_impression:
                stats['has_both'] += 1
            elif has_findings:
                stats['has_findings'] += 1
            elif has_impression:
                stats['has_impression'] += 1
            else:
                stats['has_neither'] += 1
                
        except Exception as e:
            stats['error_files'] += 1
            print(f"Error reading {study_id}: {e}")
    
    print(f"\n📊 Organized Reports Statistics:")
    print(f"Total studies: {stats['total']}")
    print(f"✅ Has both findings & impression: {stats['has_both']} ({stats['has_both']/stats['total']*100:.1f}%)")
    print(f"📝 Has findings only: {stats['has_findings']} ({stats['has_findings']/stats['total']*100:.1f}%)")
    print(f"💭 Has impression only: {stats['has_impression']} ({stats['has_impression']/stats['total']*100:.1f}%)")
    print(f"❌ Has neither: {stats['has_neither']} ({stats['has_neither']/stats['total']*100:.1f}%)")
    print(f"⚠️  Missing files: {stats['missing_files']}")
    print(f"❌ Error files: {stats['error_files']}")
    
    # Cross-check with original reports
    print(f"\n=== Cross-checking with Original Reports ===")
    original_stats = {
        'total': 0,
        'has_findings': 0,
        'has_impression': 0,
        'has_both': 0,
        'has_neither': 0,
        'missing_files': 0,
        'error_files': 0
    }
    
    # Sample first 1000 for original reports check
    sample_df = metadata_df.head(1000)
    
    for _, row in tqdm(sample_df.iterrows(), desc="Checking original reports"):
        study_id = row['study_id']
        subject_id = row['subject_id']
        
        # Original path: mimic-cxr-reports/files/p{subject_id[:2]}/p{subject_id}/s{study_id}.txt
        patient_dir = f"p{str(subject_id)[:2]}"
        original_report_path = os.path.join(original_reports_dir, patient_dir, f"p{subject_id}", f"s{study_id}.txt")
        
        original_stats['total'] += 1
        
        if not os.path.exists(original_report_path):
            original_stats['missing_files'] += 1
            continue
            
        try:
            with open(original_report_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read().lower()
            
            # Check for findings
            has_findings = any(re.search(pattern, content) for pattern in findings_patterns)
            
            # Check for impression
            has_impression = any(re.search(pattern, content) for pattern in impression_patterns)
            
            if has_findings and has_impression:
                original_stats['has_both'] += 1
            elif has_findings:
                original_stats['has_findings'] += 1
            elif has_impression:
                original_stats['has_impression'] += 1
            else:
                original_stats['has_neither'] += 1
                
        except Exception as e:
            original_stats['error_files'] += 1
            print(f"Error reading original {study_id}: {e}")
    
    print(f"\n📊 Original Reports Statistics (Sample of 1000):")
    print(f"Total studies: {original_stats['total']}")
    print(f"✅ Has both findings & impression: {original_stats['has_both']} ({original_stats['has_both']/original_stats['total']*100:.1f}%)")
    print(f"📝 Has findings only: {original_stats['has_findings']} ({original_stats['has_findings']/original_stats['total']*100:.1f}%)")
    print(f"💭 Has impression only: {original_stats['has_impression']} ({original_stats['has_impression']/original_stats['total']*100:.1f}%)")
    print(f"❌ Has neither: {original_stats['has_neither']} ({original_stats['has_neither']/original_stats['total']*100:.1f}%)")
    print(f"⚠️  Missing files: {original_stats['missing_files']}")
    print(f"❌ Error files: {original_stats['error_files']}")
    
    # Summary
    print(f"\n=== SUMMARY ===")
    print(f"Organized reports - Processable: {stats['has_both']} ({stats['has_both']/stats['total']*100:.1f}%)")
    print(f"Original reports - Processable: {original_stats['has_both']} ({original_stats['has_both']/original_stats['total']*100:.1f}%)")
    print(f"Expected final processed: ~{stats['has_both']} studies")

if __name__ == "__main__":
    analyze_report_content() 