#!/usr/bin/env python3
"""
Check why findings extraction is filtering out studies
"""

import pandas as pd
import re
import os

def check_findings():
    """Check why findings extraction is filtering out studies"""
    
    csv_path = "/home/abedin/Developments/mimic_cxr-raw-data/mimic-cxr/organized_data/metadata/processed_metadata_hybrid.csv"
    reports_dir = "/home/abedin/Developments/mimic_cxr-raw-data/mimic-cxr/organized_data/reports"
    
    df = pd.read_csv(csv_path)
    
    print("=== Checking Findings Extraction ===\n")
    
    empty_findings = 0
    no_findings_section = 0
    total_checked = 0
    
    for idx, row in df.head(100).iterrows():
        study_id = row['study_id']
        report_file = row['report_file']
        report_path = os.path.join(reports_dir, report_file)
        
        if os.path.exists(report_path):
            try:
                with open(report_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check for findings section
                findings_match = re.search(r'FINDINGS:(.*?)(?:IMPRESSION:|$)', content, re.DOTALL | re.IGNORECASE)
                
                if findings_match:
                    findings = findings_match.group(1).strip()
                    if not findings:
                        empty_findings += 1
                        print(f"Study {study_id}: Empty findings section")
                else:
                    no_findings_section += 1
                    print(f"Study {study_id}: No findings section found")
                    # Show first 200 chars of content
                    print(f"  Content preview: {content[:200]}...")
                
                total_checked += 1
                
            except Exception as e:
                print(f"Study {study_id}: Error reading file - {e}")
        else:
            print(f"Study {study_id}: Report file not found")
    
    print(f"\n=== Summary ===")
    print(f"Total checked: {total_checked}")
    print(f"Empty findings: {empty_findings}")
    print(f"No findings section: {no_findings_section}")
    print(f"Would be filtered out: {empty_findings + no_findings_section}")

if __name__ == "__main__":
    check_findings() 