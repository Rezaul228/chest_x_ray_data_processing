#!/usr/bin/env python3
"""
Test the improved ALBEF logic with MIMIC data loader approach
"""

import pandas as pd
import os
import re

def extract_text_from_report_improved(report_path):
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

def test_improved_albef_logic():
    """Test the improved ALBEF logic"""
    
    csv_path = "/home/abedin/Developments/mimic_cxr-raw-data/mimic-cxr/organized_data/metadata/processed_metadata_hybrid.csv"
    reports_dir = "/home/abedin/Developments/mimic_cxr-raw-data/mimic-cxr/organized_data/reports"
    
    df = pd.read_csv(csv_path)
    
    print("=== Testing Improved ALBEF Logic ===\n")
    
    # Test the flexible filtering logic
    total_tested = 0
    accepted_studies = 0
    findings_only = 0
    impression_only = 0
    both_found = 0
    neither_found = 0
    
    for idx, row in df.head(50).iterrows():
        study_id = row['study_id']
        report_file = row['report_file']
        report_path = os.path.join(reports_dir, report_file)
        
        if os.path.exists(report_path):
            findings, impression = extract_text_from_report_improved(report_path)
            
            # Test the flexible filtering logic (like MIMIC data loader)
            has_findings = bool(findings.strip())
            has_impression = bool(impression.strip())
            
            if has_findings and has_impression:
                both_found += 1
                accepted_studies += 1
                print(f"Study {study_id}: ✓ Both findings and impression")
            elif has_findings:
                findings_only += 1
                accepted_studies += 1
                print(f"Study {study_id}: ✓ Findings only")
            elif has_impression:
                impression_only += 1
                accepted_studies += 1
                print(f"Study {study_id}: ✓ Impression only")
            else:
                neither_found += 1
                print(f"Study {study_id}: ✗ Neither found")
            
            total_tested += 1
    
    print(f"\n=== Results ===")
    print(f"Total tested: {total_tested}")
    print(f"Accepted studies: {accepted_studies}")
    print(f"  - Both findings and impression: {both_found}")
    print(f"  - Findings only: {findings_only}")
    print(f"  - Impression only: {impression_only}")
    print(f"Rejected studies: {neither_found}")
    print(f"Acceptance rate: {accepted_studies/total_tested*100:.1f}%")
    
    # Compare with expected counts
    expected_train = len(df[df['hybrid_split'] == 'train'])
    expected_test = len(df[df['hybrid_split'] == 'test'])
    expected_val = len(df[df['hybrid_split'] == 'validate'])
    
    print(f"\n=== Expected vs Projected ===")
    print(f"Original CSV counts:")
    print(f"  Train: {expected_train:,}")
    print(f"  Test: {expected_test:,}")
    print(f"  Val: {expected_val:,}")
    print(f"  Total: {len(df):,}")
    
    # Project expected ALBEF counts based on acceptance rate
    projected_total = int(len(df) * (accepted_studies / total_tested))
    projected_train = int(expected_train * (accepted_studies / total_tested))
    projected_test = int(expected_test * (accepted_studies / total_tested))
    projected_val = int(expected_val * (accepted_studies / total_tested))
    
    print(f"\nProjected ALBEF counts (based on {accepted_studies/total_tested*100:.1f}% acceptance):")
    print(f"  Train: ~{projected_train:,}")
    print(f"  Test: ~{projected_test:,}")
    print(f"  Val: ~{projected_val:,}")
    print(f"  Total: ~{projected_total:,}")

if __name__ == "__main__":
    test_improved_albef_logic() 