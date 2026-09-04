#!/usr/bin/env python3
"""
Fix text extraction to handle different report formats
"""

import pandas as pd
import re
import os

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
            in_findings = False
            
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

def test_improved_extraction():
    """Test the improved extraction on sample reports"""
    
    csv_path = "/home/abedin/Developments/mimic_cxr-raw-data/mimic-cxr/organized_data/metadata/processed_metadata_hybrid.csv"
    reports_dir = "/home/abedin/Developments/mimic_cxr-raw-data/mimic-cxr/organized_data/reports"
    
    df = pd.read_csv(csv_path)
    
    print("=== Testing Improved Text Extraction ===\n")
    
    successful_extractions = 0
    total_tested = 0
    
    for idx, row in df.head(20).iterrows():
        study_id = row['study_id']
        report_file = row['report_file']
        report_path = os.path.join(reports_dir, report_file)
        
        if os.path.exists(report_path):
            findings, impression = extract_text_from_report_improved(report_path)
            
            if findings.strip():
                successful_extractions += 1
                print(f"Study {study_id}: ✓ Found findings ({len(findings)} chars)")
                print(f"  Preview: {findings[:100]}...")
            else:
                print(f"Study {study_id}: ✗ No findings extracted")
            
            total_tested += 1
    
    print(f"\n=== Results ===")
    print(f"Total tested: {total_tested}")
    print(f"Successful extractions: {successful_extractions}")
    print(f"Success rate: {successful_extractions/total_tested*100:.1f}%")

if __name__ == "__main__":
    test_improved_extraction() 