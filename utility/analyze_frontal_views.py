#!/usr/bin/env python3

import os
import json
import random
from collections import Counter

def analyze_frontal_views():
    """Analyze frontal view (PA + AP) percentages in the dataset"""
    
    metadata_dir = "raw_data_ReXGradient-160K/organized_data/metadata"
    
    print("="*80)
    print("Frontal View Analysis - ReXGradient Dataset")
    print("="*80)
    
    if not os.path.exists(metadata_dir):
        print(f"Error: Metadata directory not found at {metadata_dir}")
        return
    
    # Get all metadata files
    metadata_files = [f for f in os.listdir(metadata_dir) if f.endswith('.json')]
    print(f"Total metadata files: {len(metadata_files)}")
    
    # Sample files for analysis
    sample_size = min(10000, len(metadata_files))  # Larger sample for better accuracy
    sample_files = random.sample(metadata_files, sample_size)
    print(f"Analyzing {sample_size} sample files...")
    
    # Counters for different view types
    frontal_views = 0  # PA or AP
    lateral_views = 0
    portable_views = 0
    two_view_studies = 0
    single_view_studies = 0
    
    # Detailed breakdown
    pa_only = 0
    ap_only = 0
    pa_lateral = 0
    ap_lateral = 0
    lateral_only = 0
    unclear_views = 0
    
    study_descriptions = []
    
    for i, filename in enumerate(sample_files):
        if i % 2000 == 0:
            print(f"Processed {i}/{sample_size} files...")
            
        filepath = os.path.join(metadata_dir, filename)
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            study_desc = data.get('study_description', '').lower()
            findings = data.get('findings', '').lower()
            impression = data.get('impression', '').lower()
            
            study_descriptions.append(data.get('study_description', ''))
            
            # Check for view types in study description
            has_pa = 'pa' in study_desc
            has_ap = 'ap' in study_desc
            has_lateral = 'lateral' in study_desc
            has_port = 'port' in study_desc
            has_2v = '2v' in study_desc or '2 view' in study_desc
            has_1v = '1v' in study_desc or '1 view' in study_desc or 'single' in study_desc
            
            # Check for view types in clinical text
            text_has_pa = 'pa' in findings or 'pa' in impression
            text_has_ap = 'ap' in findings or 'ap' in impression
            text_has_lateral = 'lateral' in findings or 'lateral' in impression
            
            # Determine view classification
            is_frontal = has_pa or has_ap or text_has_pa or text_has_ap
            is_lateral = has_lateral or text_has_lateral
            is_portable = has_port
            is_two_view = has_2v
            is_single_view = has_1v
            
            # Count frontal views (PA or AP)
            if is_frontal:
                frontal_views += 1
                
                # Detailed breakdown
                if has_pa and has_lateral:
                    pa_lateral += 1
                elif has_ap and has_lateral:
                    ap_lateral += 1
                elif has_pa or text_has_pa:
                    pa_only += 1
                elif has_ap or text_has_ap:
                    ap_only += 1
            
            if is_lateral:
                lateral_views += 1
                if not is_frontal:
                    lateral_only += 1
                    
            if is_portable:
                portable_views += 1
                
            if is_two_view:
                two_view_studies += 1
                
            if is_single_view:
                single_view_studies += 1
                
            if not is_frontal and not is_lateral:
                unclear_views += 1
                
        except Exception as e:
            print(f"Error reading {filename}: {e}")
    
    print(f"\nAnalysis complete! Processed {len(sample_files)} files.")
    
    # Calculate percentages
    total = len(sample_files)
    frontal_percentage = (frontal_views / total) * 100
    lateral_percentage = (lateral_views / total) * 100
    portable_percentage = (portable_views / total) * 100
    two_view_percentage = (two_view_studies / total) * 100
    single_view_percentage = (single_view_studies / total) * 100
    unclear_percentage = (unclear_views / total) * 100
    
    print("\n" + "="*60)
    print("FRONTAL VIEW ANALYSIS RESULTS")
    print("="*60)
    
    print(f"Total samples analyzed: {total}")
    print(f"Frontal views (PA or AP): {frontal_views} ({frontal_percentage:.1f}%)")
    print(f"Lateral views: {lateral_views} ({lateral_percentage:.1f}%)")
    print(f"Portable studies: {portable_views} ({portable_percentage:.1f}%)")
    print(f"Two-view studies: {two_view_studies} ({two_view_percentage:.1f}%)")
    print(f"Single-view studies: {single_view_studies} ({single_view_percentage:.1f}%)")
    print(f"Unclear view types: {unclear_views} ({unclear_percentage:.1f}%)")
    
    print("\n" + "="*60)
    print("DETAILED FRONTAL VIEW BREAKDOWN")
    print("="*60)
    
    print(f"PA only: {pa_only} ({pa_only/total*100:.1f}%)")
    print(f"AP only: {ap_only} ({ap_only/total*100:.1f}%)")
    print(f"PA + Lateral: {pa_lateral} ({pa_lateral/total*100:.1f}%)")
    print(f"AP + Lateral: {ap_lateral} ({ap_lateral/total*100:.1f}%)")
    print(f"Lateral only: {lateral_only} ({lateral_only/total*100:.1f}%)")
    
    print("\n" + "="*60)
    print("STUDY DESCRIPTION ANALYSIS")
    print("="*60)
    
    desc_counter = Counter(study_descriptions)
    print("Top 15 study descriptions:")
    for desc, count in desc_counter.most_common(15):
        percentage = (count / total) * 100
        print(f"  {count:4d} ({percentage:4.1f}%): '{desc}'")
    
    print("\n" + "="*60)
    print("RECOMMENDATIONS FOR FRONTAL VIEW EXTRACTION")
    print("="*60)
    
    print(f"✅ You can extract {frontal_percentage:.1f}% of the dataset as frontal views")
    print(f"   - This includes both PA and AP orientations")
    print(f"   - Total frontal views: {frontal_views:,} out of {total:,} samples")
    
    print(f"\n📊 View Distribution:")
    print(f"   - Two-view studies (likely PA + Lateral): {two_view_percentage:.1f}%")
    print(f"   - Single-view studies (likely AP): {single_view_percentage:.1f}%")
    print(f"   - Portable studies (usually AP): {portable_percentage:.1f}%")
    
    print(f"\n🎯 Best strategies for frontal view extraction:")
    print(f"   1. Focus on 'DG CHEST 2V' studies ({desc_counter.get('DG CHEST 2V', 0)/total*100:.1f}%)")
    print(f"   2. Include 'DG CHEST 1V PORT' studies ({desc_counter.get('DG CHEST 1V PORT', 0)/total*100:.1f}%)")
    print(f"   3. Include 'Chest Single AP view' studies ({desc_counter.get('Chest Single AP view', 0)/total*100:.1f}%)")
    
    print(f"\n⚠️  Note: {unclear_percentage:.1f}% of studies have unclear view information")
    print(f"   - These may need manual review or additional criteria")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    analyze_frontal_views() 