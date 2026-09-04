#!/usr/bin/env python3

import os
import json
import re
from collections import Counter
import random

def analyze_chest_views():
    """Analyze chest X-ray view information from metadata files"""
    
    metadata_dir = "raw_data_ReXGradient-160K/organized_data/metadata"
    
    print("="*80)
    print("Chest X-Ray View Analysis - ReXGradient Dataset")
    print("="*80)
    
    if not os.path.exists(metadata_dir):
        print(f"Error: Metadata directory not found at {metadata_dir}")
        return
    
    # Get all metadata files
    metadata_files = [f for f in os.listdir(metadata_dir) if f.endswith('.json')]
    print(f"Total metadata files: {len(metadata_files)}")
    
    # Sample files for analysis (to avoid taking too long)
    sample_size = min(1000, len(metadata_files))
    sample_files = random.sample(metadata_files, sample_size)
    print(f"Analyzing {sample_size} sample files...")
    
    # Collect study descriptions
    study_descriptions = []
    view_info = []
    
    # Keywords to look for view information
    view_keywords = {
        'PA': ['PA', 'posteroanterior', 'posterior-anterior'],
        'AP': ['AP', 'anteroposterior', 'anterior-posterior'],
        'Lateral': ['lateral', 'lat', 'side'],
        'Frontal': ['frontal', 'front'],
        'Portable': ['port', 'portable', 'PORT'],
        'Single': ['single', '1v', '1 view', '1V'],
        'Two': ['2v', '2 view', '2V', 'two']
    }
    
    for i, filename in enumerate(sample_files):
        if i % 1000 == 0:
            print(f"Processed {i}/{sample_size} files...")
            
        filepath = os.path.join(metadata_dir, filename)
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            study_desc = data.get('study_description', '')
            study_descriptions.append(study_desc)
            
            # Analyze view information
            view_analysis = {
                'filename': filename,
                'study_description': study_desc,
                'patient_sex': data.get('patient_sex'),
                'patient_age': data.get('patient_age'),
                'study_date': data.get('study_date'),
                'findings': data.get('findings', ''),
                'impression': data.get('impression', ''),
                'detected_views': []
            }
            
            # Check for view keywords
            desc_lower = study_desc.lower()
            findings_lower = data.get('findings', '').lower()
            impression_lower = data.get('impression', '').lower()
            
            for view_type, keywords in view_keywords.items():
                for keyword in keywords:
                    if keyword.lower() in desc_lower:
                        view_analysis['detected_views'].append(view_type)
                        break
            
            # Also check findings and impression for view information
            for view_type, keywords in view_keywords.items():
                for keyword in keywords:
                    if keyword.lower() in findings_lower or keyword.lower() in impression_lower:
                        if view_type not in view_analysis['detected_views']:
                            view_analysis['detected_views'].append(f"{view_type}_in_text")
                        break
            
            view_info.append(view_analysis)
            
        except Exception as e:
            print(f"Error reading {filename}: {e}")
    
    print(f"\nAnalysis complete! Processed {len(view_info)} files.")
    
    # Analyze study descriptions
    print("\n" + "="*60)
    print("STUDY DESCRIPTION ANALYSIS")
    print("="*60)
    
    desc_counter = Counter(study_descriptions)
    print(f"Unique study descriptions: {len(desc_counter)}")
    print("\nTop 20 study descriptions:")
    for desc, count in desc_counter.most_common(20):
        print(f"  {count:4d}: '{desc}'")
    
    # Analyze view types
    print("\n" + "="*60)
    print("VIEW TYPE ANALYSIS")
    print("="*60)
    
    all_detected_views = []
    for info in view_info:
        all_detected_views.extend(info['detected_views'])
    
    view_counter = Counter(all_detected_views)
    print("Detected view types:")
    for view, count in view_counter.most_common():
        print(f"  {count:4d}: {view}")
    
    # Analyze specific view patterns
    print("\n" + "="*60)
    print("SPECIFIC VIEW PATTERNS")
    print("="*60)
    
    # PA and Lateral combinations
    pa_lateral_count = 0
    ap_lateral_count = 0
    single_view_count = 0
    portable_count = 0
    
    for info in view_info:
        desc = info['study_description'].lower()
        if 'pa' in desc and 'lateral' in desc:
            pa_lateral_count += 1
        elif 'ap' in desc and 'lateral' in desc:
            ap_lateral_count += 1
        elif any(word in desc for word in ['single', '1v', '1 view']):
            single_view_count += 1
        elif 'port' in desc:
            portable_count += 1
    
    print(f"PA + Lateral combinations: {pa_lateral_count}")
    print(f"AP + Lateral combinations: {ap_lateral_count}")
    print(f"Single view studies: {single_view_count}")
    print(f"Portable studies: {portable_count}")
    
    # Show sample cases with specific views
    print("\n" + "="*60)
    print("SAMPLE CASES BY VIEW TYPE")
    print("="*60)
    
    # PA and Lateral
    pa_lateral_cases = [info for info in view_info if 'PA' in info['detected_views'] and 'Lateral' in info['detected_views']]
    if pa_lateral_cases:
        print("\nPA + Lateral cases:")
        for i, case in enumerate(pa_lateral_cases[:3]):
            print(f"  {i+1}. {case['study_description']}")
            print(f"     Patient: {case['patient_sex']}, Age: {case['patient_age']}")
            print(f"     Date: {case['study_date']}")
            print(f"     Findings preview: {case['findings'][:100]}...")
            print()
    
    # AP cases
    ap_cases = [info for info in view_info if 'AP' in info['detected_views']]
    if ap_cases:
        print("\nAP view cases:")
        for i, case in enumerate(ap_cases[:3]):
            print(f"  {i+1}. {case['study_description']}")
            print(f"     Patient: {case['patient_sex']}, Age: {case['patient_age']}")
            print(f"     Date: {case['study_date']}")
            print(f"     Findings preview: {case['findings'][:100]}...")
            print()
    
    # Portable cases
    portable_cases = [info for info in view_info if 'Portable' in info['detected_views']]
    if portable_cases:
        print("\nPortable cases:")
        for i, case in enumerate(portable_cases[:3]):
            print(f"  {i+1}. {case['study_description']}")
            print(f"     Patient: {case['patient_sex']}, Age: {case['patient_age']}")
            print(f"     Date: {case['study_date']}")
            print(f"     Findings preview: {case['findings'][:100]}...")
            print()
    
    # Summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    total_analyzed = len(view_info)
    cases_with_view_info = len([info for info in view_info if info['detected_views']])
    
    print(f"Total cases analyzed: {total_analyzed}")
    print(f"Cases with detected view information: {cases_with_view_info}")
    print(f"Percentage with view info: {(cases_with_view_info/total_analyzed)*100:.1f}%")
    
    # Most common view combinations
    view_combinations = []
    for info in view_info:
        if info['detected_views']:
            view_combinations.append(' + '.join(sorted(info['detected_views'])))
    
    combo_counter = Counter(view_combinations)
    print(f"\nMost common view combinations:")
    for combo, count in combo_counter.most_common(10):
        print(f"  {count:4d}: {combo}")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    analyze_chest_views() 