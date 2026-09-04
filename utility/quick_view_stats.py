#!/usr/bin/env python3

import os
import json
import random
from collections import Counter

def quick_view_stats(sample_size=5000):
    """Quick statistics on view types from a larger sample"""
    
    metadata_dir = "raw_data_ReXGradient-160K/organized_data/metadata"
    metadata_files = [f for f in os.listdir(metadata_dir) if f.endswith('.json')]
    
    sample_files = random.sample(metadata_files, min(sample_size, len(metadata_files)))
    
    study_descriptions = []
    pa_count = 0
    ap_count = 0
    lateral_count = 0
    portable_count = 0
    two_view_count = 0
    
    for i, filename in enumerate(sample_files):
        if i % 1000 == 0:
            print(f"Processed {i}/{len(sample_files)} files...")
            
        filepath = os.path.join(metadata_dir, filename)
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            desc = data.get('study_description', '').lower()
            study_descriptions.append(data.get('study_description', ''))
            
            # Count view types
            if 'pa' in desc:
                pa_count += 1
            if 'ap' in desc:
                ap_count += 1
            if 'lateral' in desc:
                lateral_count += 1
            if 'port' in desc:
                portable_count += 1
            if '2v' in desc or '2 view' in desc:
                two_view_count += 1
                
        except Exception as e:
            continue
    
    print(f"\nQuick Statistics (Sample size: {len(sample_files)})")
    print("="*50)
    print(f"PA mentions: {pa_count} ({pa_count/len(sample_files)*100:.1f}%)")
    print(f"AP mentions: {ap_count} ({ap_count/len(sample_files)*100:.1f}%)")
    print(f"Lateral mentions: {lateral_count} ({lateral_count/len(sample_files)*100:.1f}%)")
    print(f"Portable mentions: {portable_count} ({portable_count/len(sample_files)*100:.1f}%)")
    print(f"Two-view mentions: {two_view_count} ({two_view_count/len(sample_files)*100:.1f}%)")
    
    # Top study descriptions
    desc_counter = Counter(study_descriptions)
    print(f"\nTop 10 study descriptions:")
    for desc, count in desc_counter.most_common(10):
        print(f"  {count:4d}: '{desc}'")

if __name__ == "__main__":
    import sys
    sample_size = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    quick_view_stats(sample_size) 