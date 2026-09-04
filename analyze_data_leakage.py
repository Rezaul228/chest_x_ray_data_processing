#!/usr/bin/env python3
"""
Analyze data leakage issues in processed MIMIC-CXR dataset
"""

import os
import pickle
import numpy as np
from collections import defaultdict, Counter
import pandas as pd
from tqdm import tqdm

def load_shard_data(shard_path):
    """Load data from a shard file"""
    try:
        with open(shard_path, 'rb') as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        print(f"Error loading {shard_path}: {e}")
        return None

def analyze_data_leakage(data_dir):
    """Analyze data leakage across train/val/test splits"""
    
    print("🔍 Analyzing data leakage issues...")
    
    # Collect all captions with their locations
    caption_locations = defaultdict(list)
    split_counts = {'train': 0, 'val': 0, 'test': 0}
    
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(data_dir, split)
        if not os.path.exists(split_dir):
            print(f"⚠️  {split} directory not found")
            continue
            
        print(f"\n📁 Processing {split} split...")
        shard_files = [f for f in os.listdir(split_dir) if f.endswith('.pkl')]
        
        for shard_file in tqdm(shard_files, desc=f"Loading {split} shards"):
            shard_path = os.path.join(split_dir, shard_file)
            data = load_shard_data(shard_path)
            
            if data is None:
                continue
                
            # Count samples in this shard
            if isinstance(data, list):
                split_counts[split] += len(data)
                
                # Collect captions
                for i, sample in enumerate(data):
                    if isinstance(sample, dict) and 'caption' in sample:
                        caption = tuple(sample['caption'])  # Convert to tuple for hashing
                        caption_locations[caption].append(f"{split}_shard_{shard_file}_{i}")
                    elif isinstance(sample, (list, np.ndarray)):
                        # Handle case where sample is directly the caption
                        caption = tuple(sample)
                        caption_locations[caption].append(f"{split}_shard_{shard_file}_{i}")
    
    print(f"\n📊 Dataset Statistics:")
    print(f"  Train samples: {split_counts['train']:,}")
    print(f"  Validation samples: {split_counts['val']:,}")
    print(f"  Test samples: {split_counts['test']:,}")
    print(f"  Total unique captions: {len(caption_locations):,}")
    
    # Analyze cross-split leakage
    print(f"\n🚨 CROSS-SPLIT DATA LEAKAGE ANALYSIS:")
    
    cross_split_leakage = []
    for caption, locations in caption_locations.items():
        splits_present = set()
        for location in locations:
            split = location.split('_')[0]
            splits_present.add(split)
        
        if len(splits_present) > 1:
            cross_split_leakage.append({
                'caption': caption,
                'locations': locations,
                'splits': list(splits_present)
            })
    
    print(f"  Cross-split leakage instances: {len(cross_split_leakage)}")
    
    if cross_split_leakage:
        print(f"\n🔍 EXAMPLES OF CROSS-SPLIT LEAKAGE:")
        for i, leakage in enumerate(cross_split_leakage[:5]):
            print(f"\n  Example {i+1}:")
            print(f"    Caption: {leakage['caption'][:20]}...")  # Show first 20 tokens
            print(f"    Appears in splits: {leakage['splits']}")
            print(f"    Locations: {leakage['locations'][:3]}...")  # Show first 3 locations
    
    # Analyze internal duplicates
    print(f"\n📋 INTERNAL DUPLICATES ANALYSIS:")
    
    internal_duplicates = {split: [] for split in ['train', 'val', 'test']}
    
    for caption, locations in caption_locations.items():
        split_groups = defaultdict(list)
        for location in locations:
            split = location.split('_')[0]
            split_groups[split].append(location)
        
        for split, split_locations in split_groups.items():
            if len(split_locations) > 1:
                internal_duplicates[split].append({
                    'caption': caption,
                    'locations': split_locations,
                    'count': len(split_locations)
                })
    
    for split in ['train', 'val', 'test']:
        duplicates = internal_duplicates[split]
        print(f"\n  {split.upper()} duplicates: {len(duplicates)} captions")
        
        if duplicates:
            # Sort by count
            duplicates.sort(key=lambda x: x['count'], reverse=True)
            
            print(f"    Most duplicated captions:")
            for i, dup in enumerate(duplicates[:3]):
                print(f"      {i+1}. Appears {dup['count']} times")
                print(f"         Locations: {dup['locations'][:3]}...")
    
    # Summary statistics
    print(f"\n📈 SUMMARY:")
    print(f"  Total unique captions: {len(caption_locations):,}")
    print(f"  Cross-split leakage: {len(cross_split_leakage):,}")
    print(f"  Internal duplicates:")
    for split in ['train', 'val', 'test']:
        print(f"    {split}: {len(internal_duplicates[split]):,}")
    
    return {
        'cross_split_leakage': cross_split_leakage,
        'internal_duplicates': internal_duplicates,
        'caption_locations': caption_locations,
        'split_counts': split_counts
    }

def analyze_specific_examples(data_dir, analysis_results):
    """Analyze specific examples mentioned by user"""
    
    print(f"\n🔍 ANALYZING SPECIFIC EXAMPLES:")
    
    # Example 1: [ 21 30 13 17 28 24 13 10 86 13 11 31 4 5 7 26 33 4...]
    example1 = (21, 30, 13, 17, 28, 24, 13, 10, 86, 13, 11, 31, 4, 5, 7, 26, 33, 4)
    
    # Example 2: [11 42 43 15 71 4 5 7 26 33 29 17 97 38 4 8 20 25 0 0 0...]
    example2 = (11, 42, 43, 15, 71, 4, 5, 7, 26, 33, 29, 17, 97, 38, 4, 8, 20, 25, 0, 0, 0)
    
    # Example 3: [ 62 92 56 589 4 238 126 193 0 0 0 0 0 0 0 0...]
    example3 = (62, 92, 56, 589, 4, 238, 126, 193, 0, 0, 0, 0, 0, 0, 0, 0)
    
    examples = [example1, example2, example3]
    
    for i, example in enumerate(examples, 1):
        if example in analysis_results['caption_locations']:
            locations = analysis_results['caption_locations'][example]
            print(f"\n  Example {i}:")
            print(f"    Caption: {example}")
            print(f"    Found in: {locations}")
            
            # Group by split
            splits = defaultdict(list)
            for loc in locations:
                split = loc.split('_')[0]
                splits[split].append(loc)
            
            for split, split_locs in splits.items():
                print(f"    {split.upper()}: {len(split_locs)} instances")
        else:
            print(f"\n  Example {i}: Not found in dataset")

if __name__ == "__main__":
    data_dir = "/home/abedin/Developments/chest_x_ray_data_processing/all_processed_data/mimic_shards_hybrid_full_ori"
    
    if not os.path.exists(data_dir):
        print(f"❌ Data directory not found: {data_dir}")
        exit(1)
    
    # Run analysis
    results = analyze_data_leakage(data_dir)
    
    # Analyze specific examples
    analyze_specific_examples(data_dir, results)
    
    print(f"\n✅ Analysis complete!") 