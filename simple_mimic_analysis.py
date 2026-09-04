#!/usr/bin/env python3
"""
Simple analysis of two MIMIC-CXR datasets to understand their structure and characteristics.
"""

import os
import pickle
import glob
import numpy as np
from collections import defaultdict

def analyze_dataset_structure(dataset_path, dataset_name):
    """Analyze the basic structure of a dataset"""
    print(f"\n{'='*60}")
    print(f"📊 ANALYZING: {dataset_name}")
    print(f"{'='*60}")
    
    # Check metadata
    metadata_path = os.path.join(dataset_path, 'metadata.pkl')
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            print(f"✓ Metadata loaded successfully")
            print(f"  File size: {os.path.getsize(metadata_path)/1024/1024:.1f} MB")
            
            # Print metadata keys
            print(f"  Metadata keys: {list(metadata.keys())}")
            
            # Check tokenizer info
            if 'tokenizer' in metadata:
                tokenizer = metadata['tokenizer']
                print(f"  Tokenizer type: {type(tokenizer).__name__}")
                
                if hasattr(tokenizer, 'word_index'):
                    print(f"  Vocabulary size: {len(tokenizer.word_index)}")
                elif hasattr(tokenizer, 'vocab_size'):
                    print(f"  Vocabulary size: {tokenizer.vocab_size}")
            
            # Check other metadata
            for key, value in metadata.items():
                if key != 'tokenizer':
                    print(f"  {key}: {value}")
                    
        except Exception as e:
            print(f"✗ Error loading metadata: {e}")
    else:
        print(f"✗ No metadata.pkl found")
    
    # Analyze splits
    splits = ['train', 'val', 'test']
    total_samples = 0
    total_shards = 0
    
    for split in splits:
        split_dir = os.path.join(dataset_path, split)
        if os.path.exists(split_dir):
            shard_files = glob.glob(os.path.join(split_dir, 'shard_*.pkl'))
            if shard_files:
                print(f"\n📁 {split.upper()} split:")
                print(f"  Number of shards: {len(shard_files)}")
                
                # Calculate total size
                total_size = sum(os.path.getsize(f) for f in shard_files)
                print(f"  Total size: {total_size/1024/1024:.1f} MB")
                
                # Try to load first shard to get sample count
                try:
                    with open(shard_files[0], 'rb') as f:
                        first_shard = pickle.load(f)
                    
                    if isinstance(first_shard, list):
                        samples_per_shard = len(first_shard)
                        print(f"  Samples per shard: {samples_per_shard}")
                        estimated_total = len(shard_files) * samples_per_shard
                        print(f"  Estimated total samples: {estimated_total}")
                        total_samples += estimated_total
                    else:
                        print(f"  Shard structure: {type(first_shard)}")
                        
                except Exception as e:
                    print(f"  ✗ Error loading first shard: {e}")
                
                total_shards += len(shard_files)
            else:
                print(f"\n📁 {split.upper()} split: No shards found")
        else:
            print(f"\n📁 {split.upper()} split: Directory not found")
    
    print(f"\n📈 SUMMARY for {dataset_name}:")
    print(f"  Total shards: {total_shards}")
    print(f"  Estimated total samples: {total_samples}")
    
    return {
        'total_shards': total_shards,
        'total_samples': total_samples,
        'metadata_size': os.path.getsize(metadata_path) if os.path.exists(metadata_path) else 0
    }

def compare_datasets(dataset1_path, dataset1_name, dataset2_path, dataset2_name):
    """Compare two datasets"""
    print(f"\n{'='*80}")
    print(f"🔄 COMPARISON: {dataset1_name} vs {dataset2_name}")
    print(f"{'='*80}")
    
    # Analyze both datasets
    stats1 = analyze_dataset_structure(dataset1_path, dataset1_name)
    stats2 = analyze_dataset_structure(dataset2_path, dataset2_name)
    
    # Print comparison table
    print(f"\n📊 COMPARISON SUMMARY")
    print(f"{'='*50}")
    print(f"{'Metric':<25} {dataset1_name:<20} {dataset2_name:<20}")
    print(f"{'-'*65}")
    print(f"{'Total Shards':<25} {stats1['total_shards']:<20} {stats2['total_shards']:<20}")
    print(f"{'Total Samples':<25} {stats1['total_samples']:<20} {stats2['total_samples']:<20}")
    print(f"{'Metadata Size (MB)':<25} {stats1['metadata_size']/1024/1024:<20.1f} {stats2['metadata_size']/1024/1024:<20.1f}")
    
    # Calculate ratios
    if stats1['total_samples'] > 0 and stats2['total_samples'] > 0:
        sample_ratio = stats2['total_samples'] / stats1['total_samples']
        shard_ratio = stats2['total_shards'] / stats1['total_shards']
        print(f"\n📈 RATIOS ({dataset2_name} / {dataset1_name}):")
        print(f"  Sample ratio: {sample_ratio:.2f}x")
        print(f"  Shard ratio: {shard_ratio:.2f}x")

def analyze_shard_content(dataset_path, dataset_name, num_shards=3):
    """Analyze the content of a few shards to understand data structure"""
    print(f"\n🔍 DETAILED SHARD ANALYSIS for {dataset_name}")
    print(f"{'='*60}")
    
    splits = ['train', 'val', 'test']
    
    for split in splits:
        split_dir = os.path.join(dataset_path, split)
        if os.path.exists(split_dir):
            shard_files = sorted(glob.glob(os.path.join(split_dir, 'shard_*.pkl')))
            if shard_files:
                print(f"\n📁 {split.upper()} split - analyzing first {num_shards} shards:")
                
                for i, shard_file in enumerate(shard_files[:num_shards]):
                    try:
                        with open(shard_file, 'rb') as f:
                            shard_data = pickle.load(f)
                        
                        print(f"  Shard {i+1}: {os.path.basename(shard_file)}")
                        print(f"    Type: {type(shard_data)}")
                        print(f"    Size: {os.path.getsize(shard_file)/1024/1024:.1f} MB")
                        
                        if isinstance(shard_data, list):
                            print(f"    Length: {len(shard_data)}")
                            
                            if len(shard_data) > 0:
                                first_sample = shard_data[0]
                                print(f"    First sample type: {type(first_sample)}")
                                
                                if isinstance(first_sample, dict):
                                    print(f"    Sample keys: {list(first_sample.keys())}")
                                    
                                    # Check for specific fields
                                    for key in ['caption', 'image', 'study_id']:
                                        if key in first_sample:
                                            value = first_sample[key]
                                            if key == 'caption':
                                                if isinstance(value, (list, np.ndarray)):
                                                    print(f"    Caption length: {len(value)}")
                                                    non_zero = np.count_nonzero(value)
                                                    print(f"    Non-zero tokens: {non_zero}")
                                            elif key == 'image':
                                                if hasattr(value, 'shape'):
                                                    print(f"    Image shape: {value.shape}")
                                            elif key == 'study_id':
                                                print(f"    Study ID: {value}")
                        
                    except Exception as e:
                        print(f"    ✗ Error analyzing shard: {e}")
                
                break  # Only analyze first split to avoid too much output

def main():
    # Define dataset paths
    dataset1_path = "/home/abedin/Developments/chest_x_ray_data_processing/all_processed_data/mimic_shards"
    dataset1_name = "mimic_shards"
    
    dataset2_path = "/home/abedin/Developments/chest_x_ray_data_processing/all_processed_data/mimic_shards_hufc4446-to128"
    dataset2_name = "mimic_shards_hufc4446-to128"
    
    # Run comparison
    compare_datasets(dataset1_path, dataset1_name, dataset2_path, dataset2_name)
    
    # Detailed analysis
    analyze_shard_content(dataset1_path, dataset1_name)
    analyze_shard_content(dataset2_path, dataset2_name)

if __name__ == "__main__":
    main() 