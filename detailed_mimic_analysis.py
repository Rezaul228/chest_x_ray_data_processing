#!/usr/bin/env python3
"""
Detailed analysis of two MIMIC-CXR datasets with proper handling of dictionary-structured shards.
"""

import os
import pickle
import glob
import numpy as np
from collections import defaultdict, Counter
import json

def analyze_dataset_detailed(dataset_path, dataset_name):
    """Analyze the dataset with proper handling of dictionary-structured shards"""
    print(f"\n{'='*80}")
    print(f"🔍 DETAILED ANALYSIS: {dataset_name}")
    print(f"{'='*80}")
    
    # Check metadata
    metadata_path = os.path.join(dataset_path, 'metadata.pkl')
    metadata_info = {}
    
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            
            print(f"✓ Metadata loaded successfully")
            print(f"  File size: {os.path.getsize(metadata_path)/1024/1024:.1f} MB")
            print(f"  Metadata keys: {list(metadata.keys())}")
            
            # Store metadata info
            metadata_info = {
                'file_size_mb': os.path.getsize(metadata_path)/1024/1024,
                'keys': list(metadata.keys())
            }
            
            # Check tokenizer info
            if 'tokenizer' in metadata:
                tokenizer = metadata['tokenizer']
                print(f"  Tokenizer type: {type(tokenizer).__name__}")
                
                if hasattr(tokenizer, 'word_index'):
                    vocab_size = len(tokenizer.word_index)
                    print(f"  Vocabulary size: {vocab_size}")
                    metadata_info['vocab_size'] = vocab_size
                elif hasattr(tokenizer, 'vocab_size'):
                    print(f"  Vocabulary size: {tokenizer.vocab_size}")
                    metadata_info['vocab_size'] = tokenizer.vocab_size
            
            # Check other metadata
            for key, value in metadata.items():
                if key != 'tokenizer':
                    print(f"  {key}: {value}")
                    metadata_info[key] = value
                    
        except Exception as e:
            print(f"✗ Error loading metadata: {e}")
    else:
        print(f"✗ No metadata.pkl found")
    
    # Analyze splits
    splits = ['train', 'val', 'test']
    split_stats = {}
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
                
                # Analyze first few shards to understand structure
                sample_count = 0
                token_lengths = []
                image_shapes = []
                study_ids = []
                
                # Analyze first 3 shards in detail
                for i, shard_file in enumerate(shard_files[:3]):
                    try:
                        with open(shard_file, 'rb') as f:
                            shard_data = pickle.load(f)
                        
                        print(f"  Shard {i+1}: {os.path.basename(shard_file)}")
                        print(f"    Type: {type(shard_data)}")
                        print(f"    Size: {os.path.getsize(shard_file)/1024/1024:.1f} MB")
                        
                        if isinstance(shard_data, dict):
                            print(f"    Dictionary keys: {list(shard_data.keys())}")
                            
                            # Check if it's a sample dictionary or a collection
                            if 'caption' in shard_data or 'image' in shard_data:
                                # Single sample
                                sample_count += 1
                                analyze_sample(shard_data, token_lengths, image_shapes, study_ids)
                            else:
                                # Collection of samples
                                for key, value in shard_data.items():
                                    if isinstance(value, dict) and ('caption' in value or 'image' in value):
                                        sample_count += 1
                                        analyze_sample(value, token_lengths, image_shapes, study_ids)
                        
                    except Exception as e:
                        print(f"    ✗ Error analyzing shard: {e}")
                
                # Estimate total samples based on first shard
                if sample_count > 0:
                    estimated_total = len(shard_files) * sample_count
                    print(f"  Samples per shard (estimated): {sample_count}")
                    print(f"  Estimated total samples: {estimated_total}")
                    total_samples += estimated_total
                    
                    split_stats[split] = {
                        'num_shards': len(shard_files),
                        'total_size_mb': total_size/1024/1024,
                        'samples_per_shard': sample_count,
                        'estimated_total_samples': estimated_total,
                        'token_lengths': token_lengths[:100],  # Store first 100 for analysis
                        'image_shapes': image_shapes[:100],
                        'study_ids': study_ids[:100]
                    }
                else:
                    print(f"  Could not determine sample count")
                
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
        'metadata_info': metadata_info,
        'split_stats': split_stats
    }

def analyze_sample(sample, token_lengths, image_shapes, study_ids):
    """Analyze a single sample"""
    if 'caption' in sample:
        caption = sample['caption']
        if isinstance(caption, (list, np.ndarray)):
            token_lengths.append(len(caption))
    
    if 'image' in sample:
        img = sample['image']
        if hasattr(img, 'shape'):
            image_shapes.append(img.shape)
    
    if 'study_id' in sample:
        study_ids.append(str(sample['study_id']))

def compare_datasets_detailed(dataset1_path, dataset1_name, dataset2_path, dataset2_name):
    """Compare two datasets with detailed statistics"""
    print(f"\n{'='*100}")
    print(f"🔄 DETAILED COMPARISON: {dataset1_name} vs {dataset2_name}")
    print(f"{'='*100}")
    
    # Analyze both datasets
    stats1 = analyze_dataset_detailed(dataset1_path, dataset1_name)
    stats2 = analyze_dataset_detailed(dataset2_path, dataset2_name)
    
    # Print comparison table
    print(f"\n📊 COMPARISON SUMMARY")
    print(f"{'='*80}")
    print(f"{'Metric':<35} {dataset1_name:<25} {dataset2_name:<25}")
    print(f"{'-'*85}")
    print(f"{'Total Shards':<35} {stats1['total_shards']:<25} {stats2['total_shards']:<25}")
    print(f"{'Total Samples':<35} {stats1['total_samples']:<25} {stats2['total_samples']:<25}")
    
    if 'metadata_info' in stats1 and 'metadata_info' in stats2:
        vocab1 = stats1['metadata_info'].get('vocab_size', 'N/A')
        vocab2 = stats2['metadata_info'].get('vocab_size', 'N/A')
        print(f"{'Vocabulary Size':<35} {vocab1:<25} {vocab2:<25}")
        
        meta_size1 = stats1['metadata_info'].get('file_size_mb', 0)
        meta_size2 = stats2['metadata_info'].get('file_size_mb', 0)
        print(f"{'Metadata Size (MB)':<35} {meta_size1:<25.1f} {meta_size2:<25.1f}")
    
    # Calculate ratios
    if stats1['total_samples'] > 0 and stats2['total_samples'] > 0:
        sample_ratio = stats2['total_samples'] / stats1['total_samples']
        shard_ratio = stats2['total_shards'] / stats1['total_shards']
        print(f"\n📈 RATIOS ({dataset2_name} / {dataset1_name}):")
        print(f"  Sample ratio: {sample_ratio:.2f}x")
        print(f"  Shard ratio: {shard_ratio:.2f}x")
    
    # Detailed split comparison
    print(f"\n📈 SPLIT COMPARISON:")
    splits = ['train', 'val', 'test']
    
    for split in splits:
        if split in stats1['split_stats'] and split in stats2['split_stats']:
            s1 = stats1['split_stats'][split]
            s2 = stats2['split_stats'][split]
            
            print(f"\n{split.upper()} Split:")
            print(f"  {dataset1_name}: {s1['estimated_total_samples']} samples, {s1['num_shards']} shards, {s1['total_size_mb']:.1f} MB")
            print(f"  {dataset2_name}: {s2['estimated_total_samples']} samples, {s2['num_shards']} shards, {s2['total_size_mb']:.1f} MB")
            
            # Token length comparison
            if s1['token_lengths'] and s2['token_lengths']:
                avg1 = np.mean(s1['token_lengths'])
                avg2 = np.mean(s2['token_lengths'])
                print(f"  Avg token length: {dataset1_name}={avg1:.1f}, {dataset2_name}={avg2:.1f}")
    
    return stats1, stats2

def analyze_shard_structure(dataset_path, dataset_name, num_shards=2):
    """Analyze the internal structure of shards"""
    print(f"\n🔍 SHARD STRUCTURE ANALYSIS for {dataset_name}")
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
                        
                        if isinstance(shard_data, dict):
                            print(f"    Dictionary keys: {list(shard_data.keys())}")
                            
                            # Analyze first few items
                            for j, (key, value) in enumerate(list(shard_data.items())[:3]):
                                print(f"    Item {j+1} (key: {key}):")
                                print(f"      Type: {type(value)}")
                                
                                if isinstance(value, dict):
                                    print(f"      Keys: {list(value.keys())}")
                                    
                                    if 'caption' in value:
                                        caption = value['caption']
                                        if isinstance(caption, (list, np.ndarray)):
                                            print(f"      Caption length: {len(caption)}")
                                            non_zero = np.count_nonzero(caption)
                                            print(f"      Non-zero tokens: {non_zero}")
                                    
                                    if 'image' in value:
                                        img = value['image']
                                        if hasattr(img, 'shape'):
                                            print(f"      Image shape: {img.shape}")
                                    
                                    if 'study_id' in value:
                                        print(f"      Study ID: {value['study_id']}")
                        
                    except Exception as e:
                        print(f"    ✗ Error analyzing shard: {e}")
                
                break  # Only analyze first split to avoid too much output

def main():
    # Define dataset paths
    dataset1_path = "/home/abedin/Developments/chest_x_ray_data_processing/all_processed_data/mimic_shards"
    dataset1_name = "mimic_shards"
    
    dataset2_path = "/home/abedin/Developments/chest_x_ray_data_processing/all_processed_data/mimic_shards_hufc4446-to128"
    dataset2_name = "mimic_shards_hufc4446-to128"
    
    # Run detailed comparison
    stats1, stats2 = compare_datasets_detailed(dataset1_path, dataset1_name, dataset2_path, dataset2_name)
    
    # Analyze shard structure
    analyze_shard_structure(dataset1_path, dataset1_name)
    analyze_shard_structure(dataset2_path, dataset2_name)

if __name__ == "__main__":
    main() 