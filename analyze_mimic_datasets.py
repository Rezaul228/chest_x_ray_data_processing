#!/usr/bin/env python3
"""
Comprehensive analysis of two MIMIC-CXR datasets:
1. /home/abedin/Developments/chest_x_ray_data_processing/all_processed_data/mimic_shards
2. /home/abedin/Developments/chest_x_ray_data_processing/all_processed_data/mimic_shards_hufc4446-to128
"""

import os
import pickle
import glob
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any

class DatasetAnalyzer:
    def __init__(self, dataset_path: str, dataset_name: str):
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        self.metadata = None
        self.tokenizer = None
        self.stats = defaultdict(dict)
        
    def load_metadata(self):
        """Load metadata from the dataset"""
        metadata_path = os.path.join(self.dataset_path, 'metadata.pkl')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
                self.tokenizer = self.metadata.get('tokenizer')
                print(f"✓ Loaded metadata for {self.dataset_name}")
                print(f"  Tokenizer type: {type(self.tokenizer).__name__}")
                print(f"  Vocabulary size: {self.metadata.get('vocab_size', 'N/A')}")
                print(f"  Max sequence length: {self.metadata.get('max_sequence_length', 'N/A')}")
        else:
            print(f"✗ No metadata.pkl found in {self.dataset_path}")
    
    def analyze_shards(self, split: str) -> Dict[str, Any]:
        """Analyze shards for a specific split (train/val/test)"""
        split_dir = os.path.join(self.dataset_path, split)
        if not os.path.exists(split_dir):
            print(f"✗ Split directory {split_dir} does not exist")
            return {}
        
        shard_files = sorted(glob.glob(os.path.join(split_dir, 'shard_*.pkl')))
        print(f"\n📊 Analyzing {split} split for {self.dataset_name}")
        print(f"  Number of shards: {len(shard_files)}")
        
        total_samples = 0
        token_lengths = []
        study_ids = []
        image_shapes = []
        non_zero_tokens = []
        
        for i, shard_file in enumerate(shard_files):
            try:
                with open(shard_file, 'rb') as f:
                    shard_data = pickle.load(f)
                
                shard_samples = len(shard_data)
                total_samples += shard_samples
                
                print(f"  Shard {i+1:3d}: {shard_samples:4d} samples, {os.path.getsize(shard_file)/1024/1024:.1f} MB")
                
                # Analyze each sample in the shard
                for sample in shard_data:
                    # Token length analysis
                    if 'caption' in sample:
                        caption = sample['caption']
                        if isinstance(caption, (list, np.ndarray)):
                            # Count non-zero tokens
                            non_zero_count = np.count_nonzero(caption)
                            non_zero_tokens.append(non_zero_count)
                            token_lengths.append(len(caption))
                    
                    # Study ID analysis
                    if 'study_id' in sample:
                        study_ids.append(str(sample['study_id']))
                    
                    # Image shape analysis
                    if 'image' in sample:
                        img = sample['image']
                        if hasattr(img, 'shape'):
                            image_shapes.append(img.shape)
                
                # Progress indicator
                if (i + 1) % 20 == 0:
                    print(f"    Processed {i+1}/{len(shard_files)} shards...")
                    
            except Exception as e:
                print(f"  ✗ Error loading shard {shard_file}: {e}")
                continue
        
        # Calculate statistics
        stats = {
            'total_samples': total_samples,
            'num_shards': len(shard_files),
            'token_lengths': token_lengths,
            'non_zero_tokens': non_zero_tokens,
            'study_ids': study_ids,
            'image_shapes': image_shapes,
            'avg_shard_size': total_samples / len(shard_files) if shard_files else 0
        }
        
        if token_lengths:
            stats.update({
                'min_token_length': min(token_lengths),
                'max_token_length': max(token_lengths),
                'mean_token_length': np.mean(token_lengths),
                'median_token_length': np.median(token_lengths),
                'std_token_length': np.std(token_lengths)
            })
        
        if non_zero_tokens:
            stats.update({
                'min_non_zero': min(non_zero_tokens),
                'max_non_zero': max(non_zero_tokens),
                'mean_non_zero': np.mean(non_zero_tokens),
                'median_non_zero': np.median(non_zero_tokens),
                'std_non_zero': np.std(non_zero_tokens)
            })
        
        if image_shapes:
            unique_shapes = Counter(image_shapes)
            stats['image_shapes_distribution'] = dict(unique_shapes)
        
        return stats
    
    def analyze_vocabulary(self):
        """Analyze vocabulary characteristics"""
        if not self.tokenizer:
            print("✗ No tokenizer available for vocabulary analysis")
            return
        
        vocab_stats = {}
        
        if hasattr(self.tokenizer, 'word_index'):
            vocab_stats['word_index_size'] = len(self.tokenizer.word_index)
            vocab_stats['word_index_sample'] = list(self.tokenizer.word_index.items())[:10]
        
        if hasattr(self.tokenizer, 'index_word'):
            vocab_stats['index_word_size'] = len(self.tokenizer.index_word)
            vocab_stats['index_word_sample'] = list(self.tokenizer.index_word.items())[:10]
        
        if hasattr(self.tokenizer, 'vocab_size'):
            vocab_stats['vocab_size'] = self.tokenizer.vocab_size
        
        self.stats['vocabulary'] = vocab_stats
        print(f"\n📚 Vocabulary Analysis for {self.dataset_name}:")
        for key, value in vocab_stats.items():
            print(f"  {key}: {value}")
    
    def run_comprehensive_analysis(self):
        """Run comprehensive analysis on the dataset"""
        print(f"\n{'='*80}")
        print(f"🔍 COMPREHENSIVE ANALYSIS: {self.dataset_name}")
        print(f"{'='*80}")
        
        # Load metadata
        self.load_metadata()
        
        # Analyze vocabulary
        self.analyze_vocabulary()
        
        # Analyze each split
        splits = ['train', 'val', 'test']
        for split in splits:
            split_stats = self.analyze_shards(split)
            if split_stats:
                self.stats[split] = split_stats
                
                # Print summary for this split
                print(f"\n📈 {split.upper()} Split Summary:")
                print(f"  Total samples: {split_stats['total_samples']:,}")
                print(f"  Number of shards: {split_stats['num_shards']}")
                print(f"  Average shard size: {split_stats['avg_shard_size']:.1f}")
                
                if 'token_lengths' in split_stats and split_stats['token_lengths']:
                    print(f"  Token length - Min: {split_stats['min_token_length']}, "
                          f"Max: {split_stats['max_token_length']}, "
                          f"Mean: {split_stats['mean_token_length']:.1f}, "
                          f"Median: {split_stats['median_token_length']:.1f}")
                
                if 'non_zero_tokens' in split_stats and split_stats['non_zero_tokens']:
                    print(f"  Non-zero tokens - Min: {split_stats['min_non_zero']}, "
                          f"Max: {split_stats['max_non_zero']}, "
                          f"Mean: {split_stats['mean_non_zero']:.1f}, "
                          f"Median: {split_stats['median_non_zero']:.1f}")
        
        return self.stats

def compare_datasets(dataset1_path: str, dataset1_name: str, 
                    dataset2_path: str, dataset2_name: str):
    """Compare two datasets side by side"""
    print(f"\n{'='*100}")
    print(f"🔄 COMPARISON: {dataset1_name} vs {dataset2_name}")
    print(f"{'='*100}")
    
    # Analyze both datasets
    analyzer1 = DatasetAnalyzer(dataset1_path, dataset1_name)
    analyzer2 = DatasetAnalyzer(dataset2_path, dataset2_name)
    
    stats1 = analyzer1.run_comprehensive_analysis()
    stats2 = analyzer2.run_comprehensive_analysis()
    
    # Create comparison table
    print(f"\n📊 COMPARISON SUMMARY")
    print(f"{'='*80}")
    
    comparison_data = []
    
    # Compare metadata
    print(f"\n🔧 METADATA COMPARISON:")
    print(f"{'Metric':<30} {dataset1_name:<25} {dataset2_name:<25}")
    print(f"{'-'*80}")
    
    if analyzer1.metadata and analyzer2.metadata:
        metrics = ['vocab_size', 'max_sequence_length']
        for metric in metrics:
            val1 = analyzer1.metadata.get(metric, 'N/A')
            val2 = analyzer2.metadata.get(metric, 'N/A')
            print(f"{metric:<30} {str(val1):<25} {str(val2):<25}")
    
    # Compare splits
    print(f"\n📈 SPLIT COMPARISON:")
    print(f"{'Split':<10} {'Metric':<20} {dataset1_name:<20} {dataset2_name:<20}")
    print(f"{'-'*70}")
    
    splits = ['train', 'val', 'test']
    for split in splits:
        if split in stats1 and split in stats2:
            print(f"{split:<10} {'Samples':<20} {stats1[split]['total_samples']:<20} {stats2[split]['total_samples']:<20}")
            print(f"{'':<10} {'Shards':<20} {stats1[split]['num_shards']:<20} {stats2[split]['num_shards']:<20}")
            print(f"{'':<10} {'Avg Shard Size':<20} {stats1[split]['avg_shard_size']:<20.1f} {stats2[split]['avg_shard_size']:<20.1f}")
            
            if 'token_lengths' in stats1[split] and 'token_lengths' in stats2[split]:
                print(f"{'':<10} {'Mean Token Length':<20} {stats1[split]['mean_token_length']:<20.1f} {stats2[split]['mean_token_length']:<20.1f}")
                print(f"{'':<10} {'Max Token Length':<20} {stats1[split]['max_token_length']:<20} {stats2[split]['max_token_length']:<20}")
            
            if 'non_zero_tokens' in stats1[split] and 'non_zero_tokens' in stats2[split]:
                print(f"{'':<10} {'Mean Non-zero':<20} {stats1[split]['mean_non_zero']:<20.1f} {stats2[split]['mean_non_zero']:<20.1f}")
    
    # Calculate total dataset statistics
    print(f"\n📊 TOTAL DATASET STATISTICS:")
    print(f"{'Metric':<30} {dataset1_name:<25} {dataset2_name:<25}")
    print(f"{'-'*80}")
    
    total_samples1 = sum(stats1[split]['total_samples'] for split in splits if split in stats1)
    total_samples2 = sum(stats2[split]['total_samples'] for split in splits if split in stats2)
    print(f"{'Total Samples':<30} {total_samples1:<25,} {total_samples2:<25,}")
    
    total_shards1 = sum(stats1[split]['num_shards'] for split in splits if split in stats1)
    total_shards2 = sum(stats2[split]['num_shards'] for split in splits if split in stats2)
    print(f"{'Total Shards':<30} {total_shards1:<25} {total_shards2:<25}")
    
    # Token length distribution comparison
    print(f"\n📏 TOKEN LENGTH DISTRIBUTION COMPARISON:")
    for split in splits:
        if split in stats1 and split in stats2:
            if 'token_lengths' in stats1[split] and 'token_lengths' in stats2[split]:
                print(f"\n{split.upper()} Split:")
                print(f"  {dataset1_name}: Mean={stats1[split]['mean_token_length']:.1f}, "
                      f"Std={stats1[split]['std_token_length']:.1f}, "
                      f"Range=[{stats1[split]['min_token_length']}, {stats1[split]['max_token_length']}]")
                print(f"  {dataset2_name}: Mean={stats2[split]['mean_token_length']:.1f}, "
                      f"Std={stats2[split]['std_token_length']:.1f}, "
                      f"Range=[{stats2[split]['min_token_length']}, {stats2[split]['max_token_length']}]")

def main():
    # Define dataset paths
    dataset1_path = "/home/abedin/Developments/chest_x_ray_data_processing/all_processed_data/mimic_shards"
    dataset1_name = "mimic_shards"
    
    dataset2_path = "/home/abedin/Developments/chest_x_ray_data_processing/all_processed_data/mimic_shards_hufc4446-to128"
    dataset2_name = "mimic_shards_hufc4446-to128"
    
    # Run comparison
    compare_datasets(dataset1_path, dataset1_name, dataset2_path, dataset2_name)

if __name__ == "__main__":
    main() 