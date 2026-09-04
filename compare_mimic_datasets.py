#!/usr/bin/env python3
"""
Compare Two MIMIC Datasets for Training Difficulty

This script compares two MIMIC datasets to determine which one is easier for training:
1. mimic_shards_4446_128
2. mimic_shards_hufc4446-to128

Analysis includes:
- Dataset statistics
- Vocabulary analysis
- Training difficulty factors
- Memory efficiency
- Sequence consistency
- Complexity metrics
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from typing import Dict, List, Any, Tuple
import warnings
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

class DatasetComparator:
    """Compare two datasets for training difficulty analysis."""
    
    def __init__(self, dataset1_path: str, dataset2_path: str):
        self.dataset1_path = dataset1_path
        self.dataset2_path = dataset2_path
        self.dataset1_name = os.path.basename(dataset1_path)
        self.dataset2_name = os.path.basename(dataset2_path)
        
    def load_dataset_metadata(self, dataset_path: str) -> Dict:
        """Load metadata from a dataset."""
        metadata_path = os.path.join(dataset_path, 'metadata.pkl')
        enhanced_metadata_path = os.path.join(dataset_path, 'enhanced_metadata.pkl')
        
        if os.path.exists(enhanced_metadata_path):
            with open(enhanced_metadata_path, 'rb') as f:
                return pickle.load(f)
        elif os.path.exists(metadata_path):
            with open(metadata_path, 'rb') as f:
                return pickle.load(f)
        else:
            return {}
    
    def analyze_dataset_statistics(self, dataset_path: str, dataset_name: str) -> Dict:
        """Analyze basic dataset statistics."""
        print(f"📊 Analyzing {dataset_name}...")
        
        # Load metadata
        metadata = self.load_dataset_metadata(dataset_path)
        
        # Count shards
        train_shards = len([f for f in os.listdir(os.path.join(dataset_path, 'train')) if f.endswith('.pkl')])
        val_shards = len([f for f in os.listdir(os.path.join(dataset_path, 'val')) if f.endswith('.pkl')])
        test_shards = len([f for f in os.listdir(os.path.join(dataset_path, 'test')) if f.endswith('.pkl')])
        
        # Load a sample shard to get sequence information
        sample_shard_path = os.path.join(dataset_path, 'train', os.listdir(os.path.join(dataset_path, 'train'))[0])
        with open(sample_shard_path, 'rb') as f:
            sample_data = pickle.load(f)
        
        # Analyze sequences
        sequences = sample_data.get('captions', [])
        sequence_lengths = [len(seq) for seq in sequences if len(seq) > 0]
        
        stats = {
            'dataset_name': dataset_name,
            'vocab_size': metadata.get('vocab_size', 0),
            'num_train_shards': train_shards,
            'num_val_shards': val_shards,
            'num_test_shards': test_shards,
            'total_shards': train_shards + val_shards + test_shards,
            'sequence_length_stats': {
                'mean': np.mean(sequence_lengths) if sequence_lengths else 0,
                'median': np.median(sequence_lengths) if sequence_lengths else 0,
                'std': np.std(sequence_lengths) if sequence_lengths else 0,
                'min': np.min(sequence_lengths) if sequence_lengths else 0,
                'max': np.max(sequence_lengths) if sequence_lengths else 0,
                'q25': np.percentile(sequence_lengths, 25) if sequence_lengths else 0,
                'q75': np.percentile(sequence_lengths, 75) if sequence_lengths else 0
            },
            'metadata': metadata
        }
        
        return stats
    
    def analyze_training_difficulty_factors(self, dataset_path: str, dataset_name: str) -> Dict:
        """Analyze factors that affect training difficulty."""
        print(f"🔍 Analyzing training difficulty factors for {dataset_name}...")
        
        # Load metadata
        metadata = self.load_dataset_metadata(dataset_path)
        
        # Load sample data from train, val, and test
        all_sequences = []
        all_tokens = []
        
        for split in ['train', 'val', 'test']:
            split_path = os.path.join(dataset_path, split)
            if os.path.exists(split_path):
                shard_files = [f for f in os.listdir(split_path) if f.endswith('.pkl')]
                
                # Sample a few shards for analysis
                sample_shards = shard_files[:min(3, len(shard_files))]
                
                for shard_file in sample_shards:
                    shard_path = os.path.join(split_path, shard_file)
                    with open(shard_path, 'rb') as f:
                        shard_data = pickle.load(f)
                    
                    sequences = shard_data.get('captions', [])
                    all_sequences.extend(sequences)
                    
                    # Flatten sequences to get all tokens
                    for seq in sequences:
                        all_tokens.extend(seq)
        
        # Calculate training difficulty metrics
        sequence_lengths = [len(seq) for seq in all_sequences if len(seq) > 0]
        
        if not sequence_lengths:
            return {'dataset_name': dataset_name, 'error': 'No valid sequences found'}
        
        # Vocabulary efficiency
        unique_tokens = len(set(all_tokens))
        vocab_size = metadata.get('vocab_size', unique_tokens)
        vocab_efficiency = unique_tokens / vocab_size if vocab_size > 0 else 0
        
        # Length consistency (lower is better)
        length_consistency = np.std(sequence_lengths) / np.mean(sequence_lengths) if np.mean(sequence_lengths) > 0 else 0
        
        # Training stability score (higher is better)
        # Based on sequence length consistency and vocabulary efficiency
        stability_score = (1 / (1 + length_consistency)) * vocab_efficiency
        
        # Memory efficiency
        avg_sequence_length = np.mean(sequence_lengths)
        memory_efficiency = 1 / (1 + avg_sequence_length / 100)  # Normalize to 0-1
        
        # Sequence diversity
        unique_sequences = len(set(tuple(seq) for seq in all_sequences))
        sequence_diversity = unique_sequences / len(all_sequences) if all_sequences else 0
        
        # Complexity score (lower is better)
        complexity_score = (length_consistency + (1 - vocab_efficiency) + (1 - sequence_diversity)) / 3
        
        # Repetition analysis
        token_counter = Counter(all_tokens)
        most_common_tokens = token_counter.most_common(10)
        repetition_ratio = sum(count for _, count in most_common_tokens) / len(all_tokens) if all_tokens else 0
        
        # Position consistency
        position_variance = []
        for i in range(min(50, max(sequence_lengths))):  # Analyze first 50 positions
            position_tokens = [seq[i] for seq in all_sequences if len(seq) > i]
            if position_tokens:
                position_variance.append(np.std(position_tokens))
        
        position_consistency = 1 / (1 + np.mean(position_variance)) if position_variance else 0
        
        return {
            'dataset_name': dataset_name,
            'vocab_size': vocab_size,
            'unique_tokens': unique_tokens,
            'vocab_efficiency': vocab_efficiency,
            'length_consistency': length_consistency,
            'training_stability_score': stability_score,
            'memory_efficiency': memory_efficiency,
            'sequence_diversity': sequence_diversity,
            'complexity_score': complexity_score,
            'repetition_ratio': repetition_ratio,
            'position_consistency': position_consistency,
            'avg_sequence_length': avg_sequence_length,
            'total_sequences': len(all_sequences),
            'total_tokens': len(all_tokens)
        }
    
    def compare_datasets(self) -> Dict:
        """Compare both datasets comprehensively."""
        print("🔍 Starting comprehensive dataset comparison...")
        
        # Analyze basic statistics
        stats1 = self.analyze_dataset_statistics(self.dataset1_path, self.dataset1_name)
        stats2 = self.analyze_dataset_statistics(self.dataset2_path, self.dataset2_name)
        
        # Analyze training difficulty factors
        difficulty1 = self.analyze_training_difficulty_factors(self.dataset1_path, self.dataset1_name)
        difficulty2 = self.analyze_training_difficulty_factors(self.dataset2_path, self.dataset2_name)
        
        # Calculate comparison metrics
        comparison = {
            'dataset1': {
                'name': self.dataset1_name,
                'statistics': stats1,
                'training_difficulty': difficulty1
            },
            'dataset2': {
                'name': self.dataset2_name,
                'statistics': stats2,
                'training_difficulty': difficulty2
            },
            'comparison': {}
        }
        
        # Compare key metrics
        if 'error' not in difficulty1 and 'error' not in difficulty2:
            # Training difficulty comparison (lower is better)
            diff1 = difficulty1['complexity_score']
            diff2 = difficulty2['complexity_score']
            
            if diff1 < diff2:
                easier_dataset = self.dataset1_name
                easier_score = diff1
                harder_score = diff2
            else:
                easier_dataset = self.dataset2_name
                easier_score = diff2
                harder_score = diff1
            
            comparison['comparison'] = {
                'easier_dataset': easier_dataset,
                'easier_complexity_score': easier_score,
                'harder_complexity_score': harder_score,
                'improvement_percentage': ((harder_score - easier_score) / harder_score) * 100,
                'key_differences': {
                    'vocab_size_diff': difficulty1['vocab_size'] - difficulty2['vocab_size'],
                    'length_consistency_diff': difficulty1['length_consistency'] - difficulty2['length_consistency'],
                    'stability_score_diff': difficulty1['training_stability_score'] - difficulty2['training_stability_score'],
                    'memory_efficiency_diff': difficulty1['memory_efficiency'] - difficulty2['memory_efficiency']
                }
            }
        
        return comparison
    
    def print_comparison_summary(self, comparison: Dict):
        """Print a comprehensive comparison summary."""
        print("\n" + "="*80)
        print("📊 MIMIC DATASETS COMPARISON SUMMARY")
        print("="*80)
        
        dataset1 = comparison['dataset1']
        dataset2 = comparison['dataset2']
        
        print(f"\n📈 DATASET STATISTICS:")
        print(f"{'Metric':<25} {dataset1['name']:<25} {dataset2['name']:<25}")
        print("-" * 75)
        print(f"{'Vocabulary Size':<25} {dataset1['statistics']['vocab_size']:<25} {dataset2['statistics']['vocab_size']:<25}")
        print(f"{'Total Shards':<25} {dataset1['statistics']['total_shards']:<25} {dataset2['statistics']['total_shards']:<25}")
        print(f"{'Train Shards':<25} {dataset1['statistics']['num_train_shards']:<25} {dataset2['statistics']['num_train_shards']:<25}")
        print(f"{'Val Shards':<25} {dataset1['statistics']['num_val_shards']:<25} {dataset2['statistics']['num_val_shards']:<25}")
        print(f"{'Test Shards':<25} {dataset1['statistics']['num_test_shards']:<25} {dataset2['statistics']['num_test_shards']:<25}")
        
        # Sequence length statistics
        seq1 = dataset1['statistics']['sequence_length_stats']
        seq2 = dataset2['statistics']['sequence_length_stats']
        
        print(f"\n📏 SEQUENCE LENGTH STATISTICS:")
        print(f"{'Metric':<15} {dataset1['name']:<20} {dataset2['name']:<20}")
        print("-" * 55)
        print(f"{'Mean':<15} {seq1['mean']:<20.2f} {seq2['mean']:<20.2f}")
        print(f"{'Median':<15} {seq1['median']:<20.2f} {seq2['median']:<20.2f}")
        print(f"{'Std Dev':<15} {seq1['std']:<20.2f} {seq2['std']:<20.2f}")
        print(f"{'Min':<15} {seq1['min']:<20.2f} {seq2['min']:<20.2f}")
        print(f"{'Max':<15} {seq1['max']:<20.2f} {seq2['max']:<20.2f}")
        
        # Training difficulty factors
        if 'error' not in dataset1['training_difficulty'] and 'error' not in dataset2['training_difficulty']:
            diff1 = dataset1['training_difficulty']
            diff2 = dataset2['training_difficulty']
            
            print(f"\n🎯 TRAINING DIFFICULTY FACTORS:")
            print(f"{'Factor':<25} {dataset1['name']:<20} {dataset2['name']:<20}")
            print("-" * 65)
            print(f"{'Complexity Score':<25} {diff1['complexity_score']:<20.4f} {diff2['complexity_score']:<20.4f}")
            print(f"{'Length Consistency':<25} {diff1['length_consistency']:<20.4f} {diff2['length_consistency']:<20.4f}")
            print(f"{'Vocab Efficiency':<25} {diff1['vocab_efficiency']:<20.4f} {diff2['vocab_efficiency']:<20.4f}")
            print(f"{'Training Stability':<25} {diff1['training_stability_score']:<20.4f} {diff2['training_stability_score']:<20.4f}")
            print(f"{'Memory Efficiency':<25} {diff1['memory_efficiency']:<20.4f} {diff2['memory_efficiency']:<20.4f}")
            print(f"{'Sequence Diversity':<25} {diff1['sequence_diversity']:<20.4f} {diff2['sequence_diversity']:<20.4f}")
            print(f"{'Repetition Ratio':<25} {diff1['repetition_ratio']:<20.4f} {diff2['repetition_ratio']:<20.4f}")
            print(f"{'Position Consistency':<25} {diff1['position_consistency']:<20.4f} {diff2['position_consistency']:<20.4f}")
            
            # Overall recommendation
            if 'comparison' in comparison:
                comp = comparison['comparison']
                print(f"\n🏆 OVERALL RECOMMENDATION:")
                print(f"✅ Easier Dataset: {comp['easier_dataset']}")
                print(f"📊 Improvement: {comp['improvement_percentage']:.2f}% easier")
                print(f"🎯 Complexity Score: {comp['easier_complexity_score']:.4f} vs {comp['harder_complexity_score']:.4f}")
                
                print(f"\n🔍 KEY DIFFERENCES:")
                key_diff = comp['key_differences']
                print(f"  Vocabulary Size Difference: {key_diff['vocab_size_diff']:+,}")
                print(f"  Length Consistency Difference: {key_diff['length_consistency_diff']:+.4f}")
                print(f"  Stability Score Difference: {key_diff['stability_score_diff']:+.4f}")
                print(f"  Memory Efficiency Difference: {key_diff['memory_efficiency_diff']:+.4f}")
        
        print(f"\n✅ Comparison complete!")

def main():
    """Main function to compare the two datasets."""
    
    # Dataset paths
    dataset1_path = "/home/abedin/Developments/chest_x_ray_data_processing/all_processed_data/mimic_shards_4446_128"
    dataset2_path = "/home/abedin/Developments/chest_x_ray_data_processing/all_processed_data/mimic_shards_hufc4446-to128"
    
    # Validate paths
    if not os.path.exists(dataset1_path):
        print(f"❌ Dataset 1 not found: {dataset1_path}")
        return
    
    if not os.path.exists(dataset2_path):
        print(f"❌ Dataset 2 not found: {dataset2_path}")
        return
    
    # Create comparator and run analysis
    comparator = DatasetComparator(dataset1_path, dataset2_path)
    comparison = comparator.compare_datasets()
    
    # Print results
    comparator.print_comparison_summary(comparison)
    
    # Save results
    output_path = "mimic_datasets_comparison.json"
    with open(output_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_path}")

if __name__ == "__main__":
    main() 