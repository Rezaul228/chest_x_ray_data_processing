#!/usr/bin/env python3
"""
Analyze MIMIC Original Shards Dataset

This script analyzes the mimic_original_shards dataset to extract:
- Vocabulary size
- Number of samples
- Token length statistics
"""

import os
import sys
import json
import pickle
import glob
import numpy as np
import pandas as pd
from collections import Counter
from typing import Dict, List, Any
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')


class MIMICOriginalAnalyzer:
    """
    Analyzes the MIMIC original shards dataset.
    """
    
    def __init__(self, dataset_path: str, text_key: str = "captions"):
        self.dataset_path = dataset_path
        self.text_key = text_key
        self.stats = {}
        
    def analyze_dataset(self):
        """Analyze the entire dataset."""
        print(f"🔍 Analyzing MIMIC Original Shards Dataset")
        print(f"📁 Path: {self.dataset_path}")
        
        # Check if dataset exists
        if not os.path.exists(self.dataset_path):
            print(f"❌ Dataset path does not exist: {self.dataset_path}")
            return
        
        # Load metadata first
        self._load_metadata()
        
        # Analyze each split
        splits = ['train', 'val', 'test']
        total_samples = 0
        all_tokens = []
        all_sequences = []
        
        for split in splits:
            split_path = os.path.join(self.dataset_path, split)
            if os.path.exists(split_path):
                print(f"\n📊 Analyzing {split} split...")
                split_stats = self._analyze_split(split_path, split)
                total_samples += split_stats['total_samples']
                all_tokens.extend(split_stats['all_tokens'])
                all_sequences.extend(split_stats['all_sequences'])
                self.stats[split] = split_stats
        
        # Calculate overall statistics
        print(f"\n📈 Calculating overall statistics...")
        
        # Vocabulary analysis
        vocab_counter = Counter(all_tokens)
        vocab_size = len(vocab_counter)
        
        # Token length analysis
        sequence_lengths = [len(seq) for seq in all_sequences if len(seq) > 0]
        sequence_lengths = np.array(sequence_lengths)
        
        if len(sequence_lengths) > 0:
            token_length_stats = {
                'min': float(np.min(sequence_lengths)),
                'max': float(np.max(sequence_lengths)),
                'mean': float(np.mean(sequence_lengths)),
                'median': float(np.median(sequence_lengths)),
                'std': float(np.std(sequence_lengths)),
                'q25': float(np.percentile(sequence_lengths, 25)),
                'q75': float(np.percentile(sequence_lengths, 75))
            }
        else:
            token_length_stats = {
                'min': 0, 'max': 0, 'mean': 0, 'median': 0, 
                'std': 0, 'q25': 0, 'q75': 0
            }
        
        # Overall statistics
        self.stats['overall'] = {
            'total_samples': total_samples,
            'vocabulary_size': vocab_size,
            'total_tokens': len(all_tokens),
            'avg_tokens_per_sample': len(all_tokens) / total_samples if total_samples > 0 else 0,
            'token_length_stats': token_length_stats,
            'vocabulary_distribution': {
                'most_common_tokens': dict(vocab_counter.most_common(10)),
                'least_common_tokens': dict(sorted(vocab_counter.items(), key=lambda x: x[1])[:10]),
                'token_frequency_stats': {
                    'mean': float(np.mean(list(vocab_counter.values()))),
                    'median': float(np.median(list(vocab_counter.values()))),
                    'std': float(np.std(list(vocab_counter.values()))),
                    'min': float(min(vocab_counter.values())),
                    'max': float(max(vocab_counter.values()))
                }
            }
        }
        
        print(f"✅ Analysis complete!")
    
    def _load_metadata(self):
        """Load metadata from the dataset."""
        metadata_path = os.path.join(self.dataset_path, 'metadata.pkl')
        enhanced_metadata_path = os.path.join(self.dataset_path, 'enhanced_metadata.pkl')
        
        if os.path.exists(enhanced_metadata_path):
            try:
                with open(enhanced_metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                print(f"📄 Loaded enhanced metadata")
                self.metadata = metadata
            except Exception as e:
                print(f"⚠️  Error loading enhanced metadata: {e}")
                self.metadata = {}
        elif os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                print(f"📄 Loaded standard metadata")
                self.metadata = metadata
            except Exception as e:
                print(f"⚠️  Error loading metadata: {e}")
                self.metadata = {}
        else:
            print(f"⚠️  No metadata found")
            self.metadata = {}
    
    def _analyze_split(self, split_path: str, split_name: str) -> Dict[str, Any]:
        """Analyze a specific split of the dataset."""
        # Find all pickle files
        file_pattern = os.path.join(split_path, "*.pkl")
        files = glob.glob(file_pattern)
        
        if not files:
            print(f"❌ No files found in {split_path}")
            return {
                'total_samples': 0,
                'all_tokens': [],
                'all_sequences': []
            }
        
        print(f"📄 Found {len(files)} files")
        
        # Load and process all data
        all_entries = []
        for file_path in tqdm(files, desc=f"Loading {split_name}"):
            try:
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                
                if isinstance(data, dict):
                    # Convert dict to list of entries
                    entries = []
                    for i in range(len(data.get('images', []))):
                        entry = {}
                        for key, value in data.items():
                            if isinstance(value, (list, np.ndarray)) and i < len(value):
                                entry[key] = value[i]
                            else:
                                entry[key] = value
                        entries.append(entry)
                    all_entries.extend(entries)
                elif isinstance(data, list):
                    all_entries.extend(data)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
        
        print(f"📊 Loaded {len(all_entries)} entries")
        
        # Extract text and tokenize
        all_tokens = []
        all_sequences = []
        valid_entries = 0
        
        for entry in tqdm(all_entries, desc=f"Processing {split_name}"):
            text = self._extract_text(entry)
            
            if text is not None:
                if isinstance(text, (list, np.ndarray)):
                    # Pre-tokenized text
                    tokens = [str(token) for token in text if token != 0]
                    all_tokens.extend(tokens)
                    all_sequences.append(tokens)
                    valid_entries += 1
                else:
                    # Raw text
                    tokens = text.lower().split()
                    all_tokens.extend(tokens)
                    all_sequences.append(tokens)
                    valid_entries += 1
        
        return {
            'total_samples': valid_entries,
            'all_tokens': all_tokens,
            'all_sequences': all_sequences
        }
    
    def _extract_text(self, entry: Dict) -> Any:
        """Extract text from entry."""
        if self.text_key in entry:
            return entry[self.text_key]
        
        # Try alternative keys
        for key in ['text', 'report', 'caption', 'findings', 'impression']:
            if key in entry:
                return entry[key]
        
        return None
    
    def print_summary(self):
        """Print a comprehensive summary of the analysis."""
        if 'overall' not in self.stats:
            print("❌ No analysis results available")
            return
        
        overall = self.stats['overall']
        token_stats = overall['token_length_stats']
        vocab_stats = overall['vocabulary_distribution']['token_frequency_stats']
        
        print("\n" + "="*80)
        print("📊 MIMIC ORIGINAL SHARDS DATASET ANALYSIS SUMMARY")
        print("="*80)
        
        print(f"\n📈 OVERALL STATISTICS:")
        print(f"  Total Samples: {overall['total_samples']:,}")
        print(f"  Vocabulary Size: {overall['vocabulary_size']:,}")
        print(f"  Total Tokens: {overall['total_tokens']:,}")
        print(f"  Average Tokens per Sample: {overall['avg_tokens_per_sample']:.2f}")
        
        print(f"\n📏 TOKEN LENGTH STATISTICS:")
        print(f"  Minimum Length: {token_stats['min']:.1f}")
        print(f"  Maximum Length: {token_stats['max']:.1f}")
        print(f"  Mean Length: {token_stats['mean']:.2f}")
        print(f"  Median Length: {token_stats['median']:.1f}")
        print(f"  Standard Deviation: {token_stats['std']:.2f}")
        print(f"  25th Percentile: {token_stats['q25']:.1f}")
        print(f"  75th Percentile: {token_stats['q75']:.1f}")
        
        print(f"\n📚 VOCABULARY DISTRIBUTION:")
        print(f"  Token Frequency Mean: {vocab_stats['mean']:.2f}")
        print(f"  Token Frequency Median: {vocab_stats['median']:.1f}")
        print(f"  Token Frequency Std: {vocab_stats['std']:.2f}")
        print(f"  Most Frequent Token Count: {vocab_stats['max']:,}")
        print(f"  Least Frequent Token Count: {vocab_stats['min']:,}")
        
        print(f"\n📋 SPLIT BREAKDOWN:")
        for split in ['train', 'val', 'test']:
            if split in self.stats:
                split_data = self.stats[split]
                print(f"  {split.upper()}: {split_data['total_samples']:,} samples")
        
        print(f"\n🔝 MOST COMMON TOKENS:")
        most_common = overall['vocabulary_distribution']['most_common_tokens']
        for i, (token, count) in enumerate(most_common.items(), 1):
            print(f"  {i:2d}. '{token}': {count:,}")
        
        print(f"\n🔚 LEAST COMMON TOKENS:")
        least_common = overall['vocabulary_distribution']['least_common_tokens']
        for i, (token, count) in enumerate(least_common.items(), 1):
            print(f"  {i:2d}. '{token}': {count:,}")
        
        # Metadata information if available
        if hasattr(self, 'metadata') and self.metadata:
            print(f"\n📄 METADATA INFORMATION:")
            for key, value in self.metadata.items():
                if key != 'tokenizer':  # Skip tokenizer object
                    print(f"  {key}: {value}")
    
    def save_results(self, output_path: str):
        """Save analysis results to JSON file."""
        results = {
            'analysis_timestamp': pd.Timestamp.now().isoformat(),
            'dataset_path': self.dataset_path,
            'statistics': self.stats,
            'metadata': self.metadata if hasattr(self, 'metadata') else {}
        }
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"💾 Results saved to: {output_path}")


def main():
    """Main function."""
    dataset_path = "/home/abedin/Developments/chest_x_ray_data_processing/mimic_original_shards"
    
    analyzer = MIMICOriginalAnalyzer(dataset_path, text_key="captions")
    
    # Analyze the dataset
    analyzer.analyze_dataset()
    
    # Print summary
    analyzer.print_summary()
    
    # Save results
    analyzer.save_results("mimic_original_analysis_results.json")
    
    print("\n✅ MIMIC Original Shards analysis complete!")


if __name__ == "__main__":
    main() 