#!/usr/bin/env python3
"""
Simple MIMIC Original Shards Dataset Analysis

This script analyzes the mimic_original_shards dataset to extract:
- Vocabulary size
- Number of samples
- Token length statistics
"""

import os
import sys
import pickle
import glob
import numpy as np
from collections import Counter
from typing import Dict, List, Any
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')


def analyze_mimic_original_dataset():
    """Analyze the MIMIC original shards dataset."""
    
    dataset_path = "/home/abedin/Developments/chest_x_ray_data_processing/mimic_original_shards"
    text_key = "captions"
    
    print(f"🔍 Analyzing MIMIC Original Shards Dataset")
    print(f"📁 Path: {dataset_path}")
    
    # Check if dataset exists
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset path does not exist: {dataset_path}")
        return
    
    # Load metadata first
    metadata = load_metadata(dataset_path)
    
    # Analyze each split
    splits = ['train', 'val', 'test']
    total_samples = 0
    all_tokens = []
    all_sequences = []
    
    for split in splits:
        split_path = os.path.join(dataset_path, split)
        if os.path.exists(split_path):
            print(f"\n📊 Analyzing {split} split...")
            split_stats = analyze_split(split_path, split, text_key)
            total_samples += split_stats['total_samples']
            all_tokens.extend(split_stats['all_tokens'])
            all_sequences.extend(split_stats['all_sequences'])
    
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
    
    # Print results
    print_summary(total_samples, vocab_size, all_tokens, token_length_stats, metadata)


def load_metadata(dataset_path: str) -> Dict:
    """Load metadata from the dataset."""
    metadata_path = os.path.join(dataset_path, 'metadata.pkl')
    enhanced_metadata_path = os.path.join(dataset_path, 'enhanced_metadata.pkl')
    
    if os.path.exists(enhanced_metadata_path):
        try:
            with open(enhanced_metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            print(f"📄 Loaded enhanced metadata")
            return metadata
        except Exception as e:
            print(f"⚠️  Error loading enhanced metadata: {e}")
    elif os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            print(f"📄 Loaded standard metadata")
            return metadata
        except Exception as e:
            print(f"⚠️  Error loading metadata: {e}")
    
    print(f"⚠️  No metadata found")
    return {}


def analyze_split(split_path: str, split_name: str, text_key: str) -> Dict[str, Any]:
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
        text = extract_text(entry, text_key)
        
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


def extract_text(entry: Dict, text_key: str) -> Any:
    """Extract text from entry."""
    if text_key in entry:
        return entry[text_key]
    
    # Try alternative keys
    for key in ['text', 'report', 'caption', 'findings', 'impression']:
        if key in entry:
            return entry[key]
    
    return None


def print_summary(total_samples: int, vocab_size: int, all_tokens: List[str], 
                 token_length_stats: Dict, metadata: Dict):
    """Print a comprehensive summary of the analysis."""
    
    print("\n" + "="*80)
    print("📊 MIMIC ORIGINAL SHARDS DATASET ANALYSIS SUMMARY")
    print("="*80)
    
    print(f"\n📈 OVERALL STATISTICS:")
    print(f"  Total Samples: {total_samples:,}")
    print(f"  Vocabulary Size: {vocab_size:,}")
    print(f"  Total Tokens: {len(all_tokens):,}")
    print(f"  Average Tokens per Sample: {len(all_tokens) / total_samples:.2f}" if total_samples > 0 else "  Average Tokens per Sample: 0.00")
    
    print(f"\n📏 TOKEN LENGTH STATISTICS:")
    print(f"  Minimum Length: {token_length_stats['min']:.1f}")
    print(f"  Maximum Length: {token_length_stats['max']:.1f}")
    print(f"  Mean Length: {token_length_stats['mean']:.2f}")
    print(f"  Median Length: {token_length_stats['median']:.1f}")
    print(f"  Standard Deviation: {token_length_stats['std']:.2f}")
    print(f"  25th Percentile: {token_length_stats['q25']:.1f}")
    print(f"  75th Percentile: {token_length_stats['q75']:.1f}")
    
    # Vocabulary distribution
    vocab_counter = Counter(all_tokens)
    token_freq_values = list(vocab_counter.values())
    
    if token_freq_values:
        print(f"\n📚 VOCABULARY DISTRIBUTION:")
        print(f"  Token Frequency Mean: {np.mean(token_freq_values):.2f}")
        print(f"  Token Frequency Median: {np.median(token_freq_values):.1f}")
        print(f"  Token Frequency Std: {np.std(token_freq_values):.2f}")
        print(f"  Most Frequent Token Count: {max(token_freq_values):,}")
        print(f"  Least Frequent Token Count: {min(token_freq_values):,}")
        
        print(f"\n🔝 MOST COMMON TOKENS:")
        most_common = vocab_counter.most_common(10)
        for i, (token, count) in enumerate(most_common, 1):
            print(f"  {i:2d}. '{token}': {count:,}")
        
        print(f"\n🔚 LEAST COMMON TOKENS:")
        least_common = sorted(vocab_counter.items(), key=lambda x: x[1])[:10]
        for i, (token, count) in enumerate(least_common, 1):
            print(f"  {i:2d}. '{token}': {count:,}")
    
    # Metadata information if available
    if metadata:
        print(f"\n📄 METADATA INFORMATION:")
        for key, value in metadata.items():
            if key != 'tokenizer':  # Skip tokenizer object
                if isinstance(value, (int, float, str, bool)):
                    print(f"  {key}: {value}")
                else:
                    print(f"  {key}: {type(value).__name__}")
    
    print(f"\n✅ Analysis complete!")


if __name__ == "__main__":
    analyze_mimic_original_dataset() 