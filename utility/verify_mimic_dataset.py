#!/usr/bin/env python3

import pickle
import os
import numpy as np
from collections import Counter
import json

def verify_mimic_dataset():
    """Verify token length, vocab size, and number of samples in MIMIC dataset"""
    
    dataset_path = '/home/abedin/Developments/chest_x_ray_data_processing/all_processed_data/mimic_original_shards'
    
    print("="*80)
    print("MIMIC DATASET VERIFICATION")
    print("="*80)
    print(f"Dataset path: {dataset_path}")
    
    # Check if dataset exists
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        return
    
    # Load metadata
    metadata_path = os.path.join(dataset_path, 'metadata.pkl')
    enhanced_metadata_path = os.path.join(dataset_path, 'enhanced_metadata.pkl')
    
    print(f"\nLoading metadata...")
    
    metadata = None
    enhanced_metadata = None
    
    if os.path.exists(metadata_path):
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        print(f"✓ Loaded metadata.pkl")
    
    if os.path.exists(enhanced_metadata_path):
        with open(enhanced_metadata_path, 'rb') as f:
            enhanced_metadata = pickle.load(f)
        print(f"✓ Loaded enhanced_metadata.pkl")
    
    # Analyze metadata
    print(f"\n" + "="*60)
    print("METADATA ANALYSIS")
    print("="*60)
    
    if metadata:
        print(f"Basic metadata keys: {list(metadata.keys())}")
        
        # Check vocabulary info
        if 'tokenizer' in metadata:
            tokenizer = metadata['tokenizer']
            print(f"\nTokenizer type: {type(tokenizer).__name__}")
            
            if hasattr(tokenizer, 'vocab'):
                vocab = tokenizer.vocab
                print(f"Vocabulary size: {len(vocab)}")
                print(f"Vocabulary type: {type(vocab).__name__}")
                
                # Show special tokens
                special_tokens = ['<PAD>', '<UNK>', '<START>', '<END>', '<SEP>']
                print(f"\nSpecial tokens:")
                for token in special_tokens:
                    if token in vocab:
                        print(f"  '{token}' -> {vocab[token]}")
                    else:
                        print(f"  '{token}' -> Not found")
                
                # Show sample vocabulary items
                print(f"\nSample vocabulary items (first 20):")
                sample_items = list(vocab.items())[:20]
                for word, idx in sample_items:
                    print(f"  '{word}' -> {idx}")
        
        # Check other metadata fields
        for key, value in metadata.items():
            if key != 'tokenizer':
                if isinstance(value, (int, float, str, bool)):
                    print(f"{key}: {value}")
                elif hasattr(value, '__len__'):
                    print(f"{key}: {len(value)} items")
                else:
                    print(f"{key}: {type(value).__name__}")
    
    if enhanced_metadata:
        print(f"\nEnhanced metadata keys: {list(enhanced_metadata.keys())}")
        
        # Check vocabulary info in enhanced metadata
        if 'tokenizer' in enhanced_metadata:
            tokenizer = enhanced_metadata['tokenizer']
            print(f"\nEnhanced tokenizer type: {type(tokenizer).__name__}")
            
            if hasattr(tokenizer, 'vocab'):
                vocab = tokenizer.vocab
                print(f"Enhanced vocabulary size: {len(vocab)}")
    
    # Analyze shards
    print(f"\n" + "="*60)
    print("SHARD ANALYSIS")
    print("="*60)
    
    splits = ['train', 'val', 'test']
    total_samples = 0
    token_lengths = []
    vocab_usage = Counter()
    
    for split in splits:
        split_path = os.path.join(dataset_path, split)
        if not os.path.exists(split_path):
            print(f"Split '{split}' not found")
            continue
        
        shard_files = [f for f in os.listdir(split_path) if f.endswith('.pkl')]
        shard_files.sort()
        
        print(f"\n{split.upper()} split:")
        print(f"  Number of shards: {len(shard_files)}")
        
        split_samples = 0
        
        # Analyze first few shards for detailed info
        for i, shard_file in enumerate(shard_files[:3]):  # Analyze first 3 shards
            shard_path = os.path.join(split_path, shard_file)
            
            try:
                with open(shard_path, 'rb') as f:
                    shard_data = pickle.load(f)
                
                if isinstance(shard_data, list):
                    shard_samples = len(shard_data)
                    split_samples += shard_samples
                    
                    print(f"  Shard {shard_file}: {shard_samples} samples")
                    
                    # Analyze sample structure
                    if shard_samples > 0:
                        sample = shard_data[0]
                        print(f"    Sample keys: {list(sample.keys()) if isinstance(sample, dict) else 'Not a dict'}")
                        
                        # Check token length
                        if isinstance(sample, dict) and 'tokens' in sample:
                            for sample_data in shard_data[:10]:  # Check first 10 samples
                                tokens = sample_data.get('tokens', [])
                                if isinstance(tokens, (list, np.ndarray)):
                                    token_lengths.append(len(tokens))
                                
                                # Count vocabulary usage
                                if hasattr(tokens, '__iter__'):
                                    for token in tokens:
                                        if isinstance(token, (int, np.integer)):
                                            vocab_usage[token] += 1
                        
                        # Check image data
                        if isinstance(sample, dict) and 'image' in sample:
                            image = sample['image']
                            if hasattr(image, 'shape'):
                                print(f"    Image shape: {image.shape}")
                            elif hasattr(image, '__len__'):
                                print(f"    Image length: {len(image)}")
                
                elif isinstance(shard_data, dict):
                    print(f"  Shard {shard_file}: Dictionary with keys {list(shard_data.keys())}")
                    
                    if 'data' in shard_data:
                        data = shard_data['data']
                        if isinstance(data, list):
                            shard_samples = len(data)
                            split_samples += shard_samples
                            print(f"    Data samples: {shard_samples}")
                
            except Exception as e:
                print(f"  Error reading {shard_file}: {e}")
        
        # Estimate total samples for this split
        if len(shard_files) > 3:
            avg_shard_size = split_samples / 3
            estimated_total = avg_shard_size * len(shard_files)
            print(f"  Estimated total samples: {estimated_total:.0f}")
            split_samples = estimated_total
        
        total_samples += split_samples
        print(f"  Total samples in {split}: {split_samples:.0f}")
    
    print(f"\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    print(f"Total estimated samples: {total_samples:.0f}")
    
    if token_lengths:
        print(f"\nToken length statistics:")
        print(f"  Number of samples analyzed: {len(token_lengths)}")
        print(f"  Mean token length: {np.mean(token_lengths):.2f}")
        print(f"  Median token length: {np.median(token_lengths):.2f}")
        print(f"  Min token length: {np.min(token_lengths)}")
        print(f"  Max token length: {np.max(token_lengths)}")
        print(f"  Std token length: {np.std(token_lengths):.2f}")
        
        # Token length distribution
        print(f"\nToken length distribution:")
        bins = [0, 50, 100, 150, 200, 250, 300, 400, 500, 1000]
        for i in range(len(bins)-1):
            count = sum(1 for length in token_lengths if bins[i] <= length < bins[i+1])
            percentage = (count / len(token_lengths)) * 100
            print(f"  {bins[i]}-{bins[i+1]}: {count} samples ({percentage:.1f}%)")
        
        # Long tail
        count_long = sum(1 for length in token_lengths if length >= 1000)
        percentage_long = (count_long / len(token_lengths)) * 100
        print(f"  1000+: {count_long} samples ({percentage_long:.1f}%)")
    
    if vocab_usage:
        print(f"\nVocabulary usage statistics:")
        print(f"  Unique tokens used: {len(vocab_usage)}")
        print(f"  Total token occurrences: {sum(vocab_usage.values())}")
        
        # Most common tokens
        print(f"\nMost common tokens:")
        for token_id, count in vocab_usage.most_common(10):
            print(f"  Token {token_id}: {count} occurrences")
        
        # UNK token analysis
        if metadata and 'tokenizer' in metadata:
            tokenizer = metadata['tokenizer']
            if hasattr(tokenizer, 'vocab') and '<UNK>' in tokenizer.vocab:
                unk_id = tokenizer.vocab['<UNK>']
                unk_count = vocab_usage.get(unk_id, 0)
                total_tokens = sum(vocab_usage.values())
                unk_percentage = (unk_count / total_tokens) * 100 if total_tokens > 0 else 0
                print(f"\nUNK token analysis:")
                print(f"  UNK token ID: {unk_id}")
                print(f"  UNK occurrences: {unk_count}")
                print(f"  UNK percentage: {unk_percentage:.2f}%")
    
    print(f"\n" + "="*80)

if __name__ == "__main__":
    verify_mimic_dataset() 