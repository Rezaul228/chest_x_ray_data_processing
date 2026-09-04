#!/usr/bin/env python3
"""
Quick script to check UNK token rates in processed Indiana data
"""

import os
import sys
import pickle
import numpy as np
import argparse
from pathlib import Path

def load_shard(shard_path):
    """Load a single shard file"""
    with open(shard_path, 'rb') as f:
        return pickle.load(f)

def analyze_unk_tokens(shards_dir, num_shards=5, samples_per_shard=20):
    """Analyze UNK token rates in processed shards"""
    print(f"Analyzing UNK tokens in {shards_dir}")
    print("=" * 60)
    
    # Find shard files
    shard_files = sorted([f for f in os.listdir(shards_dir) if f.endswith('.pkl')])
    
    if not shard_files:
        print(f"No shard files found in {shards_dir}")
        return
    
    print(f"Found {len(shard_files)} shard files")
    print(f"Analyzing first {min(num_shards, len(shard_files))} shards")
    
    total_unk_count = 0
    total_tokens = 0
    sample_count = 0
    
    # Analyze first few shards
    for i, shard_file in enumerate(shard_files[:num_shards]):
        shard_path = os.path.join(shards_dir, shard_file)
        print(f"\nLoading shard {i+1}: {shard_file}")
        
        try:
            shard_data = load_shard(shard_path)
            
            if 'captions' not in shard_data:
                print(f"  No captions found in shard")
                continue
                
            captions = shard_data['captions']
            print(f"  Shard contains {len(captions)} samples")
            
            # Sample some captions from this shard
            sample_indices = np.random.choice(len(captions), 
                                            min(samples_per_shard, len(captions)), 
                                            replace=False)
            
            for idx in sample_indices:
                caption = captions[idx]
                # Count UNK tokens (token ID 1)
                unk_count = np.sum(caption == 1)
                total_tokens_in_sample = np.sum(caption != 0)  # Exclude padding (0)
                
                total_unk_count += unk_count
                total_tokens += total_tokens_in_sample
                sample_count += 1
                
                # Print first few samples for inspection
                if sample_count <= 3:
                    print(f"    Sample {sample_count}: {unk_count}/{total_tokens_in_sample} UNK tokens")
                    
                    # Decode a few tokens for inspection
                    decoded_words = []
                    for token in caption[:10]:  # First 10 tokens
                        if token == 0:
                            decoded_words.append('<pad>')
                        elif token == 1:
                            decoded_words.append('<unk>')
                        else:
                            decoded_words.append(f'token_{token}')
                    
                    print(f"      First 10 tokens: {' '.join(decoded_words)}")
                    
        except Exception as e:
            print(f"  Error loading shard: {e}")
            continue
    
    # Calculate overall statistics
    if total_tokens > 0:
        unk_ratio = total_unk_count / total_tokens
        print(f"\n" + "=" * 60)
        print(f"UNK TOKEN ANALYSIS RESULTS")
        print("=" * 60)
        print(f"Total samples analyzed: {sample_count}")
        print(f"Total tokens (excluding padding): {total_tokens}")
        print(f"Total UNK tokens: {total_unk_count}")
        print(f"UNK token ratio: {unk_ratio:.2%}")
        print(f"UNK token percentage: {unk_ratio * 100:.1f}%")
        
        if unk_ratio > 0.1:
            print(f"\n⚠️  HIGH UNK RATE: {unk_ratio:.1%} - This indicates vocabulary coverage issues")
        elif unk_ratio > 0.05:
            print(f"\n⚠️  MODERATE UNK RATE: {unk_ratio:.1%} - Some vocabulary coverage issues")
        else:
            print(f"\n✅ GOOD UNK RATE: {unk_ratio:.1%} - Good vocabulary coverage")
    else:
        print("No valid tokens found for analysis")

def main():
    parser = argparse.ArgumentParser(description='Check UNK token rates in processed shards')
    parser.add_argument('--shards_dir', type=str, default='shards_simplified/train',
                       help='Directory containing shard files')
    parser.add_argument('--num_shards', type=int, default=5,
                       help='Number of shards to analyze')
    parser.add_argument('--samples_per_shard', type=int, default=20,
                       help='Number of samples to analyze per shard')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.shards_dir):
        print(f"Error: Shards directory not found: {args.shards_dir}")
        sys.exit(1)
    
    analyze_unk_tokens(args.shards_dir, args.num_shards, args.samples_per_shard)

if __name__ == "__main__":
    main() 