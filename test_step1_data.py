#!/usr/bin/env python3
"""
Test Step1 Processed Data - Test Directory
Check sequence length and vocabulary size
"""

import os
import sys
import pickle
import numpy as np
import glob

def test_step1_data():
    """Test the step1_processed_data/test directory"""
    
    test_dir = 'step1_processed_data/test'
    print(f"Testing Step1 Processed Data - Test Directory")
    print(f"Directory: {test_dir}")
    print("=" * 60)
    
    if not os.path.exists(test_dir):
        print(f"Error: Directory not found: {test_dir}")
        return
    
    # Find all shard files
    shard_files = glob.glob(os.path.join(test_dir, 'shard_*.pkl'))
    shard_files.sort()
    
    print(f"Found {len(shard_files)} shard files:")
    for shard in shard_files:
        size_mb = os.path.getsize(shard) / (1024 * 1024)
        print(f"  {os.path.basename(shard)}: {size_mb:.1f} MB")
    
    if not shard_files:
        print("No shard files found!")
        return
    
    # Load first shard to analyze structure
    print(f"\nLoading first shard: {shard_files[0]}")
    with open(shard_files[0], 'rb') as f:
        shard_data = pickle.load(f)
    
    print(f"Shard keys: {list(shard_data.keys())}")
    print(f"Number of samples: {len(shard_data['captions'])}")
    print(f"Caption shape: {shard_data['captions'].shape}")
    print(f"Image shape: {shard_data['images'].shape}")
    
    # Analyze sequence length
    caption_shape = shard_data['captions'].shape
    sequence_length = caption_shape[1] if len(caption_shape) > 1 else len(shard_data['captions'][0])
    print(f"\nSequence Length Analysis:")
    print(f"  Sequence length: {sequence_length}")
    
    # Check if sequence length is consistent
    all_lengths = []
    for caption in shard_data['captions']:
        if hasattr(caption, 'shape'):
            all_lengths.append(caption.shape[0])
        else:
            all_lengths.append(len(caption))
    
    unique_lengths = set(all_lengths)
    print(f"  Unique sequence lengths: {unique_lengths}")
    
    # Analyze vocabulary usage
    print(f"\nVocabulary Analysis:")
    all_tokens = []
    for caption in shard_data['captions']:
        if hasattr(caption, 'flatten'):
            tokens = caption.flatten()
        else:
            tokens = np.array(caption)
        all_tokens.extend(tokens.tolist())
    
    all_tokens = np.array(all_tokens)
    total_tokens = len(all_tokens)
    unique_tokens = len(np.unique(all_tokens))
    unk_tokens = np.sum(all_tokens == 1)  # Assuming 1 is UNK token
    padding_tokens = np.sum(all_tokens == 0)  # Assuming 0 is padding
    non_padding_tokens = total_tokens - padding_tokens
    
    print(f"  Total tokens: {total_tokens}")
    print(f"  Unique token values: {unique_tokens}")
    print(f"  Token range: {np.min(all_tokens)} to {np.max(all_tokens)}")
    print(f"  Padding tokens: {padding_tokens} ({padding_tokens/total_tokens*100:.1f}%)")
    print(f"  Non-padding tokens: {non_padding_tokens} ({non_padding_tokens/total_tokens*100:.1f}%)")
    print(f"  UNK tokens: {unk_tokens} ({unk_tokens/non_padding_tokens*100:.1f}% of non-padding)")
    
    # Check if there's a metadata file
    metadata_path = os.path.join(os.path.dirname(test_dir), 'metadata.pkl')
    if os.path.exists(metadata_path):
        print(f"\nLoading metadata: {metadata_path}")
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        print(f"Metadata keys: {list(metadata.keys())}")
        if 'vocab_size' in metadata:
            print(f"Vocabulary size from metadata: {metadata['vocab_size']}")
        if 'max_sequence_length' in metadata:
            print(f"Max sequence length from metadata: {metadata['max_sequence_length']}")
        if 'tokenizer' in metadata:
            tokenizer = metadata['tokenizer']
            print(f"Tokenizer word index size: {len(tokenizer.word_index)}")
            print(f"Tokenizer index word size: {len(tokenizer.index_word)}")
    else:
        print(f"\nNo metadata file found at: {metadata_path}")
    
    # Test a few sample captions
    print(f"\nSample Captions Analysis:")
    for i in range(min(3, len(shard_data['captions']))):
        caption = shard_data['captions'][i]
        study_id = shard_data['study_ids'][i] if 'study_ids' in shard_data else f"sample_{i}"
        
        print(f"\nSample {i+1} (Study ID: {study_id}):")
        print(f"  Shape: {caption.shape}")
        print(f"  Non-zero tokens: {np.count_nonzero(caption)}")
        print(f"  Non-padding tokens: {np.count_nonzero(caption != 0)}")
        print(f"  UNK tokens: {np.sum(caption == 1)}")
        print(f"  Unique tokens: {len(np.unique(caption))}")
        
        # Show first 10 tokens
        first_tokens = caption[:10]
        print(f"  First 10 tokens: {first_tokens}")
    
    # Summary
    print(f"\n" + "="*60)
    print(f"SUMMARY")
    print(f"="*60)
    print(f"Sequence Length: {sequence_length}")
    print(f"Vocabulary Size (unique tokens used): {unique_tokens}")
    print(f"UNK Ratio: {unk_tokens/non_padding_tokens*100:.2f}%")
    print(f"Padding Ratio: {padding_tokens/total_tokens*100:.1f}%")
    
    if unique_lengths == {sequence_length}:
        print(f"✅ Consistent sequence length across all samples")
    else:
        print(f"⚠️ Inconsistent sequence lengths: {unique_lengths}")

if __name__ == "__main__":
    test_step1_data() 