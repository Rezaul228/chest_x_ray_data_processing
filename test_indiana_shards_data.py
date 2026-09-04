#!/usr/bin/env python3
"""
Test Indiana Shards Data
Check sequence length and vocabulary size
"""

import os
import sys
import pickle
import numpy as np
import glob

def test_indiana_shards_data():
    """Test the all_processed_data/indiana_shards directory"""
    
    indiana_dir = 'all_processed_data/indiana_shards'
    print(f"Testing Indiana Shards Data")
    print(f"Directory: {indiana_dir}")
    print("=" * 60)
    
    if not os.path.exists(indiana_dir):
        print(f"Error: Directory not found: {indiana_dir}")
        return
    
    # Check metadata first
    metadata_path = os.path.join(indiana_dir, 'metadata.pkl')
    if os.path.exists(metadata_path):
        print(f"Loading metadata: {metadata_path}")
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
        print(f"No metadata file found at: {metadata_path}")
    
    # Test train data
    train_dir = os.path.join(indiana_dir, 'train')
    if os.path.exists(train_dir):
        print(f"\nTesting Train Data:")
        print(f"Directory: {train_dir}")
        
        train_shards = glob.glob(os.path.join(train_dir, 'shard_*.pkl'))
        train_shards.sort()
        print(f"Found {len(train_shards)} train shard files")
        
        if train_shards:
            # Load first train shard
            print(f"Loading first train shard: {train_shards[0]}")
            with open(train_shards[0], 'rb') as f:
                train_data = pickle.load(f)
            
            print(f"Train shard keys: {list(train_data.keys())}")
            print(f"Number of samples: {len(train_data['captions'])}")
            print(f"Caption shape: {train_data['captions'].shape}")
            print(f"Image shape: {train_data['images'].shape}")
            
            # Analyze sequence length
            caption_shape = train_data['captions'].shape
            sequence_length = caption_shape[1] if len(caption_shape) > 1 else len(train_data['captions'][0])
            print(f"Sequence length: {sequence_length}")
            
            # Analyze vocabulary usage
            all_tokens = []
            for caption in train_data['captions']:
                if hasattr(caption, 'flatten'):
                    tokens = caption.flatten()
                else:
                    tokens = np.array(caption)
                all_tokens.extend(tokens.tolist())
            
            all_tokens = np.array(all_tokens)
            total_tokens = len(all_tokens)
            unique_tokens = len(np.unique(all_tokens))
            unk_tokens = np.sum(all_tokens == 1)
            padding_tokens = np.sum(all_tokens == 0)
            non_padding_tokens = total_tokens - padding_tokens
            
            print(f"Train vocabulary analysis:")
            print(f"  Total tokens: {total_tokens}")
            print(f"  Unique token values: {unique_tokens}")
            print(f"  Token range: {np.min(all_tokens)} to {np.max(all_tokens)}")
            print(f"  Padding tokens: {padding_tokens} ({padding_tokens/total_tokens*100:.1f}%)")
            print(f"  Non-padding tokens: {non_padding_tokens} ({non_padding_tokens/total_tokens*100:.1f}%)")
            print(f"  UNK tokens: {unk_tokens} ({unk_tokens/non_padding_tokens*100:.1f}% of non-padding)")
    
    # Test test data
    test_dir = os.path.join(indiana_dir, 'test')
    if os.path.exists(test_dir):
        print(f"\nTesting Test Data:")
        print(f"Directory: {test_dir}")
        
        test_shards = glob.glob(os.path.join(test_dir, 'shard_*.pkl'))
        test_shards.sort()
        print(f"Found {len(test_shards)} test shard files")
        
        if test_shards:
            # Load first test shard
            print(f"Loading first test shard: {test_shards[0]}")
            with open(test_shards[0], 'rb') as f:
                test_data = pickle.load(f)
            
            print(f"Test shard keys: {list(test_data.keys())}")
            print(f"Number of samples: {len(test_data['captions'])}")
            print(f"Caption shape: {test_data['captions'].shape}")
            print(f"Image shape: {test_data['images'].shape}")
            
            # Analyze vocabulary usage for test data
            all_test_tokens = []
            for caption in test_data['captions']:
                if hasattr(caption, 'flatten'):
                    tokens = caption.flatten()
                else:
                    tokens = np.array(caption)
                all_test_tokens.extend(tokens.tolist())
            
            all_test_tokens = np.array(all_test_tokens)
            total_test_tokens = len(all_test_tokens)
            unique_test_tokens = len(np.unique(all_test_tokens))
            unk_test_tokens = np.sum(all_test_tokens == 1)
            padding_test_tokens = np.sum(all_test_tokens == 0)
            non_padding_test_tokens = total_test_tokens - padding_test_tokens
            
            print(f"Test vocabulary analysis:")
            print(f"  Total tokens: {total_test_tokens}")
            print(f"  Unique token values: {unique_test_tokens}")
            print(f"  Token range: {np.min(all_test_tokens)} to {np.max(all_test_tokens)}")
            print(f"  Padding tokens: {padding_test_tokens} ({padding_test_tokens/total_test_tokens*100:.1f}%)")
            print(f"  Non-padding tokens: {non_padding_test_tokens} ({non_padding_test_tokens/total_test_tokens*100:.1f}%)")
            print(f"  UNK tokens: {unk_test_tokens} ({unk_test_tokens/non_padding_test_tokens*100:.1f}% of non-padding)")
    
    # Test val data
    val_dir = os.path.join(indiana_dir, 'val')
    if os.path.exists(val_dir):
        print(f"\nTesting Validation Data:")
        print(f"Directory: {val_dir}")
        
        val_shards = glob.glob(os.path.join(val_dir, 'shard_*.pkl'))
        val_shards.sort()
        print(f"Found {len(val_shards)} validation shard files")
        
        if val_shards:
            # Load first val shard
            print(f"Loading first validation shard: {val_shards[0]}")
            with open(val_shards[0], 'rb') as f:
                val_data = pickle.load(f)
            
            print(f"Validation shard keys: {list(val_data.keys())}")
            print(f"Number of samples: {len(val_data['captions'])}")
            print(f"Caption shape: {val_data['captions'].shape}")
            print(f"Image shape: {val_data['images'].shape}")
    
    # Summary
    print(f"\n" + "="*60)
    print(f"SUMMARY")
    print(f"="*60)
    
    if 'train_data' in locals():
        print(f"Sequence Length: {sequence_length}")
        print(f"Vocabulary Size (unique tokens used in train): {unique_tokens}")
        print(f"UNK Ratio (train): {unk_tokens/non_padding_tokens*100:.2f}%")
        print(f"Padding Ratio (train): {padding_tokens/total_tokens*100:.1f}%")
        
        if 'test_data' in locals():
            print(f"UNK Ratio (test): {unk_test_tokens/non_padding_test_tokens*100:.2f}%")
            print(f"Padding Ratio (test): {padding_test_tokens/total_test_tokens*100:.1f}%")

if __name__ == "__main__":
    test_indiana_shards_data() 