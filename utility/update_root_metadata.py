#!/usr/bin/env python3

import pickle
import os

def update_root_metadata():
    """Update the root-level metadata.pkl with augmented tokenizer and vocabulary"""
    
    # Load the augmented tokenizer and vocab from the train split
    train_metadata_path = 'step3_augmented_data/train/train_metadata.pkl'
    if not os.path.exists(train_metadata_path):
        print(f"Error: Train metadata not found at {train_metadata_path}")
        return
    
    print("Loading augmented tokenizer from train metadata...")
    with open(train_metadata_path, 'rb') as f:
        train_meta = pickle.load(f)
    
    # Load the current root-level metadata
    root_metadata_path = 'step3_augmented_data/metadata.pkl'
    if not os.path.exists(root_metadata_path):
        print(f"Error: Root metadata not found at {root_metadata_path}")
        return
    
    print("Loading current root metadata...")
    with open(root_metadata_path, 'rb') as f:
        root_meta = pickle.load(f)
    
    # Store original values for comparison
    original_vocab_size = root_meta.get('vocab_size', 'Unknown')
    original_tokenizer_type = type(root_meta.get('tokenizer', None)).__name__ if root_meta.get('tokenizer') else 'None'
    
    # Replace tokenizer and vocab_size with the augmented ones
    root_meta['tokenizer'] = train_meta['tokenizer']
    root_meta['vocab_size'] = train_meta['vocab_size']
    
    # Save back to the root-level metadata
    print("Updating root metadata with augmented tokenizer and vocabulary...")
    with open(root_metadata_path, 'wb') as f:
        pickle.dump(root_meta, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Print summary
    print("\n" + "="*60)
    print("METADATA UPDATE SUMMARY")
    print("="*60)
    print(f"Original vocabulary size: {original_vocab_size}")
    print(f"Updated vocabulary size: {root_meta['vocab_size']}")
    print(f"Original tokenizer type: {original_tokenizer_type}")
    print(f"Updated tokenizer type: {type(root_meta['tokenizer']).__name__}")
    print(f"Root metadata updated: {root_metadata_path}")
    print("="*60)
    print("\nYour model will now use the augmented tokenizer and vocabulary!")

if __name__ == "__main__":
    update_root_metadata() 