#!/usr/bin/env python3
"""
Extract Vocabulary from MIMIC Shards Metadata

This script extracts the vocabulary directly from the metadata.pkl file
in the mimic_shards_hufc4446-to128 dataset.
"""

import os
import sys
import json
import pickle
import numpy as np
from collections import OrderedDict
from typing import Dict, Any
import warnings

warnings.filterwarnings('ignore')


def extract_vocabulary_from_metadata(metadata_path: str):
    """
    Extract vocabulary from the metadata.pkl file.
    
    Args:
        metadata_path: Path to the metadata.pkl file
    
    Returns:
        Dictionary containing vocabulary information
    """
    print(f"🔍 Extracting vocabulary from metadata: {metadata_path}")
    
    if not os.path.exists(metadata_path):
        print(f"❌ Metadata file does not exist: {metadata_path}")
        return None
    
    try:
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        print(f"📄 Successfully loaded metadata")
        print(f"📋 Metadata keys: {list(metadata.keys())}")
        
        # Check if tokenizer is in metadata
        if 'tokenizer' in metadata:
            tokenizer = metadata['tokenizer']
            print(f"🔧 Found tokenizer: {type(tokenizer).__name__}")
            
            # Extract vocabulary from tokenizer
            if hasattr(tokenizer, 'word_index') and hasattr(tokenizer, 'index_word'):
                word_index = tokenizer.word_index
                index_word = tokenizer.index_word
                
                print(f"📚 Vocabulary size from tokenizer: {len(word_index)}")
                
                # Create vocabulary dictionary
                vocabulary_dict = {
                    'word_index': word_index,
                    'index_word': index_word,
                    'vocab_size': len(word_index),
                    'source': 'metadata_tokenizer',
                    'metadata_keys': list(metadata.keys())
                }
                
                # Add other metadata information
                for key, value in metadata.items():
                    if key != 'tokenizer':
                        if isinstance(value, (int, float, str, bool, list, dict)):
                            vocabulary_dict[f'metadata_{key}'] = value
                        else:
                            vocabulary_dict[f'metadata_{key}'] = str(type(value))
                
                return vocabulary_dict
        
        # If no tokenizer, check for vocabulary in other metadata
        print(f"🔍 No tokenizer found, checking for vocabulary in metadata...")
        
        # Look for vocabulary-related keys
        vocab_keys = ['vocab', 'vocabulary', 'word_index', 'index_word', 'vocab_size']
        found_vocab = {}
        
        for key in vocab_keys:
            if key in metadata:
                found_vocab[key] = metadata[key]
                print(f"✅ Found {key} in metadata")
        
        if found_vocab:
            vocabulary_dict = {
                'source': 'metadata_direct',
                'metadata_keys': list(metadata.keys()),
                **found_vocab
            }
            
            # Add other metadata information
            for key, value in metadata.items():
                if key not in vocab_keys:
                    if isinstance(value, (int, float, str, bool, list, dict)):
                        vocabulary_dict[f'metadata_{key}'] = value
                    else:
                        vocabulary_dict[f'metadata_{key}'] = str(type(value))
            
            return vocabulary_dict
        
        # If no vocabulary found, return metadata structure
        print(f"⚠️  No vocabulary found in metadata, returning metadata structure")
        return {
            'source': 'metadata_structure_only',
            'metadata_keys': list(metadata.keys()),
            'metadata': metadata
        }
        
    except Exception as e:
        print(f"❌ Error loading metadata: {e}")
        return None


def save_vocabulary_from_metadata(vocabulary_dict: Dict, output_dir: str = "extracted_vocabulary_metadata"):
    """Save vocabulary extracted from metadata in multiple formats."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save comprehensive vocabulary dictionary
    vocab_path = os.path.join(output_dir, "vocabulary_from_metadata.json")
    with open(vocab_path, 'w') as f:
        json.dump(vocabulary_dict, f, indent=2)
    print(f"💾 Saved vocabulary from metadata to: {vocab_path}")
    
    # Save word_index mapping if available
    if 'word_index' in vocabulary_dict:
        word_index_path = os.path.join(output_dir, "word_index_from_metadata.json")
        with open(word_index_path, 'w') as f:
            json.dump(vocabulary_dict['word_index'], f, indent=2)
        print(f"💾 Saved word_index from metadata to: {word_index_path}")
    
    # Save index_word mapping if available
    if 'index_word' in vocabulary_dict:
        index_word_path = os.path.join(output_dir, "index_word_from_metadata.json")
        with open(index_word_path, 'w') as f:
            json.dump(vocabulary_dict['index_word'], f, indent=2)
        print(f"💾 Saved index_word from metadata to: {index_word_path}")
    
    # Save pickle format for compatibility
    pickle_path = os.path.join(output_dir, "vocabulary_from_metadata.pkl")
    with open(pickle_path, 'wb') as f:
        pickle.dump(vocabulary_dict, f)
    print(f"💾 Saved vocabulary pickle to: {pickle_path}")
    
    # Save simple text format for easy inspection
    txt_path = os.path.join(output_dir, "vocabulary_from_metadata.txt")
    with open(txt_path, 'w') as f:
        f.write(f"Source: {vocabulary_dict.get('source', 'unknown')}\n")
        f.write(f"Metadata Keys: {vocabulary_dict.get('metadata_keys', [])}\n\n")
        
        if 'vocab_size' in vocabulary_dict:
            f.write(f"Vocabulary Size: {vocabulary_dict['vocab_size']}\n")
        
        if 'word_index' in vocabulary_dict:
            f.write(f"\nWord Index Mapping (first 50):\n")
            f.write("-" * 50 + "\n")
            word_index = vocabulary_dict['word_index']
            for i, (word, index) in enumerate(word_index.items()):
                if i >= 50:
                    break
                f.write(f"{index:4d}: {word}\n")
        
        # Write metadata information
        f.write(f"\nMetadata Information:\n")
        f.write("-" * 30 + "\n")
        for key, value in vocabulary_dict.items():
            if not key.startswith('word_index') and not key.startswith('index_word') and key != 'source':
                f.write(f"{key}: {value}\n")
    
    print(f"💾 Saved vocabulary text to: {txt_path}")
    
    return output_dir


def print_vocabulary_summary(vocabulary_dict: Dict):
    """Print a summary of the extracted vocabulary."""
    
    print("\n" + "="*80)
    print("📚 VOCABULARY FROM METADATA SUMMARY")
    print("="*80)
    
    print(f"\n📋 SOURCE INFORMATION:")
    print(f"  Source: {vocabulary_dict.get('source', 'unknown')}")
    print(f"  Metadata Keys: {vocabulary_dict.get('metadata_keys', [])}")
    
    if 'vocab_size' in vocabulary_dict:
        print(f"\n📈 VOCABULARY STATISTICS:")
        print(f"  Vocabulary Size: {vocabulary_dict['vocab_size']:,}")
    
    if 'word_index' in vocabulary_dict:
        word_index = vocabulary_dict['word_index']
        print(f"\n📚 VOCABULARY MAPPING:")
        print(f"  Word Index Entries: {len(word_index):,}")
        
        # Show first few entries
        print(f"\n🔝 FIRST 10 WORD INDEX ENTRIES:")
        for i, (word, index) in enumerate(word_index.items()):
            if i >= 10:
                break
            print(f"  {index:4d}: '{word}'")
    
    # Show metadata information
    print(f"\n📄 METADATA INFORMATION:")
    for key, value in vocabulary_dict.items():
        if not key.startswith('word_index') and not key.startswith('index_word') and key != 'source':
            if isinstance(value, (list, dict)) and len(str(value)) > 100:
                print(f"  {key}: {type(value).__name__} (length: {len(value)})")
            else:
                print(f"  {key}: {value}")


def main():
    """Main function to extract vocabulary from metadata."""
    
    # Metadata path
    metadata_path = "/home/abedin/Developments/chest_x_ray_data_processing/all_processed_data/mimic_shards_hufc4446-to128/metadata.pkl"
    
    # Extract vocabulary from metadata
    vocabulary_dict = extract_vocabulary_from_metadata(metadata_path)
    
    if vocabulary_dict is None:
        print("❌ Failed to extract vocabulary from metadata")
        return
    
    # Print summary
    print_vocabulary_summary(vocabulary_dict)
    
    # Save vocabulary
    output_dir = save_vocabulary_from_metadata(vocabulary_dict)
    
    print(f"\n✅ Vocabulary extraction from metadata complete!")
    print(f"📁 All files saved to: {output_dir}")
    print(f"\n🎯 You can now use these vocabulary files to process the standard MIMIC dataset.")
    print(f"   Key files:")
    print(f"   - {output_dir}/vocabulary_from_metadata.json: Complete vocabulary information")
    if 'word_index' in vocabulary_dict:
        print(f"   - {output_dir}/word_index_from_metadata.json: Token to index mapping")
        print(f"   - {output_dir}/index_word_from_metadata.json: Index to token mapping")


if __name__ == "__main__":
    main() 