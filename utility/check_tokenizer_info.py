#!/usr/bin/env python3
"""
Script to check tokenizer information from processed MIMIC data
"""

import os
import sys
import pickle
import argparse
from collections import Counter

def load_metadata(metadata_path):
    """Load metadata file which contains tokenizer information"""
    with open(metadata_path, 'rb') as f:
        return pickle.load(f)

def check_tokenizer_info(shards_dir):
    """Check tokenizer information from processed shards"""
    print(f"Checking tokenizer information in {shards_dir}")
    print("=" * 80)
    
    # Check for metadata file
    metadata_path = os.path.join(shards_dir, 'metadata.pkl')
    enhanced_metadata_path = os.path.join(shards_dir, 'enhanced_metadata.pkl')
    
    metadata = None
    enhanced_metadata = None
    
    if os.path.exists(metadata_path):
        print(f"Found metadata file: {metadata_path}")
        try:
            metadata = load_metadata(metadata_path)
            print(f"✓ Loaded metadata.pkl")
        except Exception as e:
            print(f"Error loading metadata: {e}")
    
    if os.path.exists(enhanced_metadata_path):
        print(f"Found enhanced metadata file: {enhanced_metadata_path}")
        try:
            enhanced_metadata = load_metadata(enhanced_metadata_path)
            print(f"✓ Loaded enhanced_metadata.pkl")
        except Exception as e:
            print(f"Error loading enhanced metadata: {e}")
    
    # Analyze metadata in detail
    print(f"\n" + "="*60)
    print("DETAILED METADATA ANALYSIS")
    print("="*60)
    
    if metadata:
        print(f"Basic metadata keys: {list(metadata.keys())}")
        
        # Check each metadata field in detail
        for key, value in metadata.items():
            print(f"\n{key.upper()}:")
            if isinstance(value, (int, float, str, bool)):
                print(f"  Type: {type(value).__name__}")
                print(f"  Value: {value}")
            elif isinstance(value, list):
                print(f"  Type: list")
                print(f"  Length: {len(value)}")
                if len(value) > 0:
                    print(f"  Sample items (first 5): {value[:5]}")
                    if isinstance(value[0], (int, str)):
                        print(f"  Item type: {type(value[0]).__name__}")
            elif isinstance(value, dict):
                print(f"  Type: dict")
                print(f"  Keys: {list(value.keys())}")
                print(f"  Length: {len(value)}")
                if len(value) > 0:
                    sample_items = list(value.items())[:3]
                    print(f"  Sample items: {sample_items}")
            else:
                print(f"  Type: {type(value).__name__}")
                if hasattr(value, '__len__'):
                    print(f"  Length: {len(value)}")
        
        # Check vocabulary info
        if 'tokenizer' in metadata:
            tokenizer = metadata['tokenizer']
            print(f"\nTokenizer type: {type(tokenizer).__name__}")
            print(f"Tokenizer attributes: {[attr for attr in dir(tokenizer) if not attr.startswith('_')]}")
            
            if hasattr(tokenizer, 'word_index'):
                word_index = tokenizer.word_index
                print(f"Vocabulary size: {len(word_index)}")
                print(f"Vocabulary type: {type(word_index).__name__}")
                
                # Show special tokens
                special_tokens = ['<PAD>', '<UNK>', '<START>', '<END>', '<SEP>']
                print(f"\nSpecial tokens:")
                for token in special_tokens:
                    if token in word_index:
                        print(f"  '{token}' -> {word_index[token]}")
                    else:
                        print(f"  '{token}' -> Not found")
                
                # Show sample vocabulary items
                print(f"\nSample vocabulary items (first 20):")
                sample_items = list(word_index.items())[:20]
                for word, idx in sample_items:
                    print(f"  '{word}' -> {idx}")
                
                # Show medical terms if present
                medical_terms = ['pneumonia', 'effusion', 'cardiomegaly', 'edema', 'consolidation', 'opacity', 'fracture']
                print(f"\nMedical terms check:")
                for term in medical_terms:
                    if term in word_index:
                        print(f"  '{term}' -> {word_index[term]}")
                    else:
                        print(f"  '{term}' -> Not found")
                
                # Show OOV token info
                if hasattr(tokenizer, 'oov_token') and hasattr(tokenizer, 'oov_index'):
                    print(f"\nOOV token info:")
                    print(f"  OOV token: '{tokenizer.oov_token}'")
                    print(f"  OOV index: {tokenizer.oov_index}")
            else:
                print(f"No 'word_index' attribute found in tokenizer")
                # Try other possible attribute names
                for attr in ['vocab', 'index_word', 'words', 'tokens']:
                    if hasattr(tokenizer, attr):
                        attr_value = getattr(tokenizer, attr)
                        print(f"Found '{attr}' attribute: {type(attr_value).__name__}, length: {len(attr_value) if hasattr(attr_value, '__len__') else 'N/A'}")
    
    if enhanced_metadata:
        print(f"\nEnhanced metadata keys: {list(enhanced_metadata.keys())}")
        
        # Check enhanced metadata fields
        for key, value in enhanced_metadata.items():
            if key not in metadata:
                print(f"\n{key.upper()} (Enhanced):")
                if isinstance(value, (int, float, str, bool)):
                    print(f"  Type: {type(value).__name__}")
                    print(f"  Value: {value}")
                elif isinstance(value, list):
                    print(f"  Type: list")
                    print(f"  Length: {len(value)}")
                    if len(value) > 0:
                        print(f"  Sample items (first 5): {value[:5]}")
                elif isinstance(value, dict):
                    print(f"  Type: dict")
                    print(f"  Keys: {list(value.keys())}")
                    print(f"  Length: {len(value)}")
                    if len(value) > 0:
                        sample_items = list(value.items())[:3]
                        print(f"  Sample items: {sample_items}")
                else:
                    print(f"  Type: {type(value).__name__}")
                    if hasattr(value, '__len__'):
                        print(f"  Length: {len(value)}")
        
        # Check vocabulary info in enhanced metadata
        if 'tokenizer' in enhanced_metadata:
            tokenizer = enhanced_metadata['tokenizer']
            print(f"\nEnhanced tokenizer type: {type(tokenizer).__name__}")
            
            if hasattr(tokenizer, 'vocab'):
                vocab = tokenizer.vocab
                print(f"Enhanced vocabulary size: {len(vocab)}")
    
    # Detailed shard analysis - examine actual data structure
    print(f"\n" + "="*60)
    print("DETAILED SHARD DATA ANALYSIS")
    print("="*60)
    
    splits = ['train', 'val', 'test']
    
    for split in splits:
        split_path = os.path.join(shards_dir, split)
        if not os.path.exists(split_path):
            print(f"Split '{split}' not found")
            continue
        
        shard_files = [f for f in os.listdir(split_path) if f.endswith('.pkl')]
        shard_files.sort()
        
        print(f"\n{split.upper()} split:")
        print(f"  Number of shards: {len(shard_files)}")
        
        # Examine first shard in detail
        if shard_files:
            shard_path = os.path.join(split_path, shard_files[0])
            
            try:
                with open(shard_path, 'rb') as f:
                    shard_data = pickle.load(f)
                
                print(f"  First shard ({shard_files[0]}):")
                
                if isinstance(shard_data, dict):
                    print(f"    Keys: {list(shard_data.keys())}")
                    
                    # Analyze each key in detail
                    for key, value in shard_data.items():
                        print(f"\n    {key.upper()}:")
                        if hasattr(value, 'shape'):
                            print(f"      Shape: {value.shape}")
                            print(f"      Dtype: {value.dtype}")
                            print(f"      Type: {type(value).__name__}")
                            
                            # Show sample data for non-image fields
                            if key != 'images' and len(value.shape) == 2:
                                print(f"      Sample data (first 3 rows):")
                                for i in range(min(3, value.shape[0])):
                                    print(f"        Row {i}: {value[i][:10]}...")  # Show first 10 elements
                            
                            # For images, just show shape info
                            if key == 'images':
                                print(f"      Image format: {value.shape[1]}x{value.shape[2]} RGB")
                                print(f"      Value range: {value.min():.3f} to {value.max():.3f}")
                        
                        elif isinstance(value, list):
                            print(f"      Type: list")
                            print(f"      Length: {len(value)}")
                            if len(value) > 0:
                                print(f"      Sample items (first 3): {value[:3]}")
                                if isinstance(value[0], (int, str)):
                                    print(f"      Item type: {type(value[0]).__name__}")
                        
                        elif isinstance(value, (int, float, str)):
                            print(f"      Type: {type(value).__name__}")
                            print(f"      Value: {value}")
                        
                        else:
                            print(f"      Type: {type(value).__name__}")
                            if hasattr(value, '__len__'):
                                print(f"      Length: {len(value)}")
                
                elif isinstance(shard_data, list):
                    print(f"    Type: list")
                    print(f"    Length: {len(shard_data)}")
                    
                    if len(shard_data) > 0:
                        sample = shard_data[0]
                        print(f"    Sample item type: {type(sample).__name__}")
                        
                        if isinstance(sample, dict):
                            print(f"    Sample keys: {list(sample.keys())}")
                            
                            for key, value in sample.items():
                                print(f"      {key}: {type(value).__name__}")
                                if hasattr(value, '__len__'):
                                    print(f"        Length: {len(value)}")
                                if hasattr(value, 'shape'):
                                    print(f"        Shape: {value.shape}")
                
            except Exception as e:
                print(f"  Error reading {shard_files[0]}: {e}")
    
    print(f"\n" + "="*80)

def main():
    parser = argparse.ArgumentParser(description='Check tokenizer information from processed MIMIC shards')
    parser.add_argument('--shards_dir', type=str, 
                       default='/home/abedin/Developments/chest_x_ray_data_processing/all_processed_data/mimic_shards',
                       help='Directory containing MIMIC shard files')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.shards_dir):
        print(f"Error: Shards directory not found: {args.shards_dir}")
        sys.exit(1)
    
    check_tokenizer_info(args.shards_dir)

if __name__ == "__main__":
    main() 