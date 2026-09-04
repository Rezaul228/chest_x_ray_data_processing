#!/usr/bin/env python3
"""
Examine Original Indiana Data Structure Script
Check if original Indiana shards have proper content or zero tokens
"""

import os
import pickle
import numpy as np
import json

def examine_original_indiana_data():
    """Examine the original Indiana shards data structure"""
    
    data_path = '/home/abedin/Developments/chest_x_ray_data_processing/all_processed_data/indiana_shards'
    
    print(f"🔍 EXAMINING ORIGINAL INDIANA DATA STRUCTURE")
    print(f"Path: {data_path}")
    print("="*60)
    
    # Check if path exists
    if not os.path.exists(data_path):
        print(f"❌ Data path does not exist: {data_path}")
        return
    
    # Check train split
    split_path = os.path.join(data_path, 'train')
    if not os.path.exists(split_path):
        print(f"❌ Train path does not exist: {split_path}")
        return
    
    shard_files = [f for f in os.listdir(split_path) if f.endswith('.pkl')]
    if not shard_files:
        print(f"❌ No shard files found in {split_path}")
        return
    
    print(f"📂 Found {len(shard_files)} shard files")
    
    # Load first shard
    first_shard_path = os.path.join(split_path, shard_files[0])
    print(f"📂 Loading first shard: {shard_files[0]}")
    
    try:
        with open(first_shard_path, 'rb') as f:
            shard_data = pickle.load(f)
        
        print(f"✅ Shard loaded successfully")
        print(f"📊 Shard data type: {type(shard_data)}")
        
        if isinstance(shard_data, dict):
            print(f"📊 Shard keys: {list(shard_data.keys())}")
            print(f"📊 Shard contains {len(shard_data)} keys")
            
            # Check captions field specifically
            if 'captions' in shard_data:
                captions = shard_data['captions']
                print(f"\n📝 CAPTIONS ANALYSIS:")
                print(f"   Captions type: {type(captions)}")
                print(f"   Captions length: {len(captions)}")
                
                if isinstance(captions, (list, np.ndarray)):
                    print(f"   Number of captions: {len(captions)}")
                    
                    # Examine first 5 captions
                    for i in range(min(5, len(captions))):
                        caption = captions[i]
                        print(f"\n   Caption {i}:")
                        print(f"     Type: {type(caption)}")
                        
                        if isinstance(caption, (list, np.ndarray)):
                            print(f"     Length: {len(caption)}")
                            print(f"     First 10 tokens: {caption[:10]}")
                            
                            # Check for zeros
                            if len(caption) > 0:
                                zero_count = sum(1 for token in caption if token == 0)
                                non_zero_count = len(caption) - zero_count
                                zero_percentage = (zero_count / len(caption)) * 100
                                print(f"     Non-zero tokens: {non_zero_count}")
                                print(f"     Zero tokens: {zero_count}/{len(caption)} ({zero_percentage:.1f}%)")
                                
                                if non_zero_count > 0:
                                    print(f"     ✅ HAS TEXT CONTENT!")
                                else:
                                    print(f"     🚨 NO TEXT CONTENT!")
                                
                                # Show token statistics
                                if non_zero_count > 0:
                                    non_zero_tokens = [token for token in caption if token != 0]
                                    print(f"     Token range: {min(non_zero_tokens)} to {max(non_zero_tokens)}")
                                    print(f"     Unique tokens: {len(set(non_zero_tokens))}")
                        
                        elif isinstance(caption, str):
                            print(f"     Text: {caption[:100]}...")
                            tokens = caption.split()
                            print(f"     Token count: {len(tokens)}")
                            if len(tokens) == 0:
                                print(f"     ⚠️  EMPTY TEXT!")
                            else:
                                print(f"     ✅ HAS TEXT CONTENT!")
                
                # Check images field
                if 'images' in shard_data:
                    images = shard_data['images']
                    print(f"\n🖼️  IMAGES ANALYSIS:")
                    print(f"   Images type: {type(images)}")
                    print(f"   Images shape: {images.shape if hasattr(images, 'shape') else 'N/A'}")
                    
                    if hasattr(images, 'shape'):
                        print(f"   Number of images: {images.shape[0]}")
                        print(f"   Image dimensions: {images.shape[1:]}")
                        
                        # Check first few images
                        for i in range(min(3, images.shape[0])):
                            img = images[i]
                            print(f"\n   Image {i}:")
                            print(f"     Shape: {img.shape}")
                            print(f"     Data type: {img.dtype}")
                            print(f"     Value range: {img.min():.3f} to {img.max():.3f}")
                            print(f"     Non-zero pixels: {np.count_nonzero(img)}/{img.size} ({np.count_nonzero(img)/img.size*100:.1f}%)")
                
                # Check study_ids
                if 'study_ids' in shard_data:
                    study_ids = shard_data['study_ids']
                    print(f"\n🆔 STUDY IDS ANALYSIS:")
                    print(f"   Study IDs type: {type(study_ids)}")
                    print(f"   Number of study IDs: {len(study_ids)}")
                    print(f"   First 5 study IDs: {study_ids[:5]}")
            
            else:
                print(f"❌ No 'captions' key found in shard data")
                print(f"Available keys: {list(shard_data.keys())}")
        
        elif isinstance(shard_data, list):
            print(f"📊 Shard contains {len(shard_data)} samples")
            
            if len(shard_data) > 0:
                first_sample = shard_data[0]
                print(f"\n📋 FIRST SAMPLE ANALYSIS:")
                print(f"   Sample type: {type(first_sample)}")
                
                if isinstance(first_sample, dict):
                    print(f"   Sample keys: {list(first_sample.keys())}")
                    
                    # Check for caption/token fields
                    for key, value in first_sample.items():
                        print(f"\n   {key}:")
                        print(f"     Type: {type(value).__name__}")
                        
                        if isinstance(value, (list, np.ndarray)):
                            print(f"     Length: {len(value)}")
                            if len(value) > 0:
                                print(f"     First 10 elements: {value[:10]}")
                                # Check for zeros
                                zero_count = sum(1 for item in value if item == 0)
                                non_zero_count = len(value) - zero_count
                                print(f"     Non-zero elements: {non_zero_count}")
                                print(f"     Zero count: {zero_count}/{len(value)} ({zero_count/len(value)*100:.1f}%)")
                                
                                if non_zero_count > 0:
                                    print(f"     ✅ HAS CONTENT!")
                                else:
                                    print(f"     🚨 NO CONTENT!")
                        else:
                            print(f"     Value: {str(value)[:100]}...")
    
    except Exception as e:
        print(f"❌ Error loading shard: {e}")
        import traceback
        traceback.print_exc()

    # Also check metadata
    print(f"\n" + "="*60)
    print(f"📋 METADATA ANALYSIS")
    print("="*60)
    
    metadata_path = os.path.join(data_path, 'metadata.pkl')
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            
            print(f"✅ Metadata loaded successfully")
            print(f"📊 Metadata keys: {list(metadata.keys())}")
            
            if 'vocab_size' in metadata:
                print(f"📊 Vocabulary size: {metadata['vocab_size']}")
            
            if 'tokenizer' in metadata:
                tokenizer = metadata['tokenizer']
                print(f"📊 Tokenizer type: {type(tokenizer)}")
                if hasattr(tokenizer, 'word_index'):
                    print(f"📊 Word index size: {len(tokenizer.word_index)}")
                if hasattr(tokenizer, 'index_word'):
                    print(f"📊 Index word size: {len(tokenizer.index_word)}")
            
        except Exception as e:
            print(f"❌ Error loading metadata: {e}")
    else:
        print(f"❌ No metadata file found")

if __name__ == "__main__":
    examine_original_indiana_data() 