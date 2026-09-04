#!/usr/bin/env python3

import os
import numpy as np
import pickle
import glob
from tqdm import tqdm

def test_load_shards_data(shard_dir, metadata_path=None):
    """Test the load_shards_data function step by step"""
    print(f"Testing load_shards_data for {shard_dir}")
    
    if not os.path.exists(shard_dir):
        raise FileNotFoundError(f"Shard directory {shard_dir} not found")
    
    # Find all shard files
    shard_files = sorted(glob.glob(os.path.join(shard_dir, "*.pkl")))
    print(f"Found {len(shard_files)} shard files")
    
    if not shard_files:
        raise FileNotFoundError(f"No shard files found in {shard_dir}")
    
    # Initialize data containers
    all_images = []
    all_captions = []
    all_study_ids = []
    
    # Load data from each shard
    print(f"Loading data from {len(shard_files)} shards in {shard_dir}...")
    for i, shard_path in enumerate(shard_files):
        print(f"Processing shard {i+1}/{len(shard_files)}: {os.path.basename(shard_path)}")
        
        with open(shard_path, 'rb') as f:
            shard_data = pickle.load(f)
        
        print(f"  Shard data type: {type(shard_data)}")
        print(f"  Shard data keys: {list(shard_data.keys()) if isinstance(shard_data, dict) else 'Not a dict'}")
        
        # Handle new MIMIC-CXR format: {'images': np.array, 'captions': np.array, 'study_ids': np.array}
        if isinstance(shard_data, dict) and 'images' in shard_data:
            print(f"  Processing as new format")
            # New format - shard contains stacked arrays
            all_images.append(shard_data['images'])
            all_captions.append(shard_data['captions'])
            all_study_ids.extend(shard_data['study_ids'])
            print(f"  Added {len(shard_data['images'])} images, {len(shard_data['captions'])} captions, {len(shard_data['study_ids'])} study_ids")
        else:
            print(f"  Processing as old format")
            # Fallback for old format - shard contains list of individual entries
            for entry in shard_data:
                if 'frontal_img' in entry:
                    all_images.append(entry['frontal_img'])
                elif 'images' in entry:
                    all_images.append(entry['images'])
                    
                if 'caption_seq' in entry:
                    all_captions.append(entry['caption_seq'])
                elif 'captions' in entry:
                    all_captions.append(entry['captions'])
                    
                if 'study_id' in entry:
                    all_study_ids.append(entry['study_id'])
    
    print(f"Total images collected: {len(all_images)}")
    print(f"Total captions collected: {len(all_captions)}")
    print(f"Total study_ids collected: {len(all_study_ids)}")
    
    # Convert to numpy arrays
    if all_images and isinstance(all_images[0], np.ndarray) and all_images[0].ndim > 2:
        print("Converting arrays using concatenate")
        # If we have arrays from new format, concatenate them
        images = np.concatenate(all_images, axis=0)
        captions = np.concatenate(all_captions, axis=0)
    else:
        print("Converting arrays using np.array")
        # If we have individual samples, stack them
        images = np.array(all_images)
        captions = np.array(all_captions)
    
    # Ensure study_ids are strings with proper dtype for array compatibility
    study_ids = np.array([str(sid) for sid in all_study_ids], dtype='<U50')
    
    print(f"Final shapes - images: {images.shape}, captions: {captions.shape}, study_ids: {study_ids.shape}")
    
    return {
        'images': images,
        'captions': captions,
        'study_ids': study_ids
    }

if __name__ == "__main__":
    # Test with train data
    result = test_load_shards_data("step1_processed_data/train")
    print("Test completed successfully!") 