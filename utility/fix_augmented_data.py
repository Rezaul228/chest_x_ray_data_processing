#!/usr/bin/env python3

import os
import shutil
import pickle
import glob
import numpy as np
from tqdm import tqdm

def copy_metadata():
    """Copy metadata from step1 to step3"""
    source_metadata = "step1_processed_data/metadata.pkl"
    target_metadata = "step3_augmented_data/metadata.pkl"
    
    if os.path.exists(source_metadata):
        shutil.copy2(source_metadata, target_metadata)
        print(f"Copied metadata from {source_metadata} to {target_metadata}")
    else:
        print(f"Source metadata not found: {source_metadata}")

def consolidate_batches_to_shards(split_name, batch_dir, output_dir, shard_size=100):
    """Consolidate batch files into standard MIMIC-CXR format shards"""
    print(f"\nConsolidating {split_name} batches into shards...")
    
    # Find all batch files for this split
    batch_files = sorted(glob.glob(os.path.join(batch_dir, f"{split_name}_augmented_batch_*.pkl")))
    
    if not batch_files:
        print(f"No batch files found for {split_name} split")
        return
    
    # Load all augmented data from batches
    all_images = []
    all_captions = []
    all_study_ids = []
    tokenizer = None
    vocab_size = None
    
    print(f"Loading {len(batch_files)} batch files...")
    for batch_file in tqdm(batch_files, desc=f"Loading {split_name} batches"):
        with open(batch_file, 'rb') as f:
            batch_data = pickle.load(f)
        
        all_images.append(batch_data['images'])
        all_captions.append(batch_data['captions'])
        all_study_ids.extend(batch_data['study_ids'])
        
        # Save tokenizer and vocab info from first batch
        if tokenizer is None:
            tokenizer = batch_data.get('tokenizer')
            vocab_size = batch_data.get('vocab_size')
    
    # Concatenate all data
    all_images = np.concatenate(all_images, axis=0)
    all_captions = np.concatenate(all_captions, axis=0)
    # Ensure study_ids are strings with proper dtype for array compatibility
    all_study_ids = np.array([str(sid) for sid in all_study_ids], dtype='<U50')
    
    print(f"Total {split_name} augmented samples: {len(all_images)}")
    
    # Create standard shards
    num_samples = len(all_images)
    shard_idx = 0
    
    for start_idx in range(0, num_samples, shard_size):
        end_idx = min(start_idx + shard_size, num_samples)
        
        # Create shard data in MIMIC-CXR format
        shard_data = {
            'images': all_images[start_idx:end_idx],
            'captions': all_captions[start_idx:end_idx],
            'study_ids': all_study_ids[start_idx:end_idx]
        }
        
        # Save shard
        shard_filename = f"shard_{shard_idx:04d}.pkl"
        shard_path = os.path.join(output_dir, shard_filename)
        
        with open(shard_path, 'wb') as f:
            pickle.dump(shard_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        print(f"Saved {split_name} shard {shard_idx} with {len(shard_data['images'])} samples: {shard_filename}")
        shard_idx += 1
    
    # Create metadata for this split
    split_metadata = {
        'tokenizer': tokenizer,
        'vocab_size': vocab_size,
        'num_shards': shard_idx,
        'total_samples': len(all_images),
        'samples_per_shard': shard_size
    }
    
    split_metadata_path = os.path.join(output_dir, f"{split_name}_metadata.pkl")
    with open(split_metadata_path, 'wb') as f:
        pickle.dump(split_metadata, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"Created {shard_idx} standard shards for {split_name} split")
    print(f"Split metadata saved to: {split_metadata_path}")
    
    # Clean up batch files
    print(f"Cleaning up {len(batch_files)} batch files...")
    for batch_file in batch_files:
        os.remove(batch_file)
    
    # Clean up batch info files
    info_files = glob.glob(os.path.join(batch_dir, f"{split_name}_augmentation_info_*.json"))
    for file in info_files:
        os.remove(file)
    
    print(f"Cleanup completed for {split_name} split")

def main():
    # Copy metadata from step1 to step3
    copy_metadata()
    
    # Consolidate batches for each split
    splits = ['train', 'val', 'test']
    for split in splits:
        batch_dir = f"step3_augmented_data/{split}"
        output_dir = f"step3_augmented_data/{split}"
        
        if os.path.exists(batch_dir):
            consolidate_batches_to_shards(split, batch_dir, output_dir)
        else:
            print(f"Batch directory not found: {batch_dir}")
    
    print("\nAugmented data consolidation completed!")
    print("Directory structure:")
    print("  step3_augmented_data/")
    print("    ├── metadata.pkl")
    print("    ├── train/")
    print("    │   ├── shard_0000.pkl, shard_0001.pkl, ...")
    print("    │   └── train_metadata.pkl")
    print("    ├── val/")
    print("    │   ├── shard_0000.pkl, shard_0001.pkl, ...")
    print("    │   └── val_metadata.pkl")
    print("    └── test/")
    print("        ├── shard_0000.pkl, shard_0001.pkl, ...")
    print("        └── test_metadata.pkl")

if __name__ == "__main__":
    main() 