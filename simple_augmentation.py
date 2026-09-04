#!/usr/bin/env python3

import os
import numpy as np
import pickle
import random
import argparse
import datetime
from tqdm import tqdm
import json
import glob
import gc

# Project imports
from adv_aug_config import AdvAugConfig
from adv_aug_text import ADVANCED_MEDICAL_TERMS, apply_advanced_text_augmentation
from adv_aug_image import apply_advanced_image_augmentation

def load_shards_data_simple(shard_dir, metadata_path=None):
    """Load all data from shards in the specified directory (MIMIC-CXR format)"""
    if not os.path.exists(shard_dir):
        raise FileNotFoundError(f"Shard directory {shard_dir} not found")
    
    # Load metadata for tokenizer if provided
    tokenizer = None
    vocab_size = None
    if metadata_path and os.path.exists(metadata_path):
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
            tokenizer = metadata.get('tokenizer')
            vocab_size = metadata.get('vocab_size', len(tokenizer.word_index) + 1 if tokenizer else None)
    
    # Find all shard files
    shard_files = sorted(glob.glob(os.path.join(shard_dir, "*.pkl")))
    if not shard_files:
        raise FileNotFoundError(f"No shard files found in {shard_dir}")
    
    # Initialize data containers
    all_images = []
    all_captions = []
    all_study_ids = []
    
    # Load data from each shard
    print(f"Loading data from {len(shard_files)} shards in {shard_dir}...")
    for shard_path in tqdm(shard_files, desc=f"Loading {os.path.basename(shard_dir)} shards"):
        with open(shard_path, 'rb') as f:
            shard_data = pickle.load(f)
            
        # Handle new MIMIC-CXR format: {'images': np.array, 'captions': np.array, 'study_ids': np.array}
        if isinstance(shard_data, dict) and 'images' in shard_data:
            # New format - shard contains stacked arrays
            all_images.append(shard_data['images'])
            all_captions.append(shard_data['captions'])
            all_study_ids.extend(shard_data['study_ids'])
    
    # Convert to numpy arrays
    if all_images and isinstance(all_images[0], np.ndarray) and all_images[0].ndim > 2:
        # If we have arrays from new format, concatenate them
        images = np.concatenate(all_images, axis=0)
        captions = np.concatenate(all_captions, axis=0)
    else:
        # If we have individual samples, stack them
        images = np.array(all_images)
        captions = np.array(all_captions)
    
    # Ensure study_ids are strings with proper dtype for array compatibility
    study_ids = np.array([str(sid) for sid in all_study_ids], dtype='<U50')
    
    print(f"Loaded {len(images)} samples from {os.path.basename(shard_dir)}")
    
    result = {
        'images': images,
        'captions': captions,
        'study_ids': study_ids
    }
    
    if tokenizer:
        result['tokenizer'] = tokenizer
        result['vocab_size'] = vocab_size
    
    return result

def augment_dataset_simple(data, config, output_dir, data_split_name):
    """Apply augmentation to dataset and save to appropriate directory"""
    print(f"\nAugmenting {data_split_name} data with {config.num_augmentations} augmentations per sample...")
    
    # Set random seed for reproducibility
    np.random.seed(config.random_seed)
    random.seed(config.random_seed)
    
    # Process in batches to reduce memory usage
    batch_size = 25  # Very small batch size to prevent OOM errors
    num_samples = len(data['images'])
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    # Process each batch separately
    all_aug_info = []
    total_original = 0
    total_augmented = 0
    
    for batch_idx in range(num_batches):
        print(f"\nProcessing batch {batch_idx+1}/{num_batches}")
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, num_samples)
        
        # Extract batch data
        batch_data = {
            'images': data['images'][start_idx:end_idx],
            'captions': data['captions'][start_idx:end_idx],
            'study_ids': data['study_ids'][start_idx:end_idx],
            'tokenizer': data['tokenizer'],
            'vocab_size': data.get('vocab_size', len(data['tokenizer'].word_index) + 1),
            'original_vocab_size': data.get('original_vocab_size', data.get('vocab_size', len(data['tokenizer'].word_index) + 1))
        }
        
        # Initialize lists for augmented batch data
        augmented_images = []
        augmented_captions = []
        augmented_study_ids = []
        augmentation_info = []
        
        # Create augmented versions of each sample in the batch
        for idx in tqdm(range(len(batch_data['images'])), desc=f"Augmenting batch {batch_idx+1}"):
            image = batch_data['images'][idx]
            caption = batch_data['captions'][idx]
            study_id = str(batch_data['study_ids'][idx]) if isinstance(batch_data['study_ids'][idx], np.integer) else batch_data['study_ids'][idx]
            
            for aug_idx in range(config.num_augmentations):
                # Apply image augmentation
                aug_image = apply_advanced_image_augmentation(image, config)
                
                # Apply text augmentation
                aug_caption = apply_advanced_text_augmentation(caption, batch_data['tokenizer'], config)
                
                # Create augmented study ID
                aug_study_id = f"{study_id}_aug{aug_idx+1}"
                
                # Store augmented data
                augmented_images.append(aug_image)
                augmented_captions.append(aug_caption)
                augmented_study_ids.append(aug_study_id)
                
                augmentation_info.append({
                    "original_study_id": str(study_id),
                    "augmented_study_id": aug_study_id,
                    "split": data_split_name,
                    "augmentation_number": int(aug_idx + 1)
                })
        
        # Save this batch's augmented data
        batch_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_output_file = os.path.join(output_dir, f"{data_split_name}_augmented_batch_{batch_idx}_{batch_timestamp}.pkl")
        
        # Save batch data
        batch_augmented_data = {
            'images': np.array(augmented_images),
            'captions': np.array(augmented_captions),
            'study_ids': np.array(augmented_study_ids, dtype='<U50'),
            'vocab_size': batch_data.get('vocab_size'),
            'original_vocab_size': batch_data.get('original_vocab_size'),
            'tokenizer': batch_data['tokenizer'],
            'batch_index': batch_idx,
            'original_indices': list(range(start_idx, end_idx))
        }
        
        with open(batch_output_file, 'wb') as f:
            pickle.dump(batch_augmented_data, f)
        
        # Update totals
        total_original += len(batch_data['images'])
        total_augmented += len(augmented_images)
        all_aug_info.extend(augmentation_info)
        
        # Force garbage collection
        del augmented_images
        del augmented_captions
        del augmented_study_ids
        del batch_augmented_data
        gc.collect()
    
    # Save overall augmentation info
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    info_file = os.path.join(output_dir, f"{data_split_name}_augmentation_info_{timestamp}.json")
    with open(info_file, 'w') as f:
        json.dump(all_aug_info, f, indent=2)
    
    print(f"Created augmented {data_split_name} dataset with {total_original} original samples and {total_augmented} augmented samples")
    print(f"Data saved to {output_dir} in {num_batches} batch files")
    
    return info_file

def main():
    parser = argparse.ArgumentParser(description="Simple augmentation for chest X-ray dataset")
    parser.add_argument("--base_shard_dir", type=str, default="step1_processed_data",
                      help="Base directory containing train/val/test shards")
    parser.add_argument("--output_dir", type=str, default="step3_augmented_data",
                      help="Output directory for augmented data")
    parser.add_argument("--num_augmentations", type=int, default=5,
                      help="Number of augmentations per sample")
    parser.add_argument("--max_sequence_length", type=int, default=128,
                      help="Maximum sequence length for text")
    parser.add_argument("--shard_size", type=int, default=100,
                      help="Number of samples per output shard")
    args = parser.parse_args()

    # Create output directory structure
    train_out_dir = os.path.join(args.output_dir, 'train')
    val_out_dir = os.path.join(args.output_dir, 'val')
    test_out_dir = os.path.join(args.output_dir, 'test')
    
    os.makedirs(train_out_dir, exist_ok=True)
    os.makedirs(val_out_dir, exist_ok=True)
    os.makedirs(test_out_dir, exist_ok=True)

    # Load metadata for tokenizer
    metadata_path = os.path.join(args.base_shard_dir, 'metadata.pkl')
    
    # Process each split
    splits = ['train', 'val', 'test']
    for split in splits:
        print(f"\nProcessing {split} split...")
        # Input shard directory for this split
        input_shard_dir = os.path.join(args.base_shard_dir, split)
        # Output directory for this split
        output_dir = os.path.join(args.output_dir, split)
        
        # Load data from shards
        data = load_shards_data_simple(input_shard_dir, metadata_path)
        
        # Create augmentation config
        config = AdvAugConfig()
        config.num_augmentations = args.num_augmentations
        config.max_sequence_length = args.max_sequence_length
        
        # Apply augmentation
        augment_dataset_simple(data, config, output_dir, split)
        
        print(f"Completed augmentation for {split} split")
        
        # Clear memory
        del data
        gc.collect()

    print("\nAugmentation completed for all splits!")
    print(f"Augmented dataset saved to: {args.output_dir}")

if __name__ == "__main__":
    main() 