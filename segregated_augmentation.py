#!/usr/bin/env python3

import os
import numpy as np
import pickle
import random
import argparse
import datetime
from tqdm import tqdm
from scipy import ndimage
import json
import glob
import gc

# Project imports
from data_set_loader import IndianaDatasetLoader
from adv_aug_config import AdvAugConfig
from adv_aug_text import ADVANCED_MEDICAL_TERMS, apply_advanced_text_augmentation
from adv_aug_image import apply_advanced_image_augmentation

def create_directory_structure(base_dir):
    """Create the directory structure for augmented data"""
    os.makedirs(base_dir, exist_ok=True)
    
    # Create subdirectories for different data splits
    train_dir = os.path.join(base_dir, 'train')
    val_dir = os.path.join(base_dir, 'val')
    test_dir = os.path.join(base_dir, 'test')
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    return train_dir, val_dir, test_dir

def load_shards_data(shard_dir, metadata_path=None):
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
        else:
            # Fallback for old format - shard contains list of individual entries
            print(f"Warning: Old format detected in {shard_path}. Consider updating your data preprocessing.")
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

def augment_dataset(data, config, output_dir, data_split_name):
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
    
    # Create an index file that lists all batch files
    index_file = os.path.join(output_dir, f"{data_split_name}_augmented_index_{timestamp}.json")
    batch_files = sorted(glob.glob(os.path.join(output_dir, f"{data_split_name}_augmented_batch_*_*.pkl")))
    with open(index_file, 'w') as f:
        json.dump({
            'batch_files': [os.path.basename(file) for file in batch_files],
            'total_original_samples': total_original,
            'total_augmented_samples': total_augmented,
            'num_augmentations_per_sample': config.num_augmentations,
            'timestamp': timestamp
        }, f, indent=2)
    
    print(f"Created augmented {data_split_name} dataset with {total_original} original samples and {total_augmented} augmented samples")
    print(f"Data saved to {output_dir} in {num_batches} batch files")
    print(f"Index file: {index_file}")
    
    return index_file

def consolidate_augmented_batches(output_dir, data_split_name, shard_size=100):
    """
    Consolidate augmented batch files into standard MIMIC-CXR format shards
    
    Args:
        output_dir: Directory containing augmented batch files
        data_split_name: Name of the data split (train, val, test)
        shard_size: Number of samples per shard
    """
    print(f"\nConsolidating {data_split_name} augmented batches into standard shards...")
    
    # Find all batch files for this split
    batch_files = sorted(glob.glob(os.path.join(output_dir, f"{data_split_name}_augmented_batch_*.pkl")))
    
    if not batch_files:
        print(f"No batch files found for {data_split_name} split")
        return
    
    # Load all augmented data from batches
    all_images = []
    all_captions = []
    all_study_ids = []
    tokenizer = None
    vocab_size = None
    
    for batch_file in tqdm(batch_files, desc=f"Loading {data_split_name} batches"):
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
    
    print(f"Total {data_split_name} augmented samples: {len(all_images)}")
    
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
        
        print(f"Saved {data_split_name} shard {shard_idx} with {len(shard_data['images'])} samples: {shard_filename}")
        shard_idx += 1
    
    # Create metadata for augmented data
    metadata = {
        'tokenizer': tokenizer,
        'vocab_size': vocab_size,
        'num_shards': shard_idx,
        'total_samples': len(all_images),
        'samples_per_shard': shard_size
    }
    
    metadata_path = os.path.join(output_dir, f"{data_split_name}_metadata.pkl")
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"Created {shard_idx} standard shards for {data_split_name} split")
    print(f"Metadata saved to: {metadata_path}")
    
    # Clean up batch files
    print(f"Cleaning up {len(batch_files)} batch files...")
    for batch_file in batch_files:
        os.remove(batch_file)
    
    # Clean up batch index and info files
    index_files = glob.glob(os.path.join(output_dir, f"{data_split_name}_augmented_index_*.json"))
    info_files = glob.glob(os.path.join(output_dir, f"{data_split_name}_augmentation_info_*.json"))
    
    for file in index_files + info_files:
        os.remove(file)
    
    print(f"Cleanup completed for {data_split_name} split")

def expand_tokenizer_with_medical_terms(tokenizer, medical_terms_dict):
    """Expand tokenizer vocabulary with medical terms"""
    # Extract all terms and synonyms
    all_terms = []
    for term, synonyms in medical_terms_dict.items():
        all_terms.append(term)
        all_terms.extend(synonyms)
    
    # Create a text corpus with these terms
    medical_corpus = [" ".join(all_terms)]
    
    # Backup the existing word index
    original_word_index = tokenizer.word_index.copy()
    original_vocab_size = len(original_word_index) + 1
    
    # Update the tokenizer
    tokenizer.fit_on_texts(medical_corpus)
    
    new_vocab_size = len(tokenizer.word_index) + 1
    print(f"Tokenizer vocabulary expanded from {original_vocab_size} to {new_vocab_size} words")
    print(f"Added {new_vocab_size - original_vocab_size} new medical terms")
    
    return tokenizer, new_vocab_size

def main():
    parser = argparse.ArgumentParser(description="Advanced augmentation for chest X-ray dataset")
    parser.add_argument("--base_shard_dir", type=str, default="shards",
                      help="Base directory containing train/val/test shards")
    parser.add_argument("--output_dir", type=str, default="aug_shards",
                      help="Output directory for augmented data")
    parser.add_argument("--num_augmentations", type=int, default=6,
                      help="Number of augmentations per sample")
    parser.add_argument("--max_sequence_length", type=int, default=64,
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
        data = load_shards_data(input_shard_dir, metadata_path)
        
        # Create augmentation config
        config = AdvAugConfig()
        config.num_augmentations = args.num_augmentations
        config.max_sequence_length = args.max_sequence_length
        
        # Apply augmentation
        augment_dataset(data, config, output_dir, split)
        
        # Consolidate augmented batch files into standard shards
        consolidate_augmented_batches(output_dir, split, args.shard_size)
        
        print(f"Completed augmentation and consolidation for {split} split")
        
        # Clear memory
        del data
        gc.collect()

    # Create overall metadata file for augmented dataset
    overall_metadata_path = os.path.join(args.output_dir, 'metadata.pkl')
    
    # Use metadata from train split as the base (they should all be the same)
    train_metadata_path = os.path.join(args.output_dir, 'train', 'train_metadata.pkl')
    if os.path.exists(train_metadata_path):
        with open(train_metadata_path, 'rb') as f:
            base_metadata = pickle.load(f)
        
        # Create overall metadata
        overall_metadata = {
            'tokenizer': base_metadata['tokenizer'],
            'vocab_size': base_metadata['vocab_size'],
            'num_train_shards': len(glob.glob(os.path.join(args.output_dir, 'train', 'shard_*.pkl'))),
            'num_val_shards': len(glob.glob(os.path.join(args.output_dir, 'val', 'shard_*.pkl'))),
            'num_test_shards': len(glob.glob(os.path.join(args.output_dir, 'test', 'shard_*.pkl'))),
            'augmentation_settings': {
                'num_augmentations': args.num_augmentations,
                'max_sequence_length': args.max_sequence_length,
                'shard_size': args.shard_size
            }
        }
        
        with open(overall_metadata_path, 'wb') as f:
            pickle.dump(overall_metadata, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        print(f"\nOverall metadata saved to: {overall_metadata_path}")

    print("\nAugmentation completed for all splits!")
    print(f"Augmented dataset saved to: {args.output_dir}")
    print("Directory structure:")
    print(f"  {args.output_dir}/")
    print(f"    ├── metadata.pkl")
    print(f"    ├── train/")
    print(f"    │   ├── shard_0000.pkl, shard_0001.pkl, ...")
    print(f"    │   └── train_metadata.pkl")
    print(f"    ├── val/")
    print(f"    │   ├── shard_0000.pkl, shard_0001.pkl, ...")
    print(f"    │   └── val_metadata.pkl")
    print(f"    └── test/")
    print(f"        ├── shard_0000.pkl, shard_0001.pkl, ...")
    print(f"        └── test_metadata.pkl")

if __name__ == "__main__":
    main()