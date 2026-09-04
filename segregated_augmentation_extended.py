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
    """Create directory structure for augmented data"""
    os.makedirs(base_dir, exist_ok=True)
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(base_dir, split), exist_ok=True)

class SimpleTokenizer:
    """Minimal tokenizer with word_index/index_word and basic encode/decode."""
    def __init__(self, word_index, index_word):
        self.word_index = word_index
        self.index_word = index_word
        self.vocab_size = len(word_index) + 1  # +1 for padding token
    
    def texts_to_sequences(self, texts):
        sequences = []
        for text in texts:
            sequence = []
            words = text.lower().split()
            for word in words:
                if word in self.word_index:
                    sequence.append(self.word_index[word])
                else:
                    sequence.append(1)  # <unk> token id
            sequences.append(sequence)
        return sequences
    
    def sequences_to_texts(self, sequences):
        texts = []
        for sequence in sequences:
            text_tokens = []
            for token_id in sequence:
                if token_id == 0:  # <pad>
                    continue
                word = self.index_word.get(str(token_id), '<unk>')
                if word not in ['<start>', '<end>', '<pad>', '<unk>']:
                    text_tokens.append(word)
            texts.append(' '.join(text_tokens))
        return texts

def load_tokenizer_from_json(vocab_file, index_word_file):
    """Load tokenizer from JSON vocabulary files"""
    print(f"Loading tokenizer from JSON files:")
    print(f"  Vocab file: {vocab_file}")
    print(f"  Index word file: {index_word_file}")
    
    with open(vocab_file, 'r') as f:
        word_index = json.load(f)
    
    with open(index_word_file, 'r') as f:
        index_word = json.load(f)
    
    tokenizer = SimpleTokenizer(word_index, index_word)
    print(f"Tokenizer loaded with vocabulary size: {tokenizer.vocab_size}")
    
    return tokenizer

def load_shards_data(shard_dir, vocab_file=None, index_word_file=None, metadata_path=None):
    """Load all data from shards in the specified directory (MIMIC-CXR format)"""
    if not os.path.exists(shard_dir):
        raise FileNotFoundError(f"Shard directory {shard_dir} not found")
    
    # Load tokenizer from JSON files or metadata
    tokenizer = None
    vocab_size = None
    
    # First, try to load original tokenizer from metadata.pkl
    if metadata_path and os.path.exists(metadata_path):
        print(f"Loading original tokenizer from metadata: {metadata_path}")
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
            original_tokenizer = metadata.get('tokenizer')
            
            if original_tokenizer:
                # Check if we have extended vocabulary files and if the original tokenizer can use them
                if (vocab_file and index_word_file and 
                    os.path.exists(vocab_file) and os.path.exists(index_word_file) and
                    hasattr(original_tokenizer, 'load_from_files')):
                    
                    print(f"Extending original tokenizer with vocabulary from: {vocab_file}")
                    try:
                        # Create a copy of the original tokenizer and extend it
                        if hasattr(original_tokenizer, 'load_from_files'):
                            # If it's EnhancedTokenizer, we can extend it
                            extended_tokenizer = type(original_tokenizer)()
                            extended_tokenizer.load_from_files(vocab_file, index_word_file)
                            tokenizer = extended_tokenizer
                            # Fix: Use len(word_index) + 1 instead of vocab_size attribute
                            vocab_size = len(extended_tokenizer.word_index) + 1
                            print(f"Successfully extended original tokenizer. New vocab size: {vocab_size}")
                        else:
                            # Fallback to original tokenizer
                            tokenizer = original_tokenizer
                            vocab_size = metadata.get('vocab_size', len(original_tokenizer.word_index) + 1 if hasattr(original_tokenizer, 'word_index') else None)
                            print(f"Using original tokenizer without extension. Vocab size: {vocab_size}")
                    except Exception as e:
                        print(f"Warning: Failed to extend original tokenizer: {e}")
                        print("Using original tokenizer without extension")
                        tokenizer = original_tokenizer
                        vocab_size = metadata.get('vocab_size', len(original_tokenizer.word_index) + 1 if hasattr(original_tokenizer, 'word_index') else None)
                else:
                    # Use original tokenizer as-is
                    tokenizer = original_tokenizer
                    vocab_size = metadata.get('vocab_size', len(original_tokenizer.word_index) + 1 if hasattr(original_tokenizer, 'word_index') else None)
                    print(f"Using original tokenizer as-is. Vocab size: {vocab_size}")
    
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
        batch_size_actual = end_idx - start_idx
        
        # Extract batch data
        batch_images = data['images'][start_idx:end_idx]
        batch_captions = data['captions'][start_idx:end_idx]
        batch_study_ids = data['study_ids'][start_idx:end_idx]
        
        # Initialize batch results
        batch_aug_images = []
        batch_aug_captions = []
        batch_aug_study_ids = []
        
        # Process each sample in the batch
        for i in range(batch_size_actual):
            original_image = batch_images[i]
            original_caption = batch_captions[i]
            original_study_id = batch_study_ids[i]
            
            # Add original sample
            batch_aug_images.append(original_image)
            batch_aug_captions.append(original_caption)
            batch_aug_study_ids.append(f"{original_study_id}_orig")
            
            # Generate augmentations
            for aug_idx in range(config.num_augmentations):
                try:
                    # Apply image augmentation
                    aug_image = apply_advanced_image_augmentation(original_image, config)
                    
                    # Apply text augmentation
                    if 'tokenizer' in data:
                        aug_caption = apply_advanced_text_augmentation(
                            original_caption, data['tokenizer'], config
                        )
                    else:
                        # Fallback: use original caption if no tokenizer
                        aug_caption = original_caption
                    
                    # Add augmented sample
                    batch_aug_images.append(aug_image)
                    batch_aug_captions.append(aug_caption)
                    batch_aug_study_ids.append(f"{original_study_id}_aug_{aug_idx+1}")
                    
                except Exception as e:
                    print(f"Warning: Augmentation failed for sample {original_study_id}: {e}")
                    # Add original as fallback
                    batch_aug_images.append(original_image)
                    batch_aug_captions.append(original_caption)
                    batch_aug_study_ids.append(f"{original_study_id}_aug_{aug_idx+1}_failed")
        
        # Convert to numpy arrays
        batch_aug_images = np.array(batch_aug_images)
        batch_aug_captions = np.array(batch_aug_captions)
        batch_aug_study_ids = np.array(batch_aug_study_ids, dtype='<U50')
        
        # Save batch to temporary file
        batch_file = os.path.join(output_dir, f"{data_split_name}_batch_{batch_idx:04d}.pkl")
        batch_data = {
            'images': batch_aug_images,
            'captions': batch_aug_captions,
            'study_ids': batch_aug_study_ids
        }
        
        with open(batch_file, 'wb') as f:
            pickle.dump(batch_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Update statistics
        total_original += batch_size_actual
        total_augmented += len(batch_aug_images)
        
        print(f"Batch {batch_idx+1}: {batch_size_actual} original -> {len(batch_aug_images)} total samples")
        
        # Clear batch memory
        del batch_aug_images, batch_aug_captions, batch_aug_study_ids, batch_data
        gc.collect()
    
    print(f"\n{data_split_name} augmentation completed:")
    print(f"  Original samples: {total_original}")
    print(f"  Total samples (including augmentations): {total_augmented}")
    print(f"  Augmentation ratio: {total_augmented/total_original:.2f}x")

def consolidate_augmented_batches(output_dir, data_split_name, shard_size=100):
    """Consolidate augmented batch files into standard shards"""
    print(f"\nConsolidating {data_split_name} batches into shards...")
    
    # Find all batch files
    batch_files = sorted(glob.glob(os.path.join(output_dir, f"{data_split_name}_batch_*.pkl")))
    if not batch_files:
        print(f"No batch files found for {data_split_name}")
        return
    
    # Load all batch data
    all_images = []
    all_captions = []
    all_study_ids = []
    
    for batch_file in tqdm(batch_files, desc=f"Loading {data_split_name} batches"):
        with open(batch_file, 'rb') as f:
            batch_data = pickle.load(f)
        
        all_images.append(batch_data['images'])
        all_captions.append(batch_data['captions'])
        all_study_ids.extend(batch_data['study_ids'])
    
    # Concatenate all batches
    all_images = np.concatenate(all_images, axis=0)
    all_captions = np.concatenate(all_captions, axis=0)
    all_study_ids = np.array(all_study_ids, dtype='<U50')
    
    print(f"Total {data_split_name} samples: {len(all_images)}")
    
    # Create shards
    num_shards = (len(all_images) + shard_size - 1) // shard_size
    
    for shard_idx in range(num_shards):
        start_idx = shard_idx * shard_size
        end_idx = min((shard_idx + 1) * shard_size, len(all_images))
        
        shard_data = {
            'images': all_images[start_idx:end_idx],
            'captions': all_captions[start_idx:end_idx],
            'study_ids': all_study_ids[start_idx:end_idx]
        }
        
        shard_file = os.path.join(output_dir, f"shard_{shard_idx:04d}.pkl")
        with open(shard_file, 'wb') as f:
            pickle.dump(shard_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Create metadata for this split
    metadata = {
        'num_samples': len(all_images),
        'num_shards': num_shards,
        'shard_size': shard_size
    }
    
    metadata_file = os.path.join(output_dir, f"{data_split_name}_metadata.pkl")
    with open(metadata_file, 'wb') as f:
        pickle.dump(metadata, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Clean up batch files
    for batch_file in batch_files:
        os.remove(batch_file)
    
    print(f"Consolidated {data_split_name} into {num_shards} shards")

def expand_tokenizer_with_medical_terms(tokenizer, medical_terms_dict):
    """Expand tokenizer with medical terms (legacy function - not used with JSON approach)"""
    print("Note: Using JSON vocabulary files instead of tokenizer expansion")
    return tokenizer

def main():
    parser = argparse.ArgumentParser(description="Advanced augmentation for chest X-ray dataset with extended vocabulary")
    parser.add_argument("--base_shard_dir", type=str, default="shards",
                      help="Base directory containing train/val/test shards")
    parser.add_argument("--output_dir", type=str, default="aug_shards_extended",
                      help="Output directory for augmented data")
    parser.add_argument("--num_augmentations", type=int, default=6,
                      help="Number of augmentations per sample")
    parser.add_argument("--max_sequence_length", type=int, default=128,
                      help="Maximum sequence length for text")
    parser.add_argument("--shard_size", type=int, default=100,
                      help="Number of samples per output shard")
    parser.add_argument("--vocab_file", type=str, 
                      default="mimic_frontal_complete_vocab_extended_vocab.json",
                      help="Extended vocabulary JSON file")
    parser.add_argument("--index_word_file", type=str,
                      default="mimic_frontal_complete_vocab_extended_index_word.json",
                      help="Extended index word JSON file")
    args = parser.parse_args()

    # Create output directory structure
    train_out_dir = os.path.join(args.output_dir, 'train')
    val_out_dir = os.path.join(args.output_dir, 'val')
    test_out_dir = os.path.join(args.output_dir, 'test')
    
    os.makedirs(train_out_dir, exist_ok=True)
    os.makedirs(val_out_dir, exist_ok=True)
    os.makedirs(test_out_dir, exist_ok=True)

    # Check if extended vocabulary files exist
    if not os.path.exists(args.vocab_file):
        print(f"Warning: Extended vocabulary file not found: {args.vocab_file}")
        print("Falling back to metadata.pkl tokenizer")
        args.vocab_file = None
        args.index_word_file = None
    
    if not os.path.exists(args.index_word_file):
        print(f"Warning: Extended index word file not found: {args.index_word_file}")
        print("Falling back to metadata.pkl tokenizer")
        args.vocab_file = None
        args.index_word_file = None
    
    # Load metadata for fallback tokenizer
    metadata_path = os.path.join(args.base_shard_dir, 'metadata.pkl')
    
    # Process each split
    splits = ['train', 'val', 'test']
    for split in splits:
        print(f"\nProcessing {split} split...")
        # Input shard directory for this split
        input_shard_dir = os.path.join(args.base_shard_dir, split)
        # Output directory for this split
        output_dir = os.path.join(args.output_dir, split)
        
        # Load data from shards with extended vocabulary
        data = load_shards_data(input_shard_dir, args.vocab_file, args.index_word_file, metadata_path)
        
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

    # Create overall metadata file for augmented dataset (single root metadata with tokenizer)
    overall_metadata_path = os.path.join(args.output_dir, 'metadata.pkl')
    
    # Build tokenizer for metadata - preserve original tokenizer type
    tokenizer_for_metadata = None
    vocab_size_for_metadata = None
    
    if metadata_path and os.path.exists(metadata_path):
        with open(metadata_path, 'rb') as f:
            base_metadata = pickle.load(f)
        original_tokenizer = base_metadata.get('tokenizer')
        
        if original_tokenizer:
            # Try to extend original tokenizer with extended vocabulary
            if (args.vocab_file and args.index_word_file and 
                os.path.exists(args.vocab_file) and os.path.exists(args.index_word_file) and
                hasattr(original_tokenizer, 'load_from_files')):
                
                print(f"Creating extended version of original tokenizer with: {args.vocab_file}")
                try:
                    # Create extended version of original tokenizer (same class)
                    extended_tokenizer = type(original_tokenizer)()
                    extended_tokenizer.load_from_files(args.vocab_file, args.index_word_file)
                    tokenizer_for_metadata = extended_tokenizer
                    # Fix: Use len(word_index) + 1 instead of vocab_size attribute
                    vocab_size_for_metadata = len(extended_tokenizer.word_index) + 1
                    print(f"Successfully created extended tokenizer. New vocab size: {vocab_size_for_metadata}")
                except Exception as e:
                    print(f"Warning: Failed to create extended tokenizer: {e}")
                    print("Using original tokenizer without extension")
                    tokenizer_for_metadata = original_tokenizer
                    vocab_size_for_metadata = base_metadata.get('vocab_size')
            else:
                # Use original tokenizer as-is
                tokenizer_for_metadata = original_tokenizer
                vocab_size_for_metadata = base_metadata.get('vocab_size')
                print(f"Using original tokenizer as-is. Vocab size: {vocab_size_for_metadata}")
    
    # Fallback: create new tokenizer from JSON files if no original tokenizer found
    if tokenizer_for_metadata is None and args.vocab_file and args.index_word_file and os.path.exists(args.vocab_file) and os.path.exists(args.index_word_file):
        print("No original tokenizer found. Creating new SimpleTokenizer from extended JSON files.")
        tokenizer_for_metadata = load_tokenizer_from_json(args.vocab_file, args.index_word_file)
        vocab_size_for_metadata = tokenizer_for_metadata.vocab_size
    
    overall_metadata = {
        'tokenizer': tokenizer_for_metadata,
        'vocab_size': vocab_size_for_metadata,
        'num_train_shards': len(glob.glob(os.path.join(args.output_dir, 'train', 'shard_*.pkl'))),
        'num_val_shards': len(glob.glob(os.path.join(args.output_dir, 'val', 'shard_*.pkl'))),
        'num_test_shards': len(glob.glob(os.path.join(args.output_dir, 'test', 'shard_*.pkl'))),
        'augmentation_settings': {
            'num_augmentations': args.num_augmentations,
            'max_sequence_length': args.max_sequence_length,
            'shard_size': args.shard_size
        },
        'vocabulary_source': 'extended_json' if args.vocab_file else 'metadata_pkl',
        'vocab_json_paths': {
            'vocab': args.vocab_file,
            'index_word': args.index_word_file
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