#!/usr/bin/env python3
"""
Convert MIMIC-CXR Processed Data to ALBEF Format

This script converts the processed MIMIC-CXR shard data into the ALBEF format:
- Images organized in train/val/test folders
- JSON files with image paths and captions for each split
- Consistent naming convention for compatibility

ALBEF Format:
- mimic_raw_image_text/
  ├── images/
  │   ├── train/ (train_image_000000.jpg, train_image_000001.jpg, ...)
  │   ├── val/ (val_image_000000.jpg, val_image_000001.jpg, ...)
  │   └── test/ (test_image_000000.jpg, test_image_000001.jpg, ...)
  ├── train.json
  ├── val.json
  └── test.json
"""

import os
import json
import pickle
import glob
import numpy as np
from PIL import Image
import argparse
from tqdm import tqdm

def decode_caption(caption_seq, tokenizer):
    """Decode tokenized caption back to text"""
    words = []
    for token_id in caption_seq:
        if token_id == 0:  # PAD token
            continue
        token_key = str(int(token_id))
        word = tokenizer.get(token_key, '<UNK>')
        if word in ['<START>', '<END>', '<PAD>', '<UNK>']:
            continue
        words.append(word)
    return " ".join(words)

def load_shard_data(shard_path):
    """Load data from a shard file"""
    with open(shard_path, 'rb') as f:
        shard_data = pickle.load(f)
    return shard_data

def convert_shards_to_albef_format(processed_dir, output_dir, tokenizer_dict):
    """Convert processed shards to ALBEF format"""
    
    print("=== Converting MIMIC-CXR Data to ALBEF Format ===\n")
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images', 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images', 'val'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images', 'test'), exist_ok=True)
    
    # Process each split
    splits = ['train', 'val', 'test']
    split_data = {}
    
    for split in splits:
        print(f"Processing {split} split...")
        
        # Get shard files for this split
        shard_dir = os.path.join(processed_dir, split)
        shard_files = sorted(glob.glob(os.path.join(shard_dir, '*.pkl')))
        
        if not shard_files:
            print(f"No shard files found for {split} split")
            continue
        
        split_entries = []
        image_counter = 0
        
        # Process each shard
        for shard_file in tqdm(shard_files, desc=f"Processing {split} shards"):
            shard_data = load_shard_data(shard_file)
            
            images = shard_data['images']
            captions = shard_data['captions']
            study_ids = shard_data['study_ids']
            
            # Process each sample in the shard
            for i in range(len(images)):
                image = images[i]
                caption_seq = captions[i]
                study_id = study_ids[i]
                
                # Decode caption
                caption_text = decode_caption(caption_seq, tokenizer_dict)
                
                # Skip if caption is empty
                if not caption_text.strip():
                    continue
                
                # Save image
                image_filename = f"{split}_image_{image_counter:06d}.jpg"
                image_path = os.path.join(output_dir, 'images', split, image_filename)
                
                # Convert numpy array to PIL Image and save
                if len(image.shape) == 3 and image.shape[2] == 3:
                    # RGB image
                    pil_image = Image.fromarray((image * 255).astype(np.uint8))
                else:
                    # Grayscale or other format, convert to RGB
                    pil_image = Image.fromarray((image * 255).astype(np.uint8))
                    if pil_image.mode != 'RGB':
                        pil_image = pil_image.convert('RGB')
                
                pil_image.save(image_path, 'JPEG', quality=95)
                
                # Create entry for JSON
                entry = {
                    "image": f"{split}/{image_filename}",
                    "caption": caption_text,
                    "image_id": image_counter,
                    "study_id": study_id
                }
                
                split_entries.append(entry)
                image_counter += 1
        
        split_data[split] = split_entries
        print(f"  {split}: {len(split_entries)} samples processed")
    
    # Save JSON files
    print("\nSaving JSON files...")
    
    for split in splits:
        if split in split_data and split_data[split]:
            json_path = os.path.join(output_dir, f'{split}.json')
            with open(json_path, 'w') as f:
                json.dump(split_data[split], f, indent=2)
            print(f"  {split}.json: {len(split_data[split])} entries")
    
    # Print summary
    print("\n=== Conversion Summary ===")
    total_samples = sum(len(split_data.get(split, [])) for split in splits)
    print(f"Total samples converted: {total_samples:,}")
    
    for split in splits:
        count = len(split_data.get(split, []))
        print(f"{split}: {count:,} samples")
    
    print(f"\nOutput directory: {output_dir}")
    print("ALBEF format conversion completed! ✓")

def main():
    parser = argparse.ArgumentParser(description="Convert MIMIC-CXR processed data to ALBEF format")
    parser.add_argument("--processed_dir", type=str, 
                       default="/home/abedin/Developments/chest_x_ray_data_processing/all_processed_data/mimic_shards_hybrid_full_ori",
                       help="Directory containing processed shards")
    parser.add_argument("--output_dir", type=str, 
                       default="/home/abedin/Developments/chest_x_ray_data_processing/mimic_raw_image_text",
                       help="Output directory for ALBEF format")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum number of samples to convert (for testing)")
    
    args = parser.parse_args()
    
    # Load tokenizer from metadata
    metadata_path = os.path.join(args.processed_dir, 'metadata.pkl')
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    
    tokenizer_dict = metadata['tokenizer']  # This is the index_word mapping
    
    print(f"Loaded tokenizer with {len(tokenizer_dict)} tokens")
    
    # Convert to ALBEF format
    convert_shards_to_albef_format(args.processed_dir, args.output_dir, tokenizer_dict)

if __name__ == "__main__":
    main() 