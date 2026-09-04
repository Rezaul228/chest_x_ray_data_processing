#!/usr/bin/env python3
"""
Script to debug visualization loading logic and verify original vs augmented image selection
"""

import os
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt

def debug_visualization_loading(augmented_dir="/home/abedin/Developments/chest_x_ray_data_processing/all_processed_data/aug_indiana_extended"):
    """Debug the visualization loading logic to verify original vs augmented selection"""
    
    print("Debugging Visualization Loading Logic")
    print("=" * 50)
    
    # Simulate the exact loading logic from create_detailed_visualization.py
    split_dir = os.path.join(augmented_dir, 'train')
    if not os.path.exists(split_dir):
        print(f"Error: Split directory not found: {split_dir}")
        return
    
    # Find shard files
    shard_files = [f for f in os.listdir(split_dir) if f.startswith('shard_') and f.endswith('.pkl')]
    shard_files.sort()
    
    if not shard_files:
        print(f"No shard files found in {split_dir}")
        return
    
    # Randomly select a shard (same as visualization script)
    selected_shard = random.choice(shard_files)
    shard_path = os.path.join(split_dir, selected_shard)
    
    print(f"Selected shard: {selected_shard}")
    
    try:
        with open(shard_path, 'rb') as f:
            shard_data = pickle.load(f)
    except Exception as e:
        print(f"Error loading shard: {e}")
        return
    
    print(f"Loaded shard with {len(shard_data['captions'])} samples")
    
    # Extract all possible original samples (first sample in each group of 7)
    all_original_samples = []
    all_augmented_samples = []
    
    for i in range(0, len(shard_data['captions']), 7):  # 7 samples per group (1 original + 6 augmented)
        if i + 6 < len(shard_data['captions']):
            # Original sample (first in group)
            original = {
                'image': shard_data['images'][i],
                'caption': shard_data['captions'][i],
                'study_id': shard_data['study_ids'][i] if 'study_ids' in shard_data else f"sample_{i}",
                'is_original': True
            }
            all_original_samples.append(original)
            
            # Augmented samples (next 6 in group)
            aug_group = []
            for j in range(1, 7):  # 6 augmented samples
                aug_sample = {
                    'image': shard_data['images'][i + j],
                    'caption': shard_data['captions'][i + j],
                    'study_id': shard_data['study_ids'][i + j] if 'study_ids' in shard_data else f"sample_{i+j}",
                    'is_original': False,
                    'augmentation_id': j
                }
                aug_group.append(aug_sample)
            all_augmented_samples.append(aug_group)
    
    print(f"Extracted {len(all_original_samples)} original samples")
    print(f"Extracted {len(all_augmented_samples)} augmented groups")
    
    # Randomly select 3 samples (same as visualization script)
    if len(all_original_samples) >= 3:
        selected_indices = random.sample(range(len(all_original_samples)), 3)
        original_samples = [all_original_samples[i] for i in selected_indices]
        augmented_samples = [all_augmented_samples[i] for i in selected_indices]
        print(f"Randomly selected 3 samples from {len(all_original_samples)} available samples")
    else:
        original_samples = all_original_samples
        augmented_samples = all_augmented_samples
        print(f"Using all {len(all_original_samples)} available samples")
    
    # Examine the selected samples
    print(f"\nExamining selected samples:")
    
    for i, (original, aug_group) in enumerate(zip(original_samples, augmented_samples)):
        print(f"\nSample {i}:")
        print(f"  Original:")
        print(f"    Study ID: {original['study_id']}")
        print(f"    Image shape: {original['image'].shape}")
        print(f"    Image min/max: {original['image'].min():.3f}/{original['image'].max():.3f}")
        print(f"    Is original flag: {original['is_original']}")
        
        print(f"  Augmented samples:")
        for j, aug_sample in enumerate(aug_group[:3]):  # Show first 3 augmented
            print(f"    Aug {j+1}:")
            print(f"      Study ID: {aug_sample['study_id']}")
            print(f"      Image shape: {aug_sample['image'].shape}")
            print(f"      Image min/max: {aug_sample['image'].min():.3f}/{aug_sample['image'].max():.3f}")
            print(f"      Is original flag: {aug_sample['is_original']}")
            print(f"      Augmentation ID: {aug_sample['augmentation_id']}")
            
            # Calculate difference from original
            diff = np.abs(original['image'].astype(float) - aug_sample['image'].astype(float))
            mean_diff = np.mean(diff)
            max_diff = np.max(diff)
            print(f"      Difference from original: mean={mean_diff:.4f}, max={max_diff:.4f}")
    
    # Create a detailed comparison visualization
    print(f"\nCreating detailed comparison visualization...")
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle(f"Detailed Original vs Augmented Comparison\nShard: {selected_shard}", fontsize=16)
    
    for i, (original, aug_group) in enumerate(zip(original_samples, augmented_samples)):
        # Original image
        orig_image = original['image']
        if len(orig_image.shape) == 3 and orig_image.shape[2] == 1:
            orig_image = orig_image.squeeze()
        
        ax = axes[i, 0]
        ax.imshow(orig_image, cmap='gray')
        ax.set_title(f"Original {i+1}\n{original['study_id']}\nmin/max: {orig_image.min():.3f}/{orig_image.max():.3f}")
        ax.axis('off')
        
        # First 3 augmented images
        for j in range(3):
            if j < len(aug_group):
                aug_image = aug_group[j]['image']
                if len(aug_image.shape) == 3 and aug_image.shape[2] == 1:
                    aug_image = aug_image.squeeze()
                
                ax = axes[i, j+1]
                ax.imshow(aug_image, cmap='gray')
                
                # Calculate difference
                diff = np.abs(orig_image.astype(float) - aug_image.astype(float))
                mean_diff = np.mean(diff)
                
                ax.set_title(f"Aug {j+1}\n{aug_group[j]['study_id']}\nmean_diff: {mean_diff:.4f}")
                ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('visualization_debug_comparison.png', dpi=150, bbox_inches='tight')
    print(f"Detailed comparison saved to: visualization_debug_comparison.png")
    plt.close()

if __name__ == "__main__":
    debug_visualization_loading() 