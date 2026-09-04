#!/usr/bin/env python3
"""
Create Detailed Publication-Ready Augmentation Visualizations

This script generates high-quality visualizations showing actual text content
and detailed comparisons between original and augmented image-text pairs.

Author: Research Team
Date: 2024
"""

import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import rcParams
import seaborn as sns
from PIL import Image
import random
import json
from datetime import datetime
from matplotlib.gridspec import GridSpec
import textwrap

# Set up publication-quality plotting
plt.style.use('seaborn-v0_8-whitegrid')
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
rcParams['font.size'] = 10
rcParams['axes.titlesize'] = 12
rcParams['axes.labelsize'] = 10
rcParams['xtick.labelsize'] = 9
rcParams['ytick.labelsize'] = 9
rcParams['legend.fontsize'] = 9
rcParams['figure.titlesize'] = 14

class DetailedAugmentationVisualizer:
    def __init__(self, augmented_data_dir):
        """
        Initialize the visualizer with augmented data directory
        
        Args:
            augmented_data_dir: Path to augmented dataset directory
        """
        self.augmented_data_dir = augmented_data_dir
        self.tokenizer = None
        
        # Load tokenizer from metadata if available
        self.load_tokenizer()
        
        # Set up color scheme for publication
        self.colors = {
            'original': '#2E86AB',      # Blue for original
            'augmented': '#A23B72',     # Purple for augmented
            'background': '#F8F9FA',    # Light gray background
            'text': '#2C3E50',          # Dark blue for text
            'highlight': '#E74C3C'      # Red for highlights
        }
        
    def load_tokenizer(self):
        """Load tokenizer from metadata"""
        metadata_path = os.path.join(self.augmented_data_dir, 'metadata.pkl')
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                
                if 'tokenizer' in metadata:
                    self.tokenizer = metadata['tokenizer']
                    print(f"Loaded tokenizer with {len(self.tokenizer.word_index)} words")
                else:
                    print("No tokenizer found in metadata")
            except Exception as e:
                print(f"Warning: Error loading tokenizer: {e}")
    
    def decode_caption(self, caption_tokens):
        """Decode caption tokens to text"""
        if self.tokenizer is None:
            # Try to load tokenizer from metadata paths
            if hasattr(self, 'metadata') and 'vocab_json_paths' in self.metadata:
                vocab_paths = self.metadata['vocab_json_paths']
                index_word_path = vocab_paths.get('index_word')
                
                if index_word_path and os.path.exists(index_word_path):
                    try:
                        with open(index_word_path, 'r') as f:
                            index_word = json.load(f)
                        
                        decoded_words = []
                        for token in caption_tokens:
                            if token != 0:  # Skip padding
                                # Convert float to int for vocabulary lookup
                                token_int = int(token)
                                word = index_word.get(str(token_int), '<unk>')
                                decoded_words.append(word)
                        result = ' '.join(decoded_words)
                        print(f"Decoded text: {result[:50]}...")
                        return result
                    except Exception as e:
                        print(f"Error loading vocabulary from metadata path: {e}")
            
            # Fallback: try hardcoded paths
            vocab_path = "mimic_frontal_complete_vocab_extended_vocab.json"
            index_word_path = "mimic_frontal_complete_vocab_extended_index_word.json"
            
            if os.path.exists(vocab_path) and os.path.exists(index_word_path):
                try:
                    with open(vocab_path, 'r') as f:
                        vocab = json.load(f)
                    with open(index_word_path, 'r') as f:
                        index_word = json.load(f)
                    
                    decoded_words = []
                    for token in caption_tokens:
                        if token != 0:  # Skip padding
                            # Convert float to int for vocabulary lookup
                            token_int = int(token)
                            word = index_word.get(str(token_int), '<unk>')
                            decoded_words.append(word)
                    result = ' '.join(decoded_words)
                    print(f"Decoded text: {result[:50]}...")
                    return result
                except Exception as e:
                    print(f"Error loading vocabulary files: {e}")
            
            # Final fallback: show token IDs
            non_zero_tokens = [str(int(token)) for token in caption_tokens if token != 0]
            result = f"Tokens: {', '.join(non_zero_tokens[:15])}{'...' if len(non_zero_tokens) > 15 else ''}"
            print(f"Fallback text: {result}")
            return result
        
        try:
            decoded_words = []
            for token in caption_tokens:
                if token != 0:  # Skip padding
                    # Convert float to int for vocabulary lookup
                    token_int = int(token)
                    # Use string key for EnhancedTokenizer
                    word = self.tokenizer.index_word.get(str(token_int), '<unk>')
                    decoded_words.append(word)
            result = ' '.join(decoded_words)
            print(f"Tokenizer decoded text: {result[:50]}...")
            return result
        except Exception as e:
            print(f"Error decoding caption: {e}")
            return "Decoding error"
    
    def load_sample_data(self, split='train', num_samples=2):
        """Load sample data from augmented dataset with random selection"""
        split_dir = os.path.join(self.augmented_data_dir, split)
        if not os.path.exists(split_dir):
            print(f"Error: Split directory not found: {split_dir}")
            return [], []
            
        # Find shard files
        shard_files = [f for f in os.listdir(split_dir) if f.startswith('shard_') and f.endswith('.pkl')]
        shard_files.sort()
        
        if not shard_files:
            print(f"No shard files found in {split_dir}")
            return [], []
            
        # Randomly select a shard
        import random
        selected_shard = random.choice(shard_files)
        shard_path = os.path.join(split_dir, selected_shard)
        
        try:
            with open(shard_path, 'rb') as f:
                shard_data = pickle.load(f)
        except Exception as e:
            print(f"Error loading shard: {e}")
            return [], []
            
        print(f"Loaded shard {selected_shard} with {len(shard_data['captions'])} samples")
        
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
        
        # Randomly select the requested number of samples
        if len(all_original_samples) >= num_samples:
            # Get random indices
            selected_indices = random.sample(range(len(all_original_samples)), num_samples)
            
            original_samples = [all_original_samples[i] for i in selected_indices]
            augmented_samples = [all_augmented_samples[i] for i in selected_indices]
            
            print(f"Randomly selected {num_samples} samples from {len(all_original_samples)} available samples")
        else:
            # If not enough samples, take all available
            original_samples = all_original_samples
            augmented_samples = all_augmented_samples
            print(f"Using all {len(all_original_samples)} available samples")
                    
        return original_samples, augmented_samples
    
    def load_true_original_with_augmented(self, split='train', num_samples=1):
        """Load TRUE original data and find corresponding augmented versions"""
        
        # Path to TRUE original Indiana data
        original_data_dir = "/home/abedin/Developments/chest_x_ray_data_processing/all_processed_data/indiana_shards"
        original_split_dir = os.path.join(original_data_dir, split)
        
        if not os.path.exists(original_split_dir):
            print(f"Error: Original data directory not found: {original_split_dir}")
            return [], []
        
        # Load original data
        print(f"Loading TRUE original data from: {original_split_dir}")
        orig_shard_files = [f for f in os.listdir(original_split_dir) if f.startswith('shard_') and f.endswith('.pkl')]
        orig_shard_files.sort()
        
        if not orig_shard_files:
            print(f"No original shard files found in {original_split_dir}")
            return [], []
        
        # Randomly select an original shard
        import random
        selected_orig_shard = random.choice(orig_shard_files)
        orig_shard_path = os.path.join(original_split_dir, selected_orig_shard)
        
        print(f"Loading original shard: {selected_orig_shard}")
        
        try:
            with open(orig_shard_path, 'rb') as f:
                orig_shard_data = pickle.load(f)
        except Exception as e:
            print(f"Error loading original shard: {e}")
            return [], []
        
        print(f"Original shard contains {len(orig_shard_data['captions'])} samples")
        
        # Randomly select original samples
        num_orig_samples = len(orig_shard_data['captions'])
        if num_orig_samples < num_samples:
            selected_indices = list(range(num_orig_samples))
        else:
            selected_indices = random.sample(range(num_orig_samples), num_samples)
        
        print(f"Selected original sample indices: {selected_indices}")
        
        # Extract true original samples
        true_original_samples = []
        for idx in selected_indices:
            original_sample = {
                'image': orig_shard_data['images'][idx],
                'caption': orig_shard_data['captions'][idx],
                'study_id': orig_shard_data['study_ids'][idx],
                'is_original': True
            }
            true_original_samples.append(original_sample)
            print(f"True original sample: {original_sample['study_id']}")
        
        # Now find corresponding augmented versions in augmented dataset
        augmented_split_dir = os.path.join(self.augmented_data_dir, split)
        
        corresponding_augmented = []
        
        for orig_sample in true_original_samples:
            orig_study_id = str(orig_sample['study_id'])
            print(f"\nLooking for augmented versions of study ID: {orig_study_id}")
            
            # Search through augmented shards for this study ID
            aug_shard_files = [f for f in os.listdir(augmented_split_dir) if f.startswith('shard_') and f.endswith('.pkl')]
            aug_shard_files.sort()
            
            found_augmented = []
            
            for aug_shard_file in aug_shard_files:
                aug_shard_path = os.path.join(augmented_split_dir, aug_shard_file)
                
                try:
                    with open(aug_shard_path, 'rb') as f:
                        aug_shard_data = pickle.load(f)
                    
                    # Look for study IDs that match our original
                    for i, study_id in enumerate(aug_shard_data['study_ids']):
                        study_id_str = str(study_id)
                        
                        # Check if this is an augmented version of our original
                        if (study_id_str.startswith(f"{orig_study_id}_aug_") and 
                            not study_id_str.endswith('_orig')):
                            
                            aug_sample = {
                                'image': aug_shard_data['images'][i],
                                'caption': aug_shard_data['captions'][i],
                                'study_id': study_id_str,
                                'is_original': False,
                                'augmentation_id': int(study_id_str.split('_aug_')[1])
                            }
                            found_augmented.append(aug_sample)
                            
                except Exception as e:
                    print(f"Error loading augmented shard {aug_shard_file}: {e}")
                    continue
            
            # Sort augmented samples by augmentation ID
            found_augmented.sort(key=lambda x: x['augmentation_id'])
            
            print(f"Found {len(found_augmented)} augmented versions:")
            for aug in found_augmented[:6]:  # Show first 6
                print(f"  - {aug['study_id']}")
            
            corresponding_augmented.append(found_augmented)
        
        return true_original_samples, corresponding_augmented
    
    def create_detailed_comparison_figure(self, original_samples, augmented_samples, output_path=None):
        """Create a detailed comparison figure with Scientific Reports formatting"""
        if not original_samples or not augmented_samples:
            print("No samples to visualize")
            return
            
        # Set Scientific Reports font and styling
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['font.size'] = 8
        plt.rcParams['axes.titlesize'] = 9
        plt.rcParams['axes.labelsize'] = 8
        plt.rcParams['xtick.labelsize'] = 7
        plt.rcParams['ytick.labelsize'] = 7
        plt.rcParams['legend.fontsize'] = 7
        
        # Scientific Reports color palette
        colors = {
            'blue': '#1f77b4',
            'red': '#d62728', 
            'green': '#2ca02c',
            'orange': '#ff7f0e',
            'white': '#ffffff',
            'black': '#000000',
            'gray': '#cccccc',
            'original': '#1f77b4',      # Blue for original
            'augmented': '#d62728'      # Red for augmented
        }
        
        num_samples = len(original_samples)
        
        for i, (original, aug_group) in enumerate(zip(original_samples, augmented_samples)):
            # Create figure optimized for single-column width (7.2 inches) - increased size by 25%
            fig = plt.figure(figsize=(9.0, 5.94), facecolor=colors['white'])  # 7.2*1.25=9.0, 4.75*1.25=5.94
            
            # Create title with Scientific Reports formatting
            title_ax = fig.add_axes([0.1, 0.92, 0.8, 0.06])
            title_ax.text(0.5, 0.5, f"Examples of Chest X-ray Image–Text Pair Augmentation", 
                         ha='center', va='center', fontsize=9, fontweight='bold',
                         color=colors['black'],
                         bbox=dict(boxstyle="round,pad=0.3", facecolor=colors['white'], 
                                 edgecolor=colors['gray'], linewidth=1))
            title_ax.axis('off')
            
            # Create grid with two horizontal boxes
            gs = GridSpec(2, 1, figure=fig, 
                         height_ratios=[1.15, 0.85],  # Top box 15% larger, bottom box 15% smaller
                         hspace=0.1,  # Reduced spacing between boxes
                         top=0.85, bottom=0.08, left=0.08, right=0.92)  # Original margins
            
            # TOP BOX: Text content (Original + 3 Augmented) - Equal spacing and larger boxes
            ax_text_box = fig.add_subplot(gs[0, 0])
            ax_text_box.set_xlim(0, 5)  # Increased width for wider box
            ax_text_box.set_ylim(0, 1)
            
            # Removed gray border around text box
            
            # Calculate equal spacing for text boxes
            total_text_width = 5.0  # Total width of text box (increased)
            num_text_boxes = 4  # Original + 3 augmented
            text_box_width = 0.92  # Width of each text box (0.8 * 1.15 = 0.92 - 15% wider)
            text_spacing = (total_text_width - (num_text_boxes * text_box_width)) / (num_text_boxes - 1)
            
            # Original text (left side) - larger text box with equal spacing, moved left
            orig_caption = self.decode_caption(original['caption'])
            wrapped_orig = textwrap.fill(orig_caption, width=16)  # Adjusted for larger box
            orig_x_pos = text_box_width/2 + text_spacing/2 + 0.2  # Moved left by adding 0.2
            ax_text_box.text(orig_x_pos, 0.5, wrapped_orig, 
                           ha='center', va='center', fontsize=8,  # Increased font size
                           bbox=dict(boxstyle="round,pad=0.46", facecolor='lightblue', alpha=0.8),  # 15% wider padding (0.4 * 1.15 = 0.46)
                           wrap=True)
            ax_text_box.text(orig_x_pos, 0.9, 'Original Text', fontsize=9, fontweight='bold', color='blue', ha='center')
            
            # Augmented texts (right side) - only first 3, larger text boxes with equal spacing
            for j in range(min(3, len(aug_group))):
                aug_x_pos = orig_x_pos + (j + 1) * (text_box_width + text_spacing)
                aug_caption = self.decode_caption(aug_group[j]['caption'])
                wrapped_aug = textwrap.fill(aug_caption, width=16)  # Adjusted for larger box
                ax_text_box.text(aug_x_pos, 0.5, wrapped_aug, 
                               ha='center', va='center', fontsize=8,  # Increased font size
                               bbox=dict(boxstyle="round,pad=0.46", facecolor='lightgray', alpha=0.8),  # 15% wider padding (0.4 * 1.15 = 0.46)
                               wrap=True)
                ax_text_box.text(aug_x_pos, 0.9, f'Augmented Text {j+1}', fontsize=9, fontweight='bold', color='black', ha='center')
            
            ax_text_box.axis('off')
            
            # BOTTOM BOX: Image content (Original + 3 Augmented) - Adjusted for images extending outside
            ax_img_box = fig.add_subplot(gs[1, 0])
            ax_img_box.set_xlim(0, 5)  # Increased width for wider box
            ax_img_box.set_ylim(-0.1, 1)  # Extended bottom to accommodate images extending outside
            
            # Removed gray border around image box
            
            # Original image (left side)
            orig_image = original['image']
            if len(orig_image.shape) == 3 and orig_image.shape[2] == 1:
                orig_image = orig_image.squeeze()
            
            # Reduce image height by 20% - show top 80% of the image
            if len(orig_image.shape) == 2:
                height, width = orig_image.shape
            else:
                height, width = orig_image.shape[:2]
            crop_height = int(height * 0.8)
            cropped_orig = orig_image[:crop_height, :]
            
            # Create inset axes for original image - 15% larger, moved 10% down and left
            ax_orig_img = fig.add_axes([0.08, 0.02, 0.1725, 0.4025])  # Moved left from 0.12 to 0.08
            ax_orig_img.imshow(cropped_orig, cmap='gray')
            ax_orig_img.set_title("Original Image", fontsize=8, fontweight='bold', color='blue')
            ax_orig_img.axis('off')
            ax_orig_img.set_aspect('equal')
            
            # Add professional border to original image
            for spine in ax_orig_img.spines.values():
                spine.set_color('blue')
                spine.set_linewidth(2)
            
            # Augmented images (right side of image box) - only first 3, 15% larger, equal spacing
            # Calculate equal spacing: (total width - image width) / (number of images - 1)
            total_width = 0.92  # Available width (1.0 - 0.08) - increased due to wider box
            image_width = 0.1725  # Width of each image
            num_images = 4  # Original + 3 augmented
            spacing = (total_width - (num_images * image_width)) / (num_images - 1)  # Equal spacing between images
            
            for j in range(min(3, len(aug_group))):
                # Calculate position with equal spacing - moved left
                left_pos = 0.08 + (j + 1) * (image_width + spacing)  # +1 because original is at position 0, moved left from 0.12 to 0.08
                aug_image = aug_group[j]['image']
                if len(aug_image.shape) == 3 and aug_image.shape[2] == 1:
                    aug_image = aug_image.squeeze()
                
                # Reduce image height by 20% - show top 80% of the image
                if len(aug_image.shape) == 2:
                    height, width = aug_image.shape
                else:
                    height, width = aug_image.shape[:2]
                crop_height = int(height * 0.8)
                cropped_aug = aug_image[:crop_height, :]
                
                # Create inset axes for augmented image - 15% larger, moved 10% down
                ax_aug_img = fig.add_axes([left_pos, 0.02, 0.1725, 0.4025])  # Moved down from 0.12 to 0.02 (10% down)
                ax_aug_img.imshow(cropped_aug, cmap='gray')
                ax_aug_img.set_title(f"Augmented Image {j+1}", fontsize=8, fontweight='bold', color='black')
                ax_aug_img.axis('off')
                ax_aug_img.set_aspect('equal')
                
                # Add professional border to augmented image
                for spine in ax_aug_img.spines.values():
                    spine.set_color('black')
                    spine.set_linewidth(2)
            
            ax_img_box.axis('off')
            
            # Add global boundary with professional styling
            fig.patch.set_edgecolor(colors['black'])
            fig.patch.set_linewidth(1)
            
            # Save with Scientific Reports specifications - generate separate file for each sample
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            sample_output_path = f"augmentation_sample_{i+1}_{timestamp}.png"
            
            plt.savefig(sample_output_path, dpi=300, bbox_inches='tight', 
                       facecolor=colors['white'], edgecolor=colors['black'],
                       format='png', transparent=False)
            print(f"Saved Scientific Reports compliant visualization to {sample_output_path}")
            plt.close()
        
        return fig
    
    def plot_detailed_sample(self, ax, sample, title="", color="#2E86AB"):
        """Plot a detailed sample with image and full text"""
        # Plot image
        image = sample['image']
        if len(image.shape) == 3 and image.shape[2] == 1:
            image = image.squeeze()
            
        # Reduce image height by 50% - only show top 50% of the image
        if len(image.shape) == 2:
            height, width = image.shape
        else:
            height, width = image.shape[:2]
        crop_height = int(height * 0.5)  # Changed from 0.75 to 0.5
        cropped_image = image[:crop_height, :]
            
        ax.imshow(cropped_image, cmap='gray', aspect='auto')
        ax.set_title(title, fontweight='bold', color=color, fontsize=12, pad=20)
        
        # Remove ticks and labels for image
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')
        
        # Add border
        for spine in ax.spines.values():
            spine.set_color(color)
            spine.set_linewidth(2)
        
        # Decode and display text in a proper box below image
        caption_text = self.decode_caption(sample['caption'])
        
        # Create a larger, more prominent text box below the image
        # Position the text box further down to ensure it's visible
        text_box = ax.text(0.5, -0.45, caption_text, transform=ax.transAxes,
                          ha='center', va='top', fontsize=10,
                          bbox=dict(boxstyle='round,pad=1.0', 
                                  facecolor='white', 
                                  edgecolor=color, 
                                  alpha=0.95,
                                  linewidth=2),
                          wrap=True)
        
        # Add study ID in top-left corner
        study_id = sample.get('study_id', 'Unknown')
        ax.text(0.02, 0.98, f"Study ID: {study_id}", transform=ax.transAxes,
               fontsize=8, verticalalignment='top',
               bbox=dict(boxstyle='round,pad=0.3', 
                        facecolor='white', 
                        edgecolor='gray', 
                        alpha=0.8))
        
        # Add image statistics in top-right corner
        non_zero_pixels = np.count_nonzero(cropped_image)
        total_pixels = cropped_image.size
        contrast_ratio = non_zero_pixels / total_pixels if total_pixels > 0 else 0
        
        ax.text(0.98, 0.98, f"Contrast: {contrast_ratio:.3f}", transform=ax.transAxes,
               fontsize=7, verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round,pad=0.3', 
                        facecolor='white', 
                        edgecolor='gray', 
                        alpha=0.8))
        
        # Add token statistics
        non_zero_tokens = np.count_nonzero(sample['caption'])
        total_tokens = len(sample['caption'])
        token_ratio = non_zero_tokens / total_tokens if total_tokens > 0 else 0
        
        ax.text(0.98, 0.92, f"Tokens: {non_zero_tokens}/{total_tokens}", transform=ax.transAxes,
               fontsize=7, verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round,pad=0.3', 
                        facecolor='white', 
                        edgecolor='gray', 
                        alpha=0.8))
    
    def create_text_analysis_figure(self, original_samples, augmented_samples, output_path=None):
        """Create a figure analyzing text differences"""
        if not original_samples or not augmented_samples:
            return
            
        # Analyze text differences
        text_analysis = self.analyze_text_differences(original_samples, augmented_samples)
        
        # Create analysis figure
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Text length comparison
        ax1 = axes[0, 0]
        self.plot_text_length_comparison(ax1, text_analysis)
        
        # 2. Vocabulary usage
        ax2 = axes[0, 1]
        self.plot_vocabulary_usage(ax2, text_analysis)
        
        # 3. Token distribution
        ax3 = axes[0, 2]
        self.plot_token_distribution(ax3, text_analysis)
        
        # 4. Augmentation diversity
        ax4 = axes[1, 0]
        self.plot_augmentation_diversity(ax4, text_analysis)
        
        # 5. Text similarity matrix
        ax5 = axes[1, 1]
        self.plot_similarity_matrix(ax5, text_analysis)
        
        # 6. Quality metrics
        ax6 = axes[1, 2]
        self.plot_quality_metrics(ax6, text_analysis)
        
        plt.tight_layout()
        
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"text_analysis_{timestamp}.png"
            
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Text analysis figure saved to: {output_path}")
        
        return fig
    
    def analyze_text_differences(self, original_samples, augmented_samples):
        """Analyze text differences between original and augmented samples"""
        analysis = {
            'original_lengths': [],
            'augmented_lengths': [],
            'original_vocab': set(),
            'augmented_vocab': set(),
            'diversity_scores': [],
            'similarity_scores': [],
            'original_texts': [],
            'augmented_texts': []
        }
        
        for original, aug_group in zip(original_samples, augmented_samples):
            # Original text
            orig_text = self.decode_caption(original['caption'])
            orig_words = orig_text.split()
            analysis['original_lengths'].append(len(orig_words))
            analysis['original_vocab'].update(orig_words)
            analysis['original_texts'].append(orig_text)
            
            # Augmented texts
            aug_texts = []
            for aug_sample in aug_group:
                aug_text = self.decode_caption(aug_sample['caption'])
                aug_words = aug_text.split()
                analysis['augmented_lengths'].append(len(aug_words))
                analysis['augmented_vocab'].update(aug_words)
                aug_texts.append(aug_text)
                analysis['augmented_texts'].append(aug_text)
            
            # Calculate diversity and similarity
            diversity = self.calculate_text_diversity(orig_text, aug_texts)
            analysis['diversity_scores'].append(diversity)
            
            similarity = self.calculate_text_similarity(orig_text, aug_texts)
            analysis['similarity_scores'].append(similarity)
            
        return analysis
    
    def calculate_text_diversity(self, original_text, augmented_texts):
        """Calculate diversity between original and augmented texts"""
        orig_words = set(original_text.lower().split())
        total_diversity = 0
        
        for aug_text in augmented_texts:
            aug_words = set(aug_text.lower().split())
            # Jaccard distance
            intersection = len(orig_words.intersection(aug_words))
            union = len(orig_words.union(aug_words))
            diversity = 1 - (intersection / union) if union > 0 else 0
            total_diversity += diversity
            
        return total_diversity / len(augmented_texts)
    
    def calculate_text_similarity(self, original_text, augmented_texts):
        """Calculate similarity between original and augmented texts"""
        orig_words = set(original_text.lower().split())
        total_similarity = 0
        
        for aug_text in augmented_texts:
            aug_words = set(aug_text.lower().split())
            # Jaccard similarity
            intersection = len(orig_words.intersection(aug_words))
            union = len(orig_words.union(aug_words))
            similarity = intersection / union if union > 0 else 0
            total_similarity += similarity
            
        return total_similarity / len(augmented_texts)
    
    def plot_text_length_comparison(self, ax, analysis):
        """Plot text length comparison"""
        ax.hist(analysis['original_lengths'], bins=10, alpha=0.7, 
               label='Original', color=self.colors['original'])
        ax.hist(analysis['augmented_lengths'], bins=10, alpha=0.7, 
               label='Augmented', color=self.colors['augmented'])
        
        ax.set_title('Text Length Distribution', fontweight='bold')
        ax.set_xlabel('Number of Words')
        ax.set_ylabel('Frequency')
        ax.legend()
    
    def plot_vocabulary_usage(self, ax, analysis):
        """Plot vocabulary usage comparison"""
        orig_vocab_size = len(analysis['original_vocab'])
        aug_vocab_size = len(analysis['augmented_vocab'])
        
        categories = ['Original', 'Augmented']
        sizes = [orig_vocab_size, aug_vocab_size]
        colors = [self.colors['original'], self.colors['augmented']]
        
        bars = ax.bar(categories, sizes, color=colors, alpha=0.8)
        ax.set_title('Vocabulary Size Comparison', fontweight='bold')
        ax.set_ylabel('Unique Words')
        
        # Add value labels
        for bar, size in zip(bars, sizes):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01*max(sizes),
                   f'{size}', ha='center', va='bottom', fontweight='bold')
    
    def plot_token_distribution(self, ax, analysis):
        """Plot token distribution"""
        ax.hist(analysis['original_lengths'], bins=8, alpha=0.7, 
               label='Original', color=self.colors['original'])
        ax.hist(analysis['augmented_lengths'], bins=8, alpha=0.7, 
               label='Augmented', color=self.colors['augmented'])
        
        ax.set_title('Token Count Distribution', fontweight='bold')
        ax.set_xlabel('Number of Tokens')
        ax.set_ylabel('Frequency')
        ax.legend()
    
    def plot_augmentation_diversity(self, ax, analysis):
        """Plot augmentation diversity scores"""
        ax.hist(analysis['diversity_scores'], bins=10, alpha=0.8, 
               color=self.colors['augmented'])
        ax.set_title('Augmentation Diversity Scores', fontweight='bold')
        ax.set_xlabel('Diversity Score (Jaccard Distance)')
        ax.set_ylabel('Frequency')
        
        # Add mean line
        mean_diversity = np.mean(analysis['diversity_scores'])
        ax.axvline(mean_diversity, color='red', linestyle='--', 
                  label=f'Mean: {mean_diversity:.3f}')
        ax.legend()
    
    def plot_similarity_matrix(self, ax, analysis):
        """Plot text similarity matrix"""
        if len(analysis['original_texts']) > 0 and len(analysis['augmented_texts']) > 0:
            # Calculate similarity matrix
            similarities = []
            for orig_text in analysis['original_texts']:
                orig_words = set(orig_text.lower().split())
                sample_similarities = []
                for aug_text in analysis['augmented_texts']:
                    aug_words = set(aug_text.lower().split())
                    intersection = len(orig_words.intersection(aug_words))
                    union = len(orig_words.union(aug_words))
                    similarity = intersection / union if union > 0 else 0
                    sample_similarities.append(similarity)
                similarities.append(sample_similarities)
            
            if similarities:
                im = ax.imshow(similarities, cmap='viridis', aspect='auto')
                ax.set_title('Text Similarity Matrix', fontweight='bold')
                ax.set_xlabel('Augmented Samples')
                ax.set_ylabel('Original Samples')
                plt.colorbar(im, ax=ax)
    
    def plot_quality_metrics(self, ax, analysis):
        """Plot quality metrics"""
        metrics = {
            'Avg Original Length': np.mean(analysis['original_lengths']),
            'Avg Augmented Length': np.mean(analysis['augmented_lengths']),
            'Vocabulary Expansion': len(analysis['augmented_vocab']) - len(analysis['original_vocab']),
            'Avg Diversity': np.mean(analysis['diversity_scores']),
            'Avg Similarity': np.mean(analysis['similarity_scores'])
        }
        
        categories = list(metrics.keys())
        values = list(metrics.values())
        colors = [self.colors['original'], self.colors['augmented'], 
                 self.colors['highlight'], self.colors['augmented'],
                 self.colors['original']]
        
        bars = ax.bar(categories, values, color=colors, alpha=0.8)
        ax.set_title('Quality Metrics', fontweight='bold')
        ax.set_ylabel('Value')
        
        # Rotate x-axis labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01*max(values),
                   f'{value:.2f}', ha='center', va='bottom', fontweight='bold')

    def create_single_sample_comparison(self, original_sample, aug_group, output_path=None, sample_number=1):
        """Create a single comparison showing one original with its 3 augmented versions"""
        
        if not original_sample or not aug_group or len(aug_group) < 3:
            print("Error: Need original sample and at least 3 augmented samples")
            return
        
        # Set Scientific Reports font and styling - DejaVu Serif for scientific standards
        plt.rcParams['font.family'] = 'DejaVu Serif'
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['axes.labelsize'] = 10
        plt.rcParams['xtick.labelsize'] = 9
        plt.rcParams['ytick.labelsize'] = 9
        plt.rcParams['legend.fontsize'] = 9
        
        # Scientific Reports color palette
        colors = {
            'blue': '#1f77b4',
            'red': '#d62728', 
            'green': '#2ca02c',
            'orange': '#ff7f0e',
            'white': '#ffffff',
            'black': '#000000',
            'gray': '#cccccc',
            'original': '#1f77b4',      # Blue for original
            'augmented': '#d62728'      # Red for augmented
        }
        
        # Create figure with increased height for larger images
        fig = plt.figure(figsize=(11.2, 9.6), facecolor=colors['white'])  # 20% smaller (14*0.8=11.2, 12*0.8=9.6)
        
        # Create title with Scientific Reports formatting
        title_ax = fig.add_axes([0.1, 0.95, 0.8, 0.03])
        title_ax.text(0.5, 0.5, f"Figure {sample_number}: Chest X-ray Image–Text Pair Augmentation", 
                     ha='center', va='center', fontsize=14, fontweight='bold',
                     color=colors['black'],
                     bbox=dict(boxstyle="round,pad=0.3", facecolor=colors['white'], 
                             edgecolor=colors['gray'], linewidth=1))
        title_ax.axis('off')
        
        # Create grid with two horizontal sections
        gs = GridSpec(2, 1, figure=fig, 
                     height_ratios=[1.0, 2.0],  # Text section smaller, image section much larger
                     hspace=0.15,  # Increased space between sections for scientific standards
                     top=0.92, bottom=0.08, left=0.03, right=0.97)
        
        # TOP SECTION: Text content (Original + 3 Augmented) - Even wider text boxes
        ax_text_box = fig.add_subplot(gs[0, 0])
        ax_text_box.set_xlim(0, 4)  # 4 text boxes
        ax_text_box.set_ylim(0, 1)
        
        # Calculate positions for 4 even wider text boxes
        text_box_width = 1.045  # 10% wider (0.95 * 1.1 = 1.045)
        text_spacing = 0.02   # Very small spacing between boxes
        total_used_width = (4 * text_box_width) + (3 * text_spacing)
        start_x = (4 - total_used_width) / 2  # Center the text boxes
        
        text_positions = []
        for i in range(4):
            x_pos = start_x + (i * (text_box_width + text_spacing)) + (text_box_width / 2)
            text_positions.append(x_pos)
        
        # Original text - even wider box
        orig_caption = self.decode_caption(original_sample['caption'])
        wrapped_orig = textwrap.fill(orig_caption, width=30)  # 10% more characters (28 * 1.1 ≈ 30)
        ax_text_box.text(text_positions[0], 0.5, wrapped_orig, 
                       ha='center', va='center', fontsize=10,
                       bbox=dict(boxstyle="round,pad=0.44", facecolor='lightblue', alpha=0.8),  # 10% more padding
                       wrap=True)
        ax_text_box.text(text_positions[0], 0.85, 'Original Text', fontsize=12, 
                       fontweight='bold', color='blue', ha='center')
        
        # Augmented texts - even wider boxes
        for j in range(min(3, len(aug_group))):
            aug_caption = self.decode_caption(aug_group[j]['caption'])
            wrapped_aug = textwrap.fill(aug_caption, width=30)  # 10% more characters (28 * 1.1 ≈ 30)
            ax_text_box.text(text_positions[j+1], 0.5, wrapped_aug, 
                           ha='center', va='center', fontsize=10,
                           bbox=dict(boxstyle="round,pad=0.44", facecolor='lightcoral', alpha=0.8),  # 10% more padding
                           wrap=True)
            ax_text_box.text(text_positions[j+1], 0.85, f'Augmented {j+1}', fontsize=12, 
                           fontweight='bold', color='red', ha='center')
        
        ax_text_box.axis('off')
        
        # BOTTOM SECTION: Images in one line - 40% larger images with increased spacing
        ax_img_box = fig.add_subplot(gs[1, 0])
        ax_img_box.set_xlim(0, 4)  # 4 images
        ax_img_box.set_ylim(0, 1)
        
        # Calculate positions for 4 images in one line - 40% larger with increased spacing
        image_width = 0.756   # 20% larger width (0.63 * 1.2 = 0.756)
        image_height = 2.94   # 20% larger height (2.45 * 1.2 = 2.94)
        image_spacing = 0.104 # 30% more space between images (0.08 * 1.3 = 0.104)
        total_image_width = (4 * image_width) + (3 * image_spacing)
        start_img_x = (4 - total_image_width) / 2  # Center the images
        
        image_positions = []
        for i in range(4):
            x_pos = start_img_x + (i * (image_width + image_spacing))
            image_positions.append(x_pos)
        
        # Original image - 40% larger
        orig_img = original_sample['image']
        if orig_img.shape[-1] == 3:  # RGB
            orig_img_display = orig_img
        else:  # Grayscale
            orig_img_display = np.squeeze(orig_img)
        
        orig_img_ax = fig.add_axes([image_positions[0]/4, 0.15, image_width/4, image_height/4])
        orig_img_ax.imshow(orig_img_display, cmap='gray' if len(orig_img_display.shape) == 2 else None)
        orig_img_ax.set_title('Original', fontsize=13, fontweight='bold', color='blue')
        orig_img_ax.axis('off')
        
        # Augmented images in one line - 40% larger
        for j in range(min(3, len(aug_group))):
            aug_img = aug_group[j]['image']
            if aug_img.shape[-1] == 3:  # RGB
                aug_img_display = aug_img
            else:  # Grayscale
                aug_img_display = np.squeeze(aug_img)
            
            aug_img_ax = fig.add_axes([image_positions[j+1]/4, 0.15, image_width/4, image_height/4])
            aug_img_ax.imshow(aug_img_display, cmap='gray' if len(aug_img_display.shape) == 2 else None)
            aug_img_ax.set_title(f'Augmented {j+1}', fontsize=13, fontweight='bold', color='red')
            aug_img_ax.axis('off')
        
        ax_img_box.axis('off')
        
        # Save the figure with scientific standards
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if output_path is None:
            output_path = f"augmentation_sample_{sample_number}_{timestamp}.png"
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"✅ Saved visualization {sample_number}: {output_path}")
        print(f"   Original: {original_sample['study_id']}")
        print(f"   Augmented: {[aug['study_id'] for aug in aug_group[:3]]}")
        
        return output_path

def main():
    """Main function to create clean augmentation visualizations"""
    print("Creating Clean Augmentation Visualizations")
    print("=" * 60)
    print("This will create 3 separate visualizations, each showing:")
    print("- ONE TRUE original image-text pair from original Indiana dataset")
    print("- THREE augmented versions of that same sample")
    print("- 2.5x larger images and wider text boxes")
    print("- Clean, publication-ready layout")
    print()
    
    # Clean up old PNG files first
    print("Cleaning up old PNG files...")
    import glob
    old_files = glob.glob("augmentation_sample_*.png")
    for old_file in old_files:
        try:
            os.remove(old_file)
            print(f"Deleted: {old_file}")
        except Exception as e:
            print(f"Could not delete {old_file}: {e}")
    
    if old_files:
        print(f"Cleaned up {len(old_files)} old PNG files")
    else:
        print("No old PNG files found to clean up")
    print()
    
    # Data directories
    augmented_dir = "/home/abedin/Developments/chest_x_ray_data_processing/all_processed_data/aug_indiana_extended"
    
    if not os.path.exists(augmented_dir):
        print(f"Error: Augmented data directory not found: {augmented_dir}")
        return
    
    # Initialize visualizer
    visualizer = DetailedAugmentationVisualizer(augmented_dir)
    
    # Generate 3 separate visualizations
    generated_files = []
    
    for i in range(3):
        print(f"\n{'='*40}")
        print(f"GENERATING VISUALIZATION {i+1}/3")
        print(f"{'='*40}")
        
        # Load TRUE original data and find corresponding augmented versions
        print(f"Loading TRUE original data for sample {i+1}...")
        original_samples, augmented_samples = visualizer.load_true_original_with_augmented(split='train', num_samples=1)
        
        if not original_samples or not augmented_samples:
            print(f"No matching original and augmented samples found for sample {i+1}. Skipping.")
            continue
        
        if not augmented_samples[0] or len(augmented_samples[0]) < 3:
            print(f"Not enough augmented samples found for sample {i+1}. Found: {len(augmented_samples[0]) if augmented_samples[0] else 0}")
            continue
        
        print(f"Loaded 1 TRUE original sample with {len(augmented_samples[0])} augmentations")
        
        # Show which sample was selected
        original_sample = original_samples[0]
        print(f"TRUE original sample {i+1}: {original_sample['study_id']}")
        print(f"Corresponding augmented samples: {[aug['study_id'] for aug in augmented_samples[0][:3]]}")
        
        # Create single comparison figure
        print(f"Creating visualization {i+1}...")
        output_file = visualizer.create_single_sample_comparison(
            original_samples[0], 
            augmented_samples[0], 
            sample_number=i+1
        )
        
        if output_file:
            generated_files.append(output_file)
        
        print(f"✅ Completed visualization {i+1}")
    
    print("\n" + "=" * 60)
    print("ALL VISUALIZATIONS COMPLETE")
    print("=" * 60)
    print(f"Generated {len(generated_files)} files:")
    for i, file in enumerate(generated_files, 1):
        print(f"{i}. {file}")
    
    print("\nEach visualization shows:")
    print("- ONE TRUE ORIGINAL image-text pair (from original Indiana dataset)")
    print("- THREE augmented versions:")
    print("  * Augmented Image & Text 1")
    print("  * Augmented Image & Text 2") 
    print("  * Augmented Image & Text 3")
    print("- Images are 2.5x larger (height and width)")
    print("- Text boxes are even wider with more characters per line")
    print("- REAL decoded text (not tokens)")
    print("- Clean, publication-ready layout")
    print("- TRUE original vs augmented comparison")

if __name__ == "__main__":
    main() 