#!/usr/bin/env python3
"""
Extract Vocabulary from MIMIC Shards Dataset

This script extracts the vocabulary from the mimic_shards_hufc4446-to128 dataset
and saves it in a format that can be used for processing other datasets.
"""

import os
import sys
import json
import pickle
import glob
import numpy as np
from collections import Counter, OrderedDict
from typing import Dict, List, Any, Tuple
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')


def extract_vocabulary_from_dataset(dataset_path: str, text_key: str = "captions"):
    """
    Extract vocabulary from the specified dataset.
    
    Args:
        dataset_path: Path to the dataset
        text_key: Key to extract text from data entries
    
    Returns:
        Tuple of (vocabulary_dict, word_index, index_word, token_frequencies)
    """
    print(f"🔍 Extracting vocabulary from: {dataset_path}")
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset path does not exist: {dataset_path}")
        return None, None, None, None
    
    # Find all pickle files
    file_pattern = os.path.join(dataset_path, "**", "*.pkl")
    files = glob.glob(file_pattern, recursive=True)
    
    if not files:
        print(f"❌ No files found in {dataset_path}")
        return None, None, None, None
    
    print(f"📄 Found {len(files)} files")
    
    # Load and process all data
    all_tokens = []
    all_sequences = []
    
    for file_path in tqdm(files, desc="Loading files"):
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            if isinstance(data, dict):
                # Convert dict to list of entries
                entries = []
                for i in range(len(data.get('images', []))):
                    entry = {}
                    for key, value in data.items():
                        if isinstance(value, (list, np.ndarray)) and i < len(value):
                            entry[key] = value[i]
                        else:
                            entry[key] = value
                    entries.append(entry)
            elif isinstance(data, list):
                entries = data
            else:
                continue
            
            # Extract text and tokenize
            for entry in entries:
                text = extract_text(entry, text_key)
                
                if text is not None:
                    if isinstance(text, (list, np.ndarray)):
                        # Pre-tokenized text
                        tokens = [str(token) for token in text if token != 0]
                        all_tokens.extend(tokens)
                        all_sequences.append(tokens)
                    else:
                        # Raw text
                        tokens = text.lower().split()
                        all_tokens.extend(tokens)
                        all_sequences.append(tokens)
                        
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue
    
    print(f"📊 Processed {len(all_sequences)} sequences")
    
    # Build vocabulary
    vocab_counter = Counter(all_tokens)
    vocab_size = len(vocab_counter)
    
    print(f"📚 Vocabulary size: {vocab_size:,}")
    
    # Create word_index and index_word mappings
    # Sort by frequency (most frequent first), then alphabetically
    sorted_tokens = sorted(vocab_counter.items(), 
                          key=lambda x: (-x[1], x[0]))  # Sort by frequency desc, then alphabetically
    
    word_index = OrderedDict()
    index_word = OrderedDict()
    token_frequencies = {}
    
    # Add special tokens first
    special_tokens = ['<PAD>', '<UNK>', '<START>', '<END>']
    for i, token in enumerate(special_tokens):
        word_index[token] = i
        index_word[i] = token
        token_frequencies[token] = 0  # Special tokens don't appear in text
    
    # Add vocabulary tokens
    for i, (token, frequency) in enumerate(sorted_tokens):
        word_index[token] = i + len(special_tokens)
        index_word[i + len(special_tokens)] = token
        token_frequencies[token] = frequency
    
    # Create comprehensive vocabulary dictionary
    vocabulary_dict = {
        'word_index': dict(word_index),
        'index_word': dict(index_word),
        'token_frequencies': token_frequencies,
        'vocab_size': len(word_index),
        'total_tokens': len(all_tokens),
        'unique_tokens': vocab_size,
        'most_common_tokens': dict(vocab_counter.most_common(20)),
        'least_common_tokens': dict(sorted(vocab_counter.items(), key=lambda x: x[1])[:20]),
        'statistics': {
            'mean_frequency': float(np.mean(list(vocab_counter.values()))),
            'median_frequency': float(np.median(list(vocab_counter.values()))),
            'std_frequency': float(np.std(list(vocab_counter.values()))),
            'min_frequency': float(min(vocab_counter.values())),
            'max_frequency': float(max(vocab_counter.values())),
            'tokens_with_frequency_1': sum(1 for freq in vocab_counter.values() if freq == 1),
            'tokens_with_frequency_gt_10': sum(1 for freq in vocab_counter.values() if freq > 10),
            'tokens_with_frequency_gt_100': sum(1 for freq in vocab_counter.values() if freq > 100)
        }
    }
    
    return vocabulary_dict, word_index, index_word, token_frequencies


def extract_text(entry: Dict, text_key: str) -> Any:
    """Extract text from entry."""
    if text_key in entry:
        return entry[text_key]
    
    # Try alternative keys
    for key in ['text', 'report', 'caption', 'findings', 'impression']:
        if key in entry:
            return entry[key]
    
    return None


def save_vocabulary(vocabulary_dict: Dict, word_index: OrderedDict, 
                   index_word: OrderedDict, token_frequencies: Dict,
                   output_dir: str = "extracted_vocabulary"):
    """Save vocabulary in multiple formats."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save comprehensive vocabulary dictionary
    vocab_path = os.path.join(output_dir, "vocabulary.json")
    with open(vocab_path, 'w') as f:
        json.dump(vocabulary_dict, f, indent=2)
    print(f"💾 Saved comprehensive vocabulary to: {vocab_path}")
    
    # Save word_index mapping
    word_index_path = os.path.join(output_dir, "word_index.json")
    with open(word_index_path, 'w') as f:
        json.dump(dict(word_index), f, indent=2)
    print(f"💾 Saved word_index to: {word_index_path}")
    
    # Save index_word mapping
    index_word_path = os.path.join(output_dir, "index_word.json")
    with open(index_word_path, 'w') as f:
        json.dump(dict(index_word), f, indent=2)
    print(f"💾 Saved index_word to: {index_word_path}")
    
    # Save token frequencies
    frequencies_path = os.path.join(output_dir, "token_frequencies.json")
    with open(frequencies_path, 'w') as f:
        json.dump(token_frequencies, f, indent=2)
    print(f"💾 Saved token frequencies to: {frequencies_path}")
    
    # Save pickle format for compatibility
    pickle_path = os.path.join(output_dir, "vocabulary.pkl")
    with open(pickle_path, 'wb') as f:
        pickle.dump(vocabulary_dict, f)
    print(f"💾 Saved vocabulary pickle to: {pickle_path}")
    
    # Save simple text format for easy inspection
    txt_path = os.path.join(output_dir, "vocabulary.txt")
    with open(txt_path, 'w') as f:
        f.write(f"Vocabulary Size: {vocabulary_dict['vocab_size']}\n")
        f.write(f"Total Tokens: {vocabulary_dict['total_tokens']}\n")
        f.write(f"Unique Tokens: {vocabulary_dict['unique_tokens']}\n\n")
        
        f.write("Word Index Mapping (first 50):\n")
        f.write("-" * 50 + "\n")
        for i, (word, index) in enumerate(word_index.items()):
            if i >= 50:
                break
            f.write(f"{index:4d}: {word}\n")
        
        f.write(f"\nMost Common Tokens:\n")
        f.write("-" * 30 + "\n")
        for word, freq in vocabulary_dict['most_common_tokens'].items():
            f.write(f"{word}: {freq:,}\n")
    
    print(f"💾 Saved vocabulary text to: {txt_path}")
    
    return output_dir


def print_vocabulary_summary(vocabulary_dict: Dict):
    """Print a summary of the extracted vocabulary."""
    
    print("\n" + "="*80)
    print("📚 EXTRACTED VOCABULARY SUMMARY")
    print("="*80)
    
    print(f"\n📈 VOCABULARY STATISTICS:")
    print(f"  Vocabulary Size: {vocabulary_dict['vocab_size']:,}")
    print(f"  Total Tokens: {vocabulary_dict['total_tokens']:,}")
    print(f"  Unique Tokens: {vocabulary_dict['unique_tokens']:,}")
    
    stats = vocabulary_dict['statistics']
    print(f"\n📊 FREQUENCY STATISTICS:")
    print(f"  Mean Frequency: {stats['mean_frequency']:.2f}")
    print(f"  Median Frequency: {stats['median_frequency']:.1f}")
    print(f"  Standard Deviation: {stats['std_frequency']:.2f}")
    print(f"  Minimum Frequency: {stats['min_frequency']:.0f}")
    print(f"  Maximum Frequency: {stats['max_frequency']:,}")
    print(f"  Tokens with Frequency = 1: {stats['tokens_with_frequency_1']:,}")
    print(f"  Tokens with Frequency > 10: {stats['tokens_with_frequency_gt_10']:,}")
    print(f"  Tokens with Frequency > 100: {stats['tokens_with_frequency_gt_100']:,}")
    
    print(f"\n🔝 MOST COMMON TOKENS:")
    for i, (token, freq) in enumerate(vocabulary_dict['most_common_tokens'].items(), 1):
        print(f"  {i:2d}. '{token}': {freq:,}")
    
    print(f"\n🔚 LEAST COMMON TOKENS:")
    for i, (token, freq) in enumerate(vocabulary_dict['least_common_tokens'].items(), 1):
        print(f"  {i:2d}. '{token}': {freq:,}")


def main():
    """Main function to extract vocabulary."""
    
    # Dataset path
    dataset_path = "/home/abedin/Developments/chest_x_ray_data_processing/all_processed_data/mimic_shards_hufc4446-to128"
    
    # Extract vocabulary
    vocabulary_dict, word_index, index_word, token_frequencies = extract_vocabulary_from_dataset(
        dataset_path, text_key="captions"
    )
    
    if vocabulary_dict is None:
        print("❌ Failed to extract vocabulary")
        return
    
    # Print summary
    print_vocabulary_summary(vocabulary_dict)
    
    # Save vocabulary
    output_dir = save_vocabulary(vocabulary_dict, word_index, index_word, token_frequencies)
    
    print(f"\n✅ Vocabulary extraction complete!")
    print(f"📁 All files saved to: {output_dir}")
    print(f"\n🎯 You can now use these vocabulary files to process the standard MIMIC dataset.")
    print(f"   Key files:")
    print(f"   - {output_dir}/word_index.json: Token to index mapping")
    print(f"   - {output_dir}/index_word.json: Index to token mapping")
    print(f"   - {output_dir}/vocabulary.json: Complete vocabulary information")


if __name__ == "__main__":
    main() 