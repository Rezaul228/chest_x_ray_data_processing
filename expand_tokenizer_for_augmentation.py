#!/usr/bin/env python3
"""
Expand Tokenizer for Data Augmentation
This script shows how to expand the existing tokenizer with medical terms
and adapt the augmentation scripts to work with the new vocabulary.
"""

import os
import sys
import pickle
import numpy as np
import argparse
from tqdm import tqdm
import glob

# Import the medical terms from the augmentation script
from adv_aug_text import ADVANCED_MEDICAL_TERMS, STYLE_VARIATIONS

def load_existing_metadata(metadata_path):
    """Load existing metadata and tokenizer"""
    print(f"Loading existing metadata from: {metadata_path}")
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    
    tokenizer = metadata.get('tokenizer')
    vocab_size = metadata.get('vocab_size')
    
    print(f"Current vocabulary size: {vocab_size}")
    print(f"Tokenizer type: {type(tokenizer)}")
    
    return metadata, tokenizer, vocab_size

def expand_tokenizer_with_medical_terms(tokenizer, medical_terms_dict, style_variations=None):
    """
    Expand tokenizer vocabulary with medical terms from augmentation scripts
    
    Args:
        tokenizer: Existing tokenizer object
        medical_terms_dict: Dictionary of medical terms and synonyms
        style_variations: Dictionary of style variations
        
    Returns:
        Expanded tokenizer and new vocabulary size
    """
    print("Expanding tokenizer with medical terms...")
    
    # Extract all terms and synonyms
    all_terms = set()
    
    # Add main medical terms
    for term, synonyms in medical_terms_dict.items():
        all_terms.add(term.lower())
        for synonym in synonyms:
            all_terms.add(synonym.lower())
    
    # Add style variations
    if style_variations:
        for style_dict in style_variations.values():
            for term in style_dict.keys():
                all_terms.add(term.lower())
            for term in style_dict.values():
                all_terms.add(term.lower())
    
    # Convert to list and sort
    all_terms_list = sorted(list(all_terms))
    
    print(f"Found {len(all_terms_list)} unique medical terms to add")
    
    # Check which terms are already in the tokenizer
    existing_terms = []
    missing_terms = []
    
    for term in all_terms_list:
        if term in tokenizer.word_index:
            existing_terms.append(term)
        else:
            missing_terms.append(term)
    
    print(f"Terms already in tokenizer: {len(existing_terms)}")
    print(f"Terms to add: {len(missing_terms)}")
    
    if not missing_terms:
        print("All medical terms are already in the tokenizer!")
        return tokenizer, len(tokenizer.word_index) + 1
    
    # Backup original tokenizer state
    original_word_index = tokenizer.word_index.copy()
    original_index_word = tokenizer.index_word.copy()
    original_vocab_size = len(original_word_index) + 1
    
    # Create a text corpus with missing terms
    medical_corpus = [" ".join(missing_terms)]
    
    # Update the tokenizer with new terms
    tokenizer.fit_on_texts(medical_corpus)
    
    new_vocab_size = len(tokenizer.word_index) + 1
    added_terms = new_vocab_size - original_vocab_size
    
    print(f"✅ Successfully expanded tokenizer:")
    print(f"   Original vocabulary size: {original_vocab_size}")
    print(f"   New vocabulary size: {new_vocab_size}")
    print(f"   Added {added_terms} new terms")
    
    # Show some examples of added terms
    print(f"\nSample added terms:")
    for i, term in enumerate(missing_terms[:10]):
        token_id = tokenizer.word_index.get(term, "NOT_FOUND")
        print(f"   '{term}' -> token {token_id}")
    
    return tokenizer, new_vocab_size

def update_metadata_with_expanded_tokenizer(original_metadata, expanded_tokenizer, new_vocab_size):
    """Update metadata with expanded tokenizer"""
    print("Updating metadata with expanded tokenizer...")
    
    # Create new metadata
    updated_metadata = original_metadata.copy()
    updated_metadata['tokenizer'] = expanded_tokenizer
    updated_metadata['vocab_size'] = new_vocab_size
    
    # Add information about the expansion
    updated_metadata['tokenizer_expansion'] = {
        'original_vocab_size': len(original_metadata.get('tokenizer', {}).word_index) + 1,
        'new_vocab_size': new_vocab_size,
        'added_terms': new_vocab_size - len(original_metadata.get('tokenizer', {}).word_index) - 1,
        'expansion_date': str(np.datetime64('now'))
    }
    
    return updated_metadata

def test_expanded_tokenizer(tokenizer, medical_terms_dict):
    """Test the expanded tokenizer with medical terms"""
    print("\nTesting expanded tokenizer...")
    
    test_terms = [
        "opacity", "consolidation", "infiltrate", "cardiomegaly", 
        "enlarged heart", "pneumonia", "pulmonary infection",
        "effusion", "fluid collection", "normal", "unremarkable"
    ]
    
    print("Testing tokenization of medical terms:")
    for term in test_terms:
        if term in tokenizer.word_index:
            token_id = tokenizer.word_index[term]
            print(f"   '{term}' -> token {token_id}")
        else:
            print(f"   '{term}' -> NOT FOUND")
    
    # Test encoding and decoding
    test_text = "chest xray shows opacity and cardiomegaly"
    tokens = tokenizer.texts_to_sequences([test_text])[0]
    decoded = [tokenizer.index_word.get(t, '<UNK>') for t in tokens]
    
    print(f"\nTest encoding/decoding:")
    print(f"   Input: '{test_text}'")
    print(f"   Tokens: {tokens}")
    print(f"   Decoded: {decoded}")

def create_augmentation_compatible_metadata(original_metadata_path, output_metadata_path):
    """Create metadata file compatible with augmentation scripts"""
    print(f"Creating augmentation-compatible metadata...")
    
    # Load original metadata
    with open(original_metadata_path, 'rb') as f:
        original_metadata = pickle.load(f)
    
    # Expand tokenizer
    tokenizer = original_metadata['tokenizer']
    expanded_tokenizer, new_vocab_size = expand_tokenizer_with_medical_terms(
        tokenizer, ADVANCED_MEDICAL_TERMS, STYLE_VARIATIONS
    )
    
    # Update metadata
    updated_metadata = update_metadata_with_expanded_tokenizer(
        original_metadata, expanded_tokenizer, new_vocab_size
    )
    
    # Save updated metadata
    with open(output_metadata_path, 'wb') as f:
        pickle.dump(updated_metadata, f)
    
    print(f"✅ Updated metadata saved to: {output_metadata_path}")
    
    # Test the expanded tokenizer
    test_expanded_tokenizer(expanded_tokenizer, ADVANCED_MEDICAL_TERMS)
    
    return updated_metadata

def modify_augmentation_scripts_for_expanded_vocab():
    """Show how to modify existing augmentation scripts"""
    print("\n" + "="*60)
    print("HOW TO MODIFY EXISTING AUGMENTATION SCRIPTS")
    print("="*60)
    
    print("""
1. UPDATE segregated_augmentation.py:
   
   In the load_shards_data() function, add this after loading metadata:
   
   ```python
   # Check if tokenizer has been expanded
   if 'tokenizer_expansion' in metadata:
       print(f"Using expanded tokenizer with {metadata['vocab_size']} words")
       print(f"Added {metadata['tokenizer_expansion']['added_terms']} medical terms")
   ```

2. UPDATE adv_aug_text.py:
   
   In the apply_advanced_text_augmentation() function, add vocabulary coverage check:
   
   ```python
   def apply_advanced_text_augmentation(text, tokenizer, config):
       # Check vocabulary coverage for medical terms
       medical_terms_coverage = check_medical_terms_coverage(tokenizer)
       print(f"Medical terms coverage: {medical_terms_coverage:.1f}%")
       
       # Rest of the function remains the same
       ...
   
   def check_medical_terms_coverage(tokenizer):
       covered = 0
       total = 0
       for term, synonyms in ADVANCED_MEDICAL_TERMS.items():
           total += 1 + len(synonyms)
           if term in tokenizer.word_index:
               covered += 1
           for synonym in synonyms:
               if synonym in tokenizer.word_index:
                   covered += 1
       return (covered / total) * 100 if total > 0 else 0
   ```

3. UPDATE adv_aug_config.py:
   
   Add configuration for expanded vocabulary:
   
   ```python
   class AdvAugConfig:
       def __init__(self):
           # Existing settings...
           
           # New settings for expanded vocabulary
           self.use_expanded_vocab = True
           self.min_medical_terms_coverage = 80.0  # Minimum coverage required
           self.medical_terms_augmentation_weight = 0.3  # Weight for medical term replacement
   ```

4. UPDATE main augmentation script:
   
   In the main() function, add vocabulary validation:
   
   ```python
   def main():
       # ... existing code ...
       
       # Validate expanded vocabulary
       if 'tokenizer_expansion' in metadata:
           print(f"✅ Using expanded tokenizer with {metadata['vocab_size']} words")
           print(f"✅ Added {metadata['tokenizer_expansion']['added_terms']} medical terms")
       else:
           print("⚠️ Using original tokenizer - consider expanding for better augmentation")
   ```
""")

def show_usage_examples():
    """Show practical usage examples"""
    print("\n" + "="*60)
    print("PRACTICAL USAGE EXAMPLES")
    print("="*60)
    
    print("""
1. EXPAND EXISTING TOKENIZER:
   
   ```bash
   python expand_tokenizer_for_augmentation.py \\
       --input_metadata all_processed_data/indiana_shards/metadata.pkl \\
       --output_metadata all_processed_data/indiana_shards/metadata_expanded.pkl
   ```

2. RUN AUGMENTATION WITH EXPANDED VOCABULARY:
   
   ```bash
   python segregated_augmentation.py \\
       --base_shard_dir all_processed_data/indiana_shards \\
       --output_dir augmented_data_expanded \\
       --num_augmentations 6 \\
       --max_sequence_length 128
   ```

3. VERIFY EXPANDED TOKENIZER:
   
   ```python
   # Load expanded metadata
   with open('metadata_expanded.pkl', 'rb') as f:
       metadata = pickle.load(f)
   
   tokenizer = metadata['tokenizer']
   print(f"Vocabulary size: {metadata['vocab_size']}")
   
   # Test medical terms
   test_terms = ["opacity", "consolidation", "cardiomegaly"]
   for term in test_terms:
       if term in tokenizer.word_index:
           print(f"'{term}' -> token {tokenizer.word_index[term]}")
       else:
           print(f"'{term}' -> NOT FOUND")
   ```

4. COMPARE ORIGINAL vs EXPANDED:
   
   ```python
   # Load original
   with open('metadata_original.pkl', 'rb') as f:
       original = pickle.load(f)
   
   # Load expanded
   with open('metadata_expanded.pkl', 'rb') as f:
       expanded = pickle.load(f)
   
   print(f"Original vocab size: {original['vocab_size']}")
   print(f"Expanded vocab size: {expanded['vocab_size']}")
   print(f"Added terms: {expanded['vocab_size'] - original['vocab_size']}")
   ```
""")

def main():
    parser = argparse.ArgumentParser(description="Expand tokenizer for data augmentation")
    parser.add_argument("--input_metadata", type=str, required=True,
                      help="Path to original metadata.pkl file")
    parser.add_argument("--output_metadata", type=str, required=True,
                      help="Path to save expanded metadata.pkl file")
    parser.add_argument("--test_only", action="store_true",
                      help="Only test existing tokenizer without expanding")
    
    args = parser.parse_args()
    
    print("🔧 TOKENIZER EXPANSION FOR DATA AUGMENTATION")
    print("=" * 60)
    
    if not os.path.exists(args.input_metadata):
        print(f"❌ Error: Input metadata file not found: {args.input_metadata}")
        return
    
    # Load existing metadata
    metadata, tokenizer, vocab_size = load_existing_metadata(args.input_metadata)
    
    if args.test_only:
        print("\n🧪 TESTING EXISTING TOKENIZER")
        test_expanded_tokenizer(tokenizer, ADVANCED_MEDICAL_TERMS)
    else:
        # Create expanded metadata
        updated_metadata = create_augmentation_compatible_metadata(
            args.input_metadata, args.output_metadata
        )
        
        # Show modification instructions
        modify_augmentation_scripts_for_expanded_vocab()
        
        # Show usage examples
        show_usage_examples()
    
    print("\n✅ Tokenizer expansion process completed!")

if __name__ == "__main__":
    main() 