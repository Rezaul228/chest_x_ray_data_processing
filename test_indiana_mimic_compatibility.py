#!/usr/bin/env python3
"""
Test Indiana-MIMIC Compatibility

This script tests the compatibility between Indiana University data and MIMIC-CXR vocabulary
to ensure cross-dataset evaluation is possible.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime

# Add current directory to path to import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_set_loader_simplified import IndianaDatasetLoaderSimplified

def test_mimic_vocabulary_coverage():
    """Test MIMIC vocabulary coverage on Indiana data"""
    
    # Check if MIMIC vocabulary files exist
    vocab_path = 'mimic_frontal_complete_vocab_vocab.json'
    index_word_path = 'mimic_frontal_complete_vocab_index_word.json'
    
    if not os.path.exists(vocab_path):
        print(f"Error: MIMIC vocabulary file not found: {vocab_path}")
        return False
    
    if not os.path.exists(index_word_path):
        print(f"Error: MIMIC index_word file not found: {index_word_path}")
        return False
    
    # Load MIMIC vocabulary
    print("Loading MIMIC vocabulary...")
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)
    
    with open(index_word_path, 'r') as f:
        index_word = json.load(f)
    
    print(f"MIMIC vocabulary size: {len(vocab)}")
    print(f"MIMIC index_word size: {len(index_word)}")
    
    # Check for Indiana data files (you'll need to provide these paths)
    indiana_reports = input("Enter path to indiana_reports.csv: ").strip()
    indiana_projections = input("Enter path to indiana_projections.csv: ").strip()
    indiana_images = input("Enter path to Indiana images directory: ").strip()
    
    if not os.path.exists(indiana_reports):
        print(f"Error: Indiana reports file not found: {indiana_reports}")
        return False
    
    if not os.path.exists(indiana_projections):
        print(f"Error: Indiana projections file not found: {indiana_projections}")
        return False
    
    if not os.path.exists(indiana_images):
        print(f"Error: Indiana images directory not found: {indiana_images}")
        return False
    
    print("\nTesting Indiana data with MIMIC vocabulary...")
    
    try:
        # Initialize loader with MIMIC vocabulary
        loader = IndianaDatasetLoaderSimplified(
            reports_csv_path=indiana_reports,
            projections_csv_path=indiana_projections,
            image_dir=indiana_images,
            max_studies=100,  # Test with 100 studies first
            max_sequence_length=128,
            shard_size=50,
            shard_dir='test_indiana_mimic',
            skip_metadata_processing=False,
            vocab_path=vocab_path,
            index_word_path=index_word_path
        )
        
        print(f"✓ Successfully loaded {len(loader.study_entries)} Indiana studies")
        print(f"✓ Vocabulary size: {len(loader.tokenizer.word_index) + 1}")
        print(f"✓ Number of labels: {len(loader.label_names)}")
        
        # Test text processing
        print("\nTesting text processing...")
        
        # Get sample texts
        sample_texts = []
        for entry in loader.study_entries[:10]:  # First 10 entries
            findings = entry['findings'] if entry['findings'] else ''
            impression = entry['impression'] if entry['impression'] else ''
            combined_text = findings + ' ' + impression
            sample_texts.append(combined_text.strip())
        
        # Test tokenization
        sequences = loader.tokenizer.texts_to_sequences(sample_texts)
        
        # Calculate coverage statistics
        total_tokens = sum(len(seq) for seq in sequences)
        unk_tokens = sum(sum(1 for token in seq if token == 1) for seq in sequences)  # 1 is UNK token
        coverage = (total_tokens - unk_tokens) / total_tokens if total_tokens > 0 else 0
        
        print(f"✓ Text processing successful")
        print(f"✓ Total tokens in sample: {total_tokens}")
        print(f"✓ UNK tokens: {unk_tokens}")
        print(f"✓ MIMIC vocabulary coverage: {coverage:.2%}")
        
        # Show sample decoded text
        if sequences:
            print("\nSample decoded text:")
            caption_tokens = sequences[0]
            decoded_words = []
            for token in caption_tokens:
                if token != 0:  # Skip padding
                    word = loader.tokenizer.index_word.get(token, '<unk>')
                    decoded_words.append(word)
            sample_text = ' '.join(decoded_words)
            print(f"Text: {sample_text[:200]}{'...' if len(sample_text) > 200 else ''}")
        
        # Test image loading
        print("\nTesting image loading...")
        sample_entry = loader.study_entries[0]
        image_path = sample_entry['frontal_path']
        
        if os.path.exists(image_path):
            try:
                img_array = loader.load_and_preprocess_image(image_path)
                if img_array is not None:
                    print(f"✓ Image loading successful: {img_array.shape}")
                else:
                    print("✗ Image loading failed")
            except Exception as e:
                print(f"✗ Image loading error: {e}")
        else:
            print(f"✗ Image file not found: {image_path}")
        
        # Test shard creation
        print("\nTesting shard creation...")
        try:
            loader.create_shards_with_test_split(
                train_ratio=0.8,
                val_ratio=0.1,
                test_ratio=0.1,
                seed=42
            )
            print("✓ Shard creation successful")
            
            # Test data loading
            train_images, train_captions, train_study_ids = loader.get_training_data(num_samples=5)
            if train_images is not None:
                print(f"✓ Training data loading successful: {train_images.shape}")
            
            val_images, val_captions, val_study_ids = loader.get_validation_data(num_samples=5)
            if val_images is not None:
                print(f"✓ Validation data loading successful: {val_images.shape}")
            
            test_images, test_captions, test_study_ids = loader.get_test_data(num_samples=5)
            if test_images is not None:
                print(f"✓ Test data loading successful: {test_images.shape}")
                
        except Exception as e:
            print(f"✗ Shard creation error: {e}")
            return False
        
        print("\n" + "=" * 80)
        print("COMPATIBILITY TEST COMPLETE")
        print("=" * 80)
        print("✓ Indiana data can be processed with MIMIC vocabulary")
        print(f"✓ Vocabulary coverage: {coverage:.2%}")
        print("✓ Cross-dataset evaluation is possible")
        print("✓ Ready for model testing")
        print("=" * 80)
        
        return True
        
    except Exception as e:
        print(f"✗ Error during compatibility test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Indiana-MIMIC Compatibility Test")
    print("=" * 50)
    
    success = test_mimic_vocabulary_coverage()
    
    if success:
        print("\n🎉 Compatibility test PASSED!")
        print("You can now process Indiana data with MIMIC vocabulary for cross-dataset evaluation.")
    else:
        print("\n❌ Compatibility test FAILED!")
        print("Please check the error messages above.") 