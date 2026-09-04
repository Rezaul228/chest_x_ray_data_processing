#!/usr/bin/env python3
"""
Fix Tokenizer Metadata

This script fixes the broken tokenizer metadata in the mimic_shards_4446_128 dataset.
The issue is that the tokenizer was saved as a class type instead of an instance.
"""

import os
import sys
import pickle
import json
from enhanced_data_loader import EnhancedTokenizer

def fix_tokenizer_metadata(dataset_path: str):
    """Fix the tokenizer metadata in the dataset."""
    print(f"🔧 Fixing tokenizer metadata in: {dataset_path}")
    
    metadata_path = os.path.join(dataset_path, 'metadata.pkl')
    
    if not os.path.exists(metadata_path):
        print(f"❌ Metadata file not found: {metadata_path}")
        return False
    
    # Load the broken metadata
    try:
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        print(f"📄 Loaded metadata with tokenizer type: {type(metadata.get('tokenizer'))}")
        
        # Check if tokenizer is broken (saved as class type)
        tokenizer = metadata.get('tokenizer')
        if isinstance(tokenizer, type):
            print(f"❌ Tokenizer is broken (saved as class type): {tokenizer}")
            
            # Load the vocabulary from the extracted metadata
            vocab_path = "/home/abedin/Developments/chest_x_ray_data_processing/extracted_vocabulary_metadata/word_index_from_metadata.json"
            index_word_path = "/home/abedin/Developments/chest_x_ray_data_processing/extracted_vocabulary_metadata/index_word_from_metadata.json"
            
            if not os.path.exists(vocab_path):
                print(f"❌ Vocabulary file not found: {vocab_path}")
                return False
            
            # Create a proper EnhancedTokenizer instance
            print(f"🔧 Creating proper EnhancedTokenizer instance...")
            fixed_tokenizer = EnhancedTokenizer()
            fixed_tokenizer.load_from_files(vocab_path, index_word_path)
            
            # Update the metadata
            metadata['tokenizer'] = fixed_tokenizer
            
            # Save the fixed metadata
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            print(f"✅ Fixed tokenizer metadata!")
            print(f"   New tokenizer type: {type(fixed_tokenizer)}")
            print(f"   Vocabulary size: {len(fixed_tokenizer.word_index)}")
            
            return True
        else:
            print(f"✅ Tokenizer is already working: {type(tokenizer)}")
            return True
            
    except Exception as e:
        print(f"❌ Error fixing metadata: {e}")
        return False

def verify_tokenizer_fix(dataset_path: str):
    """Verify that the tokenizer fix worked."""
    print(f"🔍 Verifying tokenizer fix...")
    
    metadata_path = os.path.join(dataset_path, 'metadata.pkl')
    
    try:
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        tokenizer = metadata.get('tokenizer')
        print(f"📄 Tokenizer type: {type(tokenizer)}")
        
        if hasattr(tokenizer, 'word_index'):
            print(f"✅ Tokenizer has word_index attribute")
            print(f"   Vocabulary size: {len(tokenizer.word_index)}")
            
            # Test tokenization
            test_text = "chest x-ray shows no abnormalities"
            sequences = tokenizer.texts_to_sequences([test_text])
            print(f"✅ Tokenization test passed: {sequences[0][:10]}...")
            
            return True
        else:
            print(f"❌ Tokenizer missing word_index attribute")
            return False
            
    except Exception as e:
        print(f"❌ Error verifying tokenizer: {e}")
        return False

def main():
    """Main function to fix the tokenizer metadata."""
    
    dataset_path = "/home/abedin/Developments/chest_x_ray_data_processing/all_processed_data/mimic_shards_4446_128"
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset path not found: {dataset_path}")
        return
    
    # Fix the tokenizer metadata
    success = fix_tokenizer_metadata(dataset_path)
    
    if success:
        # Verify the fix
        verify_tokenizer_fix(dataset_path)
        print(f"\n✅ Tokenizer metadata fix completed successfully!")
    else:
        print(f"\n❌ Failed to fix tokenizer metadata")

if __name__ == "__main__":
    main() 