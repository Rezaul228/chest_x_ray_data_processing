#!/usr/bin/env python3
"""
Process MIMIC-CXR Data with Extracted Vocabulary

This script processes MIMIC-CXR data using the vocabulary extracted from metadata.
It uses the enhanced tokenizer with the pre-built vocabulary from the metadata extraction.

Usage:
    python process_with_extracted_vocabulary.py --metadata_csv path/to/metadata.csv \
                                               --reports_dir path/to/reports \
                                               --images_dir path/to/images \
                                               --output_dir mimic_shards_with_extracted_vocab
"""

import os
import sys
import argparse
import numpy as np
import random
import pickle
import json
from mimic_data_loader import MIMICDatasetLoader

# Try to import enhanced loader
try:
    from enhanced_data_loader import EnhancedTokenizer
    ENHANCED_LOADER_AVAILABLE = True
except ImportError:
    ENHANCED_LOADER_AVAILABLE = False
    print("Warning: Enhanced data loader not available. Using basic loader.")

def validate_vocabulary_files(vocab_path, index_word_path):
    """Validate that the vocabulary files exist and are properly formatted."""
    print(f"🔍 Validating vocabulary files...")
    
    # Check if files exist
    if not os.path.exists(vocab_path):
        print(f"❌ Vocabulary file not found: {vocab_path}")
        return False
    
    if not os.path.exists(index_word_path):
        print(f"❌ Index word file not found: {index_word_path}")
        return False
    
    # Load and validate vocabulary
    try:
        with open(vocab_path, 'r') as f:
            word_index = json.load(f)
        
        with open(index_word_path, 'r') as f:
            index_word = json.load(f)
        
        print(f"✅ Vocabulary loaded successfully:")
        print(f"   - Word index entries: {len(word_index):,}")
        print(f"   - Index word entries: {len(index_word):,}")
        
        # Check for special tokens
        special_tokens = ['<pad>', '<unk>', '<start>', '<end>']
        missing_tokens = [token for token in special_tokens if token not in word_index]
        
        if missing_tokens:
            print(f"⚠️  Missing special tokens: {missing_tokens}")
        else:
            print(f"✅ All special tokens present")
        
        # Show some example entries
        print(f"\n📚 Example vocabulary entries:")
        example_words = list(word_index.keys())[:10]
        for word in example_words:
            print(f"   '{word}': {word_index[word]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading vocabulary: {e}")
        return False

def create_enhanced_tokenizer_from_files(vocab_path, index_word_path):
    """Create an enhanced tokenizer from the extracted vocabulary files."""
    print(f"🔧 Creating enhanced tokenizer from extracted vocabulary...")
    
    try:
        # Load vocabulary files
        with open(vocab_path, 'r') as f:
            word_index = json.load(f)
        
        with open(index_word_path, 'r') as f:
            index_word = json.load(f)
        
        # Create enhanced tokenizer
        tokenizer = EnhancedTokenizer()
        tokenizer.word_index = word_index
        tokenizer.index_word = index_word
        tokenizer.oov_token = "<unk>"
        tokenizer.oov_index = word_index.get("<unk>", 1)
        
        print(f"✅ Enhanced tokenizer created successfully")
        print(f"   - Vocabulary size: {len(word_index):,}")
        print(f"   - OOV token: {tokenizer.oov_token} (index: {tokenizer.oov_index})")
        
        return tokenizer
        
    except Exception as e:
        print(f"❌ Error creating enhanced tokenizer: {e}")
        return None

def main(args):
    """Main data processing function"""
    print("=== MIMIC-CXR Dataset Processing with Extracted Vocabulary ===")
    print(f"Metadata CSV: {args.metadata_csv}")
    print(f"Reports Directory: {args.reports_dir}")
    print(f"Images Directory: {args.images_dir}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Vocabulary Path: {args.vocab_path}")
    print(f"Index Word Path: {args.index_word_path}")
    
    # Validate vocabulary files
    if not validate_vocabulary_files(args.vocab_path, args.index_word_path):
        print("❌ Vocabulary validation failed. Exiting.")
        return False
    
    print()
    
    # Set seed for reproducibility
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Create enhanced tokenizer
    enhanced_tokenizer = create_enhanced_tokenizer_from_files(args.vocab_path, args.index_word_path)
    if not enhanced_tokenizer:
        print("❌ Failed to create enhanced tokenizer. Exiting.")
        return False
    
    # Initialize the dataset loader with enhanced tokenizer
    print("Initializing MIMIC-CXR dataset loader with extracted vocabulary...")
    
    if args.skip_processing:
        print("Skip processing flag is set. Will try to use existing processed data.")
    
    # Create a custom MIMICDatasetLoader that uses our enhanced tokenizer
    loader = MIMICDatasetLoader(
        metadata_csv_path=args.metadata_csv,
        reports_dir=args.reports_dir,
        images_dir=args.images_dir,
        image_size=(args.image_size, args.image_size),
        batch_size=args.batch_size,
        max_studies=args.max_studies,
        max_sequence_length=args.max_seq_length,
        shard_size=args.shard_size,
        shard_dir=args.output_dir,
        skip_metadata_processing=args.skip_processing
    )
    
    # Replace the tokenizer with our enhanced one
    loader.tokenizer = enhanced_tokenizer
    
    # Check if we should skip processing
    metadata_path = os.path.join(loader.shard_base_dir, 'metadata.pkl')
    if args.skip_processing and os.path.exists(metadata_path):
        print("Existing data shards found. Skipping data processing...")
        # Mark shards as created to prevent reprocessing
        loader.shards_created = True
        
        # Load metadata to get other info
        try:
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            print("Successfully loaded metadata from existing shards.")
        except Exception as e:
            print(f"Error loading metadata: {e}")
            print("Will process data from scratch.")
            loader.create_shards_with_test_split()
    else:
        # Process data from scratch
        print("Creating data shards with train/val/test split using extracted vocabulary...")
        if not hasattr(loader, 'test_shard_dir') or not os.path.exists(loader.test_shard_dir):
            loader.create_shards_with_test_split()
    
    # Get data statistics
    print("\n=== Data Processing Complete ===")
    
    # Load and display statistics
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    
    print(f"Vocabulary size: {metadata['vocab_size']}")
    print(f"Number of train shards: {metadata['num_train_shards']}")
    print(f"Number of validation shards: {metadata['num_val_shards']}")
    print(f"Number of test shards: {metadata['num_test_shards']}")
    
    # Test loading a small sample of data
    if args.test_loading:
        print("\n=== Testing Data Loading ===")
        
        try:
            # Test validation data loading
            val_data = loader.get_validation_data(num_samples=5)
            print(f"Successfully loaded {len(val_data['images'])} validation samples")
            print(f"Image shape: {val_data['images'][0].shape}")
            print(f"Caption shape: {val_data['captions'][0].shape}")
            
            # Test training data loading
            train_data = loader.get_training_data(num_samples=5)
            print(f"Training data contains {len(train_data['images'])} samples")
            
            # Test test data loading
            test_data = loader.get_test_data(num_samples=5)
            print(f"Successfully loaded {len(test_data['images'])} test samples")
            
            print("✓ All data loading tests passed!")
            
        except Exception as e:
            print(f"✗ Error during data loading test: {e}")
            return False
    
    print(f"\n✓ Data processing completed successfully!")
    print(f"Data saved to: {loader.shard_base_dir}")
    print(f"Vocabulary used: {args.vocab_path}")
    print(f"Directory structure:")
    print(f"  {loader.shard_base_dir}/")
    print(f"    ├── metadata.pkl")
    print(f"    ├── train/")
    print(f"    │   └── shard_0000.pkl, shard_0001.pkl, ...")
    print(f"    ├── val/")
    print(f"    │   └── shard_0000.pkl, shard_0001.pkl, ...")
    print(f"    └── test/")
    print(f"        └── shard_0000.pkl, shard_0001.pkl, ...")
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process MIMIC-CXR dataset using extracted vocabulary from metadata",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Input data parameters
    parser.add_argument("--metadata_csv", type=str, 
                        default="/home/abedin/Developments/mimic_cxr-raw-data/mimic-cxr/organized_data/metadata/processed_metadata_hybrid.csv",
                        help="Path to metadata.csv file")
    parser.add_argument("--reports_dir", type=str, 
                        default="/home/abedin/Developments/mimic_cxr-raw-data/mimic-cxr/organized_data/reports",
                        help="Directory containing report text files")
    parser.add_argument("--images_dir", type=str, 
                        default="/home/abedin/Developments/mimic_cxr-raw-data/mimic-cxr/organized_data/images",
                        help="Directory containing image files")
    
    # Vocabulary parameters (using extracted vocabulary)
    parser.add_argument("--vocab_path", type=str, 
                        default="/home/abedin/Developments/chest_x_ray_data_processing/extracted_vocabulary_metadata/word_index_from_metadata.json",
                        help="Path to extracted vocabulary JSON file")
    parser.add_argument("--index_word_path", type=str, 
                        default="/home/abedin/Developments/chest_x_ray_data_processing/extracted_vocabulary_metadata/index_word_from_metadata.json",
                        help="Path to extracted index_word JSON file")
    
    # Output parameters
    parser.add_argument("--output_dir", type=str, default="mimic_shards_with_extracted_vocab",
                        help="Output directory for processed shards")
    parser.add_argument("--max_studies", type=int, default=None,
                        help="Maximum number of studies to include (None for all)")
    
    # Processing parameters
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for data loading")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    # Data format parameters
    parser.add_argument("--image_size", type=int, default=224,
                        help="Size to resize images to (square)")
    parser.add_argument("--max_seq_length", type=int, default=128,
                        help="Maximum length of text sequences")
    parser.add_argument("--shard_size", type=int, default=100,
                        help="Number of samples per shard")
    
    # Control parameters
    parser.add_argument("--skip_processing", action="store_true",
                        help="Skip processing if data shards already exist")
    parser.add_argument("--test_loading", action="store_true",
                        help="Test data loading after processing")
    
    args = parser.parse_args()
    
    # Validate input files exist
    if not args.skip_processing:
        if not os.path.exists(args.metadata_csv):
            print(f"✗ Error: Metadata CSV file not found: {args.metadata_csv}")
            sys.exit(1)
        if not os.path.exists(args.reports_dir):
            print(f"✗ Error: Reports directory not found: {args.reports_dir}")
            sys.exit(1)
        if not os.path.exists(args.images_dir):
            print(f"✗ Error: Images directory not found: {args.images_dir}")
            sys.exit(1)
    
    # Validate vocabulary files
    if not os.path.exists(args.vocab_path):
        print(f"✗ Error: Vocabulary file not found: {args.vocab_path}")
        sys.exit(1)
    if not os.path.exists(args.index_word_path):
        print(f"✗ Error: Index word file not found: {args.index_word_path}")
        sys.exit(1)
    
    # Run the main processing
    success = main(args)
    sys.exit(0 if success else 1) 