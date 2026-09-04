#!/usr/bin/env python3

import pickle
import os

def inspect_metadata():
    """Inspect the metadata to show vocabulary details and token length"""
    
    # Load the root-level metadata
    metadata_path = 'all_processed_data/step3_augmented_data/metadata.pkl'
    if not os.path.exists(metadata_path):
        print(f"Error: Metadata not found at {metadata_path}")
        return
    
    print("Loading metadata...")
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    
    print("\n" + "="*60)
    print("METADATA INSPECTION")
    print("="*60)
    
    # Basic info
    print(f"Vocabulary size: {metadata.get('vocab_size', 'Not found')}")
    print(f"Tokenizer type: {type(metadata.get('tokenizer', None)).__name__ if metadata.get('tokenizer') else 'None'}")
    
    # Get tokenizer
    tokenizer = metadata.get('tokenizer')
    if tokenizer is None:
        print("No tokenizer found in metadata!")
        return
    
    # Check if it's an EnhancedTokenizer
    if hasattr(tokenizer, 'vocab'):
        vocab = tokenizer.vocab
        print(f"\nVocabulary type: {type(vocab).__name__}")
        print(f"Number of vocabulary items: {len(vocab)}")
        
        # Show some sample words from vocabulary
        print(f"\nSample words from vocabulary (first 20):")
        sample_words = list(vocab.keys())[:20]
        for i, word in enumerate(sample_words, 1):
            print(f"  {i:2d}. '{word}' -> {vocab[word]}")
        
        # Check for special tokens
        special_tokens = ['<PAD>', '<UNK>', '<START>', '<END>', '<SEP>']
        print(f"\nSpecial tokens:")
        for token in special_tokens:
            if token in vocab:
                print(f"  '{token}' -> {vocab[token]}")
            else:
                print(f"  '{token}' -> Not found")
        
        # Check for medical terms (sample)
        medical_terms = ['pneumonia', 'effusion', 'cardiomegaly', 'edema', 'consolidation']
        print(f"\nMedical terms check:")
        for term in medical_terms:
            if term in vocab:
                print(f"  '{term}' -> {vocab[term]}")
            else:
                print(f"  '{term}' -> Not found")
    
    # Check sequence length
    if hasattr(tokenizer, 'max_length'):
        print(f"\nMaximum sequence length: {tokenizer.max_length}")
    else:
        print(f"\nMaximum sequence length: Not specified in tokenizer")
    
    # Check if there are any other relevant attributes
    print(f"\nTokenizer attributes:")
    for attr in dir(tokenizer):
        if not attr.startswith('_') and not callable(getattr(tokenizer, attr)):
            try:
                value = getattr(tokenizer, attr)
                if not hasattr(value, '__len__') or len(value) < 100:  # Avoid printing large objects
                    print(f"  {attr}: {value}")
            except:
                pass
    
    # Show available methods
    print(f"\nTokenizer methods:")
    methods = [attr for attr in dir(tokenizer) if not attr.startswith('_') and callable(getattr(tokenizer, attr))]
    for method in methods:
        print(f"  {method}")
    
    # Test tokenization on a sample text using the correct method
    print(f"\n" + "="*60)
    print("TOKENIZATION TEST")
    print("="*60)
    
    test_texts = [
        "The lungs are clear",
        "There is evidence of pneumonia in the right lower lobe",
        "Cardiomegaly is present with pulmonary edema"
    ]
    
    for text in test_texts:
        try:
            # Try different possible method names
            if hasattr(tokenizer, 'tokenize'):
                tokens = tokenizer.tokenize(text)
                print(f"\nOriginal: '{text}'")
                print(f"Tokens: {tokens}")
                print(f"Token length: {len(tokens)}")
            elif hasattr(tokenizer, 'text_to_sequence'):
                tokens = tokenizer.text_to_sequence(text)
                print(f"\nOriginal: '{text}'")
                print(f"Tokens: {tokens}")
                print(f"Token length: {len(tokens)}")
            elif hasattr(tokenizer, 'convert_tokens_to_ids'):
                # First tokenize, then convert to IDs
                if hasattr(tokenizer, 'tokenize'):
                    token_list = tokenizer.tokenize(text)
                    tokens = tokenizer.convert_tokens_to_ids(token_list)
                    print(f"\nOriginal: '{text}'")
                    print(f"Token list: {token_list}")
                    print(f"Token IDs: {tokens}")
                    print(f"Token length: {len(tokens)}")
                else:
                    print(f"\nOriginal: '{text}'")
                    print(f"Could not tokenize - no suitable method found")
            else:
                print(f"\nOriginal: '{text}'")
                print(f"Could not tokenize - no suitable method found")
        except Exception as e:
            print(f"\nError tokenizing '{text}': {e}")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    inspect_metadata() 