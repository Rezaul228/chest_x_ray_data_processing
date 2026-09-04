#!/usr/bin/env python3
"""
Check Dataset Status

This script checks the status of all datasets to verify the tokenizer fix worked.
"""

import os
import sys
import pickle
from typing import Dict, List

def check_dataset_status(dataset_path: str) -> Dict:
    """Check the status of a dataset."""
    print(f"🔍 Checking dataset: {os.path.basename(dataset_path)}")
    
    metadata_path = os.path.join(dataset_path, 'metadata.pkl')
    
    if not os.path.exists(metadata_path):
        return {
            'dataset': os.path.basename(dataset_path),
            'status': '❌ NO METADATA',
            'tokenizer_type': 'N/A',
            'vocab_size': 'N/A',
            'max_length': 'N/A'
        }
    
    try:
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        tokenizer = metadata.get('tokenizer')
        tokenizer_type = type(tokenizer).__name__ if tokenizer else 'None'
        
        # Check if tokenizer is working
        if tokenizer and hasattr(tokenizer, 'word_index'):
            status = '✅ WORKING'
            vocab_size = len(tokenizer.word_index)
            
            # Test tokenization
            try:
                test_text = "test"
                sequences = tokenizer.texts_to_sequences([test_text])
                max_length = len(sequences[0]) if sequences else 'N/A'
            except:
                status = '❌ BROKEN'
                vocab_size = 'N/A'
                max_length = 'N/A'
        else:
            status = '❌ BROKEN'
            vocab_size = 'N/A'
            max_length = 'N/A'
        
        return {
            'dataset': os.path.basename(dataset_path),
            'status': status,
            'tokenizer_type': tokenizer_type,
            'vocab_size': vocab_size,
            'max_length': max_length
        }
        
    except Exception as e:
        return {
            'dataset': os.path.basename(dataset_path),
            'status': f'❌ ERROR: {str(e)[:50]}',
            'tokenizer_type': 'N/A',
            'vocab_size': 'N/A',
            'max_length': 'N/A'
        }

def main():
    """Check all datasets."""
    datasets = [
        "/home/abedin/Developments/chest_x_ray_data_processing/all_processed_data/mimic_shards_4446_128",
        "/home/abedin/Developments/chest_x_ray_data_processing/all_processed_data/mimic_shards_hufc4446-to128",
        "/home/abedin/Developments/chest_x_ray_data_processing/all_processed_data/indiana_shards"
    ]
    
    results = []
    
    for dataset_path in datasets:
        if os.path.exists(dataset_path):
            result = check_dataset_status(dataset_path)
            results.append(result)
        else:
            results.append({
                'dataset': os.path.basename(dataset_path),
                'status': '❌ NOT FOUND',
                'tokenizer_type': 'N/A',
                'vocab_size': 'N/A',
                'max_length': 'N/A'
            })
    
    # Print results table
    print("\n" + "="*80)
    print("📊 DATASET STATUS CHECK")
    print("="*80)
    
    print(f"{'Dataset':<30} {'Status':<15} {'Tokenizer Type':<20} {'Vocab Size':<12} {'Max Length':<12}")
    print("-" * 95)
    
    for result in results:
        print(f"{result['dataset']:<30} {result['status']:<15} {result['tokenizer_type']:<20} {result['vocab_size']:<12} {result['max_length']:<12}")
    
    print("\n✅ Status check complete!")

if __name__ == "__main__":
    main() 