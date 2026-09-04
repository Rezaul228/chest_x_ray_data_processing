#!/usr/bin/env python3
"""
Example usage of the Dataset Comparison Analyzer

This script demonstrates how to use the analyzer with custom tokenizers
and different configurations.
"""

import re
from dataset_comparison_analyzer import DatasetAnalyzer

def custom_medical_tokenizer(text):
    """
    Custom tokenizer for medical text that handles special medical terms.
    
    Args:
        text: Input text string
        
    Returns:
        List of tokens
    """
    if isinstance(text, (list, tuple)):
        # Handle pre-tokenized text
        return [str(token) for token in text if token != 0]
    
    # Convert to lowercase
    text = text.lower()
    
    # Handle common medical abbreviations and terms
    text = re.sub(r'\b(chest|x-ray|xray)\b', 'chest_xray', text)
    text = re.sub(r'\b(no|negative)\b', 'negative', text)
    text = re.sub(r'\b(yes|positive)\b', 'positive', text)
    
    # Split on whitespace and filter empty tokens
    tokens = [token.strip() for token in text.split() if token.strip()]
    
    return tokens

def ngram_tokenizer(text, n=2):
    """
    N-gram tokenizer that creates overlapping n-grams.
    
    Args:
        text: Input text string
        n: Size of n-grams (default: 2 for bigrams)
        
    Returns:
        List of n-gram tokens
    """
    if isinstance(text, (list, tuple)):
        # Handle pre-tokenized text
        tokens = [str(token) for token in text if token != 0]
    else:
        tokens = text.lower().split()
    
    if len(tokens) < n:
        return tokens
    
    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngram = ' '.join(tokens[i:i+n])
        ngrams.append(ngram)
    
    return ngrams

def main():
    """Example usage with different tokenizers."""
    
    # Dataset paths
    dataset1_path = "/home/abedin/Developments/chest_x_ray_data_processing/all_processed_data/mimic_shards"
    dataset2_path = "/home/abedin/Developments/chest_x_ray_data_processing/all_processed_data/mimic_shards_hufc4446-to128"
    
    print("🔬 Example 1: Using custom medical tokenizer")
    print("="*60)
    
    # Example 1: Custom medical tokenizer
    analyzer1 = DatasetAnalyzer(
        text_key="captions",
        tokenizer_func=custom_medical_tokenizer,
        file_pattern="*.pkl"
    )
    
    analyzer1.analyze_dataset(dataset1_path, "mimic_shards_medical")
    analyzer1.analyze_dataset(dataset2_path, "mimic_shards_hufc4446_medical")
    analyzer1.print_summary_table()
    
    print("\n🔬 Example 2: Using bigram tokenizer")
    print("="*60)
    
    # Example 2: Bigram tokenizer
    bigram_tokenizer = lambda text: ngram_tokenizer(text, n=2)
    
    analyzer2 = DatasetAnalyzer(
        text_key="captions",
        tokenizer_func=bigram_tokenizer,
        file_pattern="*.pkl"
    )
    
    analyzer2.analyze_dataset(dataset1_path, "mimic_shards_bigrams")
    analyzer2.analyze_dataset(dataset2_path, "mimic_shards_hufc4446_bigrams")
    analyzer2.print_summary_table()
    
    print("\n🔬 Example 3: Using trigram tokenizer")
    print("="*60)
    
    # Example 3: Trigram tokenizer
    trigram_tokenizer = lambda text: ngram_tokenizer(text, n=3)
    
    analyzer3 = DatasetAnalyzer(
        text_key="captions",
        tokenizer_func=trigram_tokenizer,
        file_pattern="*.pkl"
    )
    
    analyzer3.analyze_dataset(dataset1_path, "mimic_shards_trigrams")
    analyzer3.analyze_dataset(dataset2_path, "mimic_shards_hufc4446_trigrams")
    analyzer3.print_summary_table()
    
    # Save results
    analyzer1.save_results("medical_tokenizer_results.json")
    analyzer2.save_results("bigram_tokenizer_results.json")
    analyzer3.save_results("trigram_tokenizer_results.json")
    
    print("\n✅ All examples completed!")

if __name__ == "__main__":
    main() 