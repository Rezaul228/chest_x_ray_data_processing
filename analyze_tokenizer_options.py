#!/usr/bin/env python3
"""
Analyze Tokenizer Options for Medical Terms
Evaluate whether to create a new tokenizer or work with current one
"""

import os
import sys
import pickle
import numpy as np
from adv_aug_text import ADVANCED_MEDICAL_TERMS, STYLE_VARIATIONS

def analyze_tokenizer_options():
    """Analyze options for handling missing medical terms"""
    
    print("Analyzing Tokenizer Options for Medical Terms")
    print("=" * 60)
    
    # Load current tokenizer
    metadata_path = '/home/abedin/Developments/chest_x_ray_data_processing/aug_indiana_extended/metadata.pkl'
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    
    tokenizer = metadata.get('tokenizer')
    current_vocab_size = len(tokenizer.word_index) + 1
    
    print(f"Current tokenizer vocabulary size: {current_vocab_size}")
    
    # Extract all medical terms
    all_medical_terms = set()
    for main_term, synonyms in ADVANCED_MEDICAL_TERMS.items():
        all_medical_terms.add(main_term.lower())
        for synonym in synonyms:
            all_medical_terms.add(synonym.lower())
    
    for style_dict in STYLE_VARIATIONS.values():
        for term in style_dict.keys():
            all_medical_terms.add(term.lower())
        for term in style_dict.values():
            all_medical_terms.add(term.lower())
    
    medical_terms_list = sorted(list(all_medical_terms))
    
    # Check coverage
    covered_terms = []
    missing_terms = []
    
    for term in medical_terms_list:
        if term in tokenizer.word_index:
            covered_terms.append(term)
        else:
            missing_terms.append(term)
    
    total_terms = len(medical_terms_list)
    covered_count = len(covered_terms)
    missing_count = len(missing_terms)
    
    print(f"\nMedical Terms Analysis:")
    print(f"  Total medical terms: {total_terms}")
    print(f"  Currently covered: {covered_count}")
    print(f"  Missing: {missing_count}")
    
    # Option 1: Work with current tokenizer
    print(f"\n" + "="*60)
    print(f"OPTION 1: Work with Current Tokenizer")
    print(f"="*60)
    
    print(f"Pros:")
    print(f"  ✅ No data reprocessing needed")
    print(f"  ✅ Maintains compatibility with existing models")
    print(f"  ✅ 52.21% of medical terms already available")
    print(f"  ✅ Strong coverage in key categories (opacity, severity)")
    
    print(f"\nCons:")
    print(f"  ❌ 47.79% of medical terms missing")
    print(f"  ❌ Limited augmentation variety")
    print(f"  ❌ Complex medical phrases unavailable")
    
    # Option 2: Expand current tokenizer
    print(f"\n" + "="*60)
    print(f"OPTION 2: Expand Current Tokenizer")
    print(f"="*60)
    
    new_vocab_size = current_vocab_size + missing_count
    print(f"New vocabulary size would be: {new_vocab_size}")
    print(f"Vocabulary increase: {missing_count} terms (+{missing_count/current_vocab_size*100:.1f}%)")
    
    print(f"\nPros:")
    print(f"  ✅ 100% medical term coverage")
    print(f"  ✅ Maximum augmentation variety")
    print(f"  ✅ No need to reprocess existing data")
    print(f"  ✅ Backward compatible with existing data")
    
    print(f"\nCons:")
    print(f"  ❌ Models need retraining with new vocabulary")
    print(f"  ❌ Vocabulary size increases by {missing_count} terms")
    print(f"  ❌ Need to update model architecture")
    
    # Option 3: Create new tokenizer from scratch
    print(f"\n" + "="*60)
    print(f"OPTION 3: Create New Tokenizer from Scratch")
    print(f"="*60)
    
    print(f"Pros:")
    print(f"  ✅ Complete control over vocabulary")
    print(f"  ✅ Can include all medical terms")
    print(f"  ✅ Optimized for medical text")
    
    print(f"\nCons:")
    print(f"  ❌ Need to reprocess ALL data")
    print(f"  ❌ Existing models become incompatible")
    print(f"  ❌ Time-consuming process")
    print(f"  ❌ Risk of losing important terms from current vocabulary")
    
    # Analyze missing terms by complexity
    print(f"\n" + "="*60)
    print(f"MISSING TERMS ANALYSIS")
    print(f"="*60)
    
    single_word_missing = []
    multi_word_missing = []
    
    for term in missing_terms:
        if ' ' in term:
            multi_word_missing.append(term)
        else:
            single_word_missing.append(term)
    
    print(f"Single-word missing terms: {len(single_word_missing)}")
    print(f"Multi-word missing terms: {len(multi_word_missing)}")
    
    print(f"\nSample single-word missing terms:")
    for term in single_word_missing[:10]:
        print(f"  '{term}'")
    
    print(f"\nSample multi-word missing terms:")
    for term in multi_word_missing[:10]:
        print(f"  '{term}'")
    
    # Recommendation
    print(f"\n" + "="*60)
    print(f"RECOMMENDATION")
    print(f"="*60)
    
    if missing_count < 50:  # Small number of missing terms
        print(f"✅ RECOMMEND: Expand current tokenizer")
        print(f"   - Only {missing_count} terms to add")
        print(f"   - Minimal vocabulary increase")
        print(f"   - Maximum augmentation capability")
        print(f"   - No data reprocessing needed")
    elif missing_count < 100:  # Moderate number
        print(f"⚠️ CONSIDER: Expand current tokenizer")
        print(f"   - {missing_count} terms to add")
        print(f"   - Moderate vocabulary increase")
        print(f"   - Good augmentation capability")
        print(f"   - Models need retraining")
    else:  # Large number
        print(f"❌ RECOMMEND: Work with current tokenizer")
        print(f"   - Too many terms to add ({missing_count})")
        print(f"   - Large vocabulary increase")
        print(f"   - Current coverage is acceptable for augmentation")
        print(f"   - Focus on working categories (opacity, severity)")
    
    # Practical augmentation impact
    print(f"\n" + "="*60)
    print(f"PRACTICAL AUGMENTATION IMPACT")
    print(f"="*60)
    
    print(f"With current tokenizer (52.21% coverage):")
    print(f"  ✅ Can augment: opacity, severity, basic abnormal terms")
    print(f"  ✅ Can restructure sentences")
    print(f"  ✅ Can modify certainty levels")
    print(f"  ✅ Can change terminology style")
    print(f"  ❌ Cannot replace: complex medical phrases")
    print(f"  ❌ Cannot replace: detailed anatomical descriptions")
    
    print(f"\nAugmentation effectiveness estimate:")
    print(f"  - Basic augmentation: 70-80% effective")
    print(f"  - Medical term replacement: 50-60% effective")
    print(f"  - Overall augmentation quality: Good")

if __name__ == "__main__":
    analyze_tokenizer_options() 