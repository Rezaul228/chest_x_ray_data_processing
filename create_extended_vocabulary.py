#!/usr/bin/env python3
"""
Create Extended Vocabulary for Data Augmentation
This script extends the existing vocabulary with medical terms from adv_aug_text.py
and creates new JSON files for use with segregated_augmentation.py
"""

import json
import os
import sys
from adv_aug_text import ADVANCED_MEDICAL_TERMS, STYLE_VARIATIONS

def load_existing_vocabulary(vocab_file, index_word_file):
    """Load existing vocabulary from JSON files"""
    print(f"Loading existing vocabulary from: {vocab_file}")
    
    with open(vocab_file, 'r') as f:
        word_index = json.load(f)
    
    with open(index_word_file, 'r') as f:
        index_word = json.load(f)
    
    print(f"Current vocabulary size: {len(word_index)}")
    return word_index, index_word

def extract_medical_terms():
    """Extract all medical terms from adv_aug_text.py"""
    print("Extracting medical terms from adv_aug_text.py...")
    
    all_medical_terms = set()
    
    # Extract from ADVANCED_MEDICAL_TERMS
    for main_term, synonyms in ADVANCED_MEDICAL_TERMS.items():
        all_medical_terms.add(main_term.lower())
        for synonym in synonyms:
            all_medical_terms.add(synonym.lower())
    
    # Extract from STYLE_VARIATIONS
    for style_dict in STYLE_VARIATIONS.values():
        for term in style_dict.keys():
            all_medical_terms.add(term.lower())
        for term in style_dict.values():
            all_medical_terms.add(term.lower())
    
    # Convert to list and sort
    medical_terms_list = sorted(list(all_medical_terms))
    print(f"Total unique medical terms: {len(medical_terms_list)}")
    
    return medical_terms_list

def create_extended_vocabulary(word_index, index_word, medical_terms):
    """Create extended vocabulary with medical terms"""
    print("Creating extended vocabulary...")
    
    # Find the highest token ID
    max_token_id = max(int(k) for k in index_word.keys())
    print(f"Current max token ID: {max_token_id}")
    
    # Create new word_index and index_word
    new_word_index = word_index.copy()
    new_index_word = index_word.copy()
    
    # Add medical terms that are not already in vocabulary
    added_terms = []
    skipped_terms = []
    
    for term in medical_terms:
        if term not in new_word_index:
            max_token_id += 1
            new_word_index[term] = max_token_id
            new_index_word[str(max_token_id)] = term
            added_terms.append(term)
        else:
            skipped_terms.append(term)
    
    print(f"Added {len(added_terms)} new terms")
    print(f"Skipped {len(skipped_terms)} existing terms")
    print(f"New vocabulary size: {len(new_word_index)}")
    
    return new_word_index, new_index_word, added_terms, skipped_terms

def save_extended_vocabulary(word_index, index_word, output_prefix):
    """Save extended vocabulary to JSON files"""
    vocab_file = f"{output_prefix}_vocab.json"
    index_word_file = f"{output_prefix}_index_word.json"
    
    print(f"Saving extended vocabulary to:")
    print(f"  {vocab_file}")
    print(f"  {index_word_file}")
    
    with open(vocab_file, 'w') as f:
        json.dump(word_index, f, indent=2)
    
    with open(index_word_file, 'w') as f:
        json.dump(index_word, f, indent=2)
    
    print("✅ Extended vocabulary saved successfully!")

def analyze_coverage(medical_terms, word_index):
    """Analyze coverage of medical terms in vocabulary"""
    print("\n" + "="*60)
    print("MEDICAL TERMS COVERAGE ANALYSIS")
    print("="*60)
    
    covered_terms = []
    missing_terms = []
    
    for term in medical_terms:
        if term in word_index:
            covered_terms.append(term)
        else:
            missing_terms.append(term)
    
    total_terms = len(medical_terms)
    covered_count = len(covered_terms)
    missing_count = len(missing_terms)
    coverage_percentage = (covered_count / total_terms) * 100
    
    print(f"Total medical terms: {total_terms}")
    print(f"Covered terms: {covered_count}")
    print(f"Missing terms: {missing_count}")
    print(f"Coverage percentage: {coverage_percentage:.2f}%")
    
    print(f"\nSample covered terms:")
    for term in covered_terms[:10]:
        token_id = word_index[term]
        print(f"  '{term}' -> token {token_id}")
    
    print(f"\nSample missing terms:")
    for term in missing_terms[:10]:
        print(f"  '{term}' -> NOT FOUND")
    
    return covered_terms, missing_terms

def main():
    """Main function"""
    print("Creating Extended Vocabulary for Data Augmentation")
    print("=" * 60)
    
    # Input files
    vocab_file = "mimic_frontal_complete_vocab_vocab.json"
    index_word_file = "mimic_frontal_complete_vocab_index_word.json"
    
    # Output prefix
    output_prefix = "mimic_frontal_complete_vocab_extended"
    
    # Check if input files exist
    if not os.path.exists(vocab_file):
        print(f"Error: Vocabulary file not found: {vocab_file}")
        return
    
    if not os.path.exists(index_word_file):
        print(f"Error: Index word file not found: {index_word_file}")
        return
    
    # Load existing vocabulary
    word_index, index_word = load_existing_vocabulary(vocab_file, index_word_file)
    
    # Extract medical terms
    medical_terms = extract_medical_terms()
    
    # Analyze current coverage
    print("\n" + "="*60)
    print("CURRENT COVERAGE ANALYSIS")
    print("="*60)
    covered_terms, missing_terms = analyze_coverage(medical_terms, word_index)
    
    # Create extended vocabulary
    print("\n" + "="*60)
    print("CREATING EXTENDED VOCABULARY")
    print("="*60)
    new_word_index, new_index_word, added_terms, skipped_terms = create_extended_vocabulary(
        word_index, index_word, medical_terms
    )
    
    # Save extended vocabulary
    print("\n" + "="*60)
    print("SAVING EXTENDED VOCABULARY")
    print("="*60)
    save_extended_vocabulary(new_word_index, new_index_word, output_prefix)
    
    # Final analysis
    print("\n" + "="*60)
    print("FINAL COVERAGE ANALYSIS")
    print("="*60)
    final_covered, final_missing = analyze_coverage(medical_terms, new_word_index)
    
    print(f"\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Original vocabulary size: {len(word_index)}")
    print(f"Extended vocabulary size: {len(new_word_index)}")
    print(f"New terms added: {len(added_terms)}")
    print(f"Medical terms coverage: {len(final_covered)}/{len(medical_terms)} ({len(final_covered)/len(medical_terms)*100:.2f}%)")
    print(f"\nNew vocabulary files created:")
    print(f"  {output_prefix}_vocab.json")
    print(f"  {output_prefix}_index_word.json")
    print(f"\nYou can now use these files with segregated_augmentation.py")

if __name__ == "__main__":
    main() 