#!/usr/bin/env python3
"""
Analyze Medical Terms Coverage in Tokenizer
Check how many medical terms from adv_aug_text.py are covered by the current tokenizer
"""

import os
import sys
import pickle
import numpy as np
from adv_aug_text import ADVANCED_MEDICAL_TERMS, STYLE_VARIATIONS

def analyze_medical_terms_coverage():
    """Analyze coverage of medical terms in the current tokenizer"""
    
    print("Analyzing Medical Terms Coverage in Tokenizer")
    print("=" * 60)
    
    # Load the tokenizer from Indiana shards metadata
    metadata_path = 'all_processed_data/indiana_shards/metadata.pkl'
    if not os.path.exists(metadata_path):
        print(f"Error: Metadata file not found: {metadata_path}")
        return
    
    print(f"Loading tokenizer from: {metadata_path}")
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    
    tokenizer = metadata.get('tokenizer')
    if not tokenizer:
        print("Error: No tokenizer found in metadata")
        return
    
    print(f"Tokenizer loaded with {len(tokenizer.word_index)} words")
    print(f"Vocabulary size: {len(tokenizer.word_index) + 1}")
    
    # Extract all medical terms from ADVANCED_MEDICAL_TERMS
    print(f"\nExtracting medical terms from adv_aug_text.py...")
    all_medical_terms = set()
    
    # Add main terms and synonyms
    for main_term, synonyms in ADVANCED_MEDICAL_TERMS.items():
        all_medical_terms.add(main_term.lower())
        for synonym in synonyms:
            all_medical_terms.add(synonym.lower())
    
    # Add style variations
    for style_dict in STYLE_VARIATIONS.values():
        for term in style_dict.keys():
            all_medical_terms.add(term.lower())
        for term in style_dict.values():
            all_medical_terms.add(term.lower())
    
    # Convert to list and sort
    medical_terms_list = sorted(list(all_medical_terms))
    print(f"Total unique medical terms: {len(medical_terms_list)}")
    
    # Check coverage in tokenizer
    print(f"\nChecking coverage in tokenizer...")
    covered_terms = []
    missing_terms = []
    
    for term in medical_terms_list:
        if term in tokenizer.word_index:
            covered_terms.append(term)
        else:
            missing_terms.append(term)
    
    # Calculate statistics
    total_terms = len(medical_terms_list)
    covered_count = len(covered_terms)
    missing_count = len(missing_terms)
    coverage_percentage = (covered_count / total_terms) * 100
    
    print(f"\nCoverage Analysis:")
    print(f"  Total medical terms: {total_terms}")
    print(f"  Covered terms: {covered_count}")
    print(f"  Missing terms: {missing_count}")
    print(f"  Coverage percentage: {coverage_percentage:.2f}%")
    
    # Show some examples
    print(f"\nSample Covered Terms (first 10):")
    for term in covered_terms[:10]:
        token_id = tokenizer.word_index[term]
        print(f"  '{term}' -> token {token_id}")
    
    print(f"\nSample Missing Terms (first 10):")
    for term in missing_terms[:10]:
        print(f"  '{term}' -> NOT FOUND")
    
    # Analyze by category
    print(f"\nDetailed Analysis by Category:")
    
    categories = {
        'opacity': ['opacity', 'consolidation', 'infiltrate', 'density', 'haziness', 'shadowing', 'opacification'],
        'cardiomegaly': ['cardiomegaly', 'enlarged heart', 'cardiac enlargement', 'heart enlargement', 'enlarged cardiac silhouette', 'prominent cardiac silhouette'],
        'pneumonia': ['pneumonia', 'pulmonary infection', 'lung infection', 'pneumonitis', 'infectious process', 'inflammatory process', 'parenchymal infection'],
        'effusion': ['effusion', 'fluid collection', 'pleural fluid', 'fluid accumulation', 'pleural effusion', 'fluid in pleural space'],
        'atelectasis': ['atelectasis', 'lung collapse', 'collapsed lung', 'lung volume loss', 'subsegmental atelectasis', 'compressive atelectasis'],
        'pneumothorax': ['pneumothorax', 'collapsed lung', 'air in pleural space', 'pleural air', 'air collection', 'pleural gas'],
        'normal': ['normal', 'unremarkable', 'no acute abnormality', 'within normal limits', 'no significant abnormality', 'no acute cardiopulmonary process', 'no active disease'],
        'abnormal': ['abnormal', 'pathologic', 'unusual', 'remarkable', 'notable finding', 'abnormality', 'pathology'],
        'severity': ['mild', 'slight', 'minimal', 'minor', 'subtle', 'trace', 'moderate', 'intermediate', 'medium', 'moderate-sized', 'moderately severe', 'severe', 'marked', 'pronounced', 'significant', 'extensive', 'profound', 'striking', 'large'],
        'location': ['bilateral', 'affecting both sides', 'on both sides', 'involving both lungs', 'in both lungs', 'both left and right', 'unilateral', 'affecting one side', 'on one side', 'involving one lung', 'in one lung', 'one-sided'],
        'certainty': ['likely', 'probable', 'suggestive of', 'consistent with', 'suspicious for', 'compatible with', 'concerning for', 'possible', 'may represent', 'cannot exclude', 'cannot rule out', 'potentially', 'possibly', 'could represent', 'may be due to'],
        'presence': ['no', 'absent', 'not seen', 'not identified', 'not detected', 'not present', 'not noted', 'not visualized', 'present', 'seen', 'identified', 'noted', 'visualized', 'detected', 'demonstrated', 'observed', 'appreciated'],
        'change': ['increased', 'elevated', 'enhanced', 'prominent', 'accentuated', 'pronounced', 'more pronounced', 'decreased', 'reduced', 'diminished', 'less prominent', 'subtle', 'limited', 'faint']
    }
    
    for category, terms in categories.items():
        covered_in_category = []
        missing_in_category = []
        
        for term in terms:
            if term in tokenizer.word_index:
                covered_in_category.append(term)
            else:
                missing_in_category.append(term)
        
        category_coverage = (len(covered_in_category) / len(terms)) * 100
        print(f"  {category.capitalize()}: {len(covered_in_category)}/{len(terms)} ({category_coverage:.1f}%)")
        
        if missing_in_category:
            print(f"    Missing: {', '.join(missing_in_category[:3])}{'...' if len(missing_in_category) > 3 else ''}")
    
    # Test with actual sample data
    print(f"\nTesting with actual sample data...")
    
    # Load a sample shard to see what terms are actually used
    sample_shard_path = 'all_processed_data/indiana_shards/train/shard_0000.pkl'
    if os.path.exists(sample_shard_path):
        with open(sample_shard_path, 'rb') as f:
            shard_data = pickle.load(f)
        
        # Decode a few captions to see actual text
        print(f"Sample decoded captions:")
        for i in range(min(3, len(shard_data['captions']))):
            caption = shard_data['captions'][i]
            decoded_text = decode_caption(caption, tokenizer)
            print(f"  Sample {i+1}: {decoded_text[:100]}{'...' if len(decoded_text) > 100 else ''}")
    
    print(f"\n" + "="*60)
    print(f"SUMMARY")
    print(f"="*60)
    print(f"Medical Terms Coverage: {coverage_percentage:.2f}%")
    print(f"Covered: {covered_count}/{total_terms}")
    print(f"Missing: {missing_count}/{total_terms}")
    
    if coverage_percentage >= 80:
        print(f"✅ GOOD COVERAGE: Most medical terms are available for augmentation")
    elif coverage_percentage >= 60:
        print(f"⚠️ MODERATE COVERAGE: Some medical terms are missing")
    else:
        print(f"❌ POOR COVERAGE: Many medical terms are missing")

def decode_caption(caption_seq, tokenizer):
    """Decode a caption from its tokenized sequence"""
    words = []
    for token_id in caption_seq:
        if token_id == 0:  # Skip padding
            continue
        
        # Get the word for this token ID
        word = tokenizer.index_word.get(token_id, '<UNK>')
        if word in ['<START>', '<END>', '<PAD>', '<UNK>']:
            continue
            
        words.append(word)
    
    return " ".join(words)

if __name__ == "__main__":
    analyze_medical_terms_coverage() 