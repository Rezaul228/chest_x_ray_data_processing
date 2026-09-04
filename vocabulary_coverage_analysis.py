#!/usr/bin/env python3
"""
Vocabulary Coverage Analysis

This script analyzes how well the extracted vocabulary from metadata
covers the actual text reports in the original MIMIC-CXR dataset.
"""

import os
import sys
import json
import glob
import re
import numpy as np
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')


def load_vocabulary_from_metadata(vocab_path: str) -> Dict[str, int]:
    """Load vocabulary from the metadata extraction."""
    print(f"📚 Loading vocabulary from: {vocab_path}")
    
    try:
        with open(vocab_path, 'r') as f:
            word_index = json.load(f)
        
        print(f"✅ Loaded vocabulary with {len(word_index)} tokens")
        
        return word_index
    except Exception as e:
        print(f"❌ Error loading vocabulary: {e}")
        return {}


def load_original_reports(reports_dir: str, max_files: int = None) -> List[str]:
    """Load text reports from the original MIMIC-CXR dataset."""
    print(f"📄 Loading reports from: {reports_dir}")
    
    # Find all text files
    file_pattern = os.path.join(reports_dir, "**", "*.txt")
    files = glob.glob(file_pattern, recursive=True)
    
    if max_files:
        files = files[:max_files]
        print(f"📊 Analyzing first {max_files} files (out of {len(glob.glob(file_pattern, recursive=True))} total)")
    else:
        print(f"📊 Found {len(files)} files")
    
    reports = []
    for file_path in tqdm(files, desc="Loading reports"):
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read().strip()
                if content:  # Only add non-empty reports
                    reports.append(content)
        except Exception as e:
            print(f"⚠️  Error reading {file_path}: {e}")
            continue
    
    print(f"✅ Loaded {len(reports)} valid reports")
    return reports


def preprocess_text(text: str) -> List[str]:
    """Preprocess text to extract tokens."""
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters but keep hyphens and apostrophes
    text = re.sub(r'[^\w\s\-\.]', ' ', text)
    
    # Split into tokens
    tokens = text.split()
    
    # Filter out empty tokens and very short tokens (likely artifacts)
    tokens = [token for token in tokens if len(token) > 1]
    
    return tokens


def analyze_vocabulary_coverage(vocabulary: Dict[str, int], reports: List[str]) -> Dict:
    """Analyze how well the vocabulary covers the reports."""
    print("🔍 Analyzing vocabulary coverage...")
    
    # Collect all tokens from reports
    all_tokens = []
    report_tokens = []
    
    for report in tqdm(reports, desc="Processing reports"):
        tokens = preprocess_text(report)
        all_tokens.extend(tokens)
        report_tokens.append(tokens)
    
    # Count token frequencies
    token_counter = Counter(all_tokens)
    total_tokens = len(all_tokens)
    unique_tokens = len(token_counter)
    
    print(f"📊 Total tokens in reports: {total_tokens:,}")
    print(f"📊 Unique tokens in reports: {unique_tokens:,}")
    
    # Analyze vocabulary coverage
    vocab_set = set(vocabulary.keys())
    report_token_set = set(token_counter.keys())
    
    # Calculate coverage metrics
    covered_tokens = vocab_set.intersection(report_token_set)
    uncovered_tokens = report_token_set - vocab_set
    
    coverage_percentage = len(covered_tokens) / len(report_token_set) * 100
    token_coverage_percentage = sum(token_counter[token] for token in covered_tokens) / total_tokens * 100
    
    # Analyze by frequency
    high_freq_tokens = {token: count for token, count in token_counter.items() if count >= 100}
    medium_freq_tokens = {token: count for token, count in token_counter.items() if 10 <= count < 100}
    low_freq_tokens = {token: count for token, count in token_counter.items() if count < 10}
    
    # Coverage by frequency
    high_freq_coverage = len(set(high_freq_tokens.keys()) & vocab_set) / len(high_freq_tokens) * 100
    medium_freq_coverage = len(set(medium_freq_tokens.keys()) & vocab_set) / len(medium_freq_tokens) * 100
    low_freq_coverage = len(set(low_freq_tokens.keys()) & vocab_set) / len(low_freq_tokens) * 100
    
    # Most common uncovered tokens
    uncovered_counter = Counter({token: token_counter[token] for token in uncovered_tokens})
    most_common_uncovered = uncovered_counter.most_common(50)
    
    # Most common covered tokens
    covered_counter = Counter({token: token_counter[token] for token in covered_tokens})
    most_common_covered = covered_counter.most_common(50)
    
    # Analyze report-level coverage
    report_coverage_stats = []
    for tokens in report_tokens:
        if tokens:
            covered_count = len([t for t in tokens if t in vocab_set])
            coverage_ratio = covered_count / len(tokens)
            report_coverage_stats.append(coverage_ratio)
    
    avg_report_coverage = np.mean(report_coverage_stats) * 100
    median_report_coverage = np.median(report_coverage_stats) * 100
    
    # Special tokens analysis
    special_tokens = ['<pad>', '<unk>', '<start>', '<end>']
    special_token_usage = {token: 0 for token in special_tokens}
    
    # Medical terminology analysis
    medical_terms = [
        'pneumonia', 'effusion', 'pneumothorax', 'atelectasis', 'consolidation',
        'cardiomegaly', 'edema', 'fracture', 'pneumonia', 'pleural', 'lung',
        'heart', 'chest', 'radiograph', 'x-ray', 'ct', 'mri', 'ultrasound'
    ]
    
    medical_coverage = {}
    for term in medical_terms:
        if term in token_counter:
            medical_coverage[term] = {
                'frequency': token_counter[term],
                'covered': term in vocab_set
            }
    
    return {
        'summary': {
            'total_tokens': total_tokens,
            'unique_tokens': unique_tokens,
            'vocabulary_size': len(vocabulary),
            'covered_tokens': len(covered_tokens),
            'uncovered_tokens': len(uncovered_tokens),
            'coverage_percentage': coverage_percentage,
            'token_coverage_percentage': token_coverage_percentage,
            'avg_report_coverage': avg_report_coverage,
            'median_report_coverage': median_report_coverage
        },
        'frequency_analysis': {
            'high_freq_tokens': len(high_freq_tokens),
            'medium_freq_tokens': len(medium_freq_tokens),
            'low_freq_tokens': len(low_freq_tokens),
            'high_freq_coverage': high_freq_coverage,
            'medium_freq_coverage': medium_freq_coverage,
            'low_freq_coverage': low_freq_coverage
        },
        'most_common_uncovered': most_common_uncovered,
        'most_common_covered': most_common_covered,
        'medical_coverage': medical_coverage,
        'report_coverage_distribution': {
            'mean': avg_report_coverage,
            'median': median_report_coverage,
            'std': np.std(report_coverage_stats) * 100,
            'min': np.min(report_coverage_stats) * 100,
            'max': np.max(report_coverage_stats) * 100
        }
    }


def save_analysis_results(results: Dict, output_path: str):
    """Save analysis results to file."""
    print(f"💾 Saving results to: {output_path}")
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Results saved to: {output_path}")


def print_analysis_summary(results: Dict):
    """Print a comprehensive summary of the analysis."""
    
    print("\n" + "="*80)
    print("📊 VOCABULARY COVERAGE ANALYSIS SUMMARY")
    print("="*80)
    
    summary = results['summary']
    freq_analysis = results['frequency_analysis']
    
    print(f"\n📈 OVERALL COVERAGE STATISTICS:")
    print(f"  Total Tokens in Reports: {summary['total_tokens']:,}")
    print(f"  Unique Tokens in Reports: {summary['unique_tokens']:,}")
    print(f"  Vocabulary Size: {summary['vocabulary_size']:,}")
    print(f"  Covered Tokens: {summary['covered_tokens']:,}")
    print(f"  Uncovered Tokens: {summary['uncovered_tokens']:,}")
    print(f"  Vocabulary Coverage: {summary['coverage_percentage']:.2f}%")
    print(f"  Token Coverage: {summary['token_coverage_percentage']:.2f}%")
    print(f"  Average Report Coverage: {summary['avg_report_coverage']:.2f}%")
    print(f"  Median Report Coverage: {summary['median_report_coverage']:.2f}%")
    
    print(f"\n📊 FREQUENCY-BASED COVERAGE:")
    print(f"  High Frequency Tokens (≥100): {freq_analysis['high_freq_tokens']:,} ({freq_analysis['high_freq_coverage']:.2f}% covered)")
    print(f"  Medium Frequency Tokens (10-99): {freq_analysis['medium_freq_tokens']:,} ({freq_analysis['medium_freq_coverage']:.2f}% covered)")
    print(f"  Low Frequency Tokens (<10): {freq_analysis['low_freq_tokens']:,} ({freq_analysis['low_freq_coverage']:.2f}% covered)")
    
    print(f"\n📏 REPORT COVERAGE DISTRIBUTION:")
    dist = results['report_coverage_distribution']
    print(f"  Mean Coverage: {dist['mean']:.2f}%")
    print(f"  Median Coverage: {dist['median']:.2f}%")
    print(f"  Standard Deviation: {dist['std']:.2f}%")
    print(f"  Min Coverage: {dist['min']:.2f}%")
    print(f"  Max Coverage: {dist['max']:.2f}%")
    
    print(f"\n🔝 MOST COMMON COVERED TOKENS:")
    for i, (token, count) in enumerate(results['most_common_covered'][:20], 1):
        print(f"  {i:2d}. '{token}': {count:,}")
    
    print(f"\n❌ MOST COMMON UNCOVERED TOKENS:")
    for i, (token, count) in enumerate(results['most_common_uncovered'][:20], 1):
        print(f"  {i:2d}. '{token}': {count:,}")
    
    print(f"\n🏥 MEDICAL TERMINOLOGY COVERAGE:")
    for term, info in results['medical_coverage'].items():
        status = "✅" if info['covered'] else "❌"
        print(f"  {status} '{term}': {info['frequency']:,} occurrences")
    
    # Calculate efficiency metrics
    vocab_efficiency = summary['token_coverage_percentage'] / (summary['vocabulary_size'] / summary['unique_tokens'])
    print(f"\n📊 VOCABULARY EFFICIENCY METRICS:")
    print(f"  Vocabulary Efficiency: {vocab_efficiency:.2f}")
    print(f"  Coverage per Vocabulary Token: {summary['coverage_percentage'] / summary['vocabulary_size']:.4f}%")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    if summary['coverage_percentage'] >= 90:
        print(f"  ✅ Excellent vocabulary coverage! The vocabulary covers {summary['coverage_percentage']:.1f}% of unique tokens.")
    elif summary['coverage_percentage'] >= 80:
        print(f"  ⚠️  Good vocabulary coverage. Consider adding {summary['uncovered_tokens']} missing tokens for better coverage.")
    elif summary['coverage_percentage'] >= 70:
        print(f"  ⚠️  Moderate vocabulary coverage. Significant improvement needed.")
    else:
        print(f"  ❌ Poor vocabulary coverage. Major vocabulary expansion required.")
    
    if freq_analysis['high_freq_coverage'] < 95:
        print(f"  ⚠️  High-frequency token coverage is only {freq_analysis['high_freq_coverage']:.1f}%. Consider adding missing high-frequency terms.")
    
    if summary['avg_report_coverage'] < 90:
        print(f"  ⚠️  Average report coverage is {summary['avg_report_coverage']:.1f}%. Some reports may have poor tokenization.")


def main():
    """Main function to analyze vocabulary coverage."""
    
    # Paths
    vocab_path = "extracted_vocabulary_metadata/word_index_from_metadata.json"
    reports_dir = "/home/abedin/Developments/chest_x_ray_data_processing/mimic-cxr_raw_original/mimic-cxr-reports/files"
    output_path = "vocabulary_coverage_analysis.json"
    
    # Load vocabulary
    vocabulary = load_vocabulary_from_metadata(vocab_path)
    if not vocabulary:
        print("❌ Failed to load vocabulary")
        return
    
    # Load reports (limit to first 1000 for faster analysis)
    reports = load_original_reports(reports_dir, max_files=1000)
    if not reports:
        print("❌ Failed to load reports")
        return
    
    # Analyze coverage
    results = analyze_vocabulary_coverage(vocabulary, reports)
    
    # Save results
    save_analysis_results(results, output_path)
    
    # Print summary
    print_analysis_summary(results)
    
    print(f"\n✅ Vocabulary coverage analysis complete!")
    print(f"📁 Results saved to: {output_path}")


if __name__ == "__main__":
    main() 