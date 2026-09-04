#!/usr/bin/env python3
"""
Vocabulary Size Impact Analysis

This script analyzes whether the differences in training difficulty factors
are caused by the vocabulary size differences between the datasets.
"""

import os
import sys
import json
import pickle
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Any
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Set up plotting
plt.style.use('default')
sns.set_palette("husl")


class VocabularySizeImpactAnalyzer:
    """
    Analyzes the impact of vocabulary size on training difficulty factors.
    """
    
    def __init__(self, text_key: str = "captions"):
        self.text_key = text_key
        self.datasets = {}
        self.vocab_analysis = {}
        
    def load_and_analyze_datasets(self):
        """Load and analyze both datasets for vocabulary size impact."""
        
        # Dataset paths
        dataset1_path = "/home/abedin/Developments/chest_x_ray_data_processing/all_processed_data/mimic_shards"
        dataset2_path = "/home/abedin/Developments/chest_x_ray_data_processing/all_processed_data/mimic_shards_hufc4446-to128"
        
        print("🔍 Loading and analyzing datasets for vocabulary size impact...")
        
        # Load both datasets
        self._load_dataset(dataset1_path, "mimic_shards")
        self._load_dataset(dataset2_path, "mimic_shards_hufc4446-to128")
        
        # Analyze vocabulary size impact
        self._analyze_vocabulary_size_impact()
        
    def _load_dataset(self, dataset_path: str, dataset_name: str):
        """Load a dataset and extract vocabulary information."""
        print(f"\n📊 Loading {dataset_name}...")
        
        # Find all pickle files
        file_pattern = os.path.join(dataset_path, "**", "*.pkl")
        files = glob.glob(file_pattern, recursive=True)
        
        if not files:
            print(f"❌ No files found in {dataset_path}")
            return
        
        # Load data
        all_entries = []
        for file_path in tqdm(files, desc=f"Loading {dataset_name}"):
            try:
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                
                if isinstance(data, dict):
                    # Convert dict to list of entries
                    entries = []
                    for i in range(len(data.get('images', []))):
                        entry = {}
                        for key, value in data.items():
                            if isinstance(value, (list, np.ndarray)) and i < len(value):
                                entry[key] = value[i]
                            else:
                                entry[key] = value
                        entries.append(entry)
                    all_entries.extend(entries)
                elif isinstance(data, list):
                    all_entries.extend(data)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
        
        # Extract text and vocabulary data
        texts = []
        token_sequences = []
        vocab_counter = Counter()
        
        for entry in tqdm(all_entries, desc=f"Processing {dataset_name}"):
            text = self._extract_text(entry)
            if text is not None:
                if isinstance(text, (list, np.ndarray)):
                    # Pre-tokenized text
                    tokens = [str(token) for token in text if token != 0]
                    token_sequences.append(tokens)
                    texts.append(" ".join(tokens))
                    vocab_counter.update(tokens)
                else:
                    # Raw text
                    tokens = text.lower().split()
                    token_sequences.append(tokens)
                    texts.append(text)
                    vocab_counter.update(tokens)
        
        # Store dataset data
        self.datasets[dataset_name] = {
            'texts': texts,
            'token_sequences': token_sequences,
            'vocabulary': vocab_counter,
            'total_entries': len(all_entries),
            'valid_entries': len(texts)
        }
        
        print(f"✅ Loaded {dataset_name}: {len(texts)} valid entries, {len(vocab_counter)} unique tokens")
    
    def _extract_text(self, entry: Dict) -> Any:
        """Extract text from entry."""
        if self.text_key in entry:
            return entry[self.text_key]
        
        # Try alternative keys
        for key in ['text', 'report', 'caption', 'findings', 'impression']:
            if key in entry:
                return entry[key]
        
        return None
    
    def _analyze_vocabulary_size_impact(self):
        """Analyze how vocabulary size differences impact training factors."""
        print("\n🔬 Analyzing vocabulary size impact on training factors...")
        
        dataset_names = list(self.datasets.keys())
        
        for name in dataset_names:
            dataset = self.datasets[name]
            vocab_counter = dataset['vocabulary']
            token_sequences = dataset['token_sequences']
            
            # Calculate vocabulary statistics
            total_tokens = sum(vocab_counter.values())
            vocab_size = len(vocab_counter)
            unique_tokens = list(vocab_counter.keys())
            
            # Token frequency analysis
            token_freq_values = list(vocab_counter.values())
            token_freq_values = np.array(token_freq_values)
            
            # Calculate vocabulary-based metrics
            vocab_metrics = {
                'vocab_size': vocab_size,
                'total_tokens': total_tokens,
                'avg_tokens_per_vocab': total_tokens / vocab_size,
                'token_freq_mean': float(np.mean(token_freq_values)),
                'token_freq_std': float(np.std(token_freq_values)),
                'token_freq_median': float(np.median(token_freq_values)),
                'most_common_freq': float(max(token_freq_values)),
                'least_common_freq': float(min(token_freq_values)),
                'vocab_efficiency': vocab_size / total_tokens,
                'frequency_skewness': self._calculate_skewness(token_freq_values),
                'frequency_kurtosis': self._calculate_kurtosis(token_freq_values),
                'rare_token_ratio': self._calculate_rare_token_ratio(vocab_counter),
                'common_token_ratio': self._calculate_common_token_ratio(vocab_counter),
                'vocabulary_coverage': self._calculate_vocabulary_coverage(token_sequences, vocab_counter)
            }
            
            # Calculate sequence-based metrics affected by vocabulary
            sequence_metrics = self._calculate_sequence_metrics(token_sequences, vocab_counter)
            
            self.vocab_analysis[name] = {
                'vocabulary_metrics': vocab_metrics,
                'sequence_metrics': sequence_metrics
            }
    
    def _calculate_skewness(self, values: np.ndarray) -> float:
        """Calculate skewness of token frequencies."""
        if len(values) < 3:
            return 0.0
        mean = np.mean(values)
        std = np.std(values)
        if std == 0:
            return 0.0
        return float(np.mean(((values - mean) / std) ** 3))
    
    def _calculate_kurtosis(self, values: np.ndarray) -> float:
        """Calculate kurtosis of token frequencies."""
        if len(values) < 4:
            return 0.0
        mean = np.mean(values)
        std = np.std(values)
        if std == 0:
            return 0.0
        return float(np.mean(((values - mean) / std) ** 4) - 3)
    
    def _calculate_rare_token_ratio(self, vocab_counter: Counter) -> float:
        """Calculate ratio of tokens that appear only once."""
        total_tokens = len(vocab_counter)
        rare_tokens = sum(1 for count in vocab_counter.values() if count == 1)
        return rare_tokens / total_tokens if total_tokens > 0 else 0.0
    
    def _calculate_common_token_ratio(self, vocab_counter: Counter) -> float:
        """Calculate ratio of tokens that appear frequently (>100 times)."""
        total_tokens = len(vocab_counter)
        common_tokens = sum(1 for count in vocab_counter.values() if count > 100)
        return common_tokens / total_tokens if total_tokens > 0 else 0.0
    
    def _calculate_vocabulary_coverage(self, token_sequences: List[List[str]], vocab_counter: Counter) -> float:
        """Calculate how much of the vocabulary is actually used in sequences."""
        used_tokens = set()
        for seq in token_sequences:
            used_tokens.update(seq)
        
        total_vocab = len(vocab_counter)
        return len(used_tokens) / total_vocab if total_vocab > 0 else 0.0
    
    def _calculate_sequence_metrics(self, token_sequences: List[List[str]], vocab_counter: Counter) -> Dict:
        """Calculate sequence metrics that are affected by vocabulary size."""
        sequence_lengths = [len(seq) for seq in token_sequences]
        unique_tokens_per_seq = [len(set(seq)) for seq in token_sequences]
        
        # Calculate vocabulary density (unique tokens / total tokens in sequence)
        vocab_density = []
        for seq in token_sequences:
            if len(seq) > 0:
                density = len(set(seq)) / len(seq)
                vocab_density.append(density)
        
        # Calculate vocabulary utilization (how much of the vocabulary is used per sequence)
        vocab_utilization = []
        for seq in token_sequences:
            if len(seq) > 0:
                unique_in_seq = len(set(seq))
                utilization = unique_in_seq / len(vocab_counter)
                vocab_utilization.append(utilization)
        
        return {
            'avg_sequence_length': float(np.mean(sequence_lengths)),
            'avg_unique_tokens_per_seq': float(np.mean(unique_tokens_per_seq)),
            'avg_vocab_density': float(np.mean(vocab_density)),
            'avg_vocab_utilization': float(np.mean(vocab_utilization)),
            'sequence_length_std': float(np.std(sequence_lengths)),
            'unique_tokens_std': float(np.std(unique_tokens_per_seq)),
            'vocab_density_std': float(np.std(vocab_density)),
            'vocab_utilization_std': float(np.std(vocab_utilization))
        }
    
    def analyze_vocabulary_size_correlation(self):
        """Analyze correlation between vocabulary size and training factors."""
        print("\n" + "="*100)
        print("🔬 VOCABULARY SIZE IMPACT ANALYSIS")
        print("="*100)
        
        dataset_names = list(self.datasets.keys())
        
        # Compare vocabulary sizes
        vocab_sizes = [self.vocab_analysis[name]['vocabulary_metrics']['vocab_size'] for name in dataset_names]
        
        print(f"\n📊 Vocabulary Size Comparison:")
        for i, name in enumerate(dataset_names):
            print(f"  {name}: {vocab_sizes[i]:,} tokens")
        
        vocab_size_ratio = vocab_sizes[0] / vocab_sizes[1]  # standard / extended
        print(f"  Ratio (Standard/Extended): {vocab_size_ratio:.2f}x")
        
        # Analyze each training factor
        print(f"\n🎯 Impact Analysis on Training Factors:")
        
        factors_to_analyze = [
            ('Token Frequency Distribution', 'token_freq_std'),
            ('Vocabulary Efficiency', 'vocab_efficiency'),
            ('Rare Token Ratio', 'rare_token_ratio'),
            ('Common Token Ratio', 'common_token_ratio'),
            ('Vocabulary Coverage', 'vocabulary_coverage'),
            ('Sequence Length Consistency', 'sequence_length_std'),
            ('Vocabulary Density', 'avg_vocab_density'),
            ('Vocabulary Utilization', 'avg_vocab_utilization')
        ]
        
        for factor_name, metric_key in factors_to_analyze:
            print(f"\n📈 {factor_name}:")
            
            values = []
            for name in dataset_names:
                if metric_key in self.vocab_analysis[name]['vocabulary_metrics']:
                    value = self.vocab_analysis[name]['vocabulary_metrics'][metric_key]
                else:
                    value = self.vocab_analysis[name]['sequence_metrics'][metric_key]
                values.append(value)
                print(f"  {name}: {value:.4f}")
            
            # Calculate impact
            if values[0] != 0:
                impact_ratio = values[1] / values[0]  # extended / standard
                print(f"  Impact Ratio (Extended/Standard): {impact_ratio:.2f}x")
                
                # Determine if vocabulary size is the likely cause
                if abs(impact_ratio - vocab_size_ratio) < 0.3:
                    print(f"  🎯 LIKELY CAUSED BY VOCABULARY SIZE DIFFERENCE")
                elif abs(impact_ratio - 1/vocab_size_ratio) < 0.3:
                    print(f"  🎯 LIKELY CAUSED BY VOCABULARY SIZE DIFFERENCE (inverse)")
                else:
                    print(f"  ❌ NOT LIKELY CAUSED BY VOCABULARY SIZE DIFFERENCE")
            else:
                print(f"  Cannot calculate impact ratio (zero value)")
        
        # Detailed correlation analysis
        print(f"\n🔍 Detailed Correlation Analysis:")
        
        # Create correlation matrix
        metrics_data = {}
        for name in dataset_names:
            metrics_data[name] = {}
            metrics_data[name].update(self.vocab_analysis[name]['vocabulary_metrics'])
            metrics_data[name].update(self.vocab_analysis[name]['sequence_metrics'])
        
        # Calculate correlations with vocabulary size
        vocab_size_correlations = {}
        for metric in metrics_data[dataset_names[0]].keys():
            if metric != 'vocab_size':
                values = [metrics_data[name][metric] for name in dataset_names]
                vocab_sizes = [metrics_data[name]['vocab_size'] for name in dataset_names]
                
                # Calculate correlation coefficient
                correlation = np.corrcoef(vocab_sizes, values)[0, 1]
                vocab_size_correlations[metric] = correlation
        
        # Sort by absolute correlation
        sorted_correlations = sorted(vocab_size_correlations.items(), 
                                   key=lambda x: abs(x[1]), reverse=True)
        
        print(f"\n📊 Correlation with Vocabulary Size (|r| > 0.5):")
        for metric, correlation in sorted_correlations:
            if abs(correlation) > 0.5:
                print(f"  {metric}: {correlation:.3f}")
        
        print(f"\n📊 Correlation with Vocabulary Size (|r| < 0.5):")
        for metric, correlation in sorted_correlations:
            if abs(correlation) <= 0.5:
                print(f"  {metric}: {correlation:.3f}")
    
    def print_detailed_comparison(self):
        """Print detailed comparison table."""
        print("\n" + "="*100)
        print("📊 DETAILED VOCABULARY IMPACT COMPARISON")
        print("="*100)
        
        dataset_names = list(self.datasets.keys())
        
        # Create comparison table
        metrics = [
            ('Vocabulary Size', 'vocab_size'),
            ('Total Tokens', 'total_tokens'),
            ('Avg Tokens per Vocab', 'avg_tokens_per_vocab'),
            ('Token Freq Mean', 'token_freq_mean'),
            ('Token Freq Std', 'token_freq_std'),
            ('Token Freq Median', 'token_freq_median'),
            ('Most Common Freq', 'most_common_freq'),
            ('Least Common Freq', 'least_common_freq'),
            ('Vocabulary Efficiency', 'vocab_efficiency'),
            ('Frequency Skewness', 'frequency_skewness'),
            ('Frequency Kurtosis', 'frequency_kurtosis'),
            ('Rare Token Ratio', 'rare_token_ratio'),
            ('Common Token Ratio', 'common_token_ratio'),
            ('Vocabulary Coverage', 'vocabulary_coverage'),
            ('Avg Sequence Length', 'avg_sequence_length'),
            ('Avg Unique Tokens/Seq', 'avg_unique_tokens_per_seq'),
            ('Avg Vocab Density', 'avg_vocab_density'),
            ('Avg Vocab Utilization', 'avg_vocab_utilization'),
            ('Sequence Length Std', 'sequence_length_std'),
            ('Unique Tokens Std', 'unique_tokens_std'),
            ('Vocab Density Std', 'vocab_density_std'),
            ('Vocab Utilization Std', 'vocab_utilization_std')
        ]
        
        # Print header
        header = f"{'Metric':<30}"
        for name in dataset_names:
            header += f"{name:>20}"
        print(header)
        print("-" * (30 + 20 * len(dataset_names)))
        
        # Print metrics
        for metric_name, metric_key in metrics:
            row = f"{metric_name:<30}"
            for name in dataset_names:
                if metric_key in self.vocab_analysis[name]['vocabulary_metrics']:
                    value = self.vocab_analysis[name]['vocabulary_metrics'][metric_key]
                else:
                    value = self.vocab_analysis[name]['sequence_metrics'][metric_key]
                
                if isinstance(value, float):
                    row += f"{value:>20.4f}"
                else:
                    row += f"{value:>20}"
            print(row)
    
    def save_analysis(self, output_path: str):
        """Save vocabulary impact analysis results."""
        results = {
            'analysis_timestamp': pd.Timestamp.now().isoformat(),
            'vocabulary_analysis': self.vocab_analysis,
            'summary': {
                'vocabulary_size_ratio': self.vocab_analysis[list(self.datasets.keys())[0]]['vocabulary_metrics']['vocab_size'] / 
                                        self.vocab_analysis[list(self.datasets.keys())[1]]['vocabulary_metrics']['vocab_size'],
                'total_tokens_ratio': self.vocab_analysis[list(self.datasets.keys())[0]]['vocabulary_metrics']['total_tokens'] / 
                                     self.vocab_analysis[list(self.datasets.keys())[1]]['vocabulary_metrics']['total_tokens']
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"💾 Vocabulary impact analysis saved to: {output_path}")


def main():
    """Main function."""
    analyzer = VocabularySizeImpactAnalyzer(text_key="captions")
    
    # Load and analyze datasets
    analyzer.load_and_analyze_datasets()
    
    # Analyze vocabulary size impact
    analyzer.analyze_vocabulary_size_correlation()
    
    # Print detailed comparison
    analyzer.print_detailed_comparison()
    
    # Save results
    analyzer.save_analysis("vocabulary_size_impact_results.json")
    
    print("\n✅ Vocabulary size impact analysis complete!")


if __name__ == "__main__":
    main() 