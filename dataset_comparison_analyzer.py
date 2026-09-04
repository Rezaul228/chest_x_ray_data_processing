#!/usr/bin/env python3
"""
Dataset Comparison Analyzer

A comprehensive tool to analyze and compare two datasets, specifically designed for
chest X-ray datasets with text reports. Supports multiple file formats and provides
detailed statistical analysis and visualization.

Author: AI Assistant
Date: 2024-12-19
"""

import os
import sys
import json
import pickle
import glob
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")


class DatasetAnalyzer:
    """
    A modular dataset analyzer that can handle different file formats and text extraction methods.
    """
    
    def __init__(self, 
                 text_key: str = "captions",
                 tokenizer_func: Optional[callable] = None,
                 file_pattern: str = "*.pkl"):
        """
        Initialize the dataset analyzer.
        
        Args:
            text_key: Key to extract text from data entries
            tokenizer_func: Function to tokenize text (default: split())
            file_pattern: Pattern to match files (e.g., "*.pkl", "*.json")
        """
        self.text_key = text_key
        self.tokenizer_func = tokenizer_func or self._default_tokenizer
        self.file_pattern = file_pattern
        
        # Statistics storage
        self.stats = {}
        self.vocabularies = {}
        self.token_lengths = {}
        self.sample_counts = {}
        
    def _default_tokenizer(self, text: str) -> List[str]:
        """
        Default tokenizer that splits on whitespace.
        
        Args:
            text: Input text string
            
        Returns:
            List of tokens
        """
        if isinstance(text, (list, np.ndarray)):
            # Handle pre-tokenized text
            return [str(token) for token in text if token != 0]  # Filter padding tokens
        elif isinstance(text, str):
            return text.lower().split()
        else:
            return []
    
    def _load_pickle_file(self, file_path: str) -> List[Dict]:
        """
        Load data from a pickle file.
        
        Args:
            file_path: Path to the pickle file
            
        Returns:
            List of data entries
        """
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            # Handle different data structures
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
                return entries
            elif isinstance(data, list):
                return data
            else:
                print(f"Warning: Unexpected data format in {file_path}")
                return []
                
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return []
    
    def _load_json_file(self, file_path: str) -> List[Dict]:
        """
        Load data from a JSON file.
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            List of data entries
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                # Handle single entry
                return [data]
            else:
                print(f"Warning: Unexpected JSON format in {file_path}")
                return []
                
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return []
    
    def _load_jsonl_file(self, file_path: str) -> List[Dict]:
        """
        Load data from a JSONL file.
        
        Args:
            file_path: Path to the JSONL file
            
        Returns:
            List of data entries
        """
        entries = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        try:
                            entry = json.loads(line)
                            entries.append(entry)
                        except json.JSONDecodeError as e:
                            print(f"Error parsing line {line_num} in {file_path}: {e}")
                            continue
            return entries
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return []
    
    def _load_parquet_file(self, file_path: str) -> List[Dict]:
        """
        Load data from a parquet file.
        
        Args:
            file_path: Path to the parquet file
            
        Returns:
            List of data entries
        """
        try:
            import pyarrow.parquet as pq
            table = pq.read_table(file_path)
            df = table.to_pandas()
            return df.to_dict('records')
        except ImportError:
            print("pyarrow not available for parquet files")
            return []
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return []
    
    def load_file(self, file_path: str) -> List[Dict]:
        """
        Load data from a file based on its extension.
        
        Args:
            file_path: Path to the file
            
        Returns:
            List of data entries
        """
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext == '.pkl':
            return self._load_pickle_file(file_path)
        elif file_ext == '.json':
            return self._load_json_file(file_path)
        elif file_ext == '.jsonl':
            return self._load_jsonl_file(file_path)
        elif file_ext == '.parquet':
            return self._load_parquet_file(file_path)
        else:
            print(f"Unsupported file format: {file_ext}")
            return []
    
    def extract_text_from_entry(self, entry: Dict) -> Optional[str]:
        """
        Extract text from a data entry.
        
        Args:
            entry: Data entry dictionary
            
        Returns:
            Extracted text or None if not found
        """
        # Try the specified text key
        if self.text_key in entry:
            text = entry[self.text_key]
            if text is not None and text != "":
                return text
        
        # Try common alternative keys
        alternative_keys = ['text', 'report', 'caption', 'findings', 'impression', 'summary']
        for key in alternative_keys:
            if key in entry:
                text = entry[key]
                if text is not None and text != "":
                    return text
        
        return None
    
    def analyze_dataset(self, dataset_path: str, dataset_name: str) -> Dict[str, Any]:
        """
        Analyze a single dataset.
        
        Args:
            dataset_path: Path to the dataset directory
            dataset_name: Name of the dataset for identification
            
        Returns:
            Dictionary containing analysis results
        """
        print(f"\n🔍 Analyzing dataset: {dataset_name}")
        print(f"📁 Path: {dataset_path}")
        
        # Find all files matching the pattern
        file_pattern = os.path.join(dataset_path, "**", self.file_pattern)
        files = glob.glob(file_pattern, recursive=True)
        
        if not files:
            print(f"❌ No files found matching pattern: {file_pattern}")
            return {}
        
        print(f"📄 Found {len(files)} files")
        
        # Load and process all data
        all_entries = []
        empty_entries = 0
        malformed_entries = 0
        
        for file_path in tqdm(files, desc=f"Loading {dataset_name}"):
            entries = self.load_file(file_path)
            all_entries.extend(entries)
        
        print(f"📊 Total entries loaded: {len(all_entries)}")
        
        # Extract text and compute statistics
        texts = []
        token_lengths = []
        vocabulary = Counter()
        
        for entry in tqdm(all_entries, desc=f"Processing {dataset_name}"):
            text = self.extract_text_from_entry(entry)
            
            if text is None or text == "":
                empty_entries += 1
                continue
            
            # Tokenize text
            tokens = self.tokenizer_func(text)
            
            if not tokens:
                malformed_entries += 1
                continue
            
            texts.append(text)
            token_lengths.append(len(tokens))
            vocabulary.update(tokens)
        
        # Compute statistics
        if token_lengths:
            token_lengths = np.array(token_lengths)
            stats = {
                'dataset_name': dataset_name,
                'total_samples': len(all_entries),
                'valid_samples': len(texts),
                'empty_entries': empty_entries,
                'malformed_entries': malformed_entries,
                'vocabulary_size': len(vocabulary),
                'token_length_min': float(np.min(token_lengths)),
                'token_length_max': float(np.max(token_lengths)),
                'token_length_mean': float(np.mean(token_lengths)),
                'token_length_median': float(np.median(token_lengths)),
                'token_length_std': float(np.std(token_lengths)),
                'token_length_q25': float(np.percentile(token_lengths, 25)),
                'token_length_q75': float(np.percentile(token_lengths, 75)),
                'total_tokens': int(np.sum(token_lengths)),
                'avg_tokens_per_sample': float(np.mean(token_lengths))
            }
        else:
            stats = {
                'dataset_name': dataset_name,
                'total_samples': len(all_entries),
                'valid_samples': 0,
                'empty_entries': empty_entries,
                'malformed_entries': malformed_entries,
                'vocabulary_size': 0,
                'token_length_min': 0,
                'token_length_max': 0,
                'token_length_mean': 0,
                'token_length_median': 0,
                'token_length_std': 0,
                'token_length_q25': 0,
                'token_length_q75': 0,
                'total_tokens': 0,
                'avg_tokens_per_sample': 0
            }
        
        # Store results
        self.stats[dataset_name] = stats
        self.vocabularies[dataset_name] = set(vocabulary.keys())
        self.token_lengths[dataset_name] = token_lengths
        
        print(f"✅ Analysis complete for {dataset_name}")
        return stats
    
    def compare_datasets(self, dataset1_name: str, dataset2_name: str) -> Dict[str, Any]:
        """
        Compare two datasets.
        
        Args:
            dataset1_name: Name of the first dataset
            dataset2_name: Name of the second dataset
            
        Returns:
            Dictionary containing comparison results
        """
        if dataset1_name not in self.stats or dataset2_name not in self.stats:
            print("❌ Both datasets must be analyzed before comparison")
            return {}
        
        vocab1 = self.vocabularies[dataset1_name]
        vocab2 = self.vocabularies[dataset2_name]
        
        # Vocabulary overlap
        vocab_overlap = vocab1.intersection(vocab2)
        vocab_union = vocab1.union(vocab2)
        vocab1_only = vocab1 - vocab2
        vocab2_only = vocab2 - vocab1
        
        # Determine smaller vocabulary for percentage calculation
        smaller_vocab_size = min(len(vocab1), len(vocab2))
        overlap_percentage = (len(vocab_overlap) / smaller_vocab_size) * 100 if smaller_vocab_size > 0 else 0
        
        comparison = {
            'vocab_overlap_absolute': len(vocab_overlap),
            'vocab_overlap_percentage': overlap_percentage,
            'vocab_union_size': len(vocab_union),
            'vocab1_only_size': len(vocab1_only),
            'vocab2_only_size': len(vocab2_only),
            'vocab1_unique_percentage': (len(vocab1_only) / len(vocab1)) * 100 if len(vocab1) > 0 else 0,
            'vocab2_unique_percentage': (len(vocab2_only) / len(vocab2)) * 100 if len(vocab2) > 0 else 0,
            'jaccard_similarity': len(vocab_overlap) / len(vocab_union) if len(vocab_union) > 0 else 0
        }
        
        return comparison
    
    def print_summary_table(self):
        """Print a side-by-side summary table of statistics."""
        if len(self.stats) < 2:
            print("❌ Need at least 2 datasets for comparison")
            return
        
        print("\n" + "="*100)
        print("📊 DATASET COMPARISON SUMMARY")
        print("="*100)
        
        # Get dataset names
        dataset_names = list(self.stats.keys())
        
        # Create comparison table
        metrics = [
            ('Total Samples', 'total_samples'),
            ('Valid Samples', 'valid_samples'),
            ('Empty Entries', 'empty_entries'),
            ('Malformed Entries', 'malformed_entries'),
            ('Vocabulary Size', 'vocabulary_size'),
            ('Total Tokens', 'total_tokens'),
            ('Avg Tokens/Sample', 'avg_tokens_per_sample'),
            ('Token Length - Min', 'token_length_min'),
            ('Token Length - Max', 'token_length_max'),
            ('Token Length - Mean', 'token_length_mean'),
            ('Token Length - Median', 'token_length_median'),
            ('Token Length - Std', 'token_length_std'),
            ('Token Length - Q25', 'token_length_q25'),
            ('Token Length - Q75', 'token_length_q75')
        ]
        
        # Print header
        header = f"{'Metric':<25}"
        for name in dataset_names:
            header += f"{name:>20}"
        print(header)
        print("-" * (25 + 20 * len(dataset_names)))
        
        # Print metrics
        for metric_name, metric_key in metrics:
            row = f"{metric_name:<25}"
            for name in dataset_names:
                value = self.stats[name].get(metric_key, 0)
                if isinstance(value, float):
                    row += f"{value:>20.2f}"
                else:
                    row += f"{value:>20}"
            print(row)
        
        # Print vocabulary comparison if available
        if len(dataset_names) == 2:
            comparison = self.compare_datasets(dataset_names[0], dataset_names[1])
            if comparison:
                print("\n" + "-" * (25 + 20 * len(dataset_names)))
                vocab_metrics = [
                    ('Vocab Overlap (Abs)', 'vocab_overlap_absolute'),
                    ('Vocab Overlap (%)', 'vocab_overlap_percentage'),
                    ('Vocab Union Size', 'vocab_union_size'),
                    ('Jaccard Similarity', 'jaccard_similarity')
                ]
                
                for metric_name, metric_key in vocab_metrics:
                    value = comparison.get(metric_key, 0)
                    if isinstance(value, float):
                        print(f"{metric_name:<25}{value:>20.2f}")
                    else:
                        print(f"{metric_name:<25}{value:>20}")
    
    def plot_token_length_distributions(self, save_path: Optional[str] = None):
        """
        Plot histograms of token length distributions for both datasets.
        
        Args:
            save_path: Optional path to save the plot
        """
        if len(self.token_lengths) < 2:
            print("❌ Need at least 2 datasets for plotting")
            return
        
        plt.figure(figsize=(15, 10))
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        
        # Plot 1: Histogram comparison
        colors = plt.cm.Set1(np.linspace(0, 1, len(self.token_lengths)))
        
        for i, (dataset_name, lengths) in enumerate(self.token_lengths.items()):
            if len(lengths) > 0:
                ax1.hist(lengths, bins=50, alpha=0.7, label=dataset_name, 
                        color=colors[i], density=True)
        
        ax1.set_xlabel('Token Length')
        ax1.set_ylabel('Density')
        ax1.set_title('Token Length Distribution Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Box plot comparison
        data_for_box = []
        labels = []
        for dataset_name, lengths in self.token_lengths.items():
            if len(lengths) > 0:
                data_for_box.append(lengths)
                labels.append(dataset_name)
        
        if data_for_box:
            ax2.boxplot(data_for_box, labels=labels, patch_artist=True)
            ax2.set_ylabel('Token Length')
            ax2.set_title('Token Length Distribution (Box Plot)')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Plot saved to: {save_path}")
        
        plt.show()
    
    def save_results(self, output_path: str):
        """
        Save analysis results to a JSON file.
        
        Args:
            output_path: Path to save the results
        """
        results = {
            'analysis_timestamp': pd.Timestamp.now().isoformat(),
            'text_key': self.text_key,
            'file_pattern': self.file_pattern,
            'statistics': self.stats,
            'vocabulary_sizes': {name: len(vocab) for name, vocab in self.vocabularies.items()},
            'comparisons': {}
        }
        
        # Add dataset comparisons if we have exactly 2 datasets
        if len(self.stats) == 2:
            dataset_names = list(self.stats.keys())
            results['comparisons'] = self.compare_datasets(dataset_names[0], dataset_names[1])
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"💾 Results saved to: {output_path}")


def main():
    """Main function to run the dataset comparison analysis."""
    parser = argparse.ArgumentParser(description='Compare two datasets')
    parser.add_argument('dataset1_path', help='Path to first dataset')
    parser.add_argument('dataset2_path', help='Path to second dataset')
    parser.add_argument('--text-key', default='captions', help='Key to extract text (default: captions)')
    parser.add_argument('--file-pattern', default='*.pkl', help='File pattern to match (default: *.pkl)')
    parser.add_argument('--output', help='Path to save results JSON')
    parser.add_argument('--plot-output', help='Path to save plot')
    parser.add_argument('--dataset1-name', help='Name for first dataset')
    parser.add_argument('--dataset2-name', help='Name for second dataset')
    
    args = parser.parse_args()
    
    # Set default dataset names if not provided
    dataset1_name = args.dataset1_name or Path(args.dataset1_path).name
    dataset2_name = args.dataset2_name or Path(args.dataset2_path).name
    
    print("🚀 Starting Dataset Comparison Analysis")
    print("="*60)
    
    # Initialize analyzer
    analyzer = DatasetAnalyzer(
        text_key=args.text_key,
        file_pattern=args.file_pattern
    )
    
    # Analyze both datasets
    analyzer.analyze_dataset(args.dataset1_path, dataset1_name)
    analyzer.analyze_dataset(args.dataset2_path, dataset2_name)
    
    # Print summary table
    analyzer.print_summary_table()
    
    # Plot distributions
    analyzer.plot_token_length_distributions(save_path=args.plot_output)
    
    # Save results
    if args.output:
        analyzer.save_results(args.output)
    
    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    main() 