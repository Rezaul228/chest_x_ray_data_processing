#!/usr/bin/env python3
"""
Dataset Training Difficulty Analysis

This script analyzes why one dataset might be easier to train than another,
examining factors beyond vocabulary size such as data distribution, sequence
length patterns, token frequency distributions, and other training-relevant metrics.
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


class TrainingDifficultyAnalyzer:
    """
    Analyzes datasets for training difficulty factors beyond vocabulary size.
    """
    
    def __init__(self, text_key: str = "captions"):
        self.text_key = text_key
        self.datasets = {}
        self.training_metrics = {}
        
    def load_dataset(self, dataset_path: str, dataset_name: str):
        """Load and analyze a dataset for training difficulty factors."""
        print(f"\n🔍 Analyzing training difficulty for: {dataset_name}")
        
        # Find all pickle files
        file_pattern = os.path.join(dataset_path, "**", "*.pkl")
        files = glob.glob(file_pattern, recursive=True)
        
        if not files:
            print(f"❌ No files found in {dataset_path}")
            return
        
        print(f"📄 Found {len(files)} files")
        
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
        
        print(f"📊 Loaded {len(all_entries)} entries")
        
        # Extract text data
        texts = []
        token_sequences = []
        
        for entry in tqdm(all_entries, desc=f"Processing {dataset_name}"):
            text = self._extract_text(entry)
            if text is not None:
                if isinstance(text, (list, np.ndarray)):
                    # Pre-tokenized text
                    tokens = [str(token) for token in text if token != 0]
                    token_sequences.append(tokens)
                    texts.append(" ".join(tokens))
                else:
                    # Raw text
                    tokens = text.lower().split()
                    token_sequences.append(tokens)
                    texts.append(text)
        
        # Store dataset data
        self.datasets[dataset_name] = {
            'texts': texts,
            'token_sequences': token_sequences,
            'total_entries': len(all_entries),
            'valid_entries': len(texts)
        }
        
        # Analyze training difficulty factors
        self._analyze_training_difficulty(dataset_name)
    
    def _extract_text(self, entry: Dict) -> Any:
        """Extract text from entry."""
        if self.text_key in entry:
            return entry[self.text_key]
        
        # Try alternative keys
        for key in ['text', 'report', 'caption', 'findings', 'impression']:
            if key in entry:
                return entry[key]
        
        return None
    
    def _analyze_training_difficulty(self, dataset_name: str):
        """Analyze various factors that affect training difficulty."""
        dataset = self.datasets[dataset_name]
        token_sequences = dataset['token_sequences']
        
        print(f"\n📈 Analyzing training difficulty factors for {dataset_name}...")
        
        # 1. Sequence Length Analysis
        sequence_lengths = [len(seq) for seq in token_sequences]
        sequence_lengths = np.array(sequence_lengths)
        
        # 2. Token Frequency Analysis
        all_tokens = []
        for seq in token_sequences:
            all_tokens.extend(seq)
        
        token_freq = Counter(all_tokens)
        total_tokens = len(all_tokens)
        
        # 3. Vocabulary Distribution Analysis
        vocab_size = len(token_freq)
        
        # 4. Sequence Complexity Analysis
        unique_tokens_per_seq = [len(set(seq)) for seq in token_sequences]
        unique_tokens_per_seq = np.array(unique_tokens_per_seq)
        
        # 5. Repetition Analysis
        repetition_ratios = []
        for seq in token_sequences:
            if len(seq) > 0:
                unique_count = len(set(seq))
                total_count = len(seq)
                repetition_ratio = 1 - (unique_count / total_count)
                repetition_ratios.append(repetition_ratio)
        
        repetition_ratios = np.array(repetition_ratios)
        
        # 6. Token Distribution Analysis
        token_freq_values = list(token_freq.values())
        token_freq_values = np.array(token_freq_values)
        
        # 7. Sequence Pattern Analysis
        avg_tokens_per_position = self._analyze_position_patterns(token_sequences)
        
        # 8. Training Stability Metrics
        length_variance = np.var(sequence_lengths)
        length_std = np.std(sequence_lengths)
        length_cv = length_std / np.mean(sequence_lengths) if np.mean(sequence_lengths) > 0 else 0
        
        # Store metrics
        self.training_metrics[dataset_name] = {
            # Basic stats
            'total_entries': dataset['total_entries'],
            'valid_entries': dataset['valid_entries'],
            'total_tokens': total_tokens,
            'vocab_size': vocab_size,
            
            # Sequence length metrics
            'avg_sequence_length': float(np.mean(sequence_lengths)),
            'median_sequence_length': float(np.median(sequence_lengths)),
            'min_sequence_length': float(np.min(sequence_lengths)),
            'max_sequence_length': float(np.max(sequence_lengths)),
            'sequence_length_std': float(np.std(sequence_lengths)),
            'sequence_length_variance': float(length_variance),
            'sequence_length_cv': float(length_cv),
            
            # Complexity metrics
            'avg_unique_tokens_per_seq': float(np.mean(unique_tokens_per_seq)),
            'avg_repetition_ratio': float(np.mean(repetition_ratios)),
            'complexity_score': float(np.mean(unique_tokens_per_seq) / np.mean(sequence_lengths)),
            
            # Token distribution metrics
            'token_freq_mean': float(np.mean(token_freq_values)),
            'token_freq_std': float(np.std(token_freq_values)),
            'token_freq_median': float(np.median(token_freq_values)),
            'most_common_token_freq': float(max(token_freq_values)),
            'least_common_token_freq': float(min(token_freq_values)),
            
            # Training difficulty indicators
            'length_consistency': 1.0 / (1.0 + length_cv),  # Higher is better
            'vocabulary_efficiency': vocab_size / total_tokens,  # Higher is better
            'sequence_diversity': float(np.mean(unique_tokens_per_seq) / vocab_size),
            'training_stability_score': self._calculate_stability_score(sequence_lengths, token_freq_values),
            
            # Position analysis
            'position_consistency': float(np.std(avg_tokens_per_position)),
            
            # Memory efficiency
            'avg_tokens_per_sample': float(total_tokens / len(token_sequences)),
            'memory_efficiency': float(vocab_size / np.mean(sequence_lengths))
        }
        
        print(f"✅ Analysis complete for {dataset_name}")
    
    def _analyze_position_patterns(self, token_sequences: List[List[str]]) -> List[float]:
        """Analyze token patterns at different positions."""
        max_len = max(len(seq) for seq in token_sequences) if token_sequences else 0
        
        if max_len == 0:
            return []
        
        position_tokens = defaultdict(list)
        
        for seq in token_sequences:
            for pos, token in enumerate(seq):
                position_tokens[pos].append(token)
        
        avg_tokens_per_position = []
        for pos in range(max_len):
            if pos in position_tokens:
                unique_tokens = len(set(position_tokens[pos]))
                avg_tokens_per_position.append(unique_tokens)
            else:
                avg_tokens_per_position.append(0)
        
        return avg_tokens_per_position
    
    def _calculate_stability_score(self, sequence_lengths: np.ndarray, token_freq_values: np.ndarray) -> float:
        """Calculate training stability score."""
        # Normalize metrics
        length_cv = np.std(sequence_lengths) / np.mean(sequence_lengths) if np.mean(sequence_lengths) > 0 else 1
        freq_cv = np.std(token_freq_values) / np.mean(token_freq_values) if np.mean(token_freq_values) > 0 else 1
        
        # Combine into stability score (lower CV = higher stability)
        stability_score = 1.0 / (1.0 + length_cv + freq_cv)
        return float(stability_score)
    
    def compare_training_difficulty(self):
        """Compare training difficulty between datasets."""
        if len(self.training_metrics) < 2:
            print("❌ Need at least 2 datasets for comparison")
            return
        
        print("\n" + "="*100)
        print("🎯 TRAINING DIFFICULTY COMPARISON ANALYSIS")
        print("="*100)
        
        dataset_names = list(self.training_metrics.keys())
        
        # Create comparison table
        metrics = [
            ('Total Entries', 'total_entries'),
            ('Valid Entries', 'valid_entries'),
            ('Total Tokens', 'total_tokens'),
            ('Vocabulary Size', 'vocab_size'),
            ('Avg Sequence Length', 'avg_sequence_length'),
            ('Sequence Length Std', 'sequence_length_std'),
            ('Length Consistency', 'length_consistency'),
            ('Vocabulary Efficiency', 'vocabulary_efficiency'),
            ('Sequence Diversity', 'sequence_diversity'),
            ('Training Stability Score', 'training_stability_score'),
            ('Memory Efficiency', 'memory_efficiency'),
            ('Avg Tokens per Sample', 'avg_tokens_per_sample'),
            ('Complexity Score', 'complexity_score'),
            ('Avg Repetition Ratio', 'avg_repetition_ratio'),
            ('Position Consistency', 'position_consistency')
        ]
        
        # Print header
        header = f"{'Training Difficulty Factor':<35}"
        for name in dataset_names:
            header += f"{name:>20}"
        print(header)
        print("-" * (35 + 20 * len(dataset_names)))
        
        # Print metrics
        for metric_name, metric_key in metrics:
            row = f"{metric_name:<35}"
            for name in dataset_names:
                value = self.training_metrics[name].get(metric_key, 0)
                if isinstance(value, float):
                    row += f"{value:>20.4f}"
                else:
                    row += f"{value:>20}"
            print(row)
        
        # Training difficulty assessment
        print("\n" + "="*100)
        print("🎯 TRAINING DIFFICULTY ASSESSMENT")
        print("="*100)
        
        for i, name in enumerate(dataset_names):
            metrics = self.training_metrics[name]
            
            print(f"\n📊 Dataset: {name}")
            print("-" * 50)
            
            # Assess each factor
            factors = []
            
            # Length consistency (higher is better)
            if metrics['length_consistency'] > 0.8:
                factors.append("✅ Good length consistency")
            elif metrics['length_consistency'] > 0.6:
                factors.append("⚠️  Moderate length consistency")
            else:
                factors.append("❌ Poor length consistency")
            
            # Vocabulary efficiency (higher is better)
            if metrics['vocabulary_efficiency'] > 0.1:
                factors.append("✅ Good vocabulary efficiency")
            elif metrics['vocabulary_efficiency'] > 0.05:
                factors.append("⚠️  Moderate vocabulary efficiency")
            else:
                factors.append("❌ Poor vocabulary efficiency")
            
            # Training stability (higher is better)
            if metrics['training_stability_score'] > 0.7:
                factors.append("✅ High training stability")
            elif metrics['training_stability_score'] > 0.5:
                factors.append("⚠️  Moderate training stability")
            else:
                factors.append("❌ Low training stability")
            
            # Memory efficiency (higher is better)
            if metrics['memory_efficiency'] > 50:
                factors.append("✅ Good memory efficiency")
            elif metrics['memory_efficiency'] > 20:
                factors.append("⚠️  Moderate memory efficiency")
            else:
                factors.append("❌ Poor memory efficiency")
            
            # Sequence diversity (higher is better)
            if metrics['sequence_diversity'] > 0.8:
                factors.append("✅ High sequence diversity")
            elif metrics['sequence_diversity'] > 0.5:
                factors.append("⚠️  Moderate sequence diversity")
            else:
                factors.append("❌ Low sequence diversity")
            
            # Print factors
            for factor in factors:
                print(f"  {factor}")
            
            # Overall assessment
            overall_score = (
                metrics['length_consistency'] * 0.25 +
                metrics['vocabulary_efficiency'] * 100 * 0.2 +
                metrics['training_stability_score'] * 0.25 +
                min(metrics['memory_efficiency'] / 100, 1.0) * 0.2 +
                metrics['sequence_diversity'] * 0.1
            )
            
            if overall_score > 0.7:
                difficulty = "🟢 EASY TO TRAIN"
            elif overall_score > 0.5:
                difficulty = "🟡 MODERATE DIFFICULTY"
            else:
                difficulty = "🔴 DIFFICULT TO TRAIN"
            
            print(f"\n🎯 Overall Training Difficulty: {difficulty} (Score: {overall_score:.3f})")
    
    def plot_training_analysis(self, save_path: str = None):
        """Plot training difficulty analysis."""
        if len(self.training_metrics) < 2:
            print("❌ Need at least 2 datasets for plotting")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Training Difficulty Analysis', fontsize=16, fontweight='bold')
        
        dataset_names = list(self.training_metrics.keys())
        colors = ['#2E86AB', '#A23B72']
        
        # Plot 1: Sequence Length Distribution
        ax1 = axes[0, 0]
        for i, name in enumerate(dataset_names):
            dataset = self.datasets[name]
            lengths = [len(seq) for seq in dataset['token_sequences']]
            ax1.hist(lengths, bins=30, alpha=0.7, label=name, color=colors[i], density=True)
        ax1.set_xlabel('Sequence Length')
        ax1.set_ylabel('Density')
        ax1.set_title('Sequence Length Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Training Stability Comparison
        ax2 = axes[0, 1]
        stability_scores = [self.training_metrics[name]['training_stability_score'] for name in dataset_names]
        ax2.bar(dataset_names, stability_scores, color=colors[:len(dataset_names)])
        ax2.set_ylabel('Training Stability Score')
        ax2.set_title('Training Stability Comparison')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Vocabulary Efficiency
        ax3 = axes[0, 2]
        vocab_efficiency = [self.training_metrics[name]['vocabulary_efficiency'] for name in dataset_names]
        ax3.bar(dataset_names, vocab_efficiency, color=colors[:len(dataset_names)])
        ax3.set_ylabel('Vocabulary Efficiency')
        ax3.set_title('Vocabulary Efficiency Comparison')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Length Consistency
        ax4 = axes[1, 0]
        length_consistency = [self.training_metrics[name]['length_consistency'] for name in dataset_names]
        ax4.bar(dataset_names, length_consistency, color=colors[:len(dataset_names)])
        ax4.set_ylabel('Length Consistency')
        ax4.set_title('Length Consistency Comparison')
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Memory Efficiency
        ax5 = axes[1, 1]
        memory_efficiency = [self.training_metrics[name]['memory_efficiency'] for name in dataset_names]
        ax5.bar(dataset_names, memory_efficiency, color=colors[:len(dataset_names)])
        ax5.set_ylabel('Memory Efficiency')
        ax5.set_title('Memory Efficiency Comparison')
        ax5.grid(True, alpha=0.3)
        
        # Plot 6: Overall Training Difficulty Score
        ax6 = axes[1, 2]
        overall_scores = []
        for name in dataset_names:
            metrics = self.training_metrics[name]
            score = (
                metrics['length_consistency'] * 0.25 +
                metrics['vocabulary_efficiency'] * 100 * 0.2 +
                metrics['training_stability_score'] * 0.25 +
                min(metrics['memory_efficiency'] / 100, 1.0) * 0.2 +
                metrics['sequence_diversity'] * 0.1
            )
            overall_scores.append(score)
        
        bars = ax6.bar(dataset_names, overall_scores, color=colors[:len(dataset_names)])
        ax6.set_ylabel('Overall Training Difficulty Score')
        ax6.set_title('Overall Training Difficulty')
        ax6.grid(True, alpha=0.3)
        
        # Add color coding to bars
        for bar, score in zip(bars, overall_scores):
            if score > 0.7:
                bar.set_color('green')
            elif score > 0.5:
                bar.set_color('orange')
            else:
                bar.set_color('red')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Training analysis plot saved to: {save_path}")
        
        plt.show()
    
    def save_analysis(self, output_path: str):
        """Save analysis results."""
        results = {
            'analysis_timestamp': pd.Timestamp.now().isoformat(),
            'training_metrics': self.training_metrics,
            'summary': {}
        }
        
        # Add summary for each dataset
        for name, metrics in self.training_metrics.items():
            overall_score = (
                metrics['length_consistency'] * 0.25 +
                metrics['vocabulary_efficiency'] * 100 * 0.2 +
                metrics['training_stability_score'] * 0.25 +
                min(metrics['memory_efficiency'] / 100, 1.0) * 0.2 +
                metrics['sequence_diversity'] * 0.1
            )
            
            if overall_score > 0.7:
                difficulty = "EASY TO TRAIN"
            elif overall_score > 0.5:
                difficulty = "MODERATE DIFFICULTY"
            else:
                difficulty = "DIFFICULT TO TRAIN"
            
            results['summary'][name] = {
                'overall_score': overall_score,
                'training_difficulty': difficulty,
                'key_strengths': [],
                'key_weaknesses': []
            }
            
            # Identify strengths and weaknesses
            if metrics['length_consistency'] > 0.8:
                results['summary'][name]['key_strengths'].append("Good length consistency")
            elif metrics['length_consistency'] < 0.6:
                results['summary'][name]['key_weaknesses'].append("Poor length consistency")
            
            if metrics['training_stability_score'] > 0.7:
                results['summary'][name]['key_strengths'].append("High training stability")
            elif metrics['training_stability_score'] < 0.5:
                results['summary'][name]['key_weaknesses'].append("Low training stability")
            
            if metrics['memory_efficiency'] > 50:
                results['summary'][name]['key_strengths'].append("Good memory efficiency")
            elif metrics['memory_efficiency'] < 20:
                results['summary'][name]['key_weaknesses'].append("Poor memory efficiency")
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"💾 Training analysis saved to: {output_path}")


def main():
    """Main function."""
    analyzer = TrainingDifficultyAnalyzer(text_key="captions")
    
    # Analyze both datasets
    dataset1_path = "/home/abedin/Developments/chest_x_ray_data_processing/all_processed_data/mimic_shards"
    dataset2_path = "/home/abedin/Developments/chest_x_ray_data_processing/all_processed_data/mimic_shards_hufc4446-to128"
    
    analyzer.load_dataset(dataset1_path, "mimic_shards")
    analyzer.load_dataset(dataset2_path, "mimic_shards_hufc4446-to128")
    
    # Compare training difficulty
    analyzer.compare_training_difficulty()
    
    # Generate plots
    analyzer.plot_training_analysis(save_path="training_difficulty_analysis.png")
    
    # Save results
    analyzer.save_analysis("training_difficulty_results.json")
    
    print("\n✅ Training difficulty analysis complete!")


if __name__ == "__main__":
    main() 