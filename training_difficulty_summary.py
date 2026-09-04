#!/usr/bin/env python3
"""
Training Difficulty Summary Visualization

Creates a visual summary of the training difficulty analysis results.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set up plotting
plt.style.use('default')
sns.set_palette("husl")

# Data from the analysis
datasets = ['mimic_shards', 'mimic_shards_hufc4446-to128']
colors = ['#2E86AB', '#A23B72']

# Key metrics for comparison
metrics = {
    'Training Stability': [0.0969, 0.1534],
    'Length Consistency': [0.6626, 0.6970],
    'Sequence Diversity': [0.0035, 0.0085],
    'Memory Efficiency': [236.07, 93.64],
    'Vocabulary Efficiency': [0.0054, 0.0018],
    'Complexity Score': [0.8376, 0.7920]
}

# Create the visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Training Difficulty Analysis: Key Factors Beyond Vocabulary Size', 
             fontsize=16, fontweight='bold')

# Plot each metric
for i, (metric_name, values) in enumerate(metrics.items()):
    row = i // 3
    col = i % 3
    ax = axes[row, col]
    
    bars = ax.bar(datasets, values, color=colors, alpha=0.8)
    ax.set_title(f'{metric_name}', fontweight='bold')
    ax.set_ylabel('Score' if 'Score' in metric_name else 'Value')
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{value:.4f}' if value < 1 else f'{value:.2f}',
                ha='center', va='bottom', fontweight='bold')
    
    # Color code based on whether higher is better
    if metric_name in ['Training Stability', 'Length Consistency', 'Sequence Diversity']:
        # Higher is better - green for higher values
        for j, (bar, value) in enumerate(zip(bars, values)):
            if value == max(values):
                bar.set_color('green')
            else:
                bar.set_color('orange')
    elif metric_name in ['Memory Efficiency', 'Vocabulary Efficiency']:
        # Higher is better - green for higher values
        for j, (bar, value) in enumerate(zip(bars, values)):
            if value == max(values):
                bar.set_color('green')
            else:
                bar.set_color('orange')
    else:  # Complexity Score
        # Lower is better - green for lower values
        for j, (bar, value) in enumerate(zip(bars, values)):
            if value == min(values):
                bar.set_color('green')
            else:
                bar.set_color('orange')
    
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_difficulty_summary.png', dpi=300, bbox_inches='tight')
plt.show()

# Create a summary table
print("\n" + "="*80)
print("🎯 TRAINING DIFFICULTY SUMMARY")
print("="*80)

print(f"{'Metric':<25} {'mimic_shards':<20} {'mimic_shards_hufc4446':<20} {'Winner':<10}")
print("-" * 80)

for metric_name, values in metrics.items():
    winner = "mimic_shards" if values[0] > values[1] else "mimic_shards_hufc4446"
    if metric_name == "Complexity Score":
        winner = "mimic_shards_hufc4446" if values[1] < values[0] else "mimic_shards"
    
    print(f"{metric_name:<25} {values[0]:<20.4f} {values[1]:<20.4f} {winner:<10}")

print("\n" + "="*80)
print("🏆 OVERALL ASSESSMENT")
print("="*80)

print("✅ mimic_shards_hufc4446-to128 advantages:")
print("   - 58% better training stability")
print("   - 2.4x higher sequence diversity") 
print("   - 5% better length consistency")
print("   - 5% lower complexity score")
print("   - More balanced token distribution")

print("\n✅ mimic_shards advantages:")
print("   - 2.5x better memory efficiency")
print("   - 3x better vocabulary efficiency")
print("   - Lower computational requirements")

print("\n🎯 CONCLUSION:")
print("The extended version (mimic_shards_hufc4446-to128) is SLIGHTLY EASIER")
print("to train despite having a smaller vocabulary, due to better training")
print("stability and more predictable patterns.") 