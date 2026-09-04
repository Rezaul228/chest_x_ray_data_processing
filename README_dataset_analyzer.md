# Dataset Comparison Analyzer

A comprehensive Python tool for analyzing and comparing datasets, specifically designed for chest X-ray datasets with text reports. The analyzer supports multiple file formats, provides detailed statistical analysis, and generates visualizations.

## Features

- **Multi-format Support**: Handles pickle, JSON, JSONL, and parquet files
- **Modular Design**: Easy to extend with custom tokenizers and text extraction methods
- **Comprehensive Statistics**: Detailed analysis of token lengths, vocabulary, and data quality
- **Visualization**: Automatic generation of distribution plots
- **Flexible Configuration**: Configurable text keys, file patterns, and analysis parameters
- **Standalone Operation**: Can run independently with minimal dependencies

## Installation

### Prerequisites

```bash
pip install numpy pandas matplotlib seaborn tqdm
```

For parquet file support:
```bash
pip install pyarrow
```

## Quick Start

### Basic Usage

```bash
python dataset_comparison_analyzer.py \
    /path/to/dataset1 \
    /path/to/dataset2 \
    --dataset1-name "dataset1" \
    --dataset2-name "dataset2" \
    --output "results.json" \
    --plot-output "distributions.png"
```

### Example with MIMIC Datasets

```bash
python dataset_comparison_analyzer.py \
    /home/abedin/Developments/chest_x_ray_data_processing/all_processed_data/mimic_shards \
    /home/abedin/Developments/chest_x_ray_data_processing/all_processed_data/mimic_shards_hufc4446-to128 \
    --dataset1-name "mimic_shards" \
    --dataset2-name "mimic_shards_hufc4446-to128" \
    --output "mimic_comparison_results.json" \
    --plot-output "mimic_token_length_distributions.png"
```

## Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `dataset1_path` | Path to first dataset directory | Required |
| `dataset2_path` | Path to second dataset directory | Required |
| `--text-key` | Key to extract text from data entries | `captions` |
| `--file-pattern` | File pattern to match (e.g., `*.pkl`, `*.json`) | `*.pkl` |
| `--output` | Path to save results JSON | Optional |
| `--plot-output` | Path to save distribution plot | Optional |
| `--dataset1-name` | Name for first dataset | Directory name |
| `--dataset2-name` | Name for second dataset | Directory name |

## Programmatic Usage

### Basic Analysis

```python
from dataset_comparison_analyzer import DatasetAnalyzer

# Initialize analyzer
analyzer = DatasetAnalyzer(
    text_key="captions",
    file_pattern="*.pkl"
)

# Analyze datasets
analyzer.analyze_dataset("/path/to/dataset1", "dataset1")
analyzer.analyze_dataset("/path/to/dataset2", "dataset2")

# Print summary
analyzer.print_summary_table()

# Generate plots
analyzer.plot_token_length_distributions(save_path="plot.png")

# Save results
analyzer.save_results("results.json")
```

### Custom Tokenizer

```python
def custom_tokenizer(text):
    """Custom tokenization function."""
    if isinstance(text, (list, tuple)):
        return [str(token) for token in text if token != 0]
    return text.lower().split()

# Use custom tokenizer
analyzer = DatasetAnalyzer(
    text_key="captions",
    tokenizer_func=custom_tokenizer,
    file_pattern="*.pkl"
)
```

### Medical Text Tokenizer Example

```python
import re

def medical_tokenizer(text):
    """Tokenizer optimized for medical text."""
    if isinstance(text, (list, tuple)):
        return [str(token) for token in text if token != 0]
    
    text = text.lower()
    
    # Handle medical abbreviations
    text = re.sub(r'\b(chest|x-ray|xray)\b', 'chest_xray', text)
    text = re.sub(r'\b(no|negative)\b', 'negative', text)
    text = re.sub(r'\b(yes|positive)\b', 'positive', text)
    
    return [token.strip() for token in text.split() if token.strip()]

analyzer = DatasetAnalyzer(
    text_key="captions",
    tokenizer_func=medical_tokenizer
)
```

## Output Format

### Console Output

The analyzer provides a comprehensive summary table:

```
📊 DATASET COMPARISON SUMMARY
================================================================================
Metric                           dataset1    dataset2
-----------------------------------------------------------------
Total Samples                        44047      53450
Valid Samples                        44047      53450
Empty Entries                            0          0
Malformed Entries                       0          0
Vocabulary Size                       9015       4442
Total Tokens                      1682040    2535578
Avg Tokens/Sample                    38.19      47.44
Token Length - Min                    8.00       5.00
Token Length - Max                  128.00     128.00
Token Length - Mean                  38.19      47.44
Token Length - Median                33.00      44.00
Token Length - Std                   19.44      20.62
Token Length - Q25                   25.00      33.00
Token Length - Q75                   47.00      58.00

-----------------------------------------------------------------
Vocab Overlap (Abs)                   4361
Vocab Overlap (%)                    98.18
Vocab Union Size                      9096
Jaccard Similarity                    0.48
```

### JSON Results

The saved JSON file contains detailed statistics:

```json
{
  "analysis_timestamp": "2025-08-19T11:03:47.012519",
  "text_key": "captions",
  "file_pattern": "*.pkl",
  "statistics": {
    "dataset1": {
      "dataset_name": "dataset1",
      "total_samples": 44047,
      "valid_samples": 44047,
      "empty_entries": 0,
      "malformed_entries": 0,
      "vocabulary_size": 9015,
      "token_length_min": 8.0,
      "token_length_max": 128.0,
      "token_length_mean": 38.19,
      "token_length_median": 33.0,
      "token_length_std": 19.44,
      "token_length_q25": 25.0,
      "token_length_q75": 47.0,
      "total_tokens": 1682040,
      "avg_tokens_per_sample": 38.19
    }
  },
  "vocabulary_sizes": {
    "dataset1": 9015,
    "dataset2": 4442
  },
  "comparisons": {
    "vocab_overlap_absolute": 4361,
    "vocab_overlap_percentage": 98.18,
    "vocab_union_size": 9096,
    "vocab1_only_size": 4654,
    "vocab2_only_size": 81,
    "vocab1_unique_percentage": 51.63,
    "vocab2_unique_percentage": 1.82,
    "jaccard_similarity": 0.48
  }
}
```

## Supported File Formats

### Pickle Files (.pkl)
- Supports both list and dictionary formats
- Automatically handles sharded data structures
- Compatible with numpy arrays

### JSON Files (.json)
- Supports single entries or lists of entries
- UTF-8 encoding

### JSONL Files (.jsonl)
- Line-by-line JSON format
- Robust error handling for malformed lines

### Parquet Files (.parquet)
- Requires pyarrow installation
- Efficient for large datasets

## Data Structure Requirements

### Expected Format

Each data entry should be a dictionary containing at least one of these text keys:
- `captions` (default)
- `text`
- `report`
- `caption`
- `findings`
- `impression`
- `summary`

### Example Entry

```python
{
    "images": [...],  # Image data
    "captions": "chest x-ray shows normal cardiac silhouette",  # Text data
    "study_ids": "study_12345",  # Metadata
    "subject_ids": "subject_67890"  # Optional metadata
}
```

## Analysis Metrics

### Basic Statistics
- **Total Samples**: Number of entries in the dataset
- **Valid Samples**: Number of entries with valid text
- **Empty Entries**: Number of entries with missing text
- **Malformed Entries**: Number of entries with invalid text

### Vocabulary Analysis
- **Vocabulary Size**: Number of unique tokens
- **Total Tokens**: Sum of all tokens across the dataset
- **Average Tokens per Sample**: Mean token count per entry

### Token Length Statistics
- **Min/Max**: Minimum and maximum token lengths
- **Mean/Median**: Central tendency measures
- **Standard Deviation**: Dispersion measure
- **Quartiles (Q25/Q75)**: Distribution spread

### Comparison Metrics
- **Vocabulary Overlap**: Number of shared tokens
- **Overlap Percentage**: Percentage relative to smaller vocabulary
- **Jaccard Similarity**: Intersection over union
- **Unique Tokens**: Tokens present in only one dataset

## Visualization

The analyzer generates two types of plots:

1. **Histogram Comparison**: Density distributions of token lengths
2. **Box Plot Comparison**: Statistical summaries of token length distributions

Plots are automatically displayed and can be saved to files.

## Extending the Analyzer

### Adding New File Formats

```python
def _load_custom_format(self, file_path: str) -> List[Dict]:
    """Load data from a custom file format."""
    # Implementation here
    pass

# Register the new loader
analyzer._load_custom_format = _load_custom_format
```

### Custom Text Extraction

```python
def custom_text_extractor(entry: Dict) -> Optional[str]:
    """Custom text extraction logic."""
    # Implementation here
    pass

analyzer.extract_text_from_entry = custom_text_extractor
```

### Custom Statistics

```python
def compute_custom_stats(self, token_lengths: List[int]) -> Dict:
    """Compute custom statistics."""
    return {
        "custom_metric": np.mean(token_lengths) * 2
    }
```

## Performance Considerations

- **Memory Usage**: Large datasets may require significant memory
- **Processing Speed**: Tokenization can be CPU-intensive
- **File I/O**: Reading many small files can be slow
- **Visualization**: Plotting large datasets may be slow

## Troubleshooting

### Common Issues

1. **No files found**: Check file pattern and directory structure
2. **Memory errors**: Process datasets in smaller chunks
3. **Import errors**: Install required dependencies
4. **Plot display issues**: Use `--plot-output` to save plots

### Debug Mode

Enable verbose output by modifying the script:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Examples

See `example_custom_usage.py` for additional examples including:
- Medical text tokenization
- N-gram tokenization
- Custom text extraction
- Multiple analysis configurations

## License

This tool is provided as-is for research and development purposes.

## Contributing

To extend the analyzer:
1. Fork the repository
2. Add new features in a modular way
3. Include tests and documentation
4. Submit a pull request

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the examples
3. Examine the source code
4. Create an issue with detailed information 