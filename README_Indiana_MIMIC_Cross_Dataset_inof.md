Generate augment visualization : source /opt/conda/etc/profile.d/conda.sh && conda activate multi_pytorch && python create_detailed_visualization.py

# Indiana University Dataset with MIMIC-CXR Vocabulary

This setup allows you to process Indiana University chest X-ray data using the MIMIC-CXR vocabulary and tokenizer, enabling cross-dataset evaluation of your MIMIC-trained models.

## Overview

The key innovation here is using the same vocabulary and preprocessing pipeline for both MIMIC-CXR and Indiana University datasets, ensuring:

1. **Consistent tokenization**: Same vocabulary, same token indices
2. **Compatible data format**: Same sequence length (128), same preprocessing
3. **Cross-dataset evaluation**: Test MIMIC-trained models on Indiana data
4. **Vocabulary coverage analysis**: Understand how well MIMIC vocabulary covers Indiana text

## Files

### Core Files
- `data_set_loader_simplified.py` - Modified Indiana dataset loader that uses MIMIC vocabulary
- `process_indiana_simplified.py` - Main processing script for Indiana data with MIMIC vocabulary
- `test_indiana_mimic_compatibility.py` - Test script to verify compatibility

### MIMIC Vocabulary Files (Required)
- `mimic_frontal_complete_vocab_vocab.json` - MIMIC vocabulary mapping
- `mimic_frontal_complete_vocab_index_word.json` - MIMIC index-to-word mapping

## Usage

### Step 1: Test Compatibility

First, test if your Indiana data can be processed with the MIMIC vocabulary:

```bash
python test_indiana_mimic_compatibility.py
```

This will:
- Load the MIMIC vocabulary
- Test processing a small sample of Indiana data
- Report vocabulary coverage statistics
- Verify image loading and shard creation

### Step 2: Process Full Dataset

Once compatibility is confirmed, process the full Indiana dataset:

```bash
python process_indiana_simplified.py \
    --reports_csv /path/to/indiana_reports.csv \
    --projections_csv /path/to/indiana_projections.csv \
    --image_dir /path/to/indiana_images \
    --output_dir shards_indiana_mimic \
    --vocab_path mimic_frontal_complete_vocab_vocab.json \
    --index_word_path mimic_frontal_complete_vocab_index_word.json
```

### Step 3: Use for Model Evaluation

The processed Indiana data will be compatible with your MIMIC-trained models:

```python
# Load Indiana test data
from data_set_loader_simplified import IndianaDatasetLoaderSimplified

loader = IndianaDatasetLoaderSimplified(
    reports_csv_path='indiana_reports.csv',
    projections_csv_path='indiana_projections.csv', 
    image_dir='indiana_images',
    vocab_path='mimic_frontal_complete_vocab_vocab.json',
    index_word_path='mimic_frontal_complete_vocab_index_word.json',
    skip_metadata_processing=True  # Load existing shards
)

# Get test data for evaluation
test_images, test_captions, test_study_ids = loader.get_test_data()

# Use with your MIMIC-trained model
predictions = model.predict(test_images, test_captions)
```

## Key Features

### 1. MIMIC Vocabulary Integration
- Uses the exact same vocabulary as MIMIC-CXR processing
- Maintains token consistency across datasets
- Reports vocabulary coverage statistics

### 2. Cross-Dataset Compatibility
- Same sequence length (128 tokens)
- Same preprocessing pipeline
- Same data format (shards)
- Compatible with MIMIC-trained models

### 3. Vocabulary Coverage Analysis
The system reports:
- Total tokens processed
- UNK token count
- Vocabulary coverage percentage
- Sample decoded text for verification

### 4. Data Integrity
- Patient-level splitting to prevent data leakage
- Proper train/validation/test splits
- Image and text alignment maintained

## Expected Vocabulary Coverage

Based on medical domain similarity, you should expect:
- **High coverage (80-95%)**: Medical terms, anatomical structures, findings
- **Medium coverage (60-80%)**: Common medical phrases, report templates
- **Lower coverage (40-60%)**: Institution-specific terminology, unique abbreviations

## Output Structure

```
shards_indiana_mimic/
├── train/
│   ├── shard_0.pkl
│   ├── shard_1.pkl
│   └── ...
├── val/
│   ├── shard_0.pkl
│   └── ...
├── test/
│   ├── shard_0.pkl
│   └── ...
└── metadata.pkl
```

Each shard contains:
- `images`: (N, 224, 224, 3) normalized image arrays
- `captions`: (N, 128) tokenized text sequences using MIMIC vocabulary
- `study_ids`: (N,) study identifier strings

## Research Benefits

### 1. Cross-Dataset Evaluation
Test your MIMIC-trained model's generalization to a different dataset with:
- Different patient population
- Different imaging protocols
- Different reporting styles

### 2. Model Robustness Assessment
- Evaluate performance degradation across datasets
- Identify dataset-specific biases
- Assess model generalization capabilities

### 3. Comparative Analysis
- Compare performance on MIMIC vs Indiana
- Analyze vocabulary coverage impact
- Study cross-institutional differences

## Troubleshooting

### Low Vocabulary Coverage
If vocabulary coverage is below 60%:
1. Check if Indiana text preprocessing matches MIMIC
2. Consider expanding MIMIC vocabulary with Indiana terms
3. Verify text cleaning is consistent

### Tokenization Errors
If tokenization fails:
1. Ensure MIMIC vocabulary files are accessible
2. Check text encoding (should be UTF-8)
3. Verify vocabulary file format

### Image Loading Issues
If images fail to load:
1. Check image file paths
2. Verify image formats (JPG, PNG)
3. Ensure sufficient disk space

## Example Output

```
Indiana Dataset Processing - MIMIC Vocabulary for Cross-Dataset Evaluation
================================================================================
Reports CSV: /path/to/indiana_reports.csv
Projections CSV: /path/to/indiana_projections.csv
Image Directory: /path/to/indiana_images
Output Directory: shards_indiana_mimic
Max Studies: All
Max Sequence Length: 128
Shard Size: 100
Train/Val/Test Split: 80.0%/10.0%/10.0%
Random Seed: 42
Skip Metadata: False
MIMIC Vocabulary: mimic_frontal_complete_vocab_vocab.json
MIMIC Index Word: mimic_frontal_complete_vocab_index_word.json
================================================================================

Loading MIMIC vocabulary from: mimic_frontal_complete_vocab_vocab.json
Loaded MIMIC vocabulary with 10804 tokens
Loaded 7,470 study entries
Vocabulary size: 10805
Number of labels: 15
MIMIC vocabulary coverage on Indiana data: 87.3%
Total tokens: 1,245,890, UNK tokens: 158,234

Creating shards with MIMIC vocabulary...
✓ Training data loaded successfully: (5, 224, 224, 3)
✓ Validation data loaded successfully: (5, 224, 224, 3)
✓ Test data loaded successfully: (5, 224, 224, 3)

PROCESSING COMPLETE - MIMIC Vocabulary Cross-Dataset Evaluation
================================================================================
Total processing time: 45.23 seconds
Vocabulary size: 10805
UNK token ratio in training sample: 12.7%

Key features for cross-dataset evaluation:
- Uses MIMIC-CXR vocabulary and tokenizer
- Same preprocessing as MIMIC-CXR (sequence length 128)
- Compatible with MIMIC-trained models
- Enables direct cross-dataset comparison
- Maintains vocabulary coverage statistics

Output directory: shards_indiana_mimic
================================================================================
```

## Next Steps

1. **Run compatibility test** to verify setup
2. **Process full dataset** with MIMIC vocabulary
3. **Evaluate your model** on Indiana test set
4. **Compare performance** between MIMIC and Indiana
5. **Analyze results** for research paper

This setup provides a robust foundation for cross-dataset evaluation in medical image-text analysis research. 