#!/bin/bash

# Step 1: Process Indiana Dataset with Minimal Cleaning
# This script processes Indiana data with 80/10/10 split and minimal text cleaning

echo "=========================================="
echo "Step 1: Indiana Dataset Processing"
echo "=========================================="

# Set your paths here
REPORTS_CSV="indiana_reports.csv"
PROJECTIONS_CSV="indiana_projections.csv"
IMAGE_DIR="images_normalized"
OUTPUT_DIR="step1_processed_data"

# Check if input files exist
if [ ! -f "$REPORTS_CSV" ]; then
    echo "Error: Reports CSV file not found: $REPORTS_CSV"
    exit 1
fi

if [ ! -f "$PROJECTIONS_CSV" ]; then
    echo "Error: Projections CSV file not found: $PROJECTIONS_CSV"
    exit 1
fi

if [ ! -d "$IMAGE_DIR" ]; then
    echo "Error: Image directory not found: $IMAGE_DIR"
    exit 1
fi

echo "Input files found:"
echo "  Reports CSV: $REPORTS_CSV"
echo "  Projections CSV: $PROJECTIONS_CSV"
echo "  Image Directory: $IMAGE_DIR"
echo "  Output Directory: $OUTPUT_DIR"
echo ""

echo "Processing settings:"
echo "  - Train/Val/Test split: 80%/10%/10%"
echo "  - Minimal text cleaning (only removes XXXX, NA, etc.)"
echo "  - Sequence length: 128 (matches MIMIC-CXR)"
echo "  - Original vocabulary (no augmentation yet)"
echo ""

# Run the processing
echo "Starting Indiana dataset processing..."
python backup_20250713_184341/process_indiana_simplified.py \
    --reports_csv "$REPORTS_CSV" \
    --projections_csv "$PROJECTIONS_CSV" \
    --image_dir "$IMAGE_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --seed 42

echo ""
echo "=========================================="
echo "Step 1 Complete!"
echo "=========================================="
echo ""
echo "Output structure:"
echo "  $OUTPUT_DIR/"
echo "    ├── train/           # 80% of data"
echo "    ├── val/             # 10% of data"
echo "    ├── test/            # 10% of data"
echo "    └── metadata.pkl     # Tokenizer and metadata"
echo ""
echo "Next steps:"
echo "  1. Check vocabulary size and UNK token rate"
echo "  2. Build augmentation vocabulary (if needed)"
echo "  3. Apply augmentation to each split" 