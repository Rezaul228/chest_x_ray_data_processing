# MIMIC-CXR Raw Data Structure Analysis

## 📁 Dataset Overview

**Path**: `/home/abedin/Developments/mimic_cxr-raw-data/mimic-cxr`

**Analysis Date**: July 28, 2025

## 🏗️ Directory Structure

```
mimic-cxr/
└── images/
    ├── p10/                    # Patient group 10
    │   ├── p10999395/         # Patient ID
    │   │   ├── s59802033/     # Study ID
    │   │   │   └── 2e9ecc5c-5c8da30e-3589f536-3809cd0f-df6631f4.jpg
    │   │   └── s56459556/
    │   │       └── [image_file].jpg
    │   └── p10998537/
    │       └── s[study_id]/
    │           └── [image_file].jpg
    ├── p11/                    # Patient group 11
    ├── p12/                    # Patient group 12
    └── ...                     # Patient groups 13-19
```

## 📊 Dataset Statistics

### **Patient Organization**
- **Total Patient Groups**: 10 (p10, p11, p12, p13, p14, p15, p16, p17, p18, p19)
- **Patient Groups**: Each group contains multiple patient directories
- **Patient IDs**: Format: `p{patient_number}` (e.g., p10999395, p10998537)

### **Study Organization**
- **Study IDs**: Format: `s{study_number}` (e.g., s59802033, s56459556)
- **Studies per Patient**: Multiple studies per patient
- **Study Structure**: Each study contains one or more image files

### **Image Files**
- **Format**: JPEG (.jpg)
- **Naming**: UUID-based filenames (e.g., `2e9ecc5c-5c8da30e-3589f536-3809cd0f-df6631f4.jpg`)
- **Size**: Various sizes (need to be resized to 224x224 for processing)

## 🔍 Key Observations

### **1. Hierarchical Organization**
```
Patient Group → Patient ID → Study ID → Image Files
     p10      → p10999395 → s59802033 → [image].jpg
```

### **2. UUID-Based Image Naming**
- Images use UUID-based filenames
- No direct correlation between filename and content
- Need to rely on directory structure for identification

### **3. Missing Reports**
- **No reports directory found** in standard locations
- **No text files found** in the dataset
- Reports may be in a separate location or need to be downloaded separately

### **4. Patient-Study Relationship**
- Each patient can have multiple studies
- Each study can have multiple images
- Patient IDs are unique within the dataset

## 🎯 Data Processing Requirements

### **Image Processing**
1. **Resize**: All images need to be resized to 224x224 pixels
2. **Normalize**: Convert to [0,1] range
3. **Format**: Convert to RGB format if needed

### **Text Processing**
1. **Reports**: Need to locate or download report files
2. **Matching**: Match reports with corresponding studies
3. **Tokenization**: Process medical text for training

### **Metadata Requirements**
1. **Patient-Study Mapping**: Track which studies belong to which patients
2. **Image-Report Mapping**: Match images with their corresponding reports
3. **Split Information**: Organize into train/val/test splits

## 📋 Processing Pipeline

### **Step 1: Image Processing**
```python
# Load and preprocess images
for patient_group in patient_groups:
    for patient_id in patient_group:
        for study_id in patient_id:
            for image_file in study_id:
                # Resize to 224x224
                # Normalize to [0,1]
                # Save processed image
```

### **Step 2: Report Processing**
```python
# Need to locate report files
# Match reports with studies
# Process and tokenize text
```

### **Step 3: Data Organization**
```python
# Create shards with aligned data
# Ensure no data leakage between splits
# Save metadata with tokenizer and split information
```

## ⚠️ Important Notes

### **Missing Components**
1. **Reports**: Text files are not present in this dataset
2. **Annotations**: No annotation files found
3. **Metadata**: No additional metadata files

### **Processing Considerations**
1. **Data Leakage**: Ensure patients don't appear in multiple splits
2. **Image Quality**: Check for corrupted or low-quality images
3. **Text Matching**: Need reliable way to match images with reports

### **Dataset Completeness**
- **Images**: ✅ Present (JPEG format)
- **Reports**: ❌ Missing (need to locate separately)
- **Annotations**: ❌ Missing
- **Metadata**: ❌ Missing (need to create)

## 🔧 Next Steps

1. **Locate Reports**: Find or download report files
2. **Create Metadata**: Build patient-study-image mappings
3. **Process Images**: Resize and normalize all images
4. **Match Data**: Align images with corresponding reports
5. **Create Splits**: Organize into train/val/test without leakage
6. **Build Tokenizer**: Create vocabulary from medical reports
7. **Generate Shards**: Create processed data files

## 📈 Dataset Statistics Summary

| Component | Status | Count |
|-----------|--------|-------|
| Patient Groups | ✅ Found | 10 |
| Patient Directories | ✅ Found | Multiple per group |
| Study Directories | ✅ Found | Multiple per patient |
| Image Files | ✅ Found | Multiple per study |
| Report Files | ❌ Missing | 0 |
| Metadata Files | ❌ Missing | 0 |

This dataset contains the **image component** of MIMIC-CXR but is missing the **text reports** that are essential for multimodal training. 