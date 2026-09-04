# ReXGradient-160K Dataset Structure Analysis

## Overview
The ReXGradient-160K dataset is a large-scale chest X-ray dataset containing 140,000 samples with images, metadata, and radiology reports. The dataset is organized in a structured format that follows DICOM-like hierarchy.

## Dataset Statistics
- **Total Samples**: 140,000
- **Patient Directories**: 80,680
- **Metadata Files**: 140,000 JSON files
- **Report Files**: 140,000 TXT files
- **Image Archive**: 101 GB (deid_png.tar)
- **Source**: https://huggingface.co/datasets/rajpurkarlab/ReXGradient-160K

## Directory Structure

```
raw_data_ReXGradient-160K/organized_data/
├── images/
│   ├── deid_png/                    # 80,680 patient directories
│   │   ├── GRDN000VSZVX0XQ2/       # Patient ID
│   │   │   ├── GRDNK9Z7WOP2E290/   # Study ID
│   │   │   │   └── studies/
│   │   │   │       └── [DICOM Study UID]/
│   │   │   │           └── series/
│   │   │   │               └── [DICOM Series UID]/
│   │   │   │                   └── instances/
│   │   │   │                       └── [DICOM Instance UID].png
│   │   ├── example/
│   │   └── .cache/
├── metadata/                        # 140,000 JSON files
├── reports/                         # 140,000 TXT files
├── deid_png.tar                     # 101 GB image archive
├── dataset_info.json
└── download_summary.json
```

## Image Structure

### Hierarchy
The images follow a DICOM-like hierarchy:
1. **Patient Level**: Each patient has a unique GRDN ID
2. **Study Level**: Each patient can have multiple studies
3. **Series Level**: Each study can have multiple series
4. **Instance Level**: Each series contains the actual PNG images

### Image Format
- **Format**: PNG files
- **Size**: ~400-500 KB per image
- **Naming**: Uses DICOM Instance UIDs as filenames
- **Content**: De-identified chest X-ray images

## Metadata Structure

Each metadata file (JSON) contains:

```json
{
  "study_id": "p[PatientID]_a[AccessionNumber]_s[DICOMStudyUID]",
  "accession_number": "GRDN[AccessionID]",
  "report_filename": "[study_id]_report.txt",
  "patient_sex": "M/F",
  "patient_age": "XXXY" (age in years) or "XXXD" (age in days),
  "study_date": YYYYMMDD,
  "study_description": "DG CHEST 1V/2V/PORT",
  "indication": "Clinical indication for the study",
  "comparison": "Comparison studies if any",
  "findings": "Radiological findings text",
  "impression": "Radiologist's impression",
  "dataset_index": [0-139999]
}
```

### Key Fields
- **Patient Demographics**: Sex, age
- **Study Information**: Date, description, indication
- **Clinical Text**: Findings and impression sections
- **Identifiers**: Accession number, study ID
- **Dataset Index**: Sequential index (0-139,999)

## Report Structure

Each report file (TXT) contains:

```
Accession Number: [AccessionID]
Study Description: [Description]
Patient Sex: [M/F]
Patient Age: [Age]
Study Date: [Date]
Indication: [Clinical Indication]
Comparison: [Comparison Studies]

Findings:
[Radiological findings text]

Impression:
[Radiologist's impression text]
```

### Text Characteristics
- **Average Findings Length**: ~130-200 characters
- **Average Impression Length**: ~20-40 characters
- **Format**: Structured text with clear sections
- **Content**: Clinical radiology reports

## Data Correlations

### Perfect Matching
- **Metadata Files**: 140,000
- **Report Files**: 140,000
- **Matching Pairs**: 140,000 (100% correlation)

Each metadata file has a corresponding report file with the same base name:
- Metadata: `[study_id]_metadata.json`
- Report: `[study_id]_report.txt`

### Image-Metadata Mapping
The image paths can be reconstructed from metadata:
- Patient ID from study_id prefix
- Study/Series/Instance UIDs from the DICOM hierarchy
- Images stored as PNG files with DICOM Instance UIDs as names

## Clinical Information

### Study Types
- **DG CHEST 1V**: Single view chest X-ray
- **DG CHEST 2V**: Two view chest X-ray (PA + Lateral)
- **DG CHEST 1V PORT**: Portable single view chest X-ray

### Patient Demographics
- **Age Range**: From newborns (XXXD) to elderly (XXXY)
- **Sex Distribution**: Both male and female patients
- **Study Dates**: Span multiple years

### Clinical Conditions
Based on sample reports, the dataset includes:
- Pneumonia
- Cardiomegaly
- Pulmonary edema
- Pleural effusions
- Fractures
- Atelectasis
- Hiatal hernia
- Sarcoidosis
- And many other chest pathologies

## Usage Notes

1. **Image Access**: Images are stored in a deep directory structure following DICOM hierarchy
2. **Text Processing**: Reports contain structured clinical text suitable for NLP tasks
3. **Metadata**: Rich metadata enables filtering by demographics, study type, and clinical conditions
4. **De-identification**: All data is de-identified for privacy compliance
5. **Scale**: Large dataset suitable for deep learning training

## File Naming Convention

### Metadata/Report Files
```
p[PatientID]_a[AccessionNumber]_s[DICOMStudyUID]_[type].json/txt
```

### Image Files
```
[DICOMInstanceUID].png
```

This structure provides a comprehensive chest X-ray dataset with rich clinical annotations, suitable for various medical AI applications including image-text understanding, report generation, and clinical decision support systems. 