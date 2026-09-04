#!/usr/bin/env python3

import os
import json
import csv
import shutil
from pathlib import Path
import re

def extract_frontal_views_advanced():
    """Advanced frontal view extraction with better image mapping"""
    
    # Configuration
    metadata_dir = "raw_data_ReXGradient-160K/organized_data/metadata"
    images_dir = "raw_data_ReXGradient-160K/organized_data/images/deid_png"
    reports_dir = "raw_data_ReXGradient-160K/organized_data/reports"
    
    # Output directories
    output_base = "extracted_frontal_views_advanced"
    output_images = os.path.join(output_base, "images")
    output_reports = os.path.join(output_base, "reports")
    output_csv = os.path.join(output_base, "frontal_views_metadata.csv")
    
    # Create output directories
    os.makedirs(output_base, exist_ok=True)
    os.makedirs(output_images, exist_ok=True)
    os.makedirs(output_reports, exist_ok=True)
    
    print("="*80)
    print("Advanced Frontal View Extraction - ReXGradient Dataset")
    print("="*80)
    
    # Get all metadata files
    metadata_files = [f for f in os.listdir(metadata_dir) if f.endswith('.json')]
    print(f"Total metadata files: {len(metadata_files)}")
    
    # Frontal view study types with confidence levels
    frontal_study_types = {
        'high_confidence': [
            'Chest Single AP view',  # Explicitly AP
            'DG CHEST 1V',           # Standard single view (PA)
        ],
        'medium_confidence': [
            'DG CHEST 1V PORT',      # Portable (AP)
            'PORTABLE CHEST - 1 VIEW', # Portable (AP)
            'XR chest 1V portable'   # Portable (AP)
        ],
        'two_view': [
            'DG CHEST 2V',           # PA + Lateral
            'Chest PA and Left Lateral', # PA + Lateral
            'Chest AP Left Lateral', # AP + Lateral
            'CHEST - 2 VIEW'         # Generic two-view
        ]
    }
    
    # CSV data
    csv_data = []
    extracted_count = 0
    skipped_count = 0
    two_view_count = 0
    
    print(f"Processing metadata files...")
    
    for i, filename in enumerate(metadata_files):
        if i % 1000 == 0:
            print(f"Processed {i}/{len(metadata_files)} files...")
            
        filepath = os.path.join(metadata_dir, filename)
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            study_desc = data.get('study_description', '')
            
            # Determine study type and confidence
            study_type = "unknown"
            confidence = "low"
            
            if any(study_type in study_desc for study_type in frontal_study_types['high_confidence']):
                study_type = "single_view"
                confidence = "high"
            elif any(study_type in study_desc for study_type in frontal_study_types['medium_confidence']):
                study_type = "portable"
                confidence = "medium"
            elif any(study_type in study_desc for study_type in frontal_study_types['two_view']):
                study_type = "two_view"
                confidence = "high"
                two_view_count += 1
            else:
                skipped_count += 1
                continue
            
            # Extract study information
            study_id = data.get('study_id', '')
            accession_number = data.get('accession_number', '')
            patient_sex = data.get('patient_sex', '')
            patient_age = data.get('patient_age', '')
            study_date = data.get('study_date', '')
            findings = data.get('findings', '')
            impression = data.get('impression', '')
            
            # Determine view type
            view_type = determine_view_type(study_desc, study_type)
            
            # Find corresponding image files
            image_files = find_study_images_advanced(study_id, images_dir)
            
            if not image_files:
                print(f"Warning: No images found for study {study_id}")
                skipped_count += 1
                continue
            
            # Create study directory
            study_dir = os.path.join(output_images, study_id)
            os.makedirs(study_dir, exist_ok=True)
            
            # Copy images based on study type
            copied_images = []
            if study_type == "two_view":
                # For two-view studies, we need to identify frontal vs lateral
                frontal_images = identify_frontal_images(image_files, study_desc)
                for img_path in frontal_images:
                    img_filename = os.path.basename(img_path)
                    dest_path = os.path.join(study_dir, img_filename)
                    shutil.copy2(img_path, dest_path)
                    copied_images.append(img_filename)
            else:
                # Single view or portable - copy all images
                for img_path in image_files:
                    if img_path.endswith('.png'):
                        img_filename = os.path.basename(img_path)
                        dest_path = os.path.join(study_dir, img_filename)
                        shutil.copy2(img_path, dest_path)
                        copied_images.append(img_filename)
            
            # Copy report if found
            report_file = find_study_report(study_id, reports_dir)
            report_dest = None
            if report_file:
                report_dest = os.path.join(output_reports, f"{study_id}_report.txt")
                shutil.copy2(report_file, report_dest)
            
            # Add to CSV data
            csv_data.append({
                'patient_id': extract_patient_id(study_id),
                'study_id': study_id,
                'accession_number': accession_number,
                'patient_sex': patient_sex,
                'patient_age': patient_age,
                'study_date': study_date,
                'study_description': study_desc,
                'study_type': study_type,
                'view_type': view_type,
                'confidence': confidence,
                'findings': findings,
                'impression': impression,
                'image_files': ';'.join(copied_images),
                'report_file': os.path.basename(report_dest) if report_dest else '',
                'total_images': len(copied_images),
                'original_total_images': len(image_files)
            })
            
            extracted_count += 1
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            skipped_count += 1
    
    # Write CSV file
    if csv_data:
        fieldnames = [
            'patient_id', 'study_id', 'accession_number', 'patient_sex', 
            'patient_age', 'study_date', 'study_description', 'study_type',
            'view_type', 'confidence', 'findings', 'impression', 
            'image_files', 'report_file', 'total_images', 'original_total_images'
        ]
        
        with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_data)
    
    print(f"\n" + "="*60)
    print("EXTRACTION SUMMARY")
    print("="*60)
    print(f"Total metadata files processed: {len(metadata_files)}")
    print(f"Frontal views extracted: {extracted_count}")
    print(f"Two-view studies processed: {two_view_count}")
    print(f"Skipped files: {skipped_count}")
    print(f"Extraction rate: {(extracted_count/len(metadata_files))*100:.1f}%")
    
    # Statistics by study type
    if csv_data:
        study_types = {}
        view_types = {}
        for entry in csv_data:
            study_type = entry['study_type']
            view_type = entry['view_type']
            study_types[study_type] = study_types.get(study_type, 0) + 1
            view_types[view_type] = view_types.get(view_type, 0) + 1
        
        print(f"\nStudy type distribution:")
        for study_type, count in study_types.items():
            print(f"  {study_type}: {count} ({count/len(csv_data)*100:.1f}%)")
        
        print(f"\nView type distribution:")
        for view_type, count in view_types.items():
            print(f"  {view_type}: {count} ({count/len(csv_data)*100:.1f}%)")
    
    print(f"\nOutput locations:")
    print(f"  Images: {output_images}")
    print(f"  Reports: {output_reports}")
    print(f"  Metadata CSV: {output_csv}")
    
    print("="*80)

def determine_view_type(study_desc, study_type):
    """Determine the view type based on study description"""
    study_desc_lower = study_desc.lower()
    
    if 'ap' in study_desc_lower:
        return "AP"
    elif 'pa' in study_desc_lower:
        return "PA"
    elif study_type == "two_view":
        return "PA"  # Two-view studies typically have PA as frontal
    elif study_type == "portable":
        return "AP"  # Portable studies are usually AP
    elif study_type == "single_view":
        return "PA"  # Standard single views are usually PA
    else:
        return "Unknown"

def identify_frontal_images(image_files, study_desc):
    """Identify frontal images from a set of images"""
    frontal_images = []
    
    # This is a simplified approach - in practice, you might need:
    # 1. Image analysis to distinguish PA from Lateral
    # 2. DICOM metadata analysis
    # 3. Machine learning-based view classification
    
    for img_path in image_files:
        img_filename = os.path.basename(img_path)
        
        # Simple heuristics (not perfect but reasonable)
        if 'lateral' in img_filename.lower():
            continue  # Skip lateral images
        elif 'pa' in img_filename.lower():
            frontal_images.append(img_path)
        elif 'ap' in img_filename.lower():
            frontal_images.append(img_path)
        else:
            # If we can't determine, include it (better to include than exclude)
            frontal_images.append(img_path)
    
    # If no frontal images identified, include all (conservative approach)
    if not frontal_images:
        frontal_images = image_files
    
    return frontal_images

def find_study_images_advanced(study_id, images_dir):
    """Advanced image finding with better study mapping"""
    image_files = []
    
    # Extract components from study ID
    # Format: p[PatientID]_a[AccessionNumber]_s[DICOMStudyUID]
    parts = study_id.split('_')
    if len(parts) >= 3:
        patient_id = parts[0]
        accession_number = parts[1]
        dicom_study_uid = parts[2]
    else:
        patient_id = study_id
        accession_number = ""
        dicom_study_uid = ""
    
    # Look for patient directory
    patient_dir = os.path.join(images_dir, patient_id)
    if not os.path.exists(patient_dir):
        return image_files
    
    # Search recursively for PNG files
    for root, dirs, files in os.walk(patient_dir):
        for file in files:
            if file.endswith('.png'):
                # Check if this image belongs to the study
                # Look for DICOM study UID in the path
                if dicom_study_uid and dicom_study_uid in root:
                    image_files.append(os.path.join(root, file))
                # Fallback: check if study ID components are in path
                elif any(part in root for part in [patient_id, accession_number]):
                    image_files.append(os.path.join(root, file))
    
    return image_files

def extract_patient_id(study_id):
    """Extract patient ID from study ID"""
    if '_' in study_id:
        return study_id.split('_')[0]
    return study_id

def find_study_report(study_id, reports_dir):
    """Find report file corresponding to a study ID"""
    report_filename = f"{study_id}_report.txt"
    report_path = os.path.join(reports_dir, report_filename)
    
    if os.path.exists(report_path):
        return report_path
    return None

if __name__ == "__main__":
    extract_frontal_views_advanced() 