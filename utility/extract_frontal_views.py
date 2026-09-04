#!/usr/bin/env python3

import os
import json
import csv
import shutil
from pathlib import Path
import random

def extract_frontal_views():
    """Extract frontal views from ReXGradient dataset with proper organization"""
    
    # Configuration
    metadata_dir = "raw_data_ReXGradient-160K/organized_data/metadata"
    images_dir = "raw_data_ReXGradient-160K/organized_data/images/deid_png"
    reports_dir = "raw_data_ReXGradient-160K/organized_data/reports"
    
    # Output directories
    output_base = "extracted_frontal_views"
    output_images = os.path.join(output_base, "images")
    output_reports = os.path.join(output_base, "reports")
    output_csv = os.path.join(output_base, "frontal_views_metadata.csv")
    
    # Create output directories
    os.makedirs(output_base, exist_ok=True)
    os.makedirs(output_images, exist_ok=True)
    os.makedirs(output_reports, exist_ok=True)
    
    print("="*80)
    print("Frontal View Extraction - ReXGradient Dataset")
    print("="*80)
    
    # Get all metadata files
    metadata_files = [f for f in os.listdir(metadata_dir) if f.endswith('.json')]
    print(f"Total metadata files: {len(metadata_files)}")
    
    # Frontal view study types
    frontal_study_types = [
        'DG CHEST 2V',           # PA frontal + lateral
        'DG CHEST 1V PORT',      # AP frontal (portable)
        'DG CHEST 1V',           # Standard frontal
        'Chest Single AP view',  # Explicit AP frontal
        'PORTABLE CHEST - 1 VIEW', # Portable frontal
        'XR chest 1V portable'   # X-ray portable frontal
    ]
    
    # CSV data
    csv_data = []
    extracted_count = 0
    skipped_count = 0
    
    print(f"Processing metadata files...")
    
    for i, filename in enumerate(metadata_files):
        if i % 1000 == 0:
            print(f"Processed {i}/{len(metadata_files)} files...")
            
        filepath = os.path.join(metadata_dir, filename)
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            study_desc = data.get('study_description', '')
            
            # Check if this is a frontal view study
            is_frontal = any(study_type in study_desc for study_type in frontal_study_types)
            
            if not is_frontal:
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
            view_type = "Unknown"
            if '2V' in study_desc:
                view_type = "PA"  # PA component of two-view study
            elif 'AP' in study_desc:
                view_type = "AP"
            elif 'PORT' in study_desc:
                view_type = "AP"  # Portable studies are usually AP
            elif '1V' in study_desc:
                view_type = "PA"  # Standard single view is usually PA
            
            # Find corresponding image files
            image_files = find_study_images(study_id, images_dir)
            
            if not image_files:
                print(f"Warning: No images found for study {study_id}")
                skipped_count += 1
                continue
            
            # Find corresponding report file
            report_file = find_study_report(study_id, reports_dir)
            
            # Create study directory
            study_dir = os.path.join(output_images, study_id)
            os.makedirs(study_dir, exist_ok=True)
            
            # Copy images (handle multiple images per study)
            copied_images = []
            for img_path in image_files:
                if img_path.endswith('.png'):
                    img_filename = os.path.basename(img_path)
                    dest_path = os.path.join(study_dir, img_filename)
                    
                    # For two-view studies, we need to identify which is frontal
                    if len(image_files) > 1 and '2V' in study_desc:
                        # This is a simplified approach - in reality, you might need
                        # more sophisticated image analysis to distinguish PA from Lateral
                        if 'PA' in img_filename or 'frontal' in img_filename.lower():
                            shutil.copy2(img_path, dest_path)
                            copied_images.append(img_filename)
                    else:
                        # Single view or portable study - copy all images
                        shutil.copy2(img_path, dest_path)
                        copied_images.append(img_filename)
            
            # Copy report if found
            report_dest = None
            if report_file:
                report_dest = os.path.join(output_reports, f"{study_id}_report.txt")
                shutil.copy2(report_file, report_dest)
            
            # Add to CSV data
            csv_data.append({
                'patient_id': study_id.split('_')[0] if '_' in study_id else study_id,
                'study_id': study_id,
                'accession_number': accession_number,
                'patient_sex': patient_sex,
                'patient_age': patient_age,
                'study_date': study_date,
                'study_description': study_desc,
                'view_type': view_type,
                'findings': findings,
                'impression': impression,
                'image_files': ';'.join(copied_images),
                'report_file': os.path.basename(report_dest) if report_dest else '',
                'total_images': len(copied_images)
            })
            
            extracted_count += 1
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            skipped_count += 1
    
    # Write CSV file
    if csv_data:
        fieldnames = [
            'patient_id', 'study_id', 'accession_number', 'patient_sex', 
            'patient_age', 'study_date', 'study_description', 'view_type',
            'findings', 'impression', 'image_files', 'report_file', 'total_images'
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
    print(f"Skipped files: {skipped_count}")
    print(f"Extraction rate: {(extracted_count/len(metadata_files))*100:.1f}%")
    
    print(f"\nOutput locations:")
    print(f"  Images: {output_images}")
    print(f"  Reports: {output_reports}")
    print(f"  Metadata CSV: {output_csv}")
    
    # Show sample of extracted data
    if csv_data:
        print(f"\nSample extracted entries:")
        for i, entry in enumerate(csv_data[:3]):
            print(f"  {i+1}. Study ID: {entry['study_id']}")
            print(f"     View Type: {entry['view_type']}")
            print(f"     Images: {entry['image_files']}")
            print(f"     Description: {entry['study_description']}")
            print()
    
    print("="*80)

def find_study_images(study_id, images_dir):
    """Find image files corresponding to a study ID"""
    image_files = []
    
    # Extract patient ID from study ID
    if '_' in study_id:
        patient_id = study_id.split('_')[0]
    else:
        patient_id = study_id
    
    # Look for patient directory
    patient_dir = os.path.join(images_dir, patient_id)
    if not os.path.exists(patient_dir):
        return image_files
    
    # Search recursively for PNG files
    for root, dirs, files in os.walk(patient_dir):
        for file in files:
            if file.endswith('.png'):
                # Check if this image belongs to the study
                # This is a simplified check - you might need more sophisticated matching
                if study_id in root or any(part in root for part in study_id.split('_')):
                    image_files.append(os.path.join(root, file))
    
    return image_files

def find_study_report(study_id, reports_dir):
    """Find report file corresponding to a study ID"""
    report_filename = f"{study_id}_report.txt"
    report_path = os.path.join(reports_dir, report_filename)
    
    if os.path.exists(report_path):
        return report_path
    return None

if __name__ == "__main__":
    extract_frontal_views() 