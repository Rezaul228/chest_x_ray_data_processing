# Data Extraction, Image-Text Pairing, and Train/Test/Val Split Analysis

## 📊 **Core Data Flow Overview**

### **1. Data Source Structure**
```
Raw MIMIC-CXR Data:
├── metadata.csv (contains study_id, image_file, report_file, hybrid_split)
├── reports/ (text files with findings + impressions)
└── images/ (chest X-ray image files)
```

### **2. Data Extraction Logic**

#### **Step 1: Metadata Loading**
```python
# Load metadata CSV containing study information
self.metadata_df = pd.read_csv(self.metadata_csv_path)
# Contains: study_id, image_file, report_file, hybrid_split
```

#### **Step 2: Study Entry Processing**
```python
for _, row in self.metadata_df.iterrows():
    study_id = row['study_id']
    image_file = row['image_file']
    report_file = row['report_file']
    
    # Construct file paths
    image_path = os.path.join(self.images_dir, image_file)
    report_path = os.path.join(self.reports_dir, report_file)
    
    # Extract text from report (findings + impressions)
    findings, impression = extract_text_from_report(report_path)
    
    # Create study entry if valid
    if findings or impression:
        self.study_entries.append({
            'study_id': study_id,
            'image_path': image_path,
            'findings': findings,
            'impression': impression,
            'combined_text': combined_text,
            'hybrid_split': row.get('hybrid_split', 'train')
        })
```

#### **Step 3: Text Extraction Function**
```python
def extract_text_from_report(report_path):
    """Extract findings and impression from MIMIC-CXR report file"""
    with open(report_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract findings section using regex
    findings_match = re.search(r'FINDINGS:(.*?)(?:IMPRESSION:|$)', content, re.DOTALL | re.IGNORECASE)
    findings = findings_match.group(1).strip() if findings_match else ""
    
    # Extract impression section using regex
    impression_match = re.search(r'IMPRESSION:(.*?)$', content, re.DOTALL | re.IGNORECASE)
    impression = impression_match.group(1).strip() if impression_match else ""
    
    return findings, impression
```

## 🔗 **Image-Text Pairing Logic**

### **Pairing Strategy:**
1. **One-to-One Mapping**: Each study has exactly one image and one report
2. **Study ID as Key**: `study_id` serves as the unique identifier linking image and text
3. **Flexible Text Inclusion**: Accepts studies with either findings OR impression (not requiring both)

### **Pairing Process:**
```python
# For each study entry:
for entry in self.study_entries:
    # Load image
    image = self.load_and_preprocess_image(entry['image_path'])
    
    # Get text (findings + impression combined)
    combined_text = entry['findings'] + ' ' + entry['impression']
    
    # Tokenize text
    sequences = self.tokenizer.texts_to_sequences([combined_text.strip()])
    caption_seq = pad_sequences(sequences, maxlen=self.max_sequence_length, padding='post')[0]
    
    # Store paired data
    images_list.append(image)
    captions_list.append(caption_seq)
    study_ids_list.append(str(entry['study_id']))
```

### **Data Validation:**
- **Image Validation**: Checks if image file exists at `image_path`
- **Text Validation**: Checks if report file exists and contains valid text
- **Pairing Validation**: Only creates pairs where both image and text are valid

## 📈 **Train/Test/Val Split Logic**

### **Split Strategy: Hybrid MIMIC-CXR Split**

#### **Key Characteristics:**
1. **Pre-defined Split**: Uses `hybrid_split` column from metadata (not random splitting)
2. **Split Values**: `'train'`, `'validate'`, `'test'`
3. **No Data Leakage**: Split is determined at the study level, not patient level
4. **Consistent Splits**: Same split used across all processing runs

#### **Split Implementation:**
```python
def create_shards_with_test_split(self):
    """Create sharded data files using hybrid MIMIC-CXR split"""
    
    # Initialize split containers
    train_entries = []
    val_entries = []
    test_entries = []
    
    # Categorize entries by hybrid_split
    for entry in self.study_entries:
        hybrid_split = entry.get('hybrid_split', 'train')
        
        if hybrid_split == 'train':
            train_entries.append(entry)
        elif hybrid_split == 'validate':
            val_entries.append(entry)
        elif hybrid_split == 'test':
            test_entries.append(entry)
        else:
            # Default to train for unknown splits
            train_entries.append(entry)
    
    # Create shards for each split
    self._create_shards_for_split(train_entries, self.train_shard_dir, "train")
    self._create_shards_for_split(val_entries, self.val_shard_dir, "val")
    self._create_shards_for_split(test_entries, self.test_shard_dir, "test")
```

### **Split Statistics:**
```python
print(f"Hybrid split data: {len(train_entries)} training samples, "
      f"{len(val_entries)} validation samples, {len(test_entries)} test samples")
```

## 📦 **Shard Creation Logic**

### **Shard Structure:**
```python
# Each shard contains:
shard_data = {
    'images': np.array(images_list, dtype=np.float32),      # (N, 224, 224, 3)
    'captions': np.array(captions_list, dtype=np.int32),    # (N, max_seq_length)
    'study_ids': np.array(study_ids_list, dtype='<U50')     # (N,) string array
}
```

### **Shard Creation Process:**
```python
def _create_shards_for_split(self, entries, shard_dir, split_name):
    """Create sharded pickle files for a specific data split"""
    
    # Process entries in batches of shard_size
    for shard_idx in range(0, len(entries), self.shard_size):
        shard_entries = entries[shard_idx:shard_idx+self.shard_size]
        
        # Initialize data containers
        images_list = []
        captions_list = []
        study_ids_list = []
        
        # Process each entry in the shard
        for entry in shard_entries:
            # Load and process image
            image = self.load_and_preprocess_image(entry['image_path'])
            
            if image is not None:
                # Get tokenized caption
                if 'caption_seq' in entry:
                    caption_seq = entry['caption_seq']
                else:
                    # Process text if not already done
                    combined_text = entry['findings'] + ' ' + entry['impression']
                    sequences = self.tokenizer.texts_to_sequences([combined_text.strip()])
                    caption_seq = pad_sequences(sequences, maxlen=self.max_sequence_length, padding='post')[0]
                
                # Store paired data
                images_list.append(image)
                captions_list.append(caption_seq)
                study_ids_list.append(str(entry['study_id']))
        
        # Save shard if it contains data
        if images_list:
            shard_data = {
                'images': np.array(images_list, dtype=np.float32),
                'captions': np.array(captions_list, dtype=np.int32),
                'study_ids': np.array(study_ids_list, dtype='<U50')
            }
            
            # Save to pickle file
            shard_path = os.path.join(shard_dir, f'shard_{shard_idx//self.shard_size:04d}.pkl')
            with open(shard_path, 'wb') as f:
                pickle.dump(shard_data, f, protocol=pickle.HIGHEST_PROTOCOL)
```

## 🔄 **Data Loading Logic**

### **Shard Loading Process:**
```python
def _get_data_from_shards(self, shard_dir, num_samples, split_name):
    """Load data from shards in the specified directory"""
    
    # Find all shard files
    shard_files = sorted(glob.glob(os.path.join(shard_dir, "*.pkl")))
    
    # Initialize data containers
    all_images = []
    all_captions = []
    all_study_ids = []
    
    # Load data from each shard
    for shard_file in shard_files:
        with open(shard_file, 'rb') as f:
            shard_data = pickle.load(f)
        
        all_images.append(shard_data['images'])
        all_captions.append(shard_data['captions'])
        all_study_ids.extend(shard_data['study_ids'])
    
    # Concatenate all data
    images = np.concatenate(all_images, axis=0)
    captions = np.concatenate(all_captions, axis=0)
    study_ids = np.array(all_study_ids, dtype='<U50')
    
    return {
        'images': images,
        'captions': captions,
        'study_ids': study_ids,
        'tokenizer': self.tokenizer,
        'vocab_size': len(self.tokenizer.word_index) + 1
    }
```

## 📊 **Key Data Flow Summary**

### **1. Input Data Sources:**
- **Metadata CSV**: Contains study_id, image_file, report_file, hybrid_split
- **Report Files**: Text files with FINDINGS and IMPRESSION sections
- **Image Files**: Chest X-ray images (various formats)

### **2. Data Extraction:**
- **Text Extraction**: Regex-based parsing of FINDINGS and IMPRESSION sections
- **Image Loading**: PIL-based image loading and resizing to 224x224
- **Pairing**: Study ID-based one-to-one mapping of images and texts

### **3. Data Splitting:**
- **Split Method**: Pre-defined hybrid_split from metadata (not random)
- **Split Ratios**: Determined by metadata, typically ~70% train, ~15% val, ~15% test
- **Split Level**: Study level (not patient level)

### **4. Data Storage:**
- **Format**: Pickle files with numpy arrays
- **Structure**: Each shard contains images, captions, and study_ids
- **Organization**: Separate directories for train/val/test splits

### **5. Data Loading:**
- **Method**: Load and concatenate shards from respective directories
- **Output**: Numpy arrays with paired image-text data
- **Access**: Separate methods for train/val/test data loading

## 🎯 **Critical Design Decisions**

### **1. Hybrid Split Usage:**
- **Why**: Ensures consistent splits across experiments
- **Benefit**: Reproducible research results
- **Alternative**: Random splitting could lead to different splits each time

### **2. Flexible Text Inclusion:**
- **Why**: Some reports may have only findings or only impressions
- **Benefit**: Maximizes data utilization
- **Alternative**: Requiring both sections would reduce dataset size

### **3. Shard-based Storage:**
- **Why**: Memory-efficient loading of large datasets
- **Benefit**: Can load subsets of data without loading entire dataset
- **Alternative**: Single large file would require loading everything

### **4. Study ID as Key:**
- **Why**: Unique identifier for each image-text pair
- **Benefit**: Enables tracking and debugging of specific samples
- **Alternative**: Index-based identification would be less traceable

This data extraction and processing pipeline ensures efficient, reproducible, and scalable handling of the MIMIC-CXR dataset for machine learning applications. 