#!/usr/bin/env python3
"""
Arthritis Dataset Preparation Script
Extracts and organizes arthritis data from the existing Osteoarthritis Knee X-ray dataset
"""

import os
import pandas as pd
import shutil
from pathlib import Path
import json
from collections import defaultdict

def prepare_arthritis_dataset():
    """
    Prepare dedicated arthritis dataset from existing osteoarthritis data
    """
    print("🔧 Preparing Arthritis Dataset...")
    print("=" * 50)
    
    # Define paths
    base_dir = Path(__file__).parent
    source_dir = base_dir / "Dataset" / "Osteoarthritis Knee X-ray"
    train_csv = source_dir / "Train.csv"
    test_csv = source_dir / "Test.csv"
    train_images = source_dir / "train"
    test_images = source_dir / "test"
    
    # Target directory for arthritis dataset
    arthritis_dir = base_dir / "Dataset" / "arthritis"
    arthritis_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📂 Source directory: {source_dir}")
    print(f"📂 Target directory: {arthritis_dir}")
    
    # Check if source files exist
    if not train_csv.exists():
        print(f"❌ Train CSV not found: {train_csv}")
        return False
    
    if not test_csv.exists():
        print(f"⚠️ Test CSV not found: {test_csv}")
        test_csv = None
    
    # Read training data
    print("\n📊 Reading training data...")
    train_df = pd.read_csv(train_csv)
    print(f"✅ Found {len(train_df)} training samples")
    
    # Read test data if available
    test_df = None
    if test_csv and test_csv.exists():
        print("📊 Reading test data...")
        test_df = pd.read_csv(test_csv)
        print(f"✅ Found {len(test_df)} test samples")
    
    # Analyze class distribution
    print("\n📈 Analyzing class distribution...")
    class_counts = train_df['label'].value_counts().sort_index()
    print("Class distribution:")
    for label, count in class_counts.items():
        class_name = get_class_name(label)
        print(f"  Class {label} ({class_name}): {count} images")
    
    # Create arthritis binary classification
    print("\n🔄 Converting to binary arthritis classification...")
    
    # Map labels: 0 = Normal, 1,2 = Arthritis (different severity levels)
    train_df['arthritis_label'] = train_df['label'].apply(lambda x: 0 if x == 0 else 1)
    
    if test_df is not None:
        test_df['arthritis_label'] = test_df['label'].apply(lambda x: 0 if x == 0 else 1)
    
    # Count arthritis binary classes
    arthritis_counts = train_df['arthritis_label'].value_counts().sort_index()
    print("Arthritis binary distribution:")
    print(f"  Normal (0): {arthritis_counts.get(0, 0)} images")
    print(f"  Arthritis (1): {arthritis_counts.get(1, 0)} images")
    
    total_images = len(train_df) + (len(test_df) if test_df is not None else 0)
    print(f"\n📊 Total images available: {total_images}")
    
    # Create directory structure
    print("\n📁 Creating directory structure...")
    
    # Create train/test directories
    (arthritis_dir / "train" / "normal").mkdir(parents=True, exist_ok=True)
    (arthritis_dir / "train" / "arthritis").mkdir(parents=True, exist_ok=True)
    
    if test_df is not None:
        (arthritis_dir / "test" / "normal").mkdir(parents=True, exist_ok=True)
        (arthritis_dir / "test" / "arthritis").mkdir(parents=True, exist_ok=True)
    
    # Copy training images
    print("\n📋 Copying training images...")
    train_copied = copy_images(train_df, train_images, arthritis_dir / "train")
    
    # Copy test images if available
    test_copied = 0
    if test_df is not None and test_images.exists():
        print("📋 Copying test images...")
        test_copied = copy_images(test_df, test_images, arthritis_dir / "test")
    
    # Create metadata files
    print("\n📄 Creating metadata files...")
    create_metadata_files(arthritis_dir, train_df, test_df, train_copied, test_copied)
    
    # Create summary CSV files
    print("📄 Creating summary CSV files...")
    create_summary_csvs(arthritis_dir, train_df, test_df)
    
    print("\n✅ Arthritis dataset preparation completed!")
    print(f"📊 Training images copied: {train_copied}")
    if test_copied > 0:
        print(f"📊 Test images copied: {test_copied}")
    
    print(f"\n📂 Dataset location: {arthritis_dir}")
    
    return True

def get_class_name(label):
    """Get human-readable class name"""
    class_names = {
        0: "Normal",
        1: "Mild Arthritis", 
        2: "Severe Arthritis"
    }
    return class_names.get(label, f"Unknown ({label})")

def copy_images(df, source_dir, target_dir):
    """Copy images based on DataFrame"""
    copied_count = 0
    
    for idx, row in df.iterrows():
        filename = row['filename']
        arthritis_label = row['arthritis_label']
        
        # Determine target folder
        target_folder = "normal" if arthritis_label == 0 else "arthritis"
        
        source_path = source_dir / filename
        target_path = target_dir / target_folder / filename
        
        if source_path.exists():
            try:
                shutil.copy2(source_path, target_path)
                copied_count += 1
                
                if copied_count % 1000 == 0:
                    print(f"  Copied {copied_count} images...")
                    
            except Exception as e:
                print(f"⚠️ Error copying {filename}: {e}")
        else:
            print(f"⚠️ Source image not found: {source_path}")
    
    return copied_count

def create_metadata_files(arthritis_dir, train_df, test_df, train_copied, test_copied):
    """Create metadata files for the arthritis dataset"""
    
    # Create dataset info
    dataset_info = {
        "name": "Arthritis Binary Classification Dataset",
        "description": "Binary classification dataset for arthritis detection from knee X-rays",
        "source": "Derived from Osteoarthritis Knee X-ray dataset",
        "classes": ["Normal", "Arthritis"],
        "class_mapping": {
            "0": "Normal",
            "1": "Arthritis (includes mild and severe cases)"
        },
        "original_classes": {
            "0": "Normal",
            "1": "Mild Arthritis",
            "2": "Severe Arthritis"
        },
        "total_images": train_copied + test_copied,
        "train_images": train_copied,
        "test_images": test_copied,
        "image_format": "JPG",
        "image_size": "Variable (will be resized to 224x224 for training)",
        "created_date": "2025-10-04",
        "version": "1.0"
    }
    
    # Calculate class distribution
    train_dist = train_df['arthritis_label'].value_counts().sort_index()
    dataset_info["train_distribution"] = {
        "normal": int(train_dist.get(0, 0)),
        "arthritis": int(train_dist.get(1, 0))
    }
    
    if test_df is not None:
        test_dist = test_df['arthritis_label'].value_counts().sort_index()
        dataset_info["test_distribution"] = {
            "normal": int(test_dist.get(0, 0)),
            "arthritis": int(test_dist.get(1, 0))
        }
    
    # Save metadata
    with open(arthritis_dir / "dataset_info.json", 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    # Create README
    readme_content = f"""# Arthritis Binary Classification Dataset

## Overview
This dataset is derived from the Osteoarthritis Knee X-ray dataset and organized for binary arthritis classification.

## Classes
- **Normal (0)**: Normal knee X-rays without arthritis
- **Arthritis (1)**: X-rays showing arthritis (combines mild and severe cases)

## Dataset Statistics
- **Total Images**: {train_copied + test_copied}
- **Training Images**: {train_copied}
- **Test Images**: {test_copied}

## Class Distribution (Training)
- Normal: {train_dist.get(0, 0)} images
- Arthritis: {train_dist.get(1, 0)} images

## Directory Structure
```
arthritis/
├── train/
│   ├── normal/     # Normal knee X-rays
│   └── arthritis/  # Arthritis cases
├── test/           # Test set (if available)
│   ├── normal/
│   └── arthritis/
├── dataset_info.json
├── train_labels.csv
├── test_labels.csv (if available)
└── README.md
```

## Usage
This dataset is ready for binary classification training with frameworks like TensorFlow, PyTorch, etc.

Generated on: 2025-10-04
"""
    
    with open(arthritis_dir / "README.md", 'w') as f:
        f.write(readme_content)

def create_summary_csvs(arthritis_dir, train_df, test_df):
    """Create summary CSV files"""
    
    # Training CSV
    train_summary = train_df[['filename', 'arthritis_label']].copy()
    train_summary.columns = ['filename', 'label']
    train_summary.to_csv(arthritis_dir / "train_labels.csv", index=False)
    
    # Test CSV if available
    if test_df is not None:
        test_summary = test_df[['filename', 'arthritis_label']].copy()
        test_summary.columns = ['filename', 'label']
        test_summary.to_csv(arthritis_dir / "test_labels.csv", index=False)

def main():
    """Main function"""
    try:
        success = prepare_arthritis_dataset()
        if success:
            print("\n🎉 Arthritis dataset preparation successful!")
            print("\n📋 Next steps:")
            print("1. The arthritis dataset is now ready for training")
            print("2. You can train a dedicated arthritis model")
            print("3. Consider data augmentation for better performance")
        else:
            print("\n❌ Arthritis dataset preparation failed!")
            return 1
            
    except Exception as e:
        print(f"\n💥 Error during preparation: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())