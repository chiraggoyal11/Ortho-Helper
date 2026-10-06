#!/usr/bin/env python3
"""Check cardiomegaly class distribution"""

import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from utils.data_loader import MedicalDataLoader
import numpy as np

def check_class_distribution():
    """Check the class distribution after normalization"""
    
    print("🧪 Checking cardiomegaly class distribution...")
    
    data_loader = MedicalDataLoader()
    
    # Test cardiomegaly dataset preparation
    dataset_info = data_loader.prepare_dataset('cardiomegaly', test_size=0.2, val_size=0.1)
    
    if dataset_info and len(dataset_info) > 0:
        y_train = dataset_info['y_train']
        y_val = dataset_info['y_val'] 
        y_test = dataset_info['y_test']
        label_encoder = dataset_info['label_encoder']
        
        print(f"\n🏷️  Label mappings:")
        for i, class_name in enumerate(label_encoder.classes_):
            print(f"  {i}: {class_name}")
        
        print(f"\n📊 Training set distribution:")
        unique, counts = np.unique(y_train, return_counts=True)
        for cls, count in zip(unique, counts):
            class_name = label_encoder.classes_[cls]
            percentage = (count / len(y_train)) * 100
            print(f"  {class_name}: {count} samples ({percentage:.1f}%)")
        
        print(f"\n📊 Validation set distribution:")
        unique, counts = np.unique(y_val, return_counts=True)
        for cls, count in zip(unique, counts):
            class_name = label_encoder.classes_[cls]
            percentage = (count / len(y_val)) * 100
            print(f"  {class_name}: {count} samples ({percentage:.1f}%)")
        
        print(f"\n📊 Test set distribution:")
        unique, counts = np.unique(y_test, return_counts=True)
        for cls, count in zip(unique, counts):
            class_name = label_encoder.classes_[cls]
            percentage = (count / len(y_test)) * 100
            print(f"  {class_name}: {count} samples ({percentage:.1f}%)")
            
        return True
    else:
        print("❌ Failed to get dataset info!")
        return False

if __name__ == "__main__":
    check_class_distribution()