#!/usr/bin/env python3
"""Test cardiomegaly dataset preparation"""

import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from utils.data_loader import MedicalDataLoader

def test_cardiomegaly_preparation():
    """Test cardiomegaly dataset preparation"""
    
    print("🧪 Testing cardiomegaly dataset preparation...")
    
    # Initialize data loader
    data_loader = MedicalDataLoader()
    
    # Test cardiomegaly dataset preparation
    try:
        dataset_info = data_loader.prepare_dataset('cardiomegaly', test_size=0.2, val_size=0.1)
        
        if dataset_info and len(dataset_info) > 0:
            print("✅ Cardiomegaly dataset preparation successful!")
            print(f"📊 Dataset info:")
            for key, value in dataset_info.items():
                if key == 'class_distribution':
                    print(f"  {key}: {value}")
                elif isinstance(value, (list, tuple)):
                    print(f"  {key}: {len(value)} items")
                else:
                    print(f"  {key}: {value}")
                    
            # Check class distributions
            if 'class_distribution' in dataset_info:
                classes = dataset_info['class_distribution']
                print(f"\n🏷️  Class distribution:")
                for class_name, count in classes.items():
                    print(f"  - {class_name}: {count} samples")
                    
        else:
            print("❌ Cardiomegaly dataset preparation failed!")
            return False
            
    except Exception as e:
        print(f"❌ Error during cardiomegaly dataset preparation: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def check_dataset_structure():
    """Check the actual dataset structure"""
    
    print("\n🔍 Checking dataset structure...")
    
    base_path = Path("Dataset/cardiomelgy")
    
    if not base_path.exists():
        print(f"❌ Base path {base_path} does not exist!")
        return False
        
    print(f"✅ Base path exists: {base_path}")
    
    # Check train structure
    train_path = base_path / "train" / "train"
    if train_path.exists():
        print(f"✅ Train path exists: {train_path}")
        
        false_path = train_path / "false" 
        true_path = train_path / "true"
        
        if false_path.exists():
            false_count = len(list(false_path.glob("*.*")))
            print(f"✅ False class: {false_count} files")
        else:
            print(f"❌ False class path missing: {false_path}")
            
        if true_path.exists():
            true_count = len(list(true_path.glob("*.*")))
            print(f"✅ True class: {true_count} files")
        else:
            print(f"❌ True class path missing: {true_path}")
    else:
        print(f"❌ Train path missing: {train_path}")
    
    # Check test structure
    test_path = base_path / "test" / "test"
    if test_path.exists():
        print(f"✅ Test path exists: {test_path}")
        
        false_path = test_path / "false"
        true_path = test_path / "true"
        
        if false_path.exists():
            false_count = len(list(false_path.glob("*.*")))
            print(f"✅ Test False class: {false_count} files")
        else:
            print(f"❌ Test False class path missing: {false_path}")
            
        if true_path.exists():
            true_count = len(list(true_path.glob("*.*")))
            print(f"✅ Test True class: {true_count} files")
        else:
            print(f"❌ Test True class path missing: {true_path}")
    else:
        print(f"❌ Test path missing: {test_path}")
    
    return True

if __name__ == "__main__":
    print("=" * 50)
    print("🫀 Cardiomegaly Dataset Test")
    print("=" * 50)
    
    # Check dataset structure first
    check_dataset_structure()
    
    # Test dataset preparation
    success = test_cardiomegaly_preparation()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 All tests passed! Cardiomegaly dataset is ready!")
    else:
        print("❌ Tests failed! Please check the dataset configuration.")
    print("=" * 50)