#!/usr/bin/env python3
"""
Test script to verify the dataset structure and code compatibility
after reorganizing the Dataset folder into ARM/CHEST/KNEE structure
"""

import os
import sys
from pathlib import Path

def test_dataset_structure():
    """Test the new dataset structure"""
    print("=" * 60)
    print("🔍 DATASET STRUCTURE VERIFICATION")
    print("=" * 60)
    
    base_path = Path("Dataset")
    
    if not base_path.exists():
        print("❌ Dataset folder not found!")
        return False
    
    # Expected structure
    expected_structure = {
        "ARM": ["MURA_Organized"],
        "CHEST": ["cardiomelgy", "Pneumonia_Organized"],
        "KNEE": ["Osteoarthritis", "Osteoporosis"]
    }
    
    all_good = True
    
    for category, expected_folders in expected_structure.items():
        category_path = base_path / category
        print(f"\n📁 Checking {category} folder...")
        
        if not category_path.exists():
            print(f"❌ {category} folder not found!")
            all_good = False
            continue
        
        print(f"✅ {category} folder exists")
        
        # Check subfolders
        for folder in expected_folders:
            folder_path = category_path / folder
            if folder_path.exists():
                print(f"  ✅ {folder} found")
                
                # Count images in each folder
                image_count = count_images(folder_path)
                print(f"    📊 Contains {image_count} image files")
            else:
                print(f"  ❌ {folder} not found!")
                all_good = False
    
    return all_good

def count_images(path):
    """Count image files recursively"""
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.dcm', '.dicom']
    count = 0
    
    for ext in image_extensions:
        count += len(list(path.rglob(f"*{ext}")))
        count += len(list(path.rglob(f"*{ext.upper()}")))
    
    return count

def test_data_loader():
    """Test the data loader with new structure"""
    print("\n" + "=" * 60)
    print("🧪 DATA LOADER COMPATIBILITY TEST")
    print("=" * 60)
    
    try:
        from utils.data_loader import MedicalDataLoader
        
        print("✅ Data loader imported successfully")
        
        # Create loader instance
        loader = MedicalDataLoader()
        print("✅ Data loader initialized")
        
        # Scan datasets
        print("\n📊 Scanning datasets...")
        dataset_info = loader.scan_datasets()
        
        if not dataset_info:
            print("❌ No datasets found!")
            return False
        
        # Check each dataset
        for dataset_name, info in dataset_info.items():
            print(f"\n📁 Dataset: {dataset_name}")
            print(f"  📊 Total images: {info['total_images']}")
            print(f"  📂 Sources found: {len(info['sources_found'])}")
            print(f"  🏷️ Classes: {list(info['class_distribution'].keys())}")
            print(f"  ✅ Ready for training: {info['ready_for_training']}")
        
        print("\n✅ Data loader test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Data loader test failed: {str(e)}")
        return False

def test_model_loading():
    """Test model loading functionality"""
    print("\n" + "=" * 60)
    print("🤖 MODEL LOADING TEST")
    print("=" * 60)
    
    try:
        from utils.model_inference import load_models
        
        print("✅ Model inference module imported successfully")
        
        # Test model loading
        print("🔄 Loading models...")
        models = load_models()
        
        if models:
            print("✅ Models loaded successfully!")
            print(f"📊 Found {len(models)} models:")
            for model_name in models.keys():
                print(f"  • {model_name}")
        else:
            print("⚠️ No models loaded - this might be expected if models haven't been trained yet")
        
        return True
        
    except Exception as e:
        print(f"❌ Model loading test failed: {str(e)}")
        return False

def test_app_imports():
    """Test main app imports"""
    print("\n" + "=" * 60)
    print("📱 MAIN APP COMPATIBILITY TEST")
    print("=" * 60)
    
    try:
        # Test critical imports
        print("🔄 Testing imports...")
        
        import streamlit as st
        print("✅ Streamlit imported")
        
        from utils.data_loader import display_dataset_overview, MedicalDataLoader
        print("✅ Data loader utilities imported")
        
        from utils.model_inference import load_models
        print("✅ Model inference imported")
        
        print("✅ All critical imports successful!")
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {str(e)}")
        return False

def check_model_files():
    """Check if model files exist"""
    print("\n" + "=" * 60)
    print("🏗️ MODEL FILES CHECK")
    print("=" * 60)
    
    models_path = Path("models")
    
    if not models_path.exists():
        print("❌ Models folder not found!")
        return False
    
    # Expected model files
    expected_models = [
        "bone_fracture_model.h5",
        "cardiomegaly_binary_model.h5",
        "cardiomegaly_DenseNet121_model.h5",
        "chest_conditions_DenseNet121_model.h5",
        "knee_conditions_DenseNet121_model.h5"
    ]
    
    found_models = []
    missing_models = []
    
    for model_file in expected_models:
        model_path = models_path / model_file
        if model_path.exists():
            found_models.append(model_file)
            print(f"✅ {model_file}")
        else:
            missing_models.append(model_file)
            print(f"❌ {model_file} - Missing")
    
    print(f"\n📊 Summary: {len(found_models)} found, {len(missing_models)} missing")
    
    return len(found_models) > 0

def main():
    """Run all tests"""
    print("🚀 COMPLETE SYSTEM VERIFICATION")
    print("Testing compatibility after Dataset folder reorganization")
    print("Date:", "October 6, 2025")
    
    tests = [
        ("Dataset Structure", test_dataset_structure),
        ("Data Loader", test_data_loader),
        ("Model Files", check_model_files),
        ("Model Loading", test_model_loading),
        ("App Imports", test_app_imports)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {str(e)}")
            results.append((test_name, False))
    
    # Final summary
    print("\n" + "=" * 60)
    print("📊 FINAL SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Your code is compatible with the new dataset structure.")
    elif passed >= total * 0.8:
        print("⚠️ Most tests passed. Minor issues may need attention.")
    else:
        print("❌ Several tests failed. Please check the issues above.")
    
    print("\n💡 Recommendations:")
    if not results[0][1]:  # Dataset structure failed
        print("• Fix the dataset folder organization")
    if not results[1][1]:  # Data loader failed
        print("• Update data loader paths")
    if not results[2][1]:  # Model files failed
        print("• Train or download the required models")
    if not results[3][1] or not results[4][1]:  # Model/app issues
        print("• Check dependencies and imports")
    
    print("• Run the Streamlit app and test each feature manually")

if __name__ == "__main__":
    main()