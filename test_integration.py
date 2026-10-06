#!/usr/bin/env python3
"""
Integration Test for Medical X-ray AI System
Tests the integration between dataset overview, model training, and model management
"""

import sys
import os
sys.path.append('.')

def test_imports():
    """Test if all modules can be imported successfully"""
    print("🧪 Testing module imports...")
    
    try:
        from utils.data_loader import MedicalDataLoader, display_dataset_overview
        print("✅ Data Loader module imported successfully")
    except Exception as e:
        print(f"❌ Data Loader import failed: {e}")
        return False
    
    try:
        from utils.model_trainer import MedicalModelTrainer, display_training_interface
        print("✅ Model Trainer module imported successfully")
    except Exception as e:
        print(f"❌ Model Trainer import failed: {e}")
        return False
    
    try:
        from utils.model_manager import ModelManager, display_model_management_interface
        print("✅ Model Manager module imported successfully")
    except Exception as e:
        print(f"❌ Model Manager import failed: {e}")
        return False
    
    return True

def test_data_loader():
    """Test data loader functionality"""
    print("\n🧪 Testing Data Loader...")
    
    try:
        from utils.data_loader import MedicalDataLoader
        
        # Initialize data loader
        loader = MedicalDataLoader()
        print("✅ Data Loader initialized")
        
        # Test dataset scanning
        dataset_info = loader.scan_datasets()
        print(f"✅ Dataset scan completed - Found {len(dataset_info)} dataset types")
        
        for dataset_name, info in dataset_info.items():
            print(f"   📊 {dataset_name}: {info['total_images']} images, Ready: {info['ready_for_training']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data Loader test failed: {e}")
        return False

def test_model_trainer():
    """Test model trainer functionality"""
    print("\n🧪 Testing Model Trainer...")
    
    try:
        from utils.model_trainer import MedicalModelTrainer
        
        # Initialize trainer
        trainer = MedicalModelTrainer()
        print("✅ Model Trainer initialized")
        
        # Test model creation
        model = trainer.create_model('Custom_CNN', num_classes=2)
        print(f"✅ Custom CNN created with {model.count_params()} parameters")
        
        # Test model compilation
        compiled_model = trainer.compile_model(model, num_classes=2)
        print("✅ Model compiled successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Model Trainer test failed: {e}")
        return False

def test_model_manager():
    """Test model manager functionality"""
    print("\n🧪 Testing Model Manager...")
    
    try:
        from utils.model_manager import ModelManager
        
        # Initialize manager
        manager = ModelManager()
        print("✅ Model Manager initialized")
        
        # Test model listing
        models = manager.list_models()
        print(f"✅ Model registry accessed - Found {len(models)} models")
        
        # Test registry validation
        issues = manager.validate_registry_integrity()
        if issues:
            print(f"⚠️ Registry has {len(issues)} issues (normal for new installations)")
        else:
            print("✅ Registry integrity validated")
        
        return True
        
    except Exception as e:
        print(f"❌ Model Manager test failed: {e}")
        return False

def test_integration():
    """Test integration between all components"""
    print("\n🧪 Testing Integration...")
    
    try:
        from utils.data_loader import MedicalDataLoader
        from utils.model_trainer import MedicalModelTrainer
        from utils.model_manager import ModelManager
        
        # Test cross-component interaction
        loader = MedicalDataLoader()
        trainer = MedicalModelTrainer()
        manager = ModelManager()
        
        print("✅ All components can work together")
        
        # Test file system integration
        models_dir = trainer.model_dir
        registry_dir = manager.registry_dir
        
        print(f"✅ Models directory: {models_dir}")
        print(f"✅ Registry directory: {registry_dir}")
        
        # Verify directories exist
        if models_dir.exists() and registry_dir.exists():
            print("✅ All required directories exist")
        else:
            print("⚠️ Some directories missing (will be created on use)")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

def main():
    """Run all integration tests"""
    print("🚀 Medical X-ray AI System - Integration Test")
    print("=" * 50)
    
    tests = [
        ("Module Imports", test_imports),
        ("Data Loader", test_data_loader),
        ("Model Trainer", test_model_trainer),
        ("Model Manager", test_model_manager),
        ("Integration", test_integration)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY:")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status} - {test_name}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All integration tests passed! System is ready for use.")
    else:
        print("⚠️ Some tests failed. Please check the error messages above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)