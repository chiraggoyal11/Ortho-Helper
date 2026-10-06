#!/usr/bin/env python3
"""
System Configuration Checker for Medical X-ray AI System
Validates integration between dataset overview, model training, and model management
"""

import os
import sys
import json
from pathlib import Path

def check_directory_structure():
    """Check if all required directories exist"""
    print("📁 Checking Directory Structure...")
    
    required_dirs = [
        "Dataset",
        "models",
        "models/registry",
        "models/active",
        "models/backups",
        "models/dataset_info",
        "utils"
    ]
    
    missing_dirs = []
    
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            missing_dirs.append(dir_path)
        else:
            print(f"   ✅ {dir_path}")
    
    if missing_dirs:
        print("\n   ⚠️ Missing directories:")
        for missing_dir in missing_dirs:
            print(f"      📂 {missing_dir}")
        print("\n   💡 These directories will be created automatically when needed.")
    
    return len(missing_dirs) == 0

def check_dataset_availability():
    """Check dataset availability"""
    print("\n📊 Checking Dataset Availability...")
    
    dataset_base = Path("Dataset")
    if not dataset_base.exists():
        print("   ❌ Dataset directory not found")
        return False
    
    expected_datasets = {
        'bone_fracture': [
            'ARM/FracAtlas',
            'Bone_Fracture_Binary_Classification'
        ],
        'chest_conditions': [
            'CHEST/cardiomelgy',
            'CHEST/chest_xray Pneumonia'
        ],
        'knee_conditions': [
            'KNEE/Knee Osteoarthritis Dataset with Severity Grading',
            'KNEE/Multi-Class Knee Osteoporosis X-Ray Dataset - OS Collected Data',
            'KNEE/Osteoarthritis Knee X-ray',
            'KNEE/Osteoporosis Knee'
        ]
    }
    
    available_datasets = {}
    
    for dataset_type, paths in expected_datasets.items():
        available_sources = []
        for path in paths:
            full_path = dataset_base / path
            if full_path.exists():
                available_sources.append(path)
        
        available_datasets[dataset_type] = available_sources
        
        if available_sources:
            print(f"   ✅ {dataset_type}: {len(available_sources)}/{len(paths)} sources available")
            for source in available_sources:
                print(f"      📂 {source}")
        else:
            print(f"   ❌ {dataset_type}: No sources available")
    
    total_available = sum(len(sources) for sources in available_datasets.values())
    total_expected = sum(len(paths) for paths in expected_datasets.values())
    
    print(f"\n   📊 Overall: {total_available}/{total_expected} dataset sources available")
    
    return total_available > 0

def check_model_registry():
    """Check model registry status"""
    print("\n🔧 Checking Model Registry...")
    
    registry_file = Path("models/registry/model_registry.json")
    
    if not registry_file.exists():
        print("   ⚠️ Model registry not yet initialized")
        print("   💡 Registry will be created when first model is trained")
        return True
    
    try:
        with open(registry_file, 'r') as f:
            registry = json.load(f)
        
        models_count = len(registry.get('models', {}))
        active_models = registry.get('active_models', {})
        active_count = sum(1 for v in active_models.values() if v is not None)
        
        print(f"   ✅ Registry loaded: {models_count} models registered")
        print(f"   ✅ Active models: {active_count}")
        
        if models_count > 0:
            print("   📋 Registered models:")
            for model_id, model_info in registry['models'].items():
                model_name = model_info.get('model_name', model_id)
                dataset_type = model_info.get('dataset_type', 'unknown')
                accuracy = model_info.get('accuracy', model_info.get('performance_metrics', {}).get('test_accuracy', 'N/A'))
                print(f"      🤖 {model_name} ({dataset_type}) - Accuracy: {accuracy}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Registry error: {e}")
        return False

def check_module_imports():
    """Check if all required modules can be imported"""
    print("\n📦 Checking Module Imports...")
    
    modules_to_test = [
        ("utils.data_loader", "MedicalDataLoader"),
        ("utils.model_trainer", "MedicalModelTrainer"),
        ("utils.model_manager", "ModelManager"),
        ("utils.image_preprocessing", "preprocess_image"),
        ("utils.model_inference", "load_models"),
        ("utils.gradcam", "generate_gradcam_heatmap")
    ]
    
    import_results = {}
    
    for module_name, class_name in modules_to_test:
        try:
            module = __import__(module_name, fromlist=[class_name])
            getattr(module, class_name)
            print(f"   ✅ {module_name}.{class_name}")
            import_results[module_name] = True
        except Exception as e:
            print(f"   ❌ {module_name}.{class_name}: {e}")
            import_results[module_name] = False
    
    successful_imports = sum(1 for success in import_results.values() if success)
    total_imports = len(import_results)
    
    print(f"\n   📊 Import Success: {successful_imports}/{total_imports}")
    
    return successful_imports == total_imports

def check_tensorflow_setup():
    """Check TensorFlow setup"""
    print("\n🤖 Checking TensorFlow Setup...")
    
    try:
        import tensorflow as tf
        print(f"   ✅ TensorFlow version: {tf.__version__}")
        
        # Check for GPU availability
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            print(f"   ✅ GPU available: {len(gpus)} device(s)")
            for i, gpu in enumerate(gpus):
                print(f"      🔥 GPU {i}: {gpu.name}")
        else:
            print("   ⚠️ No GPU detected - training will use CPU")
        
        # Test basic model creation
        test_model = tf.keras.Sequential([
            tf.keras.layers.Dense(10, activation='relu', input_shape=(5,)),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        print("   ✅ Model creation test passed")
        
        return True
        
    except Exception as e:
        print(f"   ❌ TensorFlow error: {e}")
        return False

def generate_status_report():
    """Generate comprehensive status report"""
    print("\n" + "="*60)
    print("📋 INTEGRATION STATUS REPORT")
    print("="*60)
    
    checks = [
        ("Directory Structure", check_directory_structure),
        ("Dataset Availability", check_dataset_availability),
        ("Model Registry", check_model_registry),
        ("Module Imports", check_module_imports),
        ("TensorFlow Setup", check_tensorflow_setup)
    ]
    
    results = []
    
    for check_name, check_func in checks:
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print(f"\n❌ {check_name} check failed: {e}")
            results.append((check_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("📊 SUMMARY:")
    
    passed = 0
    for check_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status} - {check_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall Status: {passed}/{len(results)} checks passed")
    
    if passed == len(results):
        print("\n🎉 System Integration: READY")
        print("   All components are properly integrated and ready for use!")
    elif passed >= len(results) - 1:
        print("\n⚠️ System Integration: MOSTLY READY")
        print("   System is functional with minor issues that can be resolved during use.")
    else:
        print("\n❌ System Integration: NEEDS ATTENTION")
        print("   Please address the failed checks before using the system.")
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS:")
    
    if not results[0][1]:  # Directory structure
        print("   • Run the application once to auto-create missing directories")
    
    if not results[1][1]:  # Dataset availability
        print("   • Download and place medical imaging datasets in the Dataset folder")
        print("   • Use the Dataset Overview page to prepare datasets for training")
    
    if not results[3][1]:  # Module imports
        print("   • Install missing dependencies using: pip install -r requirements.txt")
    
    if not results[4][1]:  # TensorFlow setup
        print("   • Reinstall TensorFlow: pip install tensorflow")
        print("   • For GPU support: pip install tensorflow-gpu")
    
    print("\n🚀 Next Steps:")
    print("   1. Address any failed checks above")
    print("   2. Start the Streamlit application: streamlit run app.py")
    print("   3. Use Dataset Overview to prepare your medical imaging data")
    print("   4. Train models using the Model Training interface")
    print("   5. Manage trained models using the Model Management interface")

def main():
    """Main function"""
    print("🔍 Medical X-ray AI System - Integration Configuration Check")
    generate_status_report()

if __name__ == "__main__":
    main()