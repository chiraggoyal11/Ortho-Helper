#!/usr/bin/env python3
"""
Training Orchestrator Script
Executes both cardiomegaly and bone fracture training with proper environment setup
"""

import os
import sys
import subprocess
import time
import datetime
from pathlib import Path

def setup_environment():
    """Setup the training environment"""
    print("🔧 Setting up training environment...")
    
    # Set TensorFlow environment variables for optimal performance
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logs
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN warnings
    
    # Create necessary directories
    directories = [
        "models/cardiomegaly/checkpoints",
        "models/bone_fracture/checkpoints",
        "models/registry",
        "training_results",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("✅ Environment setup complete")

def run_cardiomegaly_training():
    """Execute cardiomegaly training"""
    print("\\n🏥 Starting Cardiomegaly Training...")
    print("=" * 50)
    
    try:
        # Run the cardiomegaly training script
        result = subprocess.run([
            sys.executable, "train_advanced_cardiomegaly.py"
        ], capture_output=False, text=True, cwd=os.getcwd())
        
        if result.returncode == 0:
            print("✅ Cardiomegaly training completed successfully!")
            return True
        else:
            print(f"❌ Cardiomegaly training failed with code: {result.returncode}")
            return False
            
    except Exception as e:
        print(f"❌ Error running cardiomegaly training: {str(e)}")
        return False

def run_bone_fracture_training():
    """Execute bone fracture training"""
    print("\\n🦴 Starting Bone Fracture Training...")
    print("=" * 50)
    
    try:
        # Run the bone fracture training script
        result = subprocess.run([
            sys.executable, "train_advanced_bone_fracture.py"
        ], capture_output=False, text=True, cwd=os.getcwd())
        
        if result.returncode == 0:
            print("✅ Bone fracture training completed successfully!")
            return True
        else:
            print(f"❌ Bone fracture training failed with code: {result.returncode}")
            return False
            
    except Exception as e:
        print(f"❌ Error running bone fracture training: {str(e)}")
        return False

def activate_best_models():
    """Activate the best trained models"""
    print("\\n🎯 Activating best models...")
    
    try:
        # Import and use the activation manager
        from activate_trained_models import ModelActivationManager
        
        manager = ModelActivationManager()
        model_types = ['cardiomegaly', 'bone_fracture']
        
        results = manager.activate_best_models(model_types)
        
        print("\\n🎉 Model Activation Results:")
        for model_type, result in results.items():
            if result['success']:
                print(f"✅ {model_type}: {result['model_name']} ({result['accuracy']:.1%})")
            else:
                print(f"❌ {model_type}: {result.get('error', 'Failed')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error activating models: {str(e)}")
        return False

def check_dataset_availability():
    """Check if datasets are available for training"""
    print("📊 Checking dataset availability...")
    
    dataset_paths = [
        "Dataset/cardiomegaly",
        "Dataset/Cardiomegaly", 
        "Dataset/ChestXray",
        "Dataset/bone_fracture",
        "Dataset/BoneFracture",
        "Dataset/fracture",
        "Dataset"
    ]
    
    available_datasets = []
    for path in dataset_paths:
        path_obj = Path(path)
        if path_obj.exists() and any(path_obj.iterdir()):
            available_datasets.append(str(path))
            print(f"✅ Found dataset: {path}")
    
    if not available_datasets:
        print("⚠️ WARNING: No datasets found in expected locations")
        print("📁 Expected dataset structure:")
        print("   Dataset/")
        print("   ├── cardiomegaly/")
        print("   │   ├── Normal/")
        print("   │   └── Cardiomegaly/")
        print("   └── bone_fracture/")
        print("       ├── Normal/")
        print("       └── Fracture/")
        print()
        print("🔄 Training will create these directories, but you'll need to add images")
        return False
    
    return True

def main():
    """Main orchestrator function"""
    start_time = datetime.datetime.now()
    
    print("🏥 Advanced Medical AI Training Orchestrator")
    print("🎯 Training: Cardiomegaly + Bone Fracture Models")
    print("=" * 60)
    print(f"🕐 Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Setup environment
    setup_environment()
    
    # Check datasets
    datasets_available = check_dataset_availability()
    if not datasets_available:
        proceed = input("⚠️ Datasets not found. Continue anyway? (y/n): ").lower().strip()
        if proceed != 'y':
            print("🛑 Training cancelled")
            return
    
    print("\\n🚀 Starting Training Pipeline...")
    
    # Track results
    results = {
        'cardiomegaly': False,
        'bone_fracture': False,
        'activation': False
    }
    
    # Run cardiomegaly training
    print(f"\\n⏰ Stage 1/3: Cardiomegaly Training")
    results['cardiomegaly'] = run_cardiomegaly_training()
    
    if results['cardiomegaly']:
        print("✅ Cardiomegaly training successful, proceeding to bone fracture...")
        time.sleep(2)  # Brief pause
    else:
        proceed = input("❌ Cardiomegaly training failed. Continue with bone fracture? (y/n): ").lower().strip()
        if proceed != 'y':
            print("🛑 Training pipeline stopped")
            return
    
    # Run bone fracture training
    print(f"\\n⏰ Stage 2/3: Bone Fracture Training")
    results['bone_fracture'] = run_bone_fracture_training()
    
    if not results['bone_fracture']:
        print("⚠️ Bone fracture training failed, but continuing with activation...")
    
    # Activate best models
    print(f"\\n⏰ Stage 3/3: Model Activation")
    results['activation'] = activate_best_models()
    
    # Final summary
    end_time = datetime.datetime.now()
    duration = end_time - start_time
    
    print("\\n" + "=" * 60)
    print("🎉 TRAINING PIPELINE COMPLETED")
    print("=" * 60)
    print(f"🕐 Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🕑 End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⏱️ Total duration: {duration}")
    print()
    
    # Results summary
    print("📊 Results Summary:")
    for task, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"   {task.replace('_', ' ').title()}: {status}")
    
    total_success = sum(results.values())
    print(f"\\n🎯 Overall Success Rate: {total_success}/3 tasks completed")
    
    if results['cardiomegaly'] or results['bone_fracture']:
        print("\\n🚀 Next Steps:")
        print("1. Check the models/ directory for new trained models")
        print("2. Review training plots and metrics")
        print("3. Test the models in the Streamlit application")
        print("4. Run the application with: D:/Capstone/mynew/capstoneortho/.venv/Scripts/python.exe -m streamlit run app.py")
    
    if results['activation']:
        print("\\n✅ Models are ready to use in the application!")
    else:
        print("\\n⚠️ Manual model activation may be required")
        print("   Run: python activate_trained_models.py")

if __name__ == "__main__":
    main()