import os
import sys
import datetime
from pathlib import Path
import subprocess

def setup_environment():
    """Setup Python environment and validate installation"""
    
    print("🔧 Setting up training environment...")
    
    # Check if we're in a virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ Running in virtual environment")
    else:
        print("⚠️ Not in a virtual environment - this may cause package conflicts")
    
    # Check key dependencies
    required_packages = [
        'tensorflow',
        'numpy',
        'pandas',
        'matplotlib',
        'scikit-learn',
        'seaborn'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} available")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} missing")
    
    if missing_packages:
        print(f"\n⚠️ Missing packages: {missing_packages}")
        print("Please install them using: pip install " + " ".join(missing_packages))
        return False
    
    print("\n🎉 Environment setup complete!")
    return True

def run_cardiomegaly_training():
    """Execute cardiomegaly training"""
    
    print("\n" + "="*70)
    print("🏥 STARTING CARDIOMEGALY TRAINING")
    print("="*70)
    
    try:
        # Import and run the fixed cardiomegaly trainer
        from train_advanced_cardiomegaly_fixed import AdvancedCardiomegalyTrainer
        
        trainer = AdvancedCardiomegalyTrainer()
        model_path, model_id = trainer.run_complete_training()
        
        if model_path and model_id:
            print(f"✅ Cardiomegaly training successful!")
            print(f"📁 Model: {model_path}")
            print(f"🆔 ID: {model_id}")
            return True, model_id, model_path
        else:
            print("❌ Cardiomegaly training failed")
            return False, None, None
            
    except Exception as e:
        print(f"❌ Cardiomegaly training error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, None, None

def run_bone_fracture_training():
    """Execute bone fracture training"""
    
    print("\n" + "="*70)
    print("🦴 STARTING BONE FRACTURE TRAINING")
    print("="*70)
    
    try:
        # Import and run the fixed bone fracture trainer
        from train_advanced_bone_fracture_fixed import AdvancedBoneFractureTrainer
        
        trainer = AdvancedBoneFractureTrainer()
        model_path, model_id = trainer.run_complete_training()
        
        if model_path and model_id:
            print(f"✅ Bone fracture training successful!")
            print(f"📁 Model: {model_path}")
            print(f"🆔 ID: {model_id}")
            return True, model_id, model_path
        else:
            print("❌ Bone fracture training failed")
            return False, None, None
            
    except Exception as e:
        print(f"❌ Bone fracture training error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, None, None

def activate_trained_models(trained_models):
    """Activate the newly trained models"""
    
    print("\n" + "="*70)
    print("🎯 ACTIVATING TRAINED MODELS")
    print("="*70)
    
    try:
        from activate_trained_models_fixed import ModelActivationManager
        
        manager = ModelActivationManager()
        
        activation_results = {}
        
        for model_type, (success, model_id, model_path) in trained_models.items():
            if success and model_id:
                print(f"\n🔄 Activating {model_type} model: {model_id}")
                activation_success = manager.activate_model(model_id, model_type)
                activation_results[model_type] = activation_success
                
                if activation_success:
                    print(f"✅ {model_type} model activated successfully")
                else:
                    print(f"❌ Failed to activate {model_type} model")
            else:
                print(f"⏭️ Skipping {model_type} (training failed)")
                activation_results[model_type] = False
        
        return activation_results
        
    except Exception as e:
        print(f"❌ Model activation error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {}

def print_final_results(trained_models, activation_results):
    """Print comprehensive training results"""
    
    print("\n" + "="*80)
    print("🎉 COMPLETE TRAINING PIPELINE RESULTS")
    print("="*80)
    
    print("\n📊 Training Summary:")
    print("-" * 50)
    
    for model_type, (success, model_id, model_path) in trained_models.items():
        if success:
            activation_status = "✅ Activated" if activation_results.get(model_type) else "❌ Activation Failed"
            print(f"✅ {model_type.upper()}: Training Successful")
            print(f"   🆔 Model ID: {model_id}")
            print(f"   📁 Path: {model_path}")
            print(f"   🎯 Status: {activation_status}")
        else:
            print(f"❌ {model_type.upper()}: Training Failed")
        print()
    
    # Count successes
    training_successes = sum(1 for success, _, _ in trained_models.values() if success)
    activation_successes = sum(1 for result in activation_results.values() if result)
    
    print(f"📈 Training Results: {training_successes}/{len(trained_models)} models trained successfully")
    print(f"🎯 Activation Results: {activation_successes}/{len(activation_results)} models activated successfully")
    
    if training_successes == len(trained_models) and activation_successes == training_successes:
        print("\n🎉 COMPLETE SUCCESS! All models trained and activated!")
        print("🚀 Your medical AI system is ready for deployment!")
    elif training_successes > 0:
        print("\n⚠️ PARTIAL SUCCESS! Some models trained successfully.")
        print("📋 Check individual results above for details.")
    else:
        print("\n❌ TRAINING FAILED! Please check error messages above.")

def main():
    """Main training pipeline execution"""
    
    print("🏥 ADVANCED MEDICAL AI TRAINING PIPELINE")
    print("🎯 Training cardiomegaly and bone fracture detection models")
    print(f"⏰ Started at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Setup environment
    if not setup_environment():
        print("❌ Environment setup failed. Exiting.")
        return
    
    # Track training results
    trained_models = {}
    
    # Execute cardiomegaly training
    trained_models['cardiomegaly'] = run_cardiomegaly_training()
    
    # Execute bone fracture training
    trained_models['bone_fracture'] = run_bone_fracture_training()
    
    # Activate trained models
    activation_results = activate_trained_models(trained_models)
    
    # Print final results
    print_final_results(trained_models, activation_results)
    
    print(f"\n⏰ Training pipeline completed at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()