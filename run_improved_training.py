#!/usr/bin/env python3
"""
Comprehensive Improved Training Orchestrator
Runs optimized bone fracture and cardiomegaly training with better performance
"""

import os
import sys
import json
import time
import subprocess
from datetime import datetime
import tensorflow as tf

def check_environment():
    """Check training environment and requirements"""
    print("Checking training environment...")
    
    # Check TensorFlow and GPU
    print(f"   TensorFlow version: {tf.__version__}")
    
    gpu_available = len(tf.config.experimental.list_physical_devices('GPU')) > 0
    print(f"   GPU available: {'Yes' if gpu_available else 'No (CPU only)'}")
    
    # Check datasets
    datasets = {
        'Bone Fracture': 'Dataset/ARM/MURA_Organized/Forearm',
        'Cardiomegaly': 'Dataset/CHEST/cardiomelgy/train/train'
    }
    
    for name, path in datasets.items():
        exists = os.path.exists(path)
        print(f"   {name} dataset: {'Found' if exists else 'Missing'}")
        if not exists:
            print(f"      Expected path: {path}")
            return False
    
    # Check model directories
    os.makedirs('models/bone_fracture', exist_ok=True)
    os.makedirs('models/cardiomegaly', exist_ok=True)
    print("   Model directories: Ready")
    
    return True

def run_training_script(script_name, task_name):
    """Run a training script and capture results"""
    print(f"\n{'='*60}")
    print(f"Starting {task_name}")
    print(f"Script: {script_name}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # Run the training script
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=True,
            text=True,
            cwd=os.getcwd()
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n{task_name} Results:")
        print(f"   Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
        print(f"   Exit Code: {result.returncode}")
        
        if result.returncode == 0:
            print(f"   Status: SUCCESS")
            
            # Extract key metrics from output
            output_lines = result.stdout.split('\n')
            for line in output_lines[-20:]:  # Check last 20 lines for metrics
                if 'Final Test Accuracy:' in line:
                    print(f"   {line.strip()}")
                elif 'Best Validation Accuracy:' in line:
                    print(f"   {line.strip()}")
                elif 'Model ID:' in line:
                    print(f"   {line.strip()}")
        else:
            print(f"   Status: FAILED")
            print(f"   Error Output:")
            print(result.stderr[-1000:] if result.stderr else "No error output")
        
        return {
            'success': result.returncode == 0,
            'duration': duration,
            'stdout': result.stdout,
            'stderr': result.stderr
        }
        
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"   Status: EXCEPTION")
        print(f"   Error: {str(e)}")
        print(f"   Duration: {duration:.1f} seconds")
        
        return {
            'success': False,
            'duration': duration,
            'error': str(e)
        }

def analyze_training_results():
    """Analyze and summarize all training results"""
    print("\n" + "="*70)
    print("📊 COMPREHENSIVE TRAINING ANALYSIS")
    print("="*70)
    
    # Load model registry
    registry_path = 'model_registry.json'
    if not os.path.exists(registry_path):
        print("❌ No model registry found")
        return
    
    with open(registry_path, 'r') as f:
        registry = json.load(f)
    
    models = registry.get('models', {})
    if not models:
        print("❌ No models found in registry")
        return
    
    # Filter recent models (last 2 hours)
    recent_models = []
    current_time = datetime.now()
    
    for model_id, model_info in models.items():
        try:
            created_at = datetime.fromisoformat(model_info['created_at'])
            time_diff = (current_time - created_at).total_seconds() / 3600  # hours
            
            if time_diff < 2:  # Last 2 hours
                recent_models.append((model_id, model_info))
        except:
            continue
    
    if not recent_models:
        print("⚠️  No recent models found (last 2 hours)")
        return
    
    print(f"🔍 Found {len(recent_models)} recently trained models:")
    print()
    
    # Analyze each model
    bone_fracture_best = None
    cardiomegaly_best = None
    
    for model_id, model_info in recent_models:
        model_type = model_info.get('type', 'unknown')
        model_name = model_info.get('name', 'Unknown')
        metrics = model_info.get('metrics', {})
        accuracy = metrics.get('accuracy', 0)
        
        print(f"📋 {model_name}")
        print(f"   🆔 ID: {model_id}")
        print(f"   🏗️ Architecture: {model_info.get('architecture', 'unknown')}")
        print(f"   📈 Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        if metrics:
            precision = metrics.get('precision', 0)
            recall = metrics.get('recall', 0)
            f1_score = metrics.get('f1_score', 0)
            
            print(f"   🎯 Precision: {precision:.4f}")
            print(f"   🔍 Recall: {recall:.4f}")
            print(f"   📊 F1-Score: {f1_score:.4f}")
            
            # Medical-specific metrics
            if 'sensitivity' in metrics:
                print(f"   🏥 Sensitivity: {metrics['sensitivity']:.4f}")
            if 'specificity' in metrics:
                print(f"   🏥 Specificity: {metrics['specificity']:.4f}")
        
        print(f"   ⏰ Created: {model_info.get('created_at', 'unknown')}")
        print()
        
        # Track best models
        if model_type == 'bone_fracture':
            if bone_fracture_best is None or accuracy > bone_fracture_best[1]['metrics'].get('accuracy', 0):
                bone_fracture_best = (model_id, model_info)
        elif model_type == 'cardiomegaly':
            if cardiomegaly_best is None or accuracy > cardiomegaly_best[1]['metrics'].get('accuracy', 0):
                cardiomegaly_best = (model_id, model_info)
    
    # Performance summary
    print("🏆 PERFORMANCE SUMMARY:")
    print("-" * 40)
    
    # Bone fracture analysis
    if bone_fracture_best:
        bf_id, bf_info = bone_fracture_best
        bf_acc = bf_info['metrics'].get('accuracy', 0)
        target_bf = 0.90
        
        print(f"🦴 Best Bone Fracture Model:")
        print(f"   Accuracy: {bf_acc:.4f} ({bf_acc*100:.2f}%)")
        print(f"   Target: {target_bf*100:.0f}%")
        
        if bf_acc >= target_bf:
            print(f"   Status: ✅ TARGET ACHIEVED!")
        elif bf_acc >= 0.85:
            print(f"   Status: ⚠️  Close to target")
        else:
            print(f"   Status: ❌ Needs improvement")
    else:
        print(f"🦴 Bone Fracture: ❌ No recent models found")
    
    print()
    
    # Cardiomegaly analysis
    if cardiomegaly_best:
        cm_id, cm_info = cardiomegaly_best
        cm_acc = cm_info['metrics'].get('accuracy', 0)
        target_cm = 0.97
        
        print(f"❤️  Best Cardiomegaly Model:")
        print(f"   Accuracy: {cm_acc:.4f} ({cm_acc*100:.2f}%)")
        print(f"   Target: {target_cm*100:.0f}%")
        
        if cm_acc >= target_cm:
            print(f"   Status: ✅ TARGET ACHIEVED!")
        elif cm_acc >= 0.95:
            print(f"   Status: ⚠️  Close to target")
        else:
            print(f"   Status: ❌ Needs improvement")
    else:
        print(f"❤️  Cardiomegaly: ❌ No recent models found")
    
    print()
    
    # Overall assessment
    bf_success = bone_fracture_best and bone_fracture_best[1]['metrics'].get('accuracy', 0) >= 0.90
    cm_success = cardiomegaly_best and cardiomegaly_best[1]['metrics'].get('accuracy', 0) >= 0.97
    
    print("🎯 OVERALL PROJECT STATUS:")
    print("-" * 30)
    
    if bf_success and cm_success:
        print("🎉 EXCELLENT: Both models achieved their targets!")
        print("   ✅ Bone fracture detection: Ready for deployment")
        print("   ✅ Cardiomegaly detection: Ready for deployment")
    elif bf_success or cm_success:
        print("⚠️  PARTIAL SUCCESS: One model achieved target")
        if bf_success:
            print("   ✅ Bone fracture detection: Target achieved")
            print("   ❌ Cardiomegaly detection: Needs improvement")
        else:
            print("   ❌ Bone fracture detection: Needs improvement")
            print("   ✅ Cardiomegaly detection: Target achieved")
    else:
        print("❌ NEEDS IMPROVEMENT: Both models below target")
        print("   Recommendations:")
        print("   1. Increase training time/epochs")
        print("   2. Collect more training data")
        print("   3. Try different architectures")
        print("   4. Adjust hyperparameters")

def main():
    """Main orchestrator function"""
    print("COMPREHENSIVE IMPROVED MODEL TRAINING")
    print("=" * 70)
    print(f"Training session started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Environment check
    if not check_environment():
        print("Environment check failed. Please fix issues before training.")
        return False
    
    print("Environment check passed!")
    
    # Training configuration
    training_tasks = [
        {
            'script': 'improved_bone_fracture_training.py',
            'name': 'Improved Bone Fracture Detection',
            'target_accuracy': 90,
            'priority': 1
        },
        {
            'script': 'improved_cardiomegaly_training.py', 
            'name': 'Improved Cardiomegaly Detection',
            'target_accuracy': 97,
            'priority': 2
        }
    ]
    
    # Execute training tasks
    results = {}
    total_start_time = time.time()
    
    for task in training_tasks:
        script_path = task['script']
        
        if not os.path.exists(script_path):
            print(f"Script not found: {script_path}")
            results[task['name']] = {'success': False, 'error': 'Script not found'}
            continue
        
        # Run training
        result = run_training_script(script_path, task['name'])
        results[task['name']] = result
        
        # Short break between trainings
        if result['success']:
            print(f"{task['name']} completed successfully")
        else:
            print(f"{task['name']} failed")
            print("   Consider checking the error logs before proceeding")
        
        print("\nWaiting 30 seconds before next training...")
        time.sleep(30)
    
    # Calculate total time
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    
    print(f"\n{'='*70}")
    print("TRAINING SESSION SUMMARY")
    print(f"{'='*70}")
    print(f"Total Duration: {total_duration:.1f} seconds ({total_duration/60:.1f} minutes)")
    
    # Task summary
    successful_tasks = sum(1 for r in results.values() if r['success'])
    total_tasks = len(results)
    
    print(f"Task Success Rate: {successful_tasks}/{total_tasks} ({successful_tasks/total_tasks*100:.1f}%)")
    
    for task_name, result in results.items():
        status = "SUCCESS" if result['success'] else "FAILED"
        duration = result.get('duration', 0)
        print(f"   {task_name}: {status} ({duration:.1f}s)")
    
    # Analyze results
    if successful_tasks > 0:
        time.sleep(2)  # Brief pause before analysis
        analyze_training_results()
    
    print(f"\nTraining session completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if successful_tasks == total_tasks:
        print("All training tasks completed successfully!")
        return True
    else:
        print(f"{total_tasks - successful_tasks} training task(s) failed")
        return False

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Training orchestrator failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)