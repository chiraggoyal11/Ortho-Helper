#!/usr/bin/env python3
"""
Quick verification script to check if new models are activated and working
"""

import os
import sys
import json
import tensorflow as tf
from datetime import datetime

# Add the project directory to path
sys.path.append('.')

def load_registry():
    """Load model registry"""
    try:
        with open('models/registry/model_registry.json', 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Failed to load registry: {e}")
        return None

def check_active_models():
    """Check if active models are properly configured and accessible"""
    print("🔍 Checking Model Activation Status...")
    print("=" * 50)
    
    registry = load_registry()
    if not registry:
        return
    
    active_models = registry.get('active_models', {})
    models_info = registry.get('models', {})
    
    print(f"📅 Registry Last Modified: {registry.get('last_modified', 'Unknown')}")
    print()
    
    for model_type, model_id in active_models.items():
        print(f"🔬 {model_type.upper()}:")
        
        if model_id is None:
            print(f"   ❌ No active model configured")
            continue
            
        if model_id not in models_info:
            print(f"   ❌ Active model '{model_id}' not found in registry")
            continue
            
        model_info = models_info[model_id]
        
        # Check if model file exists
        model_file = model_info.get('file_path', '')
        if model_file.startswith('models\\registry\\'):
            model_path = model_file
        else:
            model_path = f"models/registry/{model_file.split('/')[-1]}"
            
        if os.path.exists(model_path):
            file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
            
            # Try to load the model
            try:
                model = tf.keras.models.load_model(model_path)
                accuracy = model_info.get('performance_metrics', {}).get('test_accuracy', 
                          model_info.get('accuracy', 'N/A'))
                
                if isinstance(accuracy, float):
                    accuracy_str = f"{accuracy:.1%}"
                else:
                    accuracy_str = str(accuracy)
                
                print(f"   ✅ ACTIVE: {model_info.get('model_name', model_id)}")
                print(f"   📊 Accuracy: {accuracy_str}")
                print(f"   🏗️  Architecture: {model_info.get('architecture', 'Unknown')}")
                print(f"   📁 File: {model_path} ({file_size_mb:.1f} MB)")
                print(f"   📅 Trained: {model_info.get('trained_date', model_info.get('created_date', 'Unknown'))}")
                
                # Check input shape
                if hasattr(model, 'input_shape'):
                    print(f"   🔧 Input Shape: {model.input_shape}")
                    
                del model  # Free memory
                
            except Exception as e:
                print(f"   ⚠️  Model file exists but failed to load: {str(e)}")
                
        else:
            print(f"   ❌ Model file not found: {model_path}")
        
        print()

def check_model_loading_function():
    """Test the actual model loading function from the system"""
    print("🧪 Testing System Model Loading...")
    print("=" * 50)
    
    try:
        # Import the actual model loading function
        from utils.model_inference import load_models
        
        print("📥 Loading models using system function...")
        models = load_models()
        
        print(f"\n📊 Successfully loaded {len(models)} models:")
        for model_key in models.keys():
            print(f"   ✅ {model_key}")
            
    except Exception as e:
        print(f"❌ Error testing model loading: {str(e)}")

if __name__ == "__main__":
    print("🏥 Medical X-ray AI System - Model Activation Verification")
    print("=" * 60)
    print(f"🕐 Verification Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Suppress TensorFlow warnings for cleaner output
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.get_logger().setLevel('ERROR')
    
    check_active_models()
    check_model_loading_function()
    
    print("✨ Verification Complete!")