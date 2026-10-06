"""
Test Active Model Loading
Verifies that only the ACTIVE model from registry is loaded for classification
"""

import json
import os

def test_active_model_loading():
    """Test that active models are correctly identified"""
    
    print("="*80)
    print("TESTING ACTIVE MODEL LOADING FROM REGISTRY")
    print("="*80)
    
    registry_path = 'models/registry/model_registry.json'
    
    if not os.path.exists(registry_path):
        print("❌ Registry file not found!")
        return
    
    with open(registry_path, 'r', encoding='utf-8') as f:
        registry = json.load(f)
    
    active_models = registry.get('active_models', {})
    all_models = registry.get('models', {})
    
    print("\n" + "="*80)
    print("ACTIVE MODELS CURRENTLY SET IN REGISTRY:")
    print("="*80)
    
    for condition, active_model_id in active_models.items():
        if active_model_id and active_model_id in all_models:
            model_info = all_models[active_model_id]
            print(f"\n📋 Condition: {condition.upper()}")
            print(f"   Active Model ID: {active_model_id}")
            print(f"   Model Name: {model_info.get('model_name', 'Unknown')}")
            print(f"   Architecture: {model_info.get('architecture', 'Unknown')}")
            print(f"   Version: {model_info.get('version', 'Unknown')}")
            print(f"   File Path: models/{model_info.get('file_path', 'Unknown')}")
            print(f"   Accuracy: {model_info.get('performance_metrics', {}).get('accuracy', 'N/A')}")
            
            # Check if file exists
            file_path = f"models/{model_info.get('file_path', '')}"
            if os.path.exists(file_path):
                print(f"   ✅ Model file EXISTS")
            else:
                print(f"   ❌ Model file NOT FOUND: {file_path}")
        else:
            print(f"\n📋 Condition: {condition.upper()}")
            print(f"   ⚠️  No active model set")
    
    print("\n" + "="*80)
    print("VERIFICATION COMPLETE")
    print("="*80)
    
    # Now test the actual loading function
    print("\n" + "="*80)
    print("TESTING load_single_model() FUNCTION")
    print("="*80)
    
    from utils.model_inference import load_single_model
    
    test_conditions = ['pneumonia', 'cardiomegaly', 'arthritis', 'osteoporosis', 'bone_fracture']
    
    for condition in test_conditions:
        print(f"\n🔍 Testing: {condition}")
        print("-" * 60)
        model = load_single_model(condition)
        if model:
            print(f"✅ Successfully loaded model for {condition}")
            print(f"   Model type: {type(model)}")
            print(f"   Input shape: {model.input_shape if hasattr(model, 'input_shape') else 'Unknown'}")
            print(f"   Output shape: {model.output_shape if hasattr(model, 'output_shape') else 'Unknown'}")
        else:
            print(f"❌ Failed to load model for {condition}")
    
    print("\n" + "="*80)
    print("ALL TESTS COMPLETE")
    print("="*80)

if __name__ == "__main__":
    test_active_model_loading()
