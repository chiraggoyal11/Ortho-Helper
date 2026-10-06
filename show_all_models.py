#!/usr/bin/env python3
"""
Display All Available Models - Comprehensive Report
Shows all intensive, fast, and quick models with their performance metrics
"""

import json
from pathlib import Path
from typing import Dict, List
import os

def load_registry():
    """Load the model registry"""
    registry_path = Path("models/registry/model_registry.json")
    with open(registry_path, 'r') as f:
        return json.load(f)

def get_model_summary(model_info: Dict) -> Dict:
    """Extract key information from model"""
    return {
        'id': model_info.get('model_id', 'Unknown'),
        'name': model_info.get('model_name', 'Unknown'),
        'type': model_info.get('model_type', 'Unknown'),
        'architecture': model_info.get('architecture', 'Unknown'),
        'accuracy': model_info.get('accuracy', model_info.get('performance_metrics', {}).get('test_accuracy', 0)),
        'input_shape': model_info.get('input_shape', [224, 224, 3]),
        'parameters': model_info.get('parameters', 0),
        'file_size_mb': model_info.get('file_size', 0) / (1024 * 1024) if isinstance(model_info.get('file_size', 0), int) else model_info.get('file_size', 0),
        'file_path': model_info.get('file_path', 'Unknown')
    }

def display_models_by_condition(registry: Dict):
    """Display all models organized by medical condition"""
    
    conditions = {
        'pneumonia': '🫁 Pneumonia Detection',
        'cardiomegaly': '❤️ Cardiomegaly Detection',
        'arthritis': '🦵 Knee Arthritis Detection',
        'osteoporosis': '🦴 Knee Osteoporosis Detection',
        'bone_fracture': '💀 Bone Fracture Detection'
    }
    
    print("\n" + "="*80)
    print("🏥 COMPREHENSIVE MODEL INVENTORY REPORT")
    print("="*80)
    
    active_models = registry.get('active_models', {})
    all_models = registry.get('models', {})
    
    total_models = 0
    
    for condition_key, condition_name in conditions.items():
        print(f"\n{'='*80}")
        print(f"{condition_name}")
        print(f"{'='*80}")
        
        # Get all models for this condition
        condition_models = [
            (model_id, model_info) 
            for model_id, model_info in all_models.items() 
            if model_info.get('dataset_type') == condition_key
        ]
        
        if not condition_models:
            print("  ⚠️  No models found for this condition")
            continue
        
        # Sort by model type
        type_order = {'Intensive': 0, 'Quick': 1, 'Fast': 2}
        condition_models.sort(key=lambda x: (
            type_order.get(x[1].get('model_type', 'Unknown'), 99),
            x[1].get('accuracy', 0)
        ), reverse=True)
        
        # Show active model
        active_id = active_models.get(condition_key)
        if active_id:
            active_info = all_models.get(active_id, {})
            summary = get_model_summary(active_info)
            print(f"\n🟢 CURRENTLY ACTIVE:")
            print(f"   Model ID: {active_id}")
            print(f"   Architecture: {summary['architecture']}")
            print(f"   Accuracy: {summary['accuracy']*100:.2f}%")
            print(f"   Type: {summary['type']}")
        
        print(f"\n📋 ALL AVAILABLE MODELS ({len(condition_models)} total):\n")
        
        for idx, (model_id, model_info) in enumerate(condition_models, 1):
            summary = get_model_summary(model_info)
            is_active = (model_id == active_id)
            
            status = "🟢 ACTIVE" if is_active else "⚪"
            
            print(f"  {status} Model #{idx}")
            print(f"     ID: {model_id}")
            print(f"     Type: {summary['type']}")
            print(f"     Architecture: {summary['architecture']}")
            print(f"     Accuracy: {summary['accuracy']*100:.2f}%")
            print(f"     Parameters: {summary['parameters']:,}")
            print(f"     Input Size: {summary['input_shape'][0]}x{summary['input_shape'][1]}")
            print(f"     File Size: {summary['file_size_mb']:.2f} MB")
            print(f"     Path: {summary['file_path']}")
            
            # Check if file exists
            file_path = Path("models") / summary['file_path']
            if not file_path.exists():
                print(f"     ⚠️  WARNING: Model file not found!")
            else:
                print(f"     ✅ File exists")
            
            print()
        
        total_models += len(condition_models)
    
    print("\n" + "="*80)
    print(f"SUMMARY: {total_models} models registered across {len(conditions)} conditions")
    print("="*80 + "\n")

def display_performance_comparison():
    """Display performance comparison between model types"""
    
    registry = load_registry()
    all_models = registry.get('models', {})
    
    print("\n" + "="*80)
    print("📊 PERFORMANCE COMPARISON BY MODEL TYPE")
    print("="*80)
    
    conditions = ['pneumonia', 'cardiomegaly', 'arthritis', 'osteoporosis', 'bone_fracture']
    
    print(f"\n{'Condition':<20} {'Intensive':<15} {'Quick':<15} {'Fast':<15} {'Best Model':<15}")
    print("-" * 80)
    
    for condition in conditions:
        models = [m for m in all_models.values() if m.get('dataset_type') == condition]
        
        intensive = next((m for m in models if m.get('model_type') == 'Intensive'), None)
        quick = next((m for m in models if m.get('model_type') == 'Quick'), None)
        fast = next((m for m in models if m.get('model_type') == 'Fast'), None)
        
        intensive_acc = intensive.get('accuracy', 0) if intensive else 0
        quick_acc = quick.get('accuracy', 0) if quick else 0
        fast_acc = fast.get('accuracy', 0) if fast else 0
        
        best_acc = max(intensive_acc, quick_acc, fast_acc)
        
        if intensive_acc == best_acc:
            best_type = "Intensive"
        elif quick_acc == best_acc:
            best_type = "Quick"
        else:
            best_type = "Fast"
        
        print(f"{condition.replace('_', ' ').title():<20} "
              f"{intensive_acc*100:>6.2f}%      "
              f"{quick_acc*100:>6.2f}%      "
              f"{fast_acc*100:>6.2f}%      "
              f"{best_type}")
    
    # Calculate averages
    intensive_avg = sum(m.get('accuracy', 0) for m in all_models.values() if m.get('model_type') == 'Intensive') / 5
    fast_avg = sum(m.get('accuracy', 0) for m in all_models.values() if m.get('model_type') == 'Fast') / 5
    
    print("-" * 80)
    print(f"{'Average':<20} {intensive_avg*100:>6.2f}%      {'N/A':<13} {fast_avg*100:>6.2f}%")
    print("\n")
    
    # Recommendations
    print("💡 RECOMMENDATIONS:")
    print("-" * 80)
    
    for condition in conditions:
        models = [m for m in all_models.values() if m.get('dataset_type') == condition]
        
        intensive = next((m for m in models if m.get('model_type') == 'Intensive'), None)
        fast = next((m for m in models if m.get('model_type') == 'Fast'), None)
        
        if intensive and fast:
            intensive_acc = intensive.get('accuracy', 0)
            fast_acc = fast.get('accuracy', 0)
            
            diff = (fast_acc - intensive_acc) * 100
            
            if diff > 0:
                print(f"✅ {condition.upper()}: Use FAST model (better by {diff:.2f}%)")
            elif abs(diff) < 2:
                print(f"⚖️  {condition.upper()}: Either model works (difference: {abs(diff):.2f}%)")
            else:
                print(f"🔬 {condition.upper()}: Use INTENSIVE model (better by {abs(diff):.2f}%)")
    
    print("\n")

def check_model_files():
    """Check which model files actually exist"""
    
    registry = load_registry()
    all_models = registry.get('models', {})
    
    print("\n" + "="*80)
    print("📁 MODEL FILE VERIFICATION")
    print("="*80 + "\n")
    
    models_dir = Path("models")
    existing = 0
    missing = 0
    
    for model_id, model_info in all_models.items():
        file_path = models_dir / model_info.get('file_path', '')
        
        if file_path.exists():
            existing += 1
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"✅ {model_id}: {file_path} ({size_mb:.2f} MB)")
        else:
            missing += 1
            print(f"❌ {model_id}: {file_path} - FILE NOT FOUND")
    
    print(f"\n{'='*80}")
    print(f"Total: {existing} files exist, {missing} files missing")
    print(f"{'='*80}\n")

def main():
    """Main function"""
    
    try:
        registry = load_registry()
        
        print("\n🏥 MEDICAL X-RAY AI - MODEL INVENTORY SYSTEM")
        print("="*80)
        print(f"Registry Version: {registry.get('version', 'Unknown')}")
        print(f"Last Modified: {registry.get('last_modified', 'Unknown')}")
        print(f"Total Registered Models: {len(registry.get('models', {}))}")
        
        # Display all models
        display_models_by_condition(registry)
        
        # Performance comparison
        display_performance_comparison()
        
        # File verification
        check_model_files()
        
        print("\n✅ Report generation complete!")
        print("\nTo view models in the web interface:")
        print("  1. Navigate to http://localhost:8502")
        print("  2. Go to '🔧 Model Management' page")
        print("  3. View models in '📋 Model Registry' tab")
        print("  4. Activate models in '🚀 Activate Models' tab\n")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
