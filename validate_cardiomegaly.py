#!/usr/bin/env python3
"""Validate cardiomegaly model registration and file integrity"""

import json
import os
from pathlib import Path

def validate_cardiomegaly_model():
    """Validate the cardiomegaly model registration"""
    
    print("🔍 Validating cardiomegaly model registration...")
    
    # Check registry file
    registry_path = Path("models/registry/model_registry.json")
    if not registry_path.exists():
        print("❌ Registry file not found!")
        return False
    
    # Load registry
    with open(registry_path, 'r') as f:
        registry = json.load(f)
    
    # Check cardiomegaly entry
    if 'cardiomegaly' not in registry['models']:
        print("❌ Cardiomegaly model not found in registry!")
        return False
    
    cardiomegaly_info = registry['models']['cardiomegaly']
    model_path = Path("models") / cardiomegaly_info['file_path']
    
    print(f"✅ Registry entry found for cardiomegaly")
    print(f"📁 Expected file: {model_path}")
    print(f"📊 Registered accuracy: {cardiomegaly_info['accuracy']:.1%}")
    print(f"🏗️  Architecture: {cardiomegaly_info['architecture']}")
    
    # Check if model file exists
    if not model_path.exists():
        print(f"❌ Model file not found: {model_path}")
        return False
    
    # Check file size
    actual_size = model_path.stat().st_size
    registered_size = cardiomegaly_info.get('file_size', 0)
    
    print(f"✅ Model file exists: {model_path.name}")
    print(f"📏 File size: {actual_size:,} bytes")
    
    if actual_size != registered_size:
        print(f"⚠️  Size mismatch - Registered: {registered_size:,}, Actual: {actual_size:,}")
    else:
        print(f"✅ File size matches registry")
    
    # Check active models
    active_models = registry.get('active_models', {})
    if active_models.get('cardiomegaly') == 'cardiomegaly':
        print(f"✅ Cardiomegaly model is active")
    else:
        print(f"⚠️  Cardiomegaly model is not active")
    
    # Check for duplicate entries
    duplicate_entries = [key for key in registry['models'].keys() if 'cardiomegaly' in key.lower()]
    if len(duplicate_entries) > 1:
        print(f"⚠️  Found {len(duplicate_entries)} cardiomegaly entries: {duplicate_entries}")
    else:
        print(f"✅ No duplicate entries found")
    
    print("\n🎉 Cardiomegaly model validation completed!")
    return True

def check_model_performance():
    """Display model performance summary"""
    
    registry_path = Path("models/registry/model_registry.json")
    with open(registry_path, 'r') as f:
        registry = json.load(f)
    
    cardiomegaly_info = registry['models']['cardiomegaly']
    
    print("📊 Cardiomegaly Model Performance Summary:")
    print("=" * 50)
    print(f"🎯 Overall Accuracy: {cardiomegaly_info['accuracy']:.1%}")
    print(f"📉 Training Loss: {cardiomegaly_info.get('training_loss', 'N/A')}")
    print(f"⏱️  Training Method: {cardiomegaly_info['training_method']}")
    print(f"📅 Trained Date: {cardiomegaly_info['trained_date']}")
    print(f"🗂️  Dataset: {cardiomegaly_info['dataset']}")
    
    if 'per_class_metrics' in cardiomegaly_info:
        print(f"\n📈 Per-Class Performance:")
        for class_name, metrics in cardiomegaly_info['per_class_metrics'].items():
            print(f"  {class_name}:")
            print(f"    Precision: {metrics['precision']:.3f}")
            print(f"    Recall: {metrics['recall']:.3f}")
            print(f"    F1-Score: {metrics['f1_score']:.3f}")

if __name__ == "__main__":
    print("🫀 Cardiomegaly Model Validation")
    print("=" * 50)
    
    if validate_cardiomegaly_model():
        print()
        check_model_performance()
    else:
        print("❌ Validation failed!")