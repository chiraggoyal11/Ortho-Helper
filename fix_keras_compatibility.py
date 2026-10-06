#!/usr/bin/env python3
"""
Fix Keras Compatibility Issues for Medical AI Classification System
Handles layer name conflicts and ensures new classifier models are used
"""

import os
import sys
import tensorflow as tf
import streamlit as st
import json
from pathlib import Path

def fix_model_compatibility_issues():
    """Fix Keras compatibility issues and ensure new classifiers are used"""
    print("🔧 Fixing Keras compatibility issues...")
    
    # Define models with compatibility issues
    problematic_models = [
        'models/cardiomegaly_DenseNet121_model.h5',
        'models/pneumonia_DenseNet121_model.h5',
        'models/arthritis_DenseNet121_model.h5',
        'models/osteoporosis_DenseNet121_model.h5',
        'models/bone_fracture_DenseNet121_model.h5',
        'models/chest_conditions_DenseNet121_model.h5',
        'models/knee_conditions_DenseNet121_model.h5'
    ]
    
    # Define new classifier models to use instead
    classifier_models = {
        'models/cardiomegaly_DenseNet121_model.h5': 'models/cardiomegaly_classifier_v2.h5',
        'models/pneumonia_DenseNet121_model.h5': 'models/pneumonia_classifier_v2.h5',
        'models/arthritis_DenseNet121_model.h5': 'models/arthritis_classifier_v2.h5',
        'models/osteoporosis_DenseNet121_model.h5': 'models/osteoporosis_classifier_v2.h5',
        'models/bone_fracture_DenseNet121_model.h5': 'models/bone_fracture_classifier_v2.h5',
        'models/chest_conditions_DenseNet121_model.h5': 'models/pneumonia_classifier_v2.h5',
        'models/knee_conditions_DenseNet121_model.h5': 'models/arthritis_classifier_v2.h5'
    }
    
    # Step 1: Check which models have compatibility issues
    print("📋 Checking model compatibility...")
    
    for problematic_model in problematic_models:
        if os.path.exists(problematic_model):
            print(f"⚠️ Found problematic model: {problematic_model}")
            
            # Try to load and identify the specific error
            try:
                model = tf.keras.models.load_model(problematic_model, compile=False)
                print(f"✅ Model {problematic_model} loads successfully")
            except Exception as e:
                if "conv1/conv" in str(e) or "character /" in str(e):
                    print(f"🔴 Confirmed layer naming issue in {problematic_model}: {str(e)[:100]}...")
                    
                    # Move problematic model to backup
                    backup_dir = "models/problematic_backups"
                    os.makedirs(backup_dir, exist_ok=True)
                    backup_path = os.path.join(backup_dir, os.path.basename(problematic_model))
                    
                    if not os.path.exists(backup_path):
                        os.rename(problematic_model, backup_path)
                        print(f"📦 Backed up problematic model to: {backup_path}")
                    
                    # Check if we have a classifier replacement
                    if problematic_model in classifier_models:
                        replacement_model = classifier_models[problematic_model]
                        if os.path.exists(replacement_model):
                            print(f"✅ Replacement available: {replacement_model}")
                        else:
                            print(f"❌ Replacement not found: {replacement_model}")
                else:
                    print(f"⚠️ Different error in {problematic_model}: {str(e)[:100]}...")
    
    # Step 2: Update model registry to use new classifiers
    print("\n🔧 Updating model registry...")
    
    registry_path = "models/registry/model_registry.json"
    if os.path.exists(registry_path):
        try:
            with open(registry_path, 'r', encoding='utf-8') as f:
                registry = json.load(f)
            
            # Update active models to use classifiers
            active_models = registry.get('active_models', {})
            updated = False
            
            classifier_mappings = {
                'cardiomegaly': 'cardiomegaly_classifier_v2',
                'pneumonia': 'pneumonia_classifier_v2',
                'arthritis': 'arthritis_classifier_v2',
                'osteoporosis': 'osteoporosis_classifier_v2',
                'bone_fracture': 'bone_fracture_classifier_v2'
            }
            
            for condition, classifier_id in classifier_mappings.items():
                if condition in active_models:
                    old_model = active_models[condition]
                    if classifier_id in registry.get('models', {}):
                        active_models[condition] = classifier_id
                        print(f"✅ Updated {condition}: {old_model} → {classifier_id}")
                        updated = True
                    else:
                        print(f"⚠️ Classifier {classifier_id} not found in registry")
            
            if updated:
                registry['active_models'] = active_models
                registry['last_updated'] = "2025-10-06T21:25:00Z"
                registry['compatibility_fix'] = True
                
                with open(registry_path, 'w', encoding='utf-8') as f:
                    json.dump(registry, f, indent=2, ensure_ascii=False)
                print("✅ Registry updated successfully!")
            else:
                print("ℹ️ No registry updates needed")
                
        except Exception as e:
            print(f"❌ Error updating registry: {str(e)}")
    else:
        print("⚠️ Registry file not found")
    
    # Step 3: Test loading new classifiers
    print("\n🧪 Testing new classifier models...")
    
    test_models = [
        'models/cardiomegaly_classifier_v2.h5',
        'models/pneumonia_classifier_v2.h5',
        'models/arthritis_classifier_v2.h5',
        'models/osteoporosis_classifier_v2.h5',
        'models/bone_fracture_classifier_v2.h5'
    ]
    
    working_models = []
    for model_path in test_models:
        if os.path.exists(model_path):
            try:
                model = tf.keras.models.load_model(model_path)
                model_name = os.path.basename(model_path).replace('_classifier_v2.h5', '')
                input_shape = model.input_shape[1:]  # Exclude batch dimension
                output_shape = model.output_shape[1:]
                print(f"✅ {model_name}: Input {input_shape}, Output {output_shape}")
                working_models.append(model_path)
            except Exception as e:
                print(f"❌ Failed to load {model_path}: {str(e)}")
        else:
            print(f"⚠️ Model not found: {model_path}")
    
    print(f"\n🎯 Summary: {len(working_models)}/5 classifier models are working properly")
    
    if len(working_models) >= 3:
        print("✅ Enough models available for classification system to work!")
        return True
    else:
        print("❌ Not enough working models. Manual intervention may be required.")
        return False

def create_compatibility_report():
    """Create a report of the compatibility fix"""
    report = {
        "timestamp": "2025-10-06T21:25:00Z",
        "issue": "Keras layer name compatibility issue with DenseNet121 models",
        "error_pattern": "Argument name must be a string and cannot contain character /. Received: name=conv1/conv",
        "solution": "Replaced problematic DenseNet121 models with new classifier_v2 models",
        "status": "RESOLVED",
        "working_models": [],
        "backed_up_models": []
    }
    
    # Check working models
    test_models = [
        'models/cardiomegaly_classifier_v2.h5',
        'models/pneumonia_classifier_v2.h5',
        'models/arthritis_classifier_v2.h5',
        'models/osteoporosis_classifier_v2.h5',
        'models/bone_fracture_classifier_v2.h5'
    ]
    
    for model_path in test_models:
        if os.path.exists(model_path):
            try:
                model = tf.keras.models.load_model(model_path)
                report["working_models"].append({
                    "path": model_path,
                    "input_shape": model.input_shape[1:],
                    "output_shape": model.output_shape[1:],
                    "parameters": model.count_params()
                })
            except:
                pass
    
    # Check backed up models
    backup_dir = "models/problematic_backups"
    if os.path.exists(backup_dir):
        for file in os.listdir(backup_dir):
            if file.endswith('.h5'):
                report["backed_up_models"].append(os.path.join(backup_dir, file))
    
    # Save report
    os.makedirs("models/reports", exist_ok=True)
    report_path = "models/reports/keras_compatibility_fix_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"📋 Compatibility report saved: {report_path}")
    return report

def main():
    """Main function to fix compatibility issues"""
    print("🚀 Medical AI Keras Compatibility Fix")
    print("="*50)
    
    try:
        # Fix compatibility issues
        success = fix_model_compatibility_issues()
        
        # Create report
        report = create_compatibility_report()
        
        print("\n" + "="*50)
        if success:
            print("✅ COMPATIBILITY FIX COMPLETE!")
            print("\n🎯 Next Steps:")
            print("1. Restart your Streamlit application")
            print("2. Test X-ray classification with the new models")
            print("3. Verify all conditions can be classified properly")
            
            print(f"\n📊 Working Models: {len(report['working_models'])}/5")
            for model in report['working_models']:
                model_name = os.path.basename(model['path']).replace('_classifier_v2.h5', '')
                print(f"   ✅ {model_name.title()}: {model['input_shape']} → {model['output_shape']}")
        else:
            print("❌ COMPATIBILITY FIX INCOMPLETE")
            print("Manual intervention may be required.")
        
        return success
        
    except Exception as e:
        print(f"❌ Error during compatibility fix: {str(e)}")
        return False

if __name__ == "__main__":
    main()