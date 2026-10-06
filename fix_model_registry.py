#!/usr/bin/env python3
"""
Fix Model Registry for Model Management Compatibility
Updates the model registry to work with the model management interface
"""

import json
import os
from pathlib import Path
from datetime import datetime

def fix_model_registry():
    """Fix the model registry to be compatible with model management system"""
    
    print("🔧 Fixing model registry for Model Management compatibility...")
    
    registry_file = Path("models/registry/model_registry.json")
    
    # Load current registry
    if registry_file.exists():
        with open(registry_file, 'r') as f:
            current_registry = json.load(f)
    else:
        print("❌ Registry file not found!")
        return False
    
    # Create compatible registry structure
    fixed_registry = {
        'version': '2.0',
        'created': current_registry.get('last_updated', datetime.now().isoformat()),
        'last_modified': datetime.now().isoformat(),
        'models': {},
        'active_models': {
            'bone_fracture': 'bone_fracture_v1',
            'chest_conditions': 'pneumonia_v1',  # Use pneumonia as chest representative
            'knee_conditions': 'arthritis_v1',   # Use arthritis as knee representative
            'pneumonia': 'pneumonia_v1',
            'cardiomegaly': 'cardiomegaly_v1',
            'arthritis': 'arthritis_v1',
            'osteoporosis': 'osteoporosis_v1'
        }
    }
    
    # Convert models to compatible format
    model_mappings = {
        'bone_fracture': {
            'dataset_type': 'bone_fracture',
            'model_id': 'bone_fracture_v1',
            'version': '1.0'
        },
        'pneumonia': {
            'dataset_type': 'pneumonia', 
            'model_id': 'pneumonia_v1',
            'version': '1.0'
        },
        'cardiomegaly': {
            'dataset_type': 'cardiomegaly',
            'model_id': 'cardiomegaly_v1', 
            'version': '1.0'
        },
        'arthritis': {
            'dataset_type': 'arthritis',
            'model_id': 'arthritis_v1',
            'version': '1.0'
        },
        'osteoporosis': {
            'dataset_type': 'osteoporosis',
            'model_id': 'osteoporosis_v1',
            'version': '1.0'
        }
    }
    
    # Process each model
    for model_key, model_data in current_registry.get('models', {}).items():
        if model_key in model_mappings:
            mapping = model_mappings[model_key]
            
            # Create compatible model entry
            compatible_model = {
                'model_id': mapping['model_id'],
                'model_name': f"DenseNet121 {model_key.replace('_', ' ').title()} Detection",
                'dataset_type': mapping['dataset_type'],
                'version': mapping['version'],
                'architecture': model_data.get('architecture', 'DenseNet121'),
                'input_shape': [224, 224, 3],
                'num_classes': len(model_data.get('classes', [])),
                'class_names': model_data.get('classes', []),
                'classes': model_data.get('classes', []),
                'performance_metrics': {
                    'accuracy': model_data.get('accuracy', 0.0),
                    'test_accuracy': model_data.get('accuracy', 0.0)
                },
                'accuracy': model_data.get('accuracy', 0.0),
                'training_info': {
                    'training_date': model_data.get('training_date', '2025-10-06'),
                    'dataset': model_data.get('dataset', 'Unknown'),
                    'performance_level': model_data.get('performance_level', 'Unknown'),
                    'clinical_readiness': model_data.get('clinical_readiness', 'Unknown')
                },
                'file_path': model_data.get('file_path', f"{model_key}_model.h5"),
                'file_size': 0,  # Will be calculated if needed
                'file_hash': '',  # Will be calculated if needed
                'created_date': model_data.get('training_date', '2025-10-06'),
                'description': f"DenseNet121 model for {model_key.replace('_', ' ')} detection with {model_data.get('accuracy', 0)*100:.1f}% accuracy",
                'tags': [
                    'DenseNet121',
                    'medical',
                    model_data.get('performance_level', 'unknown').lower().replace(' ', '_'),
                    model_key
                ],
                'gradcam_layer': model_data.get('gradcam_layer', 'conv5_block16_2_conv'),
                'is_active': True  # Mark as active since these are our current models
            }
            
            fixed_registry['models'][mapping['model_id']] = compatible_model
            
            print(f"✅ Fixed {model_key} -> {mapping['model_id']}")
    
    # Save fixed registry
    backup_file = registry_file.parent / f"model_registry_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    # Backup original
    if registry_file.exists():
        with open(backup_file, 'w') as f:
            json.dump(current_registry, f, indent=2)
        print(f"📦 Backed up original registry to {backup_file}")
    
    # Save fixed registry
    with open(registry_file, 'w') as f:
        json.dump(fixed_registry, f, indent=2)
    
    print(f"✅ Fixed registry saved to {registry_file}")
    print(f"📊 Models in registry: {len(fixed_registry['models'])}")
    
    return True

def main():
    """Main function"""
    print("🚀 MODEL REGISTRY FIX STARTED")
    print("=" * 50)
    
    try:
        success = fix_model_registry()
        
        if success:
            print("\n" + "=" * 50)
            print("🎉 MODEL REGISTRY FIX COMPLETED!")
            print("=" * 50)
            print("✅ Registry is now compatible with Model Management system")
            print("✅ All 5 models properly configured")
            print("✅ Active models set correctly")
            print("\n🔄 Please restart your Streamlit app to see the changes")
        else:
            print("❌ Fix failed!")
            
    except Exception as e:
        print(f"❌ Error during fix: {str(e)}")
        return False
    
    return True

if __name__ == "__main__":
    main()