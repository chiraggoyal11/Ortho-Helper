# 🔧 Fix Model Registry Paths and Compatibility

import json
import os
from datetime import datetime
import shutil

def main():
    print("🔧 Fixing Model Registry Paths and Compatibility")
    print("=" * 55)
    
    # Paths
    registry_path = "models/registry/model_registry.json"
    backup_path = f"models/registry/model_registry_path_fix_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    # Load current registry
    with open(registry_path, 'r') as f:
        registry = json.load(f)
    
    # Backup current registry
    shutil.copy2(registry_path, backup_path)
    print(f"📦 Backed up registry to: {backup_path}")
    
    # Check which model files actually exist
    models_dir = "models"
    existing_files = []
    for file in os.listdir(models_dir):
        if file.endswith('.h5'):
            existing_files.append(file)
    
    print(f"\n📁 Found {len(existing_files)} .h5 model files in models/ directory:")
    for file in sorted(existing_files):
        size = round(os.path.getsize(os.path.join(models_dir, file)) / (1024 * 1024), 2)
        print(f"   • {file} ({size} MB)")
    
    # Fix model file paths and use the best available models
    model_fixes = {
        "bone_fracture_v2": {
            "file_path": "bone_fracture_densenet121_intensive_v2.h5",
            "fallback": "bone_fracture_model.h5"
        },
        "pneumonia_v2": {
            "file_path": "pneumonia_densenet121_intensive_v2.h5", 
            "fallback": "pneumonia_binary_model.h5"
        },
        "cardiomegaly_v2": {
            "file_path": "cardiomegaly_densenet121_intensive_v2.h5",
            "fallback": "cardiomegaly_binary_model.h5"
        },
        "arthritis_v2": {
            "file_path": "arthritis_densenet121_intensive_v2.h5",
            "fallback": "arthritis_binary_model.h5"
        },
        "osteoporosis_v2": {
            "file_path": "osteoporosis_densenet121_intensive_v2.h5",
            "fallback": "osteoporosis_binary_model.h5"
        },
        # Also fix v1 models
        "bone_fracture_v1": {
            "file_path": "bone_fracture_model.h5",
            "fallback": "bone_fracture_model.h5"
        },
        "pneumonia_v1": {
            "file_path": "pneumonia_binary_model.h5",
            "fallback": "pneumonia_DenseNet121_model.h5"
        },
        "cardiomegaly_v1": {
            "file_path": "cardiomegaly_binary_model.h5",
            "fallback": "cardiomegaly_DenseNet121_model.h5"
        },
        "arthritis_v1": {
            "file_path": "arthritis_binary_model.h5",
            "fallback": "knee_conditions_DenseNet121_model.h5"
        },
        "osteoporosis_v1": {
            "file_path": "osteoporosis_binary_model.h5",
            "fallback": "osteoporosis_DenseNet121_model.h5"
        }
    }
    
    print(f"\n🔧 Fixing model file paths...")
    
    for model_id, fix_info in model_fixes.items():
        if model_id in registry["models"]:
            model_entry = registry["models"][model_id]
            
            # Check which file exists
            primary_file = fix_info["file_path"]
            fallback_file = fix_info["fallback"]
            
            if primary_file in existing_files:
                model_entry["file_path"] = primary_file
                file_size = round(os.path.getsize(os.path.join(models_dir, primary_file)) / (1024 * 1024), 2)
                model_entry["file_size"] = file_size
                print(f"   ✅ {model_id}: Using {primary_file} ({file_size} MB)")
            elif fallback_file in existing_files:
                model_entry["file_path"] = fallback_file
                file_size = round(os.path.getsize(os.path.join(models_dir, fallback_file)) / (1024 * 1024), 2)
                model_entry["file_size"] = file_size
                print(f"   ⚠️ {model_id}: Using fallback {fallback_file} ({file_size} MB)")
            else:
                print(f"   ❌ {model_id}: No file found!")
                
    # Update active models to use working models
    # For now, let's use the binary models that are known to work
    safe_active_models = {
        "bone_fracture": "bone_fracture_v1",  # Use v1 (binary model) for stability
        "pneumonia": "pneumonia_v1",          # Use v1 (binary model) for stability  
        "cardiomegaly": "cardiomegaly_v1",    # Use v1 (binary model) for stability
        "arthritis": "arthritis_v1",          # Use v1 (binary model) for stability
        "osteoporosis": "osteoporosis_v1",    # Use v1 (binary model) for stability
        "chest_conditions": "pneumonia_v1",   # Map to working pneumonia model
        "knee_conditions": "arthritis_v1"     # Map to working arthritis model
    }
    
    print(f"\n🎯 Setting safe active models...")
    
    for condition, model_id in safe_active_models.items():
        if model_id in registry["models"]:
            registry["active_models"][condition] = model_id
            model_file = registry["models"][model_id]["file_path"]
            print(f"   ✅ {condition}: {model_id} -> {model_file}")
        else:
            print(f"   ❌ {condition}: {model_id} not found in registry!")
    
    # Update registry metadata
    registry["last_modified"] = datetime.now().isoformat()
    registry["version"] = "2.2"
    
    # Add compatibility notes
    registry["compatibility_notes"] = {
        "keras_version_issues": "DenseNet121 v2 models may have compatibility issues",
        "recommended_models": "Binary v1 models for stable operation",
        "fixed_paths": "Model file paths corrected to models/ directory",
        "last_path_fix": datetime.now().isoformat()
    }
    
    # Save updated registry
    with open(registry_path, 'w') as f:
        json.dump(registry, f, indent=2)
    
    print(f"\n✅ Registry paths fixed and saved!")
    print(f"📊 Registry version: {registry['version']}")
    print(f"🎯 Active models set to stable binary versions")
    
    # Verify files exist
    print(f"\n🔍 Verifying model files exist...")
    missing_files = []
    
    for model_id, model_info in registry["models"].items():
        file_path = model_info["file_path"]
        full_path = os.path.join(models_dir, file_path)
        
        if os.path.exists(full_path):
            size = round(os.path.getsize(full_path) / (1024 * 1024), 2)
            print(f"   ✅ {model_id}: {file_path} ({size} MB)")
        else:
            missing_files.append(f"{model_id}: {file_path}")
            print(f"   ❌ {model_id}: {file_path} - FILE NOT FOUND!")
    
    if missing_files:
        print(f"\n⚠️ Missing files found:")
        for missing in missing_files:
            print(f"   • {missing}")
    else:
        print(f"\n🎉 All model files verified and exist!")
    
    print(f"\n💡 Now restart the Streamlit app to test X-ray classification!")

if __name__ == "__main__":
    main()