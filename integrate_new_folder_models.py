# 🔄 Properly Integrate New Trained Models from "new" Folder

import json
import os
import shutil
from datetime import datetime

def main():
    print("🔄 Integrating New Trained Models from 'new' Folder")
    print("=" * 60)
    
    # Paths
    registry_path = "models/registry/model_registry.json"
    backup_path = f"models/registry/model_registry_new_integration_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    models_dir = "models"
    new_models_dir = "new"
    
    # Load current registry
    with open(registry_path, 'r') as f:
        registry = json.load(f)
    
    # Backup current registry
    shutil.copy2(registry_path, backup_path)
    print(f"📦 Backed up registry to: {backup_path}")
    
    # Find and copy the actual trained models from new folder
    new_model_mappings = {
        "pneumonia_v2_new": {
            "source_path": "new/densenet121_pneumonia_intensive_models/models/densenet121_pneumonia_intensive_20251006_182328.h5",
            "target_file": "pneumonia_densenet121_new_v2.h5",
            "dataset_type": "pneumonia",
            "model_name": "DenseNet121 Pneumonia Detection (New Training v2)",
            "details_path": "new/densenet121_pneumonia_intensive_models/model_details.json"
        },
        "arthritis_v2_new": {
            "source_path": "new/densenet121_osteoarthritis_intensive_models/models/densenet121_osteoarthritis_intensive_20251006_185456.h5",
            "target_file": "arthritis_densenet121_new_v2.h5", 
            "dataset_type": "arthritis",
            "model_name": "DenseNet121 Arthritis Detection (New Training v2)",
            "details_path": "new/densenet121_osteoarthritis_intensive_models/model_details.json"
        },
        "osteoporosis_v2_new": {
            "source_path": "new/densenet121_osteoporosis_intensive_models/models/densenet121_osteoporosis_intensive_20251006_183913.h5",
            "target_file": "osteoporosis_densenet121_new_v2.h5",
            "dataset_type": "osteoporosis", 
            "model_name": "DenseNet121 Osteoporosis Detection (New Training v2)",
            "details_path": "new/densenet121_osteoporosis_intensive_models/model_details.json"
        },
        "bone_fracture_v2_new": {
            "source_path": "new/densenet121_limbabnormalities_intensive_models/models/densenet121_limbabnormalities_intensive_20251006_190347.h5",
            "target_file": "bone_fracture_densenet121_new_v2.h5",
            "dataset_type": "bone_fracture",
            "model_name": "DenseNet121 Bone Fracture Detection (New Training v2)", 
            "details_path": "new/densenet121_limbabnormalities_intensive_models/model_details.json"
        },
        "cardiomegaly_v2_new": {
            "source_path": "new/cardiomegaly_densenet121/cardiomegaly_intensive_20251006_192404/cardiomegaly_densenet121_intensive_20251006_192404.h5",
            "target_file": "cardiomegaly_densenet121_new_v2.h5",
            "dataset_type": "cardiomegaly",
            "model_name": "DenseNet121 Cardiomegaly Detection (New Training v2)",
            "details_path": "new/cardiomegaly_densenet121/cardiomegaly_intensive_20251006_192404/model_details.json"
        }
    }
    
    print(f"\n🎯 Processing {len(new_model_mappings)} new trained models from 'new' folder...")
    
    for model_id, config in new_model_mappings.items():
        print(f"\n📋 Processing {model_id}...")
        
        source_path = config["source_path"]
        target_path = os.path.join(models_dir, config["target_file"])
        details_path = config["details_path"]
        
        # Check if source file exists
        if not os.path.exists(source_path):
            print(f"   ❌ Source model not found: {source_path}")
            continue
            
        # Copy the model file
        try:
            shutil.copy2(source_path, target_path)
            source_size = round(os.path.getsize(source_path) / (1024 * 1024), 2)
            target_size = round(os.path.getsize(target_path) / (1024 * 1024), 2)
            print(f"   ✅ Copied model: {config['target_file']} ({target_size} MB)")
            
            if target_size < 10:  # If less than 10 MB, might be incomplete
                print(f"   ⚠️ Warning: Model size ({target_size} MB) seems small for DenseNet121")
                
        except Exception as e:
            print(f"   ❌ Failed to copy model: {str(e)}")
            continue
        
        # Load model details if available
        model_details = {}
        if os.path.exists(details_path):
            try:
                with open(details_path, 'r') as f:
                    model_details = json.load(f)
                print(f"   ✅ Loaded model details from: {details_path}")
            except Exception as e:
                print(f"   ⚠️ Could not load model details: {str(e)}")
        
        # Extract performance metrics
        performance = model_details.get("performance", {})
        accuracy = performance.get("test_accuracy", 0.0)
        
        # Determine performance level
        if accuracy >= 0.90:
            performance_level = "Medical Grade"
            clinical_readiness = "Ready for Clinical Use"
            grade_tag = "medical_grade"
        elif accuracy >= 0.75:
            performance_level = "Clinical Assistant"
            clinical_readiness = "Clinical Support"
            grade_tag = "clinical_assistant"
        else:
            performance_level = "Research Grade"
            clinical_readiness = "Research and Development"
            grade_tag = "research_grade"
        
        # Create comprehensive registry entry
        registry_entry = {
            "model_id": model_id,
            "model_name": config["model_name"],
            "dataset_type": config["dataset_type"],
            "version": "2.0_new",
            "architecture": "DenseNet121",
            "input_shape": [224, 224, 3],
            "num_classes": 2,
            "class_names": model_details.get("dataset_info", {}).get("classes", ["Normal", "Abnormal"]),
            "classes": model_details.get("dataset_info", {}).get("classes", ["Normal", "Abnormal"]),
            "performance_metrics": {
                "accuracy": accuracy,
                "test_accuracy": accuracy,
                "precision": performance.get("test_precision", 0.0),
                "recall": performance.get("test_recall", 0.0)
            },
            "accuracy": accuracy,
            "training_info": {
                "training_date": "2025-10-06",
                "dataset": model_details.get("dataset_info", {}).get("path", ""),
                "performance_level": performance_level,
                "clinical_readiness": clinical_readiness,
                "training_time_minutes": performance.get("training_time_minutes", 0),
                "epochs_trained": performance.get("epochs_trained", "Unknown"),
                "configuration": model_details.get("training_config", {}).get("name", "Intensive"),
                "early_stopping": performance.get("early_stopping", False)
            },
            "file_path": config["target_file"],
            "file_size": round(os.path.getsize(target_path) / (1024 * 1024), 2),
            "file_hash": "",
            "created_date": "2025-10-06",
            "description": f"{config['model_name']} with {accuracy*100:.1f}% accuracy from new training",
            "tags": [
                "DenseNet121",
                "medical",
                grade_tag,
                config["dataset_type"],
                "intensive_training",
                "new_folder",
                "v2_new"
            ],
            "gradcam_layer": "conv5_block16_2_conv",
            "is_active": False,  # Don't activate yet, let user choose
            "training_type": "intensive",
            "model_source": "new_folder_actual_training",
            "source_path": source_path,
            "integration_date": datetime.now().isoformat()
        }
        
        # Add to registry
        registry["models"][model_id] = registry_entry
        
        print(f"   ✅ Added to registry: {model_id}")
        print(f"   📊 Accuracy: {accuracy*100:.1f}% ({performance_level})")
        print(f"   📁 File size: {registry_entry['file_size']} MB")
        print(f"   🎯 Status: Available for activation")
    
    # Update registry metadata
    registry["last_modified"] = datetime.now().isoformat()
    registry["version"] = "2.3"
    
    # Add integration notes
    if "integration_history" not in registry:
        registry["integration_history"] = []
    
    registry["integration_history"].append({
        "date": datetime.now().isoformat(),
        "action": "new_folder_integration",
        "models_added": list(new_model_mappings.keys()),
        "description": "Integrated actual trained models from new/ folder"
    })
    
    # Save updated registry
    with open(registry_path, 'w') as f:
        json.dump(registry, f, indent=2)
    
    print(f"\n✅ Registry updated successfully!")
    print(f"📊 Total models in registry: {len(registry['models'])}")
    
    # Show summary of new models
    print(f"\n🎉 NEW MODELS AVAILABLE FOR ACTIVATION:")
    new_models = [m for m in registry["models"].keys() if "_v2_new" in m]
    for model_id in new_models:
        model = registry["models"][model_id]
        accuracy = model['accuracy']*100
        size = model['file_size']
        status = model['training_info']['performance_level']
        print(f"   • {model['model_name']}: {accuracy:.1f}% accuracy ({size} MB) - {status}")
    
    print(f"\n🎯 ACTIVATION INSTRUCTIONS:")
    print(f"   1. Go to Model Management → Activate Models tab")
    print(f"   2. Choose which models to activate for each condition:")
    
    for model_id in new_models:
        model = registry["models"][model_id]
        dataset_type = model['dataset_type']
        accuracy = model['accuracy']*100
        current_active = registry["active_models"].get(dataset_type, "N/A")
        print(f"      • {dataset_type}: Activate {model_id} ({accuracy:.1f}%) vs Current {current_active}")
    
    print(f"\n💡 These new models are now available in the registry but not yet active.")
    print(f"🔄 Use the Model Management interface to activate them when ready!")

if __name__ == "__main__":
    main()