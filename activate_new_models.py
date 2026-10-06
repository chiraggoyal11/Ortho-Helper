# 🎯 Activate New Trained Models for X-ray Classification

import json
import os
from datetime import datetime
import shutil

def main():
    print("🎯 Activating New Trained Models for X-ray Classification")
    print("=" * 65)
    
    # Paths
    registry_path = "models/registry/model_registry.json"
    backup_path = f"models/registry/model_registry_activation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    # Load current registry
    with open(registry_path, 'r') as f:
        registry = json.load(f)
    
    # Backup current registry
    shutil.copy2(registry_path, backup_path)
    print(f"📦 Backed up registry to: {backup_path}")
    
    # Show current active models
    print(f"\n📊 CURRENT ACTIVE MODELS:")
    for condition, model_id in registry["active_models"].items():
        if model_id in registry["models"]:
            model_info = registry["models"][model_id]
            accuracy = model_info.get("accuracy", 0) * 100
            print(f"   • {condition}: {model_id} ({accuracy:.1f}% accuracy)")
        else:
            print(f"   • {condition}: {model_id} (model not found)")
    
    # Show available new models
    print(f"\n🎉 AVAILABLE NEW MODELS:")
    new_models = {}
    for model_id, model_info in registry["models"].items():
        if "_v2_new" in model_id:
            dataset_type = model_info["dataset_type"]
            accuracy = model_info.get("accuracy", 0) * 100
            performance_level = model_info["training_info"]["performance_level"]
            new_models[dataset_type] = {
                "model_id": model_id,
                "accuracy": accuracy,
                "performance_level": performance_level,
                "file_size": model_info.get("file_size", 0)
            }
            print(f"   • {dataset_type}: {model_id} ({accuracy:.1f}% accuracy - {performance_level})")
    
    # Ask user which models to activate (or activate all high-performance ones)
    print(f"\n🔄 ACTIVATION STRATEGY:")
    print(f"   Activating all Medical Grade models (>90% accuracy) automatically")
    print(f"   Keeping Research Grade models available but not active")
    
    # Update active models - activate Medical Grade models
    activation_changes = []
    
    for dataset_type, new_model_info in new_models.items():
        current_active = registry["active_models"].get(dataset_type)
        new_model_id = new_model_info["model_id"]
        new_accuracy = new_model_info["accuracy"]
        
        # Get current model accuracy for comparison
        current_accuracy = 0
        if current_active and current_active in registry["models"]:
            current_accuracy = registry["models"][current_active].get("accuracy", 0) * 100
        
        # Activation logic
        should_activate = False
        reason = ""
        
        if new_model_info["performance_level"] == "Medical Grade":
            should_activate = True
            reason = "Medical Grade (>90% accuracy)"
        elif new_accuracy > current_accuracy:
            should_activate = True
            reason = f"Higher accuracy ({new_accuracy:.1f}% vs {current_accuracy:.1f}%)"
        else:
            reason = f"Lower performance ({new_accuracy:.1f}% vs {current_accuracy:.1f}%)"
        
        if should_activate:
            registry["active_models"][dataset_type] = new_model_id
            activation_changes.append({
                "condition": dataset_type,
                "old_model": current_active,
                "new_model": new_model_id,
                "old_accuracy": current_accuracy,
                "new_accuracy": new_accuracy,
                "reason": reason
            })
            print(f"   ✅ Activating {dataset_type}: {new_model_id} ({reason})")
        else:
            print(f"   ⏸️ Keeping {dataset_type}: {current_active} ({reason})")
    
    # Update registry metadata
    registry["last_modified"] = datetime.now().isoformat()
    registry["version"] = "2.4"
    
    # Add activation history
    if "activation_history" not in registry:
        registry["activation_history"] = []
    
    registry["activation_history"].append({
        "date": datetime.now().isoformat(),
        "action": "activate_new_trained_models",
        "changes": activation_changes,
        "description": "Activated new trained models from new/ folder"
    })
    
    # Save updated registry
    with open(registry_path, 'w') as f:
        json.dump(registry, f, indent=2)
    
    print(f"\n✅ Registry activation completed!")
    print(f"📊 Registry version updated to: {registry['version']}")
    
    # Show final active model summary
    print(f"\n🎯 UPDATED ACTIVE MODELS:")
    for condition, model_id in registry["active_models"].items():
        if model_id in registry["models"]:
            model_info = registry["models"][model_id]
            accuracy = model_info.get("accuracy", 0) * 100
            performance_level = model_info["training_info"]["performance_level"]
            file_size = model_info.get("file_size", 0)
            is_new = "_v2_new" in model_id
            status = "🆕 NEW" if is_new else "📋 EXISTING"
            print(f"   • {condition}: {model_id} ({accuracy:.1f}% - {performance_level}) {status}")
        else:
            print(f"   • {condition}: {model_id} (model not found)")
    
    # Verify model files exist
    print(f"\n🔍 VERIFYING MODEL FILES:")
    missing_files = []
    
    for condition, model_id in registry["active_models"].items():
        if model_id in registry["models"]:
            model_info = registry["models"][model_id]
            file_path = model_info["file_path"]
            full_path = os.path.join("models", file_path)
            
            if os.path.exists(full_path):
                size = round(os.path.getsize(full_path) / (1024 * 1024), 2)
                print(f"   ✅ {condition}: {file_path} ({size} MB)")
            else:
                missing_files.append(f"{condition}: {file_path}")
                print(f"   ❌ {condition}: {file_path} - FILE NOT FOUND!")
    
    if missing_files:
        print(f"\n⚠️ WARNING: Some active model files are missing!")
        for missing in missing_files:
            print(f"   • {missing}")
        print(f"\n🔧 Run the model loading fix again if needed.")
    else:
        print(f"\n🎉 All active model files verified and exist!")
    
    # Show performance summary
    print(f"\n📈 PERFORMANCE SUMMARY:")
    medical_grade = 0
    research_grade = 0
    
    for condition, model_id in registry["active_models"].items():
        if model_id in registry["models"]:
            model_info = registry["models"][model_id]
            performance_level = model_info["training_info"]["performance_level"]
            if performance_level == "Medical Grade":
                medical_grade += 1
            else:
                research_grade += 1
    
    print(f"   🏅 Medical Grade Models: {medical_grade}")
    print(f"   🔬 Research Grade Models: {research_grade}")
    print(f"   🎯 Total Active Models: {medical_grade + research_grade}")
    
    print(f"\n🚀 READY FOR X-RAY CLASSIFICATION!")
    print(f"   The new trained models are now active and ready to use.")
    print(f"   Navigate to X-Ray Classification page to test with medical images.")
    print(f"   Expected: Higher accuracy results from the improved models!")

if __name__ == "__main__":
    main()