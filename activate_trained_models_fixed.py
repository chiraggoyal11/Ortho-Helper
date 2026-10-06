import os
import json
import datetime
from pathlib import Path

class ModelActivationManager:
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.models_dir = self.base_dir / "models"
        self.registry_path = self.models_dir / "registry" / "model_registry.json"
        
        # Ensure registry directory exists
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        
        print("🔧 Model Activation Manager Initialized")
        print(f"📁 Registry path: {self.registry_path}")
    
    def load_registry(self):
        """Load the model registry"""
        if not self.registry_path.exists():
            print("⚠️ Registry file not found. Creating empty registry.")
            return {"models": {}, "active_models": {}}
        
        try:
            with open(self.registry_path, 'r') as f:
                registry = json.load(f)
                return registry
        except Exception as e:
            print(f"❌ Error loading registry: {e}")
            return None
    
    def save_registry(self, registry):
        """Save the model registry"""
        try:
            with open(self.registry_path, 'w') as f:
                json.dump(registry, f, indent=2)
            return True
        except Exception as e:
            print(f"❌ Error saving registry: {e}")
            return False
    
    def list_available_models(self, model_type=None):
        """List all available models or filter by type"""
        registry = self.load_registry()
        if not registry:
            return []
        
        models = []
        for model_id, model_data in registry.get('models', {}).items():
            if not model_type or model_data.get('dataset_type') == model_type:
                models.append({
                    'model_id': model_id,
                    'name': model_data.get('model_name'),
                    'type': model_data.get('dataset_type'),
                    'accuracy': model_data.get('performance_metrics', {}).get('accuracy', 0),
                    'is_active': model_data.get('is_active', False),
                    'training_date': model_data.get('training_info', {}).get('training_date', 'Unknown')
                })
        
        return sorted(models, key=lambda x: x['accuracy'], reverse=True)
    
    def activate_model(self, model_id, model_type):
        """Activate a specific model and deactivate others of the same type"""
        registry = self.load_registry()
        if not registry:
            return False
        
        if model_id not in registry.get('models', {}):
            print(f"❌ Model {model_id} not found in registry")
            return False
        
        # Deactivate all models of this type
        for mid, model_data in registry['models'].items():
            if model_data.get('dataset_type') == model_type:
                model_data['is_active'] = False
        
        # Activate the selected model
        registry['models'][model_id]['is_active'] = True
        
        # Update active_models section
        if 'active_models' not in registry:
            registry['active_models'] = {}
        
        registry['active_models'][model_type] = model_id
        registry['last_modified'] = datetime.datetime.now().isoformat()
        
        # Save registry
        self.save_registry(registry)
        
        model_name = registry['models'][model_id].get('model_name', model_id)
        print(f"✅ Activated {model_type} model: {model_name}")
        return True
    
    def get_best_model(self, model_type):
        """Get the best performing model for a given type"""
        models = self.list_available_models(model_type)
        if models:
            return models[0]  # Already sorted by accuracy
        return None
    
    def activate_best_models(self, model_types):
        """Activate the best model for each specified type"""
        results = {}
        
        for model_type in model_types:
            best_model = self.get_best_model(model_type)
            if best_model:
                success = self.activate_model(best_model['model_id'], model_type)
                results[model_type] = {
                    'success': success,
                    'model_id': best_model['model_id'],
                    'model_name': best_model['name'],
                    'accuracy': best_model['accuracy']
                }
            else:
                results[model_type] = {
                    'success': False,
                    'error': 'No models found'
                }
        
        return results
    
    def print_model_status(self):
        """Print current status of all models"""
        registry = self.load_registry()
        if not registry:
            return
        
        print("\n📋 Current Model Registry Status")
        print("=" * 60)
        
        # Group by type
        by_type = {}
        for model_id, model_data in registry.get('models', {}).items():
            model_type = model_data.get('dataset_type', 'unknown')
            if model_type not in by_type:
                by_type[model_type] = []
            by_type[model_type].append((model_id, model_data))
        
        for model_type, models in by_type.items():
            print(f"\n🏥 {model_type.upper()} Models:")
            print("-" * 40)
            
            # Sort by accuracy
            models.sort(key=lambda x: x[1].get('performance_metrics', {}).get('accuracy', 0), reverse=True)
            
            for model_id, model_data in models:
                name = model_data.get('model_name', model_id)
                accuracy = model_data.get('performance_metrics', {}).get('accuracy', 0)
                is_active = model_data.get('is_active', False)
                training_date = model_data.get('training_info', {}).get('training_date', 'Unknown')[:10]
                
                status = "✅ ACTIVE" if is_active else "⭕ Inactive"
                print(f"  {status} {name}")
                print(f"    📊 Accuracy: {accuracy:.1%} | 📅 Date: {training_date}")
                print(f"    🆔 ID: {model_id}")
                print()
        
        # Show active models summary
        active_models = registry.get('active_models', {})
        if active_models:
            print("\n🎯 Currently Active Models:")
            print("-" * 30)
            for model_type, model_id in active_models.items():
                model_data = registry.get('models', {}).get(model_id, {})
                name = model_data.get('model_name', model_id)
                accuracy = model_data.get('performance_metrics', {}).get('accuracy', 0)
                print(f"  {model_type}: {name} ({accuracy:.1%})")

def main():
    """Main model activation interface"""
    
    print("🏥 Medical AI Model Activation Manager")
    print("=" * 50)
    
    manager = ModelActivationManager()
    
    # Print current status
    manager.print_model_status()
    
    print("\n🎯 Available Actions:")
    print("1. Activate best models automatically")
    print("2. Activate specific model")
    print("3. Show model details")
    print("4. Exit")
    
    while True:
        choice = input("\nSelect an option (1-4): ").strip()
        
        if choice == '1':
            # Auto-activate best models
            model_types = ['cardiomegaly', 'bone_fracture', 'pneumonia', 'arthritis', 'osteoporosis']
            results = manager.activate_best_models(model_types)
            
            print("\n🎉 Auto-Activation Results:")
            for model_type, result in results.items():
                if result['success']:
                    print(f"✅ {model_type}: {result['model_name']} ({result['accuracy']:.1%})")
                else:
                    print(f"❌ {model_type}: {result.get('error', 'Failed')}")
            break
            
        elif choice == '2':
            # Manual activation
            model_type = input("Enter model type (cardiomegaly/bone_fracture/pneumonia/arthritis/osteoporosis): ").strip()
            models = manager.list_available_models(model_type)
            
            if not models:
                print(f"❌ No models found for type: {model_type}")
                continue
            
            print(f"\nAvailable {model_type} models:")
            for i, model in enumerate(models):
                status = "✅ ACTIVE" if model['is_active'] else "⭕ Inactive"
                print(f"{i+1}. {status} {model['name']} ({model['accuracy']:.1%})")
            
            try:
                selection = int(input("Select model number: ")) - 1
                if 0 <= selection < len(models):
                    selected_model = models[selection]
                    success = manager.activate_model(selected_model['model_id'], model_type)
                    if success:
                        print(f"✅ Successfully activated: {selected_model['name']}")
                    else:
                        print("❌ Activation failed")
                else:
                    print("❌ Invalid selection")
            except ValueError:
                print("❌ Invalid input")
                
        elif choice == '3':
            # Show details
            manager.print_model_status()
            
        elif choice == '4':
            print("👋 Goodbye!")
            break
            
        else:
            print("❌ Invalid choice. Please select 1-4.")

if __name__ == "__main__":
    main()