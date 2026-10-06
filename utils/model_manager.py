# Model Management System for Swappable AI Models

import os
import json
import shutil
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import tensorflow as tf
import numpy as np
import streamlit as st
import zipfile
import requests
from urllib.parse import urlparse
import pickle

class ModelManager:
    """
    Advanced model management system for swappable AI models
    Handles model versioning, validation, and deployment
    """
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.active_models_dir = self.models_dir / "active"
        self.backup_models_dir = self.models_dir / "backups"
        self.registry_dir = self.models_dir / "registry"
        self.temp_dir = self.models_dir / "temp"
        
        for directory in [self.active_models_dir, self.backup_models_dir, self.registry_dir, self.temp_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Model registry file
        self.registry_file = self.registry_dir / "model_registry.json"
        self._initialize_registry()
        
        # Supported model formats
        self.supported_formats = ['.h5', '.pb', '.tflite', '.onnx', '.pkl']
        
        # Model requirements schema
        self.model_schema = {
            'model_name': str,
            'dataset_type': str,  # bone_fracture, pneumonia, cardiomegaly, arthritis, osteoporosis
            'version': str,
            'architecture': str,
            'input_shape': list,
            'num_classes': int,
            'class_names': list,
            'performance_metrics': dict,
            'training_info': dict,
            'file_path': str,
            'file_size': int,
            'file_hash': str,
            'created_date': str,
            'description': str,
            'tags': list
        }
    
    def _initialize_registry(self):
        """Initialize model registry if it doesn't exist"""
        if not self.registry_file.exists():
            initial_registry = {
                'version': '1.0',
                'created': datetime.now().isoformat(),
                'models': {},
                'active_models': {
                    'bone_fracture': None,
                    'pneumonia': None,
                    'cardiomegaly': None,
                    'arthritis': None,
                    'osteoporosis': None
                }
            }
            with open(self.registry_file, 'w') as f:
                json.dump(initial_registry, f, indent=2)
    
    def _load_registry(self) -> Dict[str, Any]:
        """Load model registry"""
        try:
            with open(self.registry_file, 'r') as f:
                registry = json.load(f)
            
            # Ensure all model types are present in active_models
            expected_models = ['bone_fracture', 'pneumonia', 'cardiomegaly', 'arthritis', 'osteoporosis']
            
            if 'active_models' not in registry:
                registry['active_models'] = {}
            
            # Add missing model types
            updated = False
            for model_type in expected_models:
                if model_type not in registry['active_models']:
                    registry['active_models'][model_type] = None
                    updated = True
            
            # Save updated registry if changes were made
            if updated:
                self._save_registry(registry)
            
            return registry
        except FileNotFoundError:
            # Initialize registry if it doesn't exist
            self._initialize_registry()
            with open(self.registry_file, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            st.error(f"Registry file is corrupted: {e}")
            # Backup and reinitialize
            backup_path = self.registry_file.with_suffix('.backup')
            if self.registry_file.exists():
                shutil.copy2(self.registry_file, backup_path)
            self._initialize_registry()
            with open(self.registry_file, 'r') as f:
                return json.load(f)
    
    def _save_registry(self, registry: Dict[str, Any]):
        """Save model registry"""
        try:
            registry['last_modified'] = datetime.now().isoformat()
            # Create backup before saving
            if self.registry_file.exists():
                backup_path = self.registry_file.with_suffix('.backup')
                shutil.copy2(self.registry_file, backup_path)
            
            with open(self.registry_file, 'w') as f:
                json.dump(registry, f, indent=2)
        except Exception as e:
            st.error(f"Failed to save registry: {e}")
            raise
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of a file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def validate_model_structure(self, model_info: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate model information against schema"""
        errors = []
        
        # Check required fields
        for field, expected_type in self.model_schema.items():
            if field not in model_info:
                errors.append(f"Missing required field: {field}")
            elif not isinstance(model_info[field], expected_type):
                errors.append(f"Invalid type for {field}: expected {expected_type.__name__}, got {type(model_info[field]).__name__}")
        
        # Validate specific constraints
        if 'dataset_type' in model_info:
            valid_datasets = ['bone_fracture', 'pneumonia', 'cardiomegaly', 'arthritis', 'osteoporosis']
            if model_info['dataset_type'] not in valid_datasets:
                errors.append(f"Invalid dataset_type: {model_info['dataset_type']}. Must be one of {valid_datasets}")
        
        if 'input_shape' in model_info:
            if len(model_info['input_shape']) != 3:
                errors.append("input_shape must be a 3D tuple (height, width, channels)")
        
        if 'num_classes' in model_info and 'class_names' in model_info:
            if len(model_info['class_names']) != model_info['num_classes']:
                errors.append("Number of class_names must match num_classes")
        
        return len(errors) == 0, errors
    
    def register_model(self, model_file_path: str, model_info: Dict[str, Any]) -> bool:
        """Register a new model in the system"""
        
        model_path = Path(model_file_path)
        
        # Validate file exists and format
        if not model_path.exists():
            st.error(f"❌ Model file not found: {model_file_path}")
            return False
        
        if model_path.suffix not in self.supported_formats:
            st.error(f"❌ Unsupported model format: {model_path.suffix}")
            return False
        
        # Validate model information
        is_valid, errors = self.validate_model_structure(model_info)
        if not is_valid:
            st.error(f"❌ Model validation failed: {'; '.join(errors)}")
            return False
        
        try:
            # Load registry
            registry = self._load_registry()
            
            # Generate model ID
            model_id = f"{model_info['dataset_type']}_{model_info['model_name']}_{model_info['version']}"
            
            # Check if model already exists
            if model_id in registry['models']:
                st.warning(f"⚠️ Model {model_id} already exists. Creating new version.")
                # Increment version
                base_id = f"{model_info['dataset_type']}_{model_info['model_name']}"
                version_num = 1
                while f"{base_id}_v{version_num}" in registry['models']:
                    version_num += 1
                model_id = f"{base_id}_v{version_num}"
                model_info['version'] = f"v{version_num}"
            
            # Copy model to registry
            registered_model_path = self.registry_dir / f"{model_id}{model_path.suffix}"
            shutil.copy2(model_path, registered_model_path)
            
            # Calculate file hash and size
            model_info['file_path'] = str(registered_model_path)
            model_info['file_size'] = registered_model_path.stat().st_size
            model_info['file_hash'] = self._calculate_file_hash(registered_model_path)
            model_info['registered_date'] = datetime.now().isoformat()
            model_info['model_id'] = model_id
            
            # Add to registry
            registry['models'][model_id] = model_info
            self._save_registry(registry)
            
            st.success(f"✅ Model registered successfully: {model_id}")
            return True
            
        except Exception as e:
            st.error(f"❌ Failed to register model: {str(e)}")
            return False
    
    def activate_model(self, model_id: str) -> bool:
        """Activate a model for a specific dataset type"""
        
        registry = self._load_registry()
        
        if model_id not in registry['models']:
            st.error(f"❌ Model not found: {model_id}")
            return False
        
        model_info = registry['models'][model_id]
        dataset_type = model_info['dataset_type']
        
        try:
            # Backup current active model if exists
            current_active = registry['active_models'].get(dataset_type)
            if current_active:
                self._backup_active_model(dataset_type, current_active)
            
            # Copy model to active directory
            # Handle both 'file_path' and 'model_path' for compatibility
            source_path_str = model_info.get('file_path') or model_info.get('model_path')
            source_path = self.models_dir / source_path_str if not Path(source_path_str).is_absolute() else Path(source_path_str)
            active_path = self.active_models_dir / f"{dataset_type}_model{source_path.suffix}"
            
            shutil.copy2(source_path, active_path)
            
            # Update registry
            registry['active_models'][dataset_type] = model_id
            self._save_registry(registry)
            
            # Save model metadata for inference
            metadata_path = self.active_models_dir / f"{dataset_type}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(model_info, f, indent=2)
            
            st.success(f"✅ Model {model_id} activated for {dataset_type}")
            return True
            
        except Exception as e:
            st.error(f"❌ Failed to activate model: {str(e)}")
            return False
    
    def _backup_active_model(self, dataset_type: str, model_id: str):
        """Backup currently active model"""
        try:
            backup_dir = self.backup_models_dir / dataset_type
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Create timestamped backup
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"{model_id}_{timestamp}"
            
            active_model_path = self.active_models_dir / f"{dataset_type}_model.h5"
            active_metadata_path = self.active_models_dir / f"{dataset_type}_metadata.json"
            
            if active_model_path.exists():
                backup_model_path = backup_dir / f"{backup_name}.h5"
                shutil.copy2(active_model_path, backup_model_path)
            
            if active_metadata_path.exists():
                backup_metadata_path = backup_dir / f"{backup_name}_metadata.json"
                shutil.copy2(active_metadata_path, backup_metadata_path)
            
            st.info(f"📦 Previous model backed up: {backup_name}")
            
        except Exception as e:
            st.warning(f"⚠️ Failed to backup previous model: {str(e)}")
    
    def load_active_model(self, dataset_type: str) -> Tuple[Optional[tf.keras.Model], Optional[Dict[str, Any]]]:
        """Load currently active model for a dataset type"""
        
        model_path = self.active_models_dir / f"{dataset_type}_model.h5"
        metadata_path = self.active_models_dir / f"{dataset_type}_metadata.json"
        
        if not model_path.exists() or not metadata_path.exists():
            return None, None
        
        try:
            # Load model
            model = tf.keras.models.load_model(str(model_path))
            
            # Load metadata
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            return model, metadata
            
        except Exception as e:
            st.error(f"❌ Failed to load active model for {dataset_type}: {str(e)}")
            return None, None
    
    def list_models(self, dataset_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all registered models, optionally filtered by dataset type"""
        
        registry = self._load_registry()
        models = []
        
        for model_id, model_info in registry['models'].items():
            if dataset_type is None or model_info['dataset_type'] == dataset_type:
                # Add active status
                is_active = registry['active_models'].get(model_info['dataset_type']) == model_id
                model_info_copy = model_info.copy()
                model_info_copy['is_active'] = is_active
                model_info_copy['model_id'] = model_id
                models.append(model_info_copy)
        
        return models
    
    def delete_model(self, model_id: str) -> bool:
        """Delete a model from the registry"""
        
        registry = self._load_registry()
        
        if model_id not in registry['models']:
            st.error(f"❌ Model not found: {model_id}")
            return False
        
        try:
            model_info = registry['models'][model_id]
            
            # Check if model is currently active
            dataset_type = model_info['dataset_type']
            if registry['active_models'].get(dataset_type) == model_id:
                st.error(f"❌ Cannot delete active model. Please activate a different model first.")
                return False
            
            # Delete model file
            # Handle both 'file_path' and 'model_path' for compatibility
            model_path_str = model_info.get('file_path') or model_info.get('model_path')
            if model_path_str:
                model_path = self.models_dir / model_path_str if not Path(model_path_str).is_absolute() else Path(model_path_str)
                if model_path.exists():
                    model_path.unlink()
            
            # Remove from registry
            del registry['models'][model_id]
            self._save_registry(registry)
            
            st.success(f"✅ Model deleted: {model_id}")
            return True
            
        except Exception as e:
            st.error(f"❌ Failed to delete model: {str(e)}")
            return False
    
    def export_model(self, model_id: str, export_path: str) -> bool:
        """Export a model with its metadata"""
        
        registry = self._load_registry()
        
        if model_id not in registry['models']:
            st.error(f"❌ Model not found: {model_id}")
            return False
        
        try:
            model_info = registry['models'][model_id]
            export_path = Path(export_path)
            export_path.mkdir(parents=True, exist_ok=True)
            
            # Create export package
            package_path = export_path / f"{model_id}_package.zip"
            
            with zipfile.ZipFile(package_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Add model file
                # Handle both 'file_path' and 'model_path' for compatibility
                model_path_key = model_info.get('file_path') or model_info.get('model_path')
                if model_path_key:
                    model_file_path = self.models_dir / model_path_key if not Path(model_path_key).is_absolute() else Path(model_path_key)
                    if model_file_path.exists():
                        zipf.write(model_file_path, f"model{model_file_path.suffix}")
                    else:
                        st.warning(f"⚠️ Model file not found: {model_file_path}")
                else:
                    st.error("❌ No model file path found in model info")
                
                # Add metadata
                metadata_str = json.dumps(model_info, indent=2)
                zipf.writestr("metadata.json", metadata_str)
                
                # Add readme
                readme_content = f"""
# Medical X-ray Model Package: {model_id}

## Model Information
- **Name:** {model_info['model_name']}
- **Dataset Type:** {model_info['dataset_type']}
- **Version:** {model_info['version']}
- **Architecture:** {model_info['architecture']}

## Usage
1. Extract this package
2. Load the model using TensorFlow: `tf.keras.models.load_model('model.h5')`
3. Check metadata.json for input requirements and class names

## Performance
{json.dumps(model_info.get('performance_metrics', {}), indent=2)}
"""
                zipf.writestr("README.md", readme_content)
            
            st.success(f"✅ Model exported: {package_path}")
            return True
            
        except Exception as e:
            st.error(f"❌ Failed to export model: {str(e)}")
            return False
    
    def import_model_package(self, package_path: str) -> bool:
        """Import a model package"""
        
        package_path = Path(package_path)
        
        if not package_path.exists() or package_path.suffix != '.zip':
            st.error("❌ Invalid package file")
            return False
        
        try:
            # Extract to temp directory
            temp_extract_dir = self.temp_dir / f"import_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            temp_extract_dir.mkdir(parents=True, exist_ok=True)
            
            with zipfile.ZipFile(package_path, 'r') as zipf:
                zipf.extractall(temp_extract_dir)
            
            # Load metadata
            metadata_file = temp_extract_dir / "metadata.json"
            if not metadata_file.exists():
                st.error("❌ Invalid package: metadata.json not found")
                return False
            
            with open(metadata_file, 'r') as f:
                model_info = json.load(f)
            
            # Find model file
            model_files = [f for f in temp_extract_dir.glob("model*") if f.suffix in self.supported_formats]
            if not model_files:
                st.error("❌ Invalid package: model file not found")
                return False
            
            model_file = model_files[0]
            
            # Register the model
            success = self.register_model(str(model_file), model_info)
            
            # Cleanup temp directory
            shutil.rmtree(temp_extract_dir, ignore_errors=True)
            
            return success
            
        except Exception as e:
            st.error(f"❌ Failed to import model package: {str(e)}")
            return False
    
    def get_model_performance_comparison(self) -> Dict[str, Any]:
        """Get performance comparison of binary classification models only"""
        
        registry = self._load_registry()
        comparison = {}
        
        # Only show the 5 binary classification models
        binary_datasets = ['pneumonia', 'cardiomegaly', 'arthritis', 'osteoporosis', 'bone_fracture']
        
        for dataset_type in binary_datasets:
            comparison[dataset_type] = {
                'models': [],
                'active_model': registry['active_models'].get(dataset_type)
            }
            
            # Get models for this dataset type
            for model_id, model_info in registry['models'].items():
                if model_info['dataset_type'] == dataset_type:
                    performance = model_info.get('performance_metrics', {})
                    # Handle both 'accuracy' (float 0-1) and 'test_accuracy' (float 0-1)
                    # Also handle if accuracy is stored as percentage > 1
                    raw_accuracy = model_info.get('accuracy', performance.get('test_accuracy', performance.get('accuracy', 0)))
                    
                    # Normalize accuracy to 0-1 range if it's > 1 (percentage)
                    if raw_accuracy > 1.0:
                        accuracy = raw_accuracy / 100.0
                    else:
                        accuracy = raw_accuracy
                        
                    comparison[dataset_type]['models'].append({
                        'model_id': model_id,
                        'model_name': model_info['model_name'],
                        'version': model_info['version'],
                        'architecture': model_info['architecture'],
                        'test_accuracy': accuracy,
                        'is_active': registry['active_models'].get(dataset_type) == model_id
                    })
            
            # Sort by test accuracy
            comparison[dataset_type]['models'].sort(key=lambda x: x['test_accuracy'], reverse=True)
        
        return comparison
    
    def cleanup_orphaned_files(self) -> Tuple[int, List[str]]:
        """Clean up orphaned model files that aren't in registry"""
        registry = self._load_registry()
        
        # Get registered files
        registry_files = set()
        for model_info in registry['models'].values():
            file_path = Path(model_info['file_path'])
            if not file_path.is_absolute():
                file_path = self.models_dir / file_path
            registry_files.add(str(file_path))
        
        # Find orphaned files
        actual_files = set(str(f) for f in self.models_dir.glob('*.h5'))
        orphaned = actual_files - registry_files
        
        # Remove orphaned files
        removed_files = []
        for orphan_path in orphaned:
            try:
                Path(orphan_path).unlink()
                removed_files.append(orphan_path)
            except Exception as e:
                st.warning(f"Failed to remove {orphan_path}: {e}")
        
        return len(removed_files), removed_files
    
    def validate_registry_integrity(self) -> List[str]:
        """Validate registry integrity and return list of issues"""
        issues = []
        registry = self._load_registry()
        
        # Check for missing files
        for model_id, model_info in registry['models'].items():
            file_path = Path(model_info['file_path'])
            if not file_path.is_absolute():
                file_path = self.models_dir / file_path
            
            if not file_path.exists():
                issues.append(f"Missing file for {model_id}: {file_path}")
            else:
                # Check if file size matches (if recorded)
                if 'file_size' in model_info:
                    actual_size = file_path.stat().st_size
                    recorded_size = model_info['file_size']
                    if actual_size != recorded_size:
                        issues.append(f"File size mismatch for {model_id}: recorded={recorded_size}, actual={actual_size}")
        
        return issues

def display_model_management_interface():
    """Display model management interface in Streamlit"""
    
    st.markdown("## 🔧 Model Management System")
    
    manager = ModelManager()
    
    # Tabs for different management functions
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 Model Registry", 
        "🚀 Activate Models", 
        "📦 Import/Export", 
        "📊 Performance Comparison",
        "🛠️ Model Utilities"
    ])
    
    with tab1:
        st.markdown("### 📋 Registered Models")
        
        # Filter options
        col1, col2 = st.columns([1, 1])
        with col1:
            dataset_filter = st.selectbox(
                "Filter by dataset type:",
                ["All"] + ['pneumonia', 'cardiomegaly', 'arthritis', 'osteoporosis', 'bone_fracture']
            )
        with col2:
            show_active_only = st.checkbox("Show active models only")
        
        # Get models
        filter_type = None if dataset_filter == "All" else dataset_filter
        models = manager.list_models(filter_type)
        
        if show_active_only:
            models = [m for m in models if m.get('is_active', False)]
        
        if models:
            for model in models:
                with st.expander(f"{'🟢 ' if model.get('is_active') else '⚪ '}{model['model_id']}", expanded=False):
                    col1, col2, col3 = st.columns([2, 1, 1])
                    
                    with col1:
                        model_name = model.get('model_name', f"{model.get('dataset_type', 'Unknown')} Model")
                        st.write(f"**Name:** {model_name}")
                        st.write(f"**Architecture:** {model.get('architecture', 'N/A')}")
                        st.write(f"**Classes:** {len(model.get('classes', model.get('class_names', [])))}")
                        st.write(f"**Description:** {model.get('description', 'N/A')}")
                    
                    with col2:
                        performance = model.get('performance_metrics', {})
                        raw_accuracy = model.get('accuracy', performance.get('test_accuracy', performance.get('accuracy', 0)))
                        
                        if raw_accuracy > 1.0:
                            accuracy = raw_accuracy / 100.0
                        else:
                            accuracy = raw_accuracy
                            
                        st.metric("Test Accuracy", f"{accuracy:.1%}")
                        st.write(f"**Size:** {model.get('file_size', 0) / 1024 / 1024:.1f} MB")
                        st.write(f"**Version:** {model['version']}")
                    
                    with col3:
                        if model.get('is_active'):
                            st.success("🟢 Active")
                        else:
                            if st.button(f"Activate", key=f"activate_{model['model_id']}"):
                                manager.activate_model(model['model_id'])
                                st.rerun()
                        
                        if st.button(f"Delete", key=f"delete_{model['model_id']}", type="secondary"):
                            manager.delete_model(model['model_id'])
                            st.rerun()
        else:
            st.info("No models found matching the selected criteria.")
    
    with tab2:
        st.markdown("### 🚀 Activate Binary Classification Models")
        st.info("💡 **Binary Classification System**: Each model specializes in detecting one specific condition vs normal.")
        
        # Define the 5 binary models with descriptions
        binary_models = [
            ('pneumonia', '🫁 Pneumonia Detection', 'Detects pneumonia in chest X-rays'),
            ('cardiomegaly', '❤️ Heart Enlargement Detection', 'Detects cardiomegaly in chest X-rays'),
            ('arthritis', '🦵 Knee Arthritis Detection', 'Detects osteoarthritis in knee X-rays'),
            ('osteoporosis', '🦴 Knee Osteoporosis Detection', 'Detects osteoporosis in knee X-rays'),
            ('bone_fracture', '💀 Bone Fracture Detection', 'Detects fractures in limb X-rays')
        ]
        
        for dataset_type, display_name, description in binary_models:
            st.markdown(f"#### {display_name}")
            st.caption(description)
            
            models = manager.list_models(dataset_type)
            active_model = next((m for m in models if m.get('is_active')), None)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                if active_model:
                    # Handle accuracy retrieval and normalization
                    performance = active_model.get('performance_metrics', {})
                    raw_accuracy = active_model.get('accuracy', performance.get('test_accuracy', performance.get('accuracy', 0)))
                    
                    if raw_accuracy > 1.0:
                        accuracy = raw_accuracy / 100.0
                    else:
                        accuracy = raw_accuracy
                        
                    st.success(f"🟢 Active: {active_model.get('model_name', active_model['model_id'])} ({active_model.get('version', 'v1.0')}) - Accuracy: {accuracy:.1%}")
                else:
                    st.warning("⚠️ No active model")
                
                # Model selection
                available_models = []
                for m in models:
                    perf = m.get('performance_metrics', {})
                    raw_acc = m.get('accuracy', perf.get('test_accuracy', perf.get('accuracy', 0)))
                    if raw_acc > 1.0:
                        acc = raw_acc / 100.0
                    else:
                        acc = raw_acc
                    available_models.append((m['model_id'], f"{m.get('model_name', m.get('dataset_type', 'Unknown'))} ({m.get('version', 'v1.0')}) - Acc: {acc:.1%}"))
                
                if available_models:
                    selected_model = st.selectbox(
                        f"Available models:",
                        available_models,
                        format_func=lambda x: x[1],
                        key=f"select_{dataset_type}"
                    )
            
            with col2:
                if available_models:
                    if st.button(f"Activate Selected", key=f"activate_btn_{dataset_type}"):
                        manager.activate_model(selected_model[0])
                        st.rerun()
                else:
                    st.info("No models available")
            
            st.markdown("---")
    
    with tab3:
        st.markdown("### 📦 Import/Export Models")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📥 Import Model")
            uploaded_file = st.file_uploader("Upload model package (.zip)", type="zip")
            
            if uploaded_file and st.button("Import Model"):
                # Save uploaded file temporarily
                temp_path = manager.temp_dir / uploaded_file.name
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                success = manager.import_model_package(str(temp_path))
                if success:
                    st.rerun()
        
        with col2:
            st.markdown("#### 📤 Export Model")
            models = manager.list_models()
            if models:
                model_options = [(m['model_id'], f"{m['model_name']} ({m['version']})") for m in models]
                selected_export = st.selectbox(
                    "Select model to export:",
                    model_options,
                    format_func=lambda x: x[1]
                )
                
                if st.button("Export Model"):
                    export_path = manager.models_dir / "exports"
                    success = manager.export_model(selected_export[0], str(export_path))
                    if success:
                        st.success("Model exported successfully!")
    
    with tab4:
        st.markdown("### 📊 Binary Classification Models Performance")
        st.info("💡 **Performance comparison for all 5 binary classification models**")
        
        comparison = manager.get_model_performance_comparison()
        
        # Define display names for binary models
        model_display_names = {
            'pneumonia': '🫁 Pneumonia Detection',
            'cardiomegaly': '❤️ Heart Enlargement Detection',
            'arthritis': '🦵 Knee Arthritis Detection', 
            'osteoporosis': '🦴 Knee Osteoporosis Detection',
            'bone_fracture': '💀 Bone Fracture Detection'
        }
        
        for dataset_type, data in comparison.items():
            display_name = model_display_names.get(dataset_type, dataset_type.replace('_', ' ').title())
            st.markdown(f"#### {display_name}")
            
            if data['models']:
                # Display model information in a clean format
                for model in data['models']:
                    accuracy = model['test_accuracy']
                    performance_level = "🟢 Medical Grade" if accuracy >= 0.9 else "🟡 Research Grade" if accuracy >= 0.7 else "🔴 Development"
                    
                    col1, col2, col3 = st.columns([2, 1, 1])
                    
                    with col1:
                        st.markdown(f"**{model['model_name']}** v{model['version']}")
                        st.text(f"Architecture: {model['architecture']}")
                    
                    with col2:
                        st.metric("Accuracy", f"{accuracy:.1%}")
                        st.text(performance_level)
                    
                    with col3:
                        status_icon = '🟢 Active' if model['is_active'] else '⚪ Inactive'
                        st.text(status_icon)
                
                # Best model highlight
                if data['models']:
                    best_model = max(data['models'], key=lambda x: x['test_accuracy'])
                    accuracy = best_model['test_accuracy']
                    st.success(f"🏆 Best performing: **{best_model['model_name']}** - {accuracy:.1%} accuracy")
            else:
                st.info(f"No models available for {display_name}")
            
            st.markdown("---")
    
    with tab5:
        st.markdown("### 🛠️ Model Utilities")
        
        # Registry statistics
        registry = manager._load_registry()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Models", len(registry['models']))
        with col2:
            active_count = sum(1 for v in registry['active_models'].values() if v is not None)
            st.metric("Active Models", active_count)
        with col3:
            total_size = sum(model.get('file_size', 0) for model in registry['models'].values())
            st.metric("Total Size", f"{total_size / 1024 / 1024:.1f} MB")
        
        # Registry health check
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔍 Run Registry Health Check"):
                with st.spinner("Checking registry health..."):
                    issues = manager.validate_registry_integrity()
                    
                    if issues:
                        st.warning(f"⚠️ Found {len(issues)} issues:")
                        for issue in issues:
                            st.write(f"• {issue}")
                    else:
                        st.success("✅ Registry is healthy!")
        
        with col2:
            if st.button("🧹 Cleanup Orphaned Files"):
                with st.spinner("Cleaning up orphaned files..."):
                    removed_count, removed_files = manager.cleanup_orphaned_files()
                    
                    if removed_count > 0:
                        st.success(f"✅ Removed {removed_count} orphaned files:")
                        for file in removed_files:
                            st.write(f"• {file}")
                    else:
                        st.info("ℹ️ No orphaned files found to clean up.")

# Example usage
if __name__ == "__main__":
    print("Model Management System initialized!")
    
    # Test model manager
    manager = ModelManager()
    models = manager.list_models()
    print(f"Found {len(models)} registered models")