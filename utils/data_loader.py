# Data Loading and Preparation Module for Medical X-ray AI System

import os
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import streamlit as st
from typing import Tuple, List, Dict, Any, Optional
import glob
import json
from pathlib import Path
import shutil

class MedicalDataLoader:
    """
    Comprehensive data loader for medical X-ray datasets
    Supports multiple dataset formats and automatic organization
    """
    
    def __init__(self, base_data_path: str = "Dataset"):
        self.base_path = Path(base_data_path)
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.dcm', '.dicom']
        
        # Dataset configurations - Updated for 5 Binary Models with New Organized Structure
        self.dataset_configs = {
            # 5 Binary Models - Matching Actual Dataset Structure
            'bone_fracture': {
                'sources': [
                    'ARM/MURA_Organized/Forearm',
                    'ARM/MURA_Organized/Humerus'
                ],
                'classes': ['Normal', 'Fracture'],
                'target_size': (224, 224),
                'binary': True,
                'description': '🦴 Binary bone fracture detection dataset (ARM)'
            },
            'pneumonia': {
                'sources': [
                    'CHEST/Pneumonia_Organized'
                ],
                'classes': ['Normal', 'Pneumonia'],
                'target_size': (224, 224),
                'binary': True,
                'description': '🫁 Binary pneumonia detection dataset (CHEST)'
            },
            'cardiomegaly': {
                'sources': [
                    'CHEST/cardiomelgy'
                ],
                'classes': ['Normal', 'Cardiomegaly'],
                'target_size': (224, 224),
                'binary': True,
                'description': '❤️ Binary cardiomegaly detection dataset (CHEST)'
            },
            'arthritis': {
                'sources': [
                    'KNEE/Osteoarthritis/Combined_Osteoarthritis_Dataset'
                ],
                'classes': ['Normal', 'Arthritis'],
                'target_size': (224, 224),
                'binary': True,
                'description': '🦵 Binary arthritis detection dataset (KNEE)'
            },
            'osteoporosis': {
                'sources': [
                    'KNEE/Osteoporosis/Combined_Osteoporosis_Dataset'
                ],
                'classes': ['Normal', 'Osteoporosis'],
                'target_size': (224, 224),
                'binary': True,
                'description': '🦴 Binary osteoporosis detection dataset (KNEE)'
            }
        }
    
    def scan_datasets(self) -> Dict[str, Any]:
        """Scan and analyze available datasets"""
        dataset_info = {}
        
        # Check if base path exists
        if not self.base_path.exists():
            return {}  # Return empty dict, error will be shown in display function
        
        for dataset_name, config in self.dataset_configs.items():
            dataset_info[dataset_name] = {
                'sources_found': [],
                'total_images': 0,
                'class_distribution': {},
                'file_formats': set(),
                'ready_for_training': False
            }
            
            for source in config['sources']:
                source_path = self.base_path / source
                if source_path.exists():
                    dataset_info[dataset_name]['sources_found'].append(str(source_path))
                    
                    # Count images
                    image_count = self._count_images_in_directory(source_path)
                    dataset_info[dataset_name]['total_images'] += image_count
                    
                    # Analyze structure
                    class_dist = self._analyze_directory_structure(source_path)
                    for class_name, count in class_dist.items():
                        if class_name in dataset_info[dataset_name]['class_distribution']:
                            dataset_info[dataset_name]['class_distribution'][class_name] += count
                        else:
                            dataset_info[dataset_name]['class_distribution'][class_name] = count
                else:
                    st.warning(f"⚠️ Dataset not found: {source}")
            
            # Check if ready for training
            if (dataset_info[dataset_name]['total_images'] > 100 and 
                len(dataset_info[dataset_name]['class_distribution']) >= 2):
                dataset_info[dataset_name]['ready_for_training'] = True
        
        return dataset_info
    
    def _count_images_in_directory(self, directory: Path) -> int:
        """Count images in a directory recursively"""
        count = 0
        for ext in self.supported_formats:
            count += len(list(directory.rglob(f"*{ext}")))
            count += len(list(directory.rglob(f"*{ext.upper()}")))
        return count
    
    def _analyze_directory_structure(self, directory: Path) -> Dict[str, int]:
        """Analyze directory structure to understand class distribution"""
        class_distribution = {}
        
        # Look for common directory patterns
        subdirs = [d for d in directory.iterdir() if d.is_dir()]
        
        if subdirs:
            # If subdirectories exist, assume they represent classes
            for subdir in subdirs:
                if subdir.name.lower() in ['train', 'test', 'val', 'valid', 'validation']:
                    # Handle train/test/val structure
                    train_test_subdirs = [d for d in subdir.iterdir() if d.is_dir()]
                    for class_dir in train_test_subdirs:
                        class_name = self._normalize_class_name(class_dir.name)
                        count = self._count_images_in_directory(class_dir)
                        if class_name in class_distribution:
                            class_distribution[class_name] += count
                        else:
                            class_distribution[class_name] = count
                else:
                    # Direct class directories
                    class_name = self._normalize_class_name(subdir.name)
                    count = self._count_images_in_directory(subdir)
                    class_distribution[class_name] = count
        else:
            # All images in one directory
            count = self._count_images_in_directory(directory)
            class_distribution['unknown'] = count
        
        return class_distribution
    
    def _normalize_class_name(self, name: str) -> str:
        """Normalize class names for consistency"""
        name = name.lower().strip()
        
        # Mapping for common variations
        name_mappings = {
            # Normal variants
            'normal': 'Normal',
            'no_fracture': 'Normal',
            'not_fractured': 'Normal',
            'not fractured': 'Normal',
            'healthy': 'Normal',
            'negative': 'Normal',    # For ARM dataset
            'false': 'Normal',       # For cardiomegaly dataset
            
            # Fracture variants
            'fracture': 'Fracture',
            'fractured': 'Fracture',
            'broken': 'Fracture',
            'crack': 'Fracture',
            'positive': 'Fracture',  # For ARM dataset
            
            # Pneumonia variants
            'pneumonia': 'Pneumonia',
            'bacterial': 'Pneumonia',
            'viral': 'Pneumonia',
            
            # Cardiomegaly variants
            'cardiomegaly': 'Cardiomegaly',
            'enlarged_heart': 'Cardiomegaly',
            'heart_enlargement': 'Cardiomegaly',
            'true': 'Cardiomegaly',  # For cardiomegaly dataset
            
            # Osteoporosis variants
            'osteoporosis': 'Osteoporosis',
            'osteopenia': 'Osteoporosis',
            'bone_loss': 'Osteoporosis',
            
            # Arthritis variants
            'arthritis': 'Arthritis',
            'osteoarthritis': 'Arthritis',
            'joint_degeneration': 'Arthritis'
        }
        
        return name_mappings.get(name, name.capitalize())
    
    def prepare_dataset(self, dataset_name: str, test_size: float = 0.2, val_size: float = 0.1) -> Dict[str, Any]:
        """Prepare dataset for training with proper train/val/test splits"""
        
        if dataset_name not in self.dataset_configs:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        config = self.dataset_configs[dataset_name]
        
        st.info(f"📁 Preparing {dataset_name} dataset...")
        
        # Collect all image paths and labels
        image_paths = []
        labels = []
        
        for source in config['sources']:
            source_path = self.base_path / source
            if source_path.exists():
                paths, lbls = self._collect_images_and_labels(source_path, dataset_name)
                image_paths.extend(paths)
                labels.extend(lbls)
        
        if len(image_paths) == 0:
            st.error(f"❌ No images found for {dataset_name}")
            return {}
        
        st.success(f"✅ Found {len(image_paths)} images for {dataset_name}")
        
        # Convert labels to indices
        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(labels)
        
        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            image_paths, encoded_labels, 
            test_size=test_size, 
            stratify=encoded_labels, 
            random_state=42
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, 
            test_size=val_size/(1-test_size), 
            stratify=y_temp, 
            random_state=42
        )
        
        dataset_info = {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'label_encoder': label_encoder,
            'classes': list(label_encoder.classes_),
            'num_classes': len(label_encoder.classes_),
            'target_size': config['target_size'],
            'total_samples': len(image_paths),
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'test_samples': len(X_test)
        }
        
        # Save dataset info
        self._save_dataset_info(dataset_name, dataset_info)
        
        st.success(f"📊 Dataset prepared: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
        
        return dataset_info
    
    def _collect_images_and_labels(self, source_path: Path, dataset_name: str) -> Tuple[List[str], List[str]]:
        """Collect image paths and their corresponding labels"""
        image_paths = []
        labels = []
        
        # Special handling for cardiomegaly dataset with nested structure
        if 'cardiomelgy' in str(source_path) or 'CHEST/cardiomelgy' in str(source_path):
            # Handle cardiomegaly's nested train/train and test/test structure
            for split_dir in source_path.iterdir():
                if split_dir.is_dir() and split_dir.name.lower() in ['train', 'test']:
                    # Check for nested directory with same name
                    nested_split = split_dir / split_dir.name
                    if nested_split.exists() and nested_split.is_dir():
                        for class_dir in nested_split.iterdir():
                            if class_dir.is_dir():
                                class_name = self._normalize_class_name(class_dir.name)
                                for image_path in self._get_images_in_directory(class_dir):
                                    image_paths.append(str(image_path))
                                    labels.append(class_name)
        # Check if it's a structured dataset (train/val/test subdirs)
        elif any(subdir.name.lower() in ['train', 'test', 'val', 'valid'] for subdir in source_path.iterdir() if subdir.is_dir()):
            # Handle structured dataset
            for split_dir in source_path.iterdir():
                if split_dir.is_dir() and split_dir.name.lower() in ['train', 'test', 'val', 'valid']:
                    for class_dir in split_dir.iterdir():
                        if class_dir.is_dir():
                            class_name = self._normalize_class_name(class_dir.name)
                            for image_path in self._get_images_in_directory(class_dir):
                                image_paths.append(str(image_path))
                                labels.append(class_name)
        else:
            # Handle direct class directories
            for class_dir in source_path.iterdir():
                if class_dir.is_dir():
                    class_name = self._normalize_class_name(class_dir.name)
                    for image_path in self._get_images_in_directory(class_dir):
                        image_paths.append(str(image_path))
                        labels.append(class_name)
        
        return image_paths, labels
    
    def _get_images_in_directory(self, directory: Path) -> List[Path]:
        """Get all image files in a directory"""
        images = []
        for ext in self.supported_formats:
            images.extend(directory.glob(f"*{ext}"))
            images.extend(directory.glob(f"*{ext.upper()}"))
        return images
    
    def _save_dataset_info(self, dataset_name: str, dataset_info: Dict[str, Any]):
        """Save dataset information to JSON file"""
        # Create info directory if it doesn't exist
        info_dir = Path("models") / "dataset_info"
        info_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare serializable data
        serializable_info = {
            'classes': dataset_info['classes'],
            'num_classes': dataset_info['num_classes'],
            'target_size': dataset_info['target_size'],
            'total_samples': dataset_info['total_samples'],
            'train_samples': dataset_info['train_samples'],
            'val_samples': dataset_info['val_samples'],
            'test_samples': dataset_info['test_samples'],
            'dataset_name': dataset_name
        }
        
        info_file = info_dir / f"{dataset_name}_info.json"
        with open(info_file, 'w') as f:
            json.dump(serializable_info, f, indent=2)
        
        st.success(f"💾 Dataset info saved: {info_file}")
    
    def create_tf_dataset(self, dataset_info: Dict[str, Any], batch_size: int = 32, shuffle: bool = True):
        """Create TensorFlow datasets for training"""
        
        # Ensure target_size is in the correct format
        target_size = dataset_info.get('target_size', (224, 224))
        if isinstance(target_size, tuple):
            target_size = list(target_size)
        elif not isinstance(target_size, list):
            target_size = [224, 224]
        
        # Update dataset_info with proper target_size format
        dataset_info = dataset_info.copy()
        dataset_info['target_size'] = target_size
        
        def load_and_preprocess_image(path, label):
            try:
                # Read and decode image file
                image = tf.io.read_file(path)
                
                # Try different decoding methods for medical images
                try:
                    # First try with automatic format detection
                    image = tf.image.decode_image(image, channels=3, expand_animations=False)
                except tf.errors.InvalidArgumentError:
                    # If that fails, try JPEG decoding
                    try:
                        image = tf.image.decode_jpeg(image, channels=3)
                    except tf.errors.InvalidArgumentError:
                        # Finally try PNG decoding
                        try:
                            image = tf.image.decode_png(image, channels=3)
                        except tf.errors.InvalidArgumentError:
                            # If all fail, create a black placeholder image
                            print(f"Warning: Could not decode image {path}, using placeholder")
                            image = tf.zeros(dataset_info['target_size'] + [3], dtype=tf.uint8)
                
                # Ensure image has proper shape
                image = tf.ensure_shape(image, [None, None, 3])
                
                # Resize to target size
                image = tf.image.resize(image, dataset_info['target_size'])
                
                # Normalize to [0, 1] range
                image = tf.cast(image, tf.float32) / 255.0
                
                return image, label
                
            except Exception as e:
                # If everything fails, create a placeholder
                print(f"Error processing image {path}: {e}")
                placeholder = tf.zeros(dataset_info['target_size'] + [3], dtype=tf.float32)
                return placeholder, label
        
        # Create datasets
        train_ds = tf.data.Dataset.from_tensor_slices((dataset_info['X_train'], dataset_info['y_train']))
        val_ds = tf.data.Dataset.from_tensor_slices((dataset_info['X_val'], dataset_info['y_val']))
        test_ds = tf.data.Dataset.from_tensor_slices((dataset_info['X_test'], dataset_info['y_test']))
        
        # Apply preprocessing
        train_ds = train_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
        val_ds = val_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
        test_ds = test_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
        
        # Configure for performance
        if shuffle:
            train_ds = train_ds.shuffle(1000)
        
        # Ensure minimum dataset size for batching
        min_train_size = len(dataset_info['X_train'])
        min_val_size = len(dataset_info['X_val'])
        
        # Adjust batch size if necessary
        effective_batch_size = min(batch_size, min_train_size // 2, min_val_size // 2)
        effective_batch_size = max(1, effective_batch_size)  # At least 1
        
        if effective_batch_size != batch_size:
            print(f"Adjusted batch size from {batch_size} to {effective_batch_size} for dataset compatibility")
        
        # Use drop_remainder=True to ensure consistent batch sizes
        train_ds = train_ds.batch(effective_batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
        val_ds = val_ds.batch(effective_batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
        test_ds = test_ds.batch(effective_batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
        
        return train_ds, val_ds, test_ds

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_cached_dataset_info():
    """Get cached dataset information"""
    loader = MedicalDataLoader()
    return loader.scan_datasets()

def display_dataset_overview():
    """Display dataset overview in Streamlit"""
    
    # Add refresh button for manual cache clearing
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🔄 Refresh", key="refresh_datasets"):
            st.cache_data.clear()
            st.rerun()
    
    # Use cached dataset info
    dataset_info = get_cached_dataset_info()
    
    # Check if datasets were found
    if not dataset_info:
        st.error("❌ Dataset directory not found or no datasets available.")
        st.info("💡 Please ensure the Dataset folder exists in your project directory and contains medical imaging data.")
        return
    
    # Show cache status
    st.caption("📋 Dataset information cached for faster loading. Use refresh button to update.")
    
    for dataset_name, info in dataset_info.items():
        with st.expander(f"📁 {dataset_name.replace('_', ' ').title()} Dataset"):
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("**Dataset Status:**")
                if info['ready_for_training']:
                    st.success("✅ Ready for training")
                else:
                    st.warning("⚠️ Needs more data")
                
                st.markdown(f"**Total Images:** {info['total_images']}")
                st.markdown(f"**Sources Found:** {len(info['sources_found'])}")
            
            with col2:
                st.markdown("**Class Distribution:**")
                for class_name, count in info['class_distribution'].items():
                    st.markdown(f"• {class_name}: {count}")
            
            # Initialize preparation state
            prep_key = f"preparing_{dataset_name}"
            if prep_key not in st.session_state:
                st.session_state[prep_key] = False
            
            if not st.session_state[prep_key] and st.button(f"🚀 Prepare {dataset_name} for Training", key=f"prep_{dataset_name}"):
                st.session_state[prep_key] = True
                with st.spinner(f"Preparing {dataset_name} dataset..."):
                    try:
                        # Create loader instance within button scope
                        data_loader = MedicalDataLoader()
                        prepared_info = data_loader.prepare_dataset(dataset_name)
                        if prepared_info:
                            st.balloons()
                            st.success(f"🎉 **{dataset_name.replace('_', ' ').title()} Dataset Prepared Successfully!**")
                            
                            # Show preparation summary
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Training Samples", prepared_info.get('train_samples', 'N/A'))
                            with col2:
                                st.metric("Validation Samples", prepared_info.get('val_samples', 'N/A'))
                            with col3:
                                st.metric("Test Samples", prepared_info.get('test_samples', 'N/A'))
                            
                            st.info(f"📋 **Dataset Ready:** You can now use this dataset for model training!")
                            
                            # Show detailed info in expander
                            with st.expander("📊 View Detailed Preparation Info", expanded=False):
                                st.json(prepared_info)
                        else:
                            st.error(f"❌ Failed to prepare {dataset_name} dataset. Please check the data sources.")
                    except Exception as e:
                        st.error(f"❌ Error preparing dataset: {str(e)}")
                        st.info("💡 Tip: Ensure the dataset folder structure is correct and contains valid image files.")
                    finally:
                        # Reset preparation state
                        st.session_state[prep_key] = False
            
            elif st.session_state[prep_key]:
                st.info("📊 Dataset preparation in progress... Please wait.")

# Example usage
if __name__ == "__main__":
    print("Medical Data Loader initialized!")
    
    # Test dataset scanning
    loader = MedicalDataLoader()
    info = loader.scan_datasets()
    
    for dataset, details in info.items():
        print(f"\n{dataset}: {details['total_images']} images, Ready: {details['ready_for_training']}")