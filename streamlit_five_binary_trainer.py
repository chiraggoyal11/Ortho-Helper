#!/usr/bin/env python3
"""
Streamlit GUI for Training 5 Binary Classification Models
Interactive Medical X-ray AI Training Interface (Updated with Bone Fracture Model)
"""

import streamlit as st
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import time

# Page configuration
st.set_page_config(
    page_title="5 Binary Models Trainer",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set memory growth for GPU
@st.cache_resource
def configure_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            return f"✅ GPU configured: {len(gpus)} device(s)"
        except RuntimeError as e:
            return f"⚠️ GPU configuration error: {e}"
    return "💻 Running on CPU"

class StreamlitFiveBinaryModelsTrainer:
    """
    Streamlit GUI trainer for 5 binary classification models
    """
    
    def __init__(self):
        self.input_shape = (224, 224, 3)
        self.model_configs = {
            'pneumonia': {
                'name': 'Pneumonia Detection',
                'dataset_path': 'Dataset/CHEST/chest_xray Pneumonia',
                'classes': ['NORMAL', 'PNEUMONIA'],
                'model_file': 'models/pneumonia_binary_model.h5',
                'description': 'Binary classification: Normal vs Pneumonia chest X-rays'
            },
            'cardiomegaly': {
                'name': 'Cardiomegaly Detection', 
                'dataset_path': 'Dataset/CHEST/cardiomelgy',
                'classes': ['normal', 'Cardiomegaly'],
                'model_file': 'models/cardiomegaly_binary_model.h5',
                'description': 'Binary classification: Normal vs Cardiomegaly chest X-rays'
            },
            'arthritis': {
                'name': 'Arthritis Detection',
                'dataset_path': 'Dataset/KNEE/Osteoarthritis Knee X-ray',
                'classes': ['Normal', 'Osteoarthritis'],  
                'model_file': 'models/arthritis_binary_model.h5',
                'description': 'Binary classification: Normal vs Arthritis knee X-rays'
            },
            'osteoporosis': {
                'name': 'Osteoporosis Detection',
                'dataset_path': 'Dataset/KNEE/Osteoporosis Knee', 
                'classes': ['Normal', 'Osteoporosis'],
                'model_file': 'models/osteoporosis_binary_model.h5',
                'description': 'Binary classification: Normal vs Osteoporosis knee X-rays'
            },
            'bone_fracture': {
                'name': 'Bone Fracture Detection',
                'dataset_path': 'Dataset/Bone_Fracture_Binary_Classification/Bone_Fracture_Binary_Classification',
                'classes': ['not fractured', 'fractured'],
                'model_file': 'models/bone_fracture_binary_model.h5',
                'description': 'Binary classification: Normal vs Fractured bones (hand/leg bones only)',
                'uses_subdirs': True
            }
        }
        
        # Create directories
        os.makedirs('models', exist_ok=True)
        os.makedirs('models/binary_backups', exist_ok=True)
        os.makedirs('models/registry', exist_ok=True)
    
    def check_dataset_availability(self):
        """Check which datasets are available"""
        dataset_status = {}
        
        for model_key, config in self.model_configs.items():
            path = config['dataset_path']
            uses_subdirs = config.get('uses_subdirs', False)
            
            if os.path.exists(path):
                try:
                    sample_count = 0
                    folders = []
                    
                    if uses_subdirs:
                        # Check train/val/test structure
                        subdirs = ['train', 'val', 'test']
                        missing_subdirs = []
                        
                        for subdir in subdirs:
                            subdir_path = os.path.join(path, subdir)
                            if os.path.exists(subdir_path):
                                subdir_folders = os.listdir(subdir_path)
                                folders.extend([f"{subdir}/{f}" for f in subdir_folders])
                                
                                for folder in subdir_folders:
                                    folder_path = os.path.join(subdir_path, folder)
                                    if os.path.isdir(folder_path):
                                        sample_count += len([f for f in os.listdir(folder_path) 
                                                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                            else:
                                missing_subdirs.append(subdir)
                        
                        if missing_subdirs:
                            dataset_status[model_key] = {
                                'available': False,
                                'error': f'Missing subdirectories: {missing_subdirs}',
                                'path': path
                            }
                        else:
                            dataset_status[model_key] = {
                                'available': True,
                                'folders': folders,
                                'sample_count': sample_count,
                                'path': path,
                                'structure': 'train/val/test'
                            }
                    else:
                        # Simple structure
                        folders = os.listdir(path)
                        for folder in folders:
                            folder_path = os.path.join(path, folder)
                            if os.path.isdir(folder_path):
                                sample_count += len([f for f in os.listdir(folder_path) 
                                                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                        
                        dataset_status[model_key] = {
                            'available': True,
                            'folders': folders,
                            'sample_count': sample_count,
                            'path': path,
                            'structure': 'simple'
                        }
                        
                except Exception as e:
                    dataset_status[model_key] = {
                        'available': False,
                        'error': str(e),
                        'path': path
                    }
            else:
                dataset_status[model_key] = {
                    'available': False,
                    'error': 'Path not found',
                    'path': path
                }
        
        return dataset_status
    
    def create_densenet121_model(self, model_name: str) -> keras.Model:
        """Create DenseNet121 model with dropout"""
        
        # Input layer
        inputs = keras.Input(shape=self.input_shape, name='input_layer')
        
        # Data augmentation
        data_augmentation = keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
            layers.RandomContrast(0.1),
        ], name="data_augmentation")
        
        x = data_augmentation(inputs)
        x = layers.Rescaling(1./255)(x)
        
        # DenseNet121 base
        base_model = DenseNet121(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape,
            pooling=None
        )
        
        base_model.trainable = False
        x = base_model(x, training=False)
        
        # Custom top with dropout
        x = layers.GlobalAveragePooling2D(name='global_avg_pooling')(x)
        x = layers.Dense(512, activation='relu', name='dense_1')(x)
        x = layers.BatchNormalization(name='batch_norm_1')(x)
        x = layers.Dropout(0.5, name='dropout_1')(x)
        
        x = layers.Dense(256, activation='relu', name='dense_2')(x)
        x = layers.BatchNormalization(name='batch_norm_2')(x)
        x = layers.Dropout(0.4, name='dropout_2')(x)
        
        x = layers.Dense(128, activation='relu', name='dense_3')(x)
        x = layers.BatchNormalization(name='batch_norm_3')(x)
        x = layers.Dropout(0.3, name='dropout_3')(x)
        
        # Grad-CAM target layer
        x = layers.Activation('relu', name='gradcam_target_layer')(x)
        x = layers.Dropout(0.2, name='dropout_final')(x)
        
        # Binary classification output
        outputs = layers.Dense(2, activation='softmax', name='predictions')(x)
        
        model = keras.Model(inputs, outputs, name=f'densenet121_{model_name}_binary')
        return model, base_model
    
    def prepare_data_generators(self, model_key: str, batch_size: int):
        """Prepare data generators"""
        
        config = self.model_configs[model_key]
        dataset_path = config['dataset_path']
        uses_subdirs = config.get('uses_subdirs', False)
        
        if uses_subdirs:
            # Handle train/val/test structure
            train_path = os.path.join(dataset_path, 'train')
            val_path = os.path.join(dataset_path, 'val')
            
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=15,
                width_shift_range=0.1,
                height_shift_range=0.1,
                shear_range=0.1,
                zoom_range=0.1,
                horizontal_flip=True,
                fill_mode='nearest'
            )
            
            val_datagen = ImageDataGenerator(rescale=1./255)
            
            train_generator = train_datagen.flow_from_directory(
                train_path,
                target_size=(224, 224),
                batch_size=batch_size,
                class_mode='categorical',
                shuffle=True,
                seed=42
            )
            
            val_generator = val_datagen.flow_from_directory(
                val_path,
                target_size=(224, 224),
                batch_size=batch_size,
                class_mode='categorical',
                shuffle=False,
                seed=42
            )
            
        else:
            # Handle simple structure with validation split
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=15,
                width_shift_range=0.1,
                height_shift_range=0.1,
                shear_range=0.1,
                zoom_range=0.1,
                horizontal_flip=True,
                fill_mode='nearest',
                validation_split=0.2
            )
            
            val_datagen = ImageDataGenerator(
                rescale=1./255,
                validation_split=0.2
            )
            
            train_generator = train_datagen.flow_from_directory(
                dataset_path,
                target_size=(224, 224),
                batch_size=batch_size,
                class_mode='categorical',
                subset='training',
                shuffle=True,
                seed=42
            )
            
            val_generator = val_datagen.flow_from_directory(
                dataset_path,
                target_size=(224, 224),
                batch_size=batch_size,
                class_mode='categorical',
                subset='validation',
                shuffle=False,
                seed=42
            )
        
        return train_generator, val_generator
    
    def train_single_model(self, model_key: str, epochs: int, batch_size: int, 
                          learning_rate: float, progress_placeholder, log_placeholder):
        """Train a single model with progress updates"""
        
        config = self.model_configs[model_key]
        
        with log_placeholder.container():
            st.write(f"🚀 Training {config['name']}...")
        
        try:
            # Prepare data
            train_gen, val_gen = self.prepare_data_generators(model_key, batch_size)
            
            with log_placeholder.container():
                st.write(f"✅ Data loaded: {train_gen.samples} train, {val_gen.samples} val samples")
            
            # Create model
            model, base_model = self.create_densenet121_model(model_key)
            
            # Calculate class weights
            class_weights = compute_class_weight(
                'balanced',
                classes=np.unique(train_gen.classes),
                y=train_gen.classes
            )
            class_weight_dict = dict(enumerate(class_weights))
            
            # Compile model
            model.compile(
                optimizer=Adam(learning_rate=learning_rate),
                loss='categorical_crossentropy',
                metrics=['accuracy', 'precision', 'recall']
            )
            
            # Callbacks
            callbacks = [
                EarlyStopping(
                    monitor='val_accuracy',
                    patience=10,
                    restore_best_weights=True,
                    verbose=0,
                    mode='max'
                ),
                ModelCheckpoint(
                    config['model_file'],
                    monitor='val_accuracy',
                    save_best_only=True,
                    verbose=0,
                    mode='max'
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.2,
                    patience=5,
                    min_lr=1e-7,
                    verbose=0
                )
            ]
            
            # Training progress tracking
            progress_bar = progress_placeholder.progress(0)
            
            # Custom callback for progress updates
            class StreamlitCallback(keras.callbacks.Callback):
                def on_epoch_end(self, epoch, logs=None):
                    progress_bar.progress((epoch + 1) / epochs)
                    
                    with log_placeholder.container():
                        st.write(f"Epoch {epoch + 1}/{epochs}")
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Accuracy", f"{logs.get('accuracy', 0):.4f}")
                        with col2:
                            st.metric("Val Accuracy", f"{logs.get('val_accuracy', 0):.4f}")
                        with col3:
                            st.metric("Loss", f"{logs.get('loss', 0):.4f}")
                        with col4:
                            st.metric("Val Loss", f"{logs.get('val_loss', 0):.4f}")
            
            callbacks.append(StreamlitCallback())
            
            # Train model
            history = model.fit(
                train_gen,
                epochs=epochs,
                validation_data=val_gen,
                callbacks=callbacks,
                class_weight=class_weight_dict,
                verbose=0
            )
            
            # Fine-tuning
            with log_placeholder.container():
                st.write("🔧 Starting fine-tuning...")
            
            base_model.trainable = True
            for layer in base_model.layers[:-20]:
                layer.trainable = False
            
            model.compile(
                optimizer=Adam(learning_rate=learning_rate * 0.1),
                loss='categorical_crossentropy',
                metrics=['accuracy', 'precision', 'recall']
            )
            
            # Fine-tuning with fewer epochs
            finetune_epochs = min(10, epochs // 2)
            
            history_finetune = model.fit(
                train_gen,
                epochs=finetune_epochs,
                validation_data=val_gen,
                callbacks=callbacks,
                class_weight=class_weight_dict,
                verbose=0
            )
            
            # Evaluate model
            val_gen.reset()
            test_loss, test_accuracy, test_precision, test_recall = model.evaluate(
                val_gen, verbose=0
            )
            
            # Generate predictions for classification report
            val_gen.reset()
            predictions = model.predict(val_gen, verbose=0)
            predicted_classes = np.argmax(predictions, axis=1)
            true_classes = val_gen.classes[:len(predicted_classes)]
            
            class_labels = list(val_gen.class_indices.keys())
            report = classification_report(
                true_classes, 
                predicted_classes,
                target_names=class_labels,
                output_dict=True
            )
            
            results = {
                'model': model,
                'history': history,
                'history_finetune': history_finetune,
                'test_accuracy': test_accuracy,
                'test_precision': test_precision,
                'test_recall': test_recall,
                'classification_report': report,
                'config': config
            }
            
            with log_placeholder.container():
                st.success(f"✅ {config['name']} completed!")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Final Accuracy", f"{test_accuracy:.4f}")
                with col2:
                    st.metric("Final Precision", f"{test_precision:.4f}")
                with col3:
                    st.metric("Final Recall", f"{test_recall:.4f}")
            
            return results
            
        except Exception as e:
            with log_placeholder.container():
                st.error(f"❌ Error training {config['name']}: {str(e)}")
            return None
    
    def update_model_registry(self, results_dict):
        """Update model registry with results"""
        
        registry_path = 'models/registry/model_registry.json'
        
        # Load existing registry
        if os.path.exists(registry_path):
            with open(registry_path, 'r') as f:
                registry = json.load(f)
        else:
            registry = {
                "version": "2.0",
                "created": datetime.now().isoformat(),
                "models": {},
                "active_models": {}
            }
        
        # Update registry for each successful model
        for model_key, model_data in results_dict.items():
            if model_data is not None:
                config = model_data['config']
                
                model_info = {
                    "model_path": config['model_file'],
                    "file_path": config['model_file'],
                    "dataset_type": model_key,
                    "model_name": f"{config['name']} - DenseNet121 Binary",
                    "architecture": "DenseNet121_Binary",
                    "version": "v1.0",
                    "accuracy": float(model_data['test_accuracy']),
                    "precision": float(model_data['test_precision']),
                    "recall": float(model_data['test_recall']),
                    "f1_score": float(model_data['classification_report']['weighted avg']['f1-score']),
                    "classes": config['classes'],
                    "input_shape": list(self.input_shape),
                    "trained_date": datetime.now().isoformat(),
                    "dataset": f"{config['name']} Binary Classification Dataset",
                    "training_method": "Streamlit_Binary_DenseNet121_Dropout_EarlyStopping",
                    "gradcam_target_layer": "gradcam_target_layer",
                    "classification_type": "binary",
                    "description": config['description']
                }
                
                # Add file size
                if os.path.exists(config['model_file']):
                    model_info["file_size"] = os.path.getsize(config['model_file'])
                
                registry["models"][model_key] = model_info
                registry["active_models"][model_key] = model_key
        
        registry["last_modified"] = datetime.now().isoformat()
        
        # Save registry
        with open(registry_path, 'w') as f:
            json.dump(registry, f, indent=2)
        
        return registry_path

def main():
    """Main Streamlit application"""
    
    st.title("🏥 Medical X-ray AI: 5 Binary Models Trainer")
    st.markdown("**DenseNet121 Architecture with Early Stopping and Dropout**")
    st.markdown("*Updated with Bone Fracture Detection Model*")
    
    # GPU configuration
    gpu_status = configure_gpu()
    st.sidebar.success(gpu_status)
    
    # Initialize trainer
    trainer = StreamlitFiveBinaryModelsTrainer()
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Training Configuration")
    
    # Training parameters
    epochs = st.sidebar.slider("Epochs", 10, 100, 30, 5)
    batch_size = st.sidebar.selectbox("Batch Size", [16, 32, 64], index=1)
    learning_rate = st.sidebar.select_slider(
        "Learning Rate",
        options=[0.0001, 0.0005, 0.001, 0.005, 0.01],
        value=0.001,
        format_func=lambda x: f"{x:.4f}"
    )
    
    # Model selection
    st.sidebar.header("🎯 Model Selection")
    
    # Check dataset availability
    dataset_status = trainer.check_dataset_availability()
    
    selected_models = {}
    for model_key, config in trainer.model_configs.items():
        status = dataset_status[model_key]
        
        if status['available']:
            structure_info = f" ({status.get('structure', 'simple')})" if 'structure' in status else ""
            checkbox_label = f"✅ {config['name']} ({status['sample_count']} samples{structure_info})"
            selected_models[model_key] = st.sidebar.checkbox(
                checkbox_label,
                value=True,
                key=f"select_{model_key}"
            )
        else:
            st.sidebar.error(f"❌ {config['name']}: {status['error']}")
            selected_models[model_key] = False
    
    # Dataset overview
    st.header("📊 Dataset Overview")
    
    dataset_df_data = []
    for model_key, config in trainer.model_configs.items():
        status = dataset_status[model_key]
        dataset_df_data.append({
            'Model': config['name'],
            'Status': '✅ Available' if status['available'] else '❌ Missing',
            'Samples': status.get('sample_count', 0) if status['available'] else 0,
            'Structure': status.get('structure', 'N/A') if status['available'] else 'N/A',
            'Classes': ', '.join(config['classes']),
            'Description': config['description'],
            'Path': config['dataset_path']
        })
    
    dataset_df = pd.DataFrame(dataset_df_data)
    st.dataframe(dataset_df, use_container_width=True)
    
    # Training section
    st.header("🚀 Model Training")
    
    if st.button("🏋️ Start Training Selected Models", type="primary"):
        
        selected_count = sum(selected_models.values())
        
        if selected_count == 0:
            st.warning("⚠️ Please select at least one model to train.")
            return
        
        st.info(f"🎯 Training {selected_count} model(s) with the following configuration:")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Epochs", epochs)
        with col2:
            st.metric("Batch Size", batch_size)
        with col3:
            st.metric("Learning Rate", f"{learning_rate:.4f}")
        
        # Training progress containers
        progress_container = st.container()
        results_dict = {}
        
        with progress_container:
            
            for model_key, should_train in selected_models.items():
                if should_train:
                    config = trainer.model_configs[model_key]
                    
                    st.subheader(f"🔄 {config['name']}")
                    
                    progress_col, log_col = st.columns([1, 2])
                    
                    with progress_col:
                        progress_placeholder = st.empty()
                    
                    with log_col:
                        log_placeholder = st.empty()
                    
                    # Train model
                    result = trainer.train_single_model(
                        model_key, epochs, batch_size, learning_rate,
                        progress_placeholder, log_placeholder
                    )
                    
                    results_dict[model_key] = result
                    
                    st.divider()
        
        # Training complete - show results
        successful_models = [k for k, v in results_dict.items() if v is not None]
        
        if successful_models:
            st.success(f"🎉 Training Complete! {len(successful_models)}/{selected_count} models trained successfully.")
            
            # Update registry
            registry_path = trainer.update_model_registry(results_dict)
            st.info(f"✅ Model registry updated: {registry_path}")
            
            # Results summary
            st.header("📊 Training Results Summary")
            
            results_data = []
            for model_key in successful_models:
                result = results_dict[model_key]
                config = result['config']
                
                results_data.append({
                    'Model': config['name'],
                    'Accuracy': f"{result['test_accuracy']:.4f}",
                    'Precision': f"{result['test_precision']:.4f}",
                    'Recall': f"{result['test_recall']:.4f}",
                    'F1-Score': f"{result['classification_report']['weighted avg']['f1-score']:.4f}",
                    'Model File': config['model_file']
                })
            
            results_df = pd.DataFrame(results_data)
            st.dataframe(results_df, use_container_width=True)
            
            # Plot training curves
            st.header("📈 Training Curves")
            
            fig, axes = plt.subplots(2, len(successful_models), figsize=(5*len(successful_models), 8))
            
            if len(successful_models) == 1:
                axes = axes.reshape(-1, 1)
            
            for idx, model_key in enumerate(successful_models):
                result = results_dict[model_key]
                history = result['history']
                config = result['config']
                
                # Accuracy plot
                axes[0, idx].plot(history.history['accuracy'], label='Training', color='blue')
                axes[0, idx].plot(history.history['val_accuracy'], label='Validation', color='red')
                axes[0, idx].set_title(f"{config['name']} - Accuracy")
                axes[0, idx].set_xlabel('Epochs')
                axes[0, idx].set_ylabel('Accuracy')
                axes[0, idx].legend()
                axes[0, idx].grid(True, alpha=0.3)
                
                # Loss plot
                axes[1, idx].plot(history.history['loss'], label='Training', color='blue')
                axes[1, idx].plot(history.history['val_loss'], label='Validation', color='red')
                axes[1, idx].set_title(f"{config['name']} - Loss")
                axes[1, idx].set_xlabel('Epochs')
                axes[1, idx].set_ylabel('Loss')
                axes[1, idx].legend()
                axes[1, idx].grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            st.balloons()
            
        else:
            st.error("❌ No models were successfully trained. Please check the logs above for details.")

if __name__ == "__main__":
    main()