#!/usr/bin/env python3
"""
Quick 3-Class Chest Model Training Script
Combines Pneumonia and Cardiomegaly datasets for comprehensive detection
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from PIL import Image
import json
from datetime import datetime

def load_chest_datasets():
    """Load and combine pneumonia and cardiomegaly datasets"""
    print("📂 Loading chest datasets...")
    
    # Load pneumonia dataset using existing loader
    sys.path.append('utils')
    from data_loader import MedicalDataLoader
    
    data_loader = MedicalDataLoader("Dataset")
    pneumonia_data = data_loader.load_chest_data()
    
    pneumonia_images = pneumonia_data['images']
    pneumonia_labels = pneumonia_data['labels']
    
    print(f"🫁 Pneumonia dataset: {len(pneumonia_images)} samples")
    
    # Load cardiomegaly dataset
    cardiomegaly_path = "Dataset/CHEST/cardiomelgy"
    cardiomegaly_images = []
    cardiomegaly_labels = []
    
    if os.path.exists(cardiomegaly_path):
        for root, dirs, files in os.walk(cardiomegaly_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    file_path = os.path.join(root, file)
                    
                    try:
                        img = Image.open(file_path).convert('RGB')
                        img = img.resize((224, 224))
                        img_array = np.array(img) / 255.0
                        
                        # Label as cardiomegaly (class 2) or normal (class 0)
                        folder_name = os.path.basename(root).lower()
                        if 'normal' in folder_name or 'healthy' in folder_name:
                            label = 0  # Normal
                        else:
                            label = 2  # Cardiomegaly
                        
                        cardiomegaly_images.append(img_array)
                        cardiomegaly_labels.append(label)
                        
                    except Exception as e:
                        continue
    
    cardiomegaly_images = np.array(cardiomegaly_images)
    cardiomegaly_labels = np.array(cardiomegaly_labels)
    
    print(f"💗 Cardiomegaly dataset: {len(cardiomegaly_images)} samples")
    
    # Combine datasets
    all_images = np.vstack([pneumonia_images, cardiomegaly_images])
    all_labels = np.hstack([pneumonia_labels, cardiomegaly_labels])
    
    print(f"📊 Combined dataset: {len(all_images)} samples")
    print(f"🏷️ Class distribution: {np.bincount(all_labels)}")
    
    return all_images, all_labels

def create_3class_model():
    """Create DenseNet121 model for 3-class classification"""
    print("🏗️ Creating 3-class chest model...")
    
    # Base model
    base_model = DenseNet121(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    
    # Freeze most layers
    for layer in base_model.layers[:-20]:
        layer.trainable = False
    
    # Add custom layers
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu', name='gradcam_target_layer')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    outputs = Dense(3, activation='softmax', name='chest_3class_output')(x)
    
    model = Model(inputs, outputs, name='Chest3ClassModel')
    
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("✅ Model created successfully!")
    print(f"📊 Parameters: {model.count_params():,}")
    
    return model

def train_model(model, images, labels, epochs=20, batch_size=16):
    """Train the 3-class model"""
    print(f"🚀 Training model for {epochs} epochs...")
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        images, labels, 
        test_size=0.2, 
        stratify=labels, 
        random_state=42
    )
    
    print(f"📊 Training: {len(X_train)}, Validation: {len(X_val)}")
    
    # Callbacks
    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(patience=3, factor=0.5, min_lr=1e-7, verbose=1)
    ]
    
    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    return history

def save_model_and_update_registry(model, history):
    """Save model and update registry"""
    print("💾 Saving model...")
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('models/registry', exist_ok=True)
    
    # Save model
    model_path = 'models/chest_conditions_3class_model.h5'
    model.save(model_path)
    
    # Get final accuracy
    final_accuracy = max(history.history['val_accuracy'])
    
    # Update registry
    registry_path = 'models/registry/model_registry.json'
    
    if os.path.exists(registry_path):
        with open(registry_path, 'r') as f:
            registry = json.load(f)
    else:
        registry = {"version": "1.0", "models": {}, "active_models": {}}
    
    # Add 3-class model info
    registry["models"]["chest_conditions"] = {
        "model_path": "chest_conditions_3class_model.h5",
        "file_path": "chest_conditions_3class_model.h5",
        "dataset_type": "chest_conditions",
        "model_name": "Chest 3-Class DenseNet121",
        "architecture": "DenseNet121",
        "version": "v2.0",
        "accuracy": float(final_accuracy),
        "classes": ["Normal", "Pneumonia", "Cardiomegaly"],
        "input_shape": [224, 224, 3],
        "trained_date": datetime.now().isoformat(),
        "dataset": "Combined Pneumonia + Cardiomegaly Dataset",
        "training_method": "3-Class Combined Training",
        "gradcam_target_layer": "gradcam_target_layer",
        "file_size": os.path.getsize(model_path)
    }
    
    registry["active_models"]["chest_conditions"] = "chest_conditions"
    registry["last_modified"] = datetime.now().isoformat()
    
    with open(registry_path, 'w') as f:
        json.dump(registry, f, indent=2)
    
    print(f"✅ Model saved: {model_path}")
    print(f"🎯 Final accuracy: {final_accuracy:.4f}")
    
    return model_path

def main():
    """Main training function"""
    print("🫁 3-Class Chest Model Training")
    print("=" * 50)
    
    try:
        # Load datasets
        images, labels = load_chest_datasets()
        
        # Create model
        model = create_3class_model()
        
        # Train model
        history = train_model(model, images, labels, epochs=15)
        
        # Save model
        model_path = save_model_and_update_registry(model, history)
        
        print("\n🎉 Training completed successfully!")
        print(f"📁 Model saved: {model_path}")
        print("📝 Your chest model can now detect:")
        print("   • Normal chest X-rays")
        print("   • Pneumonia")
        print("   • Cardiomegaly (enlarged heart)")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()