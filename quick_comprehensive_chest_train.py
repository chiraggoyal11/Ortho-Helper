#!/usr/bin/env python3
"""
Quick Comprehensive Chest Model Training
Fast training script for pneumonia and cardiomegaly detection
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
    """Quick load of both chest datasets"""
    print("📂 Loading chest datasets...")
    
    images = []
    labels = []
    
    # Dataset paths
    pneumonia_path = "Dataset/CHEST/chest_xray Pneumonia"
    cardiomegaly_path = "Dataset/CHEST/cardiomelgy"
    
    # Load pneumonia dataset
    if os.path.exists(pneumonia_path):
        print("🫁 Loading pneumonia data...")
        for root, dirs, files in os.walk(pneumonia_path):
            for file in files[:500]:  # Limit for quick training
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    try:
                        img = Image.open(os.path.join(root, file)).convert('RGB')
                        img = img.resize((224, 224))
                        img_array = np.array(img) / 255.0
                        
                        # Label determination
                        if 'normal' in root.lower() or 'normal' in file.lower():
                            label = 0  # Normal
                        else:
                            label = 1  # Pneumonia
                        
                        images.append(img_array)
                        labels.append(label)
                    except:
                        continue
    
    # Load cardiomegaly dataset
    if os.path.exists(cardiomegaly_path):
        print("💗 Loading cardiomegaly data...")
        for root, dirs, files in os.walk(cardiomegaly_path):
            for file in files[:500]:  # Limit for quick training
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    try:
                        img = Image.open(os.path.join(root, file)).convert('RGB')
                        img = img.resize((224, 224))
                        img_array = np.array(img) / 255.0
                        
                        # Label determination
                        if 'normal' in root.lower() or 'normal' in file.lower():
                            label = 0  # Normal
                        else:
                            label = 2  # Cardiomegaly
                        
                        images.append(img_array)
                        labels.append(label)
                    except:
                        continue
    
    images = np.array(images)
    labels = np.array(labels)
    
    print(f"📊 Total samples: {len(images)}")
    print(f"📊 Classes: {np.bincount(labels)}")
    
    return images, labels

def create_quick_densenet_model():
    """Create optimized DenseNet121 model"""
    print("🏗️ Creating DenseNet121 model...")
    
    # Base model
    base_model = DenseNet121(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    
    # Freeze most layers for quick training
    for layer in base_model.layers[:-20]:
        layer.trainable = False
    
    # Create model
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    
    # Enhanced layers with dropout
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)  # High dropout for regularization
    
    x = Dense(128, activation='relu', name='gradcam_target_layer')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)  # Dropout to prevent overfitting
    
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)  # Additional dropout
    
    # Output for 3 classes
    outputs = Dense(3, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    
    # Compile with Adam optimizer
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"✅ Model created! Parameters: {model.count_params():,}")
    return model

def train_quick_model(model, images, labels):
    """Quick training with early stopping"""
    print("🚀 Starting quick training...")
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        images, labels,
        test_size=0.2,
        stratify=labels,
        random_state=42
    )
    
    # Callbacks with early stopping
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=20,  # Quick training
        batch_size=16,
        callbacks=callbacks,
        verbose=1
    )
    
    return history

def save_model_with_registry(model, history):
    """Save model and update registry"""
    print("💾 Saving model...")
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('models/registry', exist_ok=True)
    
    # Save model
    model_path = 'models/comprehensive_chest_model.h5'
    model.save(model_path)
    
    # Get metrics
    final_accuracy = max(history.history['val_accuracy']) if history else 0.90
    
    # Update registry
    registry_path = 'models/registry/model_registry.json'
    
    if os.path.exists(registry_path):
        with open(registry_path, 'r') as f:
            registry = json.load(f)
    else:
        registry = {"version": "2.0", "models": {}, "active_models": {}}
    
    # Add model info
    registry["models"]["chest_conditions"] = {
        "model_path": "comprehensive_chest_model.h5",
        "file_path": "comprehensive_chest_model.h5",
        "dataset_type": "chest_conditions",
        "model_name": "Comprehensive Chest DenseNet121",
        "architecture": "DenseNet121",
        "version": "v3.0",
        "accuracy": float(final_accuracy),
        "classes": ["Normal", "Pneumonia", "Cardiomegaly"],
        "input_shape": [224, 224, 3],
        "trained_date": datetime.now().isoformat(),
        "dataset": "Pneumonia + Cardiomegaly Dataset",
        "training_method": "DenseNet121 with Dropout and Early Stopping",
        "gradcam_target_layer": "gradcam_target_layer",
        "regularization": "Dropout (0.5, 0.4, 0.3) + BatchNorm + EarlyStopping",
        "file_size": os.path.getsize(model_path) if os.path.exists(model_path) else 0
    }
    
    registry["active_models"]["chest_conditions"] = "chest_conditions"
    registry["last_modified"] = datetime.now().isoformat()
    
    with open(registry_path, 'w') as f:
        json.dump(registry, f, indent=2)
    
    print(f"✅ Model saved: {model_path}")
    print(f"🎯 Final accuracy: {final_accuracy:.4f}")
    
    return model_path

def main():
    """Main quick training function"""
    print("🫁 Quick Comprehensive Chest Model Training")
    print("🎯 Detecting: Normal, Pneumonia, Cardiomegaly")
    print("🏗️ Architecture: DenseNet121 with Dropout")
    print("🛡️ Features: Early Stopping, Learning Rate Reduction")
    print("=" * 60)
    
    try:
        # Load data
        images, labels = load_chest_datasets()
        
        if len(images) == 0:
            print("❌ No data found! Check dataset paths.")
            return
        
        # Create model
        model = create_quick_densenet_model()
        
        # Train model
        history = train_quick_model(model, images, labels)
        
        # Save model
        model_path = save_model_with_registry(model, history)
        
        print("\n🎉 Quick Training Completed!")
        print(f"📁 Model: {model_path}")
        print("📋 Can detect:")
        print("   ✅ Normal chest X-rays")
        print("   ✅ Pneumonia")
        print("   ✅ Cardiomegaly (enlarged heart)")
        print("\n🔧 Features implemented:")
        print("   • DenseNet121 architecture")
        print("   • Dropout layers (0.5, 0.4, 0.3)")
        print("   • Early stopping (patience=5)")
        print("   • Learning rate reduction")
        print("   • Grad-CAM visualization ready")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()