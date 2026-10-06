#!/usr/bin/env python3
"""
🦴 Lightning-Fast Knee Osteoarthritis Classification Training
Parallel training script optimized for speed while maintaining 95%+ accuracy

Medical-Grade Performance in Minimal Time:
- Multi-class classification (0: Normal, 1: Mild, 2: Severe)
- EfficientNetB0 architecture for speed
- Mixed precision training
- Advanced augmentation strategies
- Smart class weighting for medical accuracy

Author: Medical AI Training System
Target: 95%+ accuracy in 15-30 minutes
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("🦴 KNEE OSTEOARTHRITIS FAST TRAINING SYSTEM")
print("=" * 60)
print("🎯 Target: 95%+ accuracy classification")
print("⚡ Optimized for parallel training with cardiomegaly model")
print("🔬 Multi-class: Normal, Mild OA, Severe OA")
print("=" * 60)

# Enable mixed precision for speed
# Disable mixed precision for metric compatibility
print("🚀 Using float32 precision for metric compatibility")
# try:
#     policy = tf.keras.mixed_precision.Policy('mixed_float16')
#     tf.keras.mixed_precision.set_global_policy(policy)
#     print("🚀 Mixed precision enabled for faster training")
# except:
#     print("⚠️  Mixed precision not available, using float32")

# Configuration
CONFIG = {
    'data_path': r'Dataset/Osteoarthritis Knee X-ray',
    'image_size': (224, 224),
    'batch_size': 16,  # Optimized for speed
    'epochs': 20,      # Reduced for fast training
    'learning_rate': 0.001,
    'validation_split': 0.2,
    'test_split': 0.1,
    'num_classes': 5,  # 0: Normal, 1: Doubtful, 2: Minimal, 3: Moderate, 4: Severe
    'model_name': 'fast_knee_arthritis_95.h5'
}

def load_and_prepare_data():
    """Load and prepare knee osteoarthritis dataset"""
    print("\n📊 Loading knee osteoarthritis dataset...")
    
    # Load CSV files
    train_df = pd.read_csv(os.path.join(CONFIG['data_path'], 'Train.csv'))
    
    print(f"📈 Dataset statistics:")
    print(f"   • Total images: {len(train_df)}")
    print(f"   • Classes: {sorted(train_df['label'].unique())}")
    
    # Class distribution
    class_counts = train_df['label'].value_counts().sort_index()
    print(f"   • Class distribution:")
    class_names = {0: 'Normal', 1: 'Doubtful', 2: 'Minimal', 3: 'Moderate', 4: 'Severe'}
    for class_idx, count in class_counts.items():
        class_name = class_names.get(class_idx, f'Class_{class_idx}')
        percentage = (count / len(train_df)) * 100
        print(f"     - Class {class_idx} ({class_name}): {count} images ({percentage:.1f}%)")
    
    # Add full image paths
    train_df['filepath'] = train_df['filename'].apply(
        lambda x: os.path.join(CONFIG['data_path'], 'train', x)
    )
    
    # Split data for training
    train_data, temp_data = train_test_split(
        train_df, test_size=CONFIG['validation_split'] + CONFIG['test_split'], 
        stratify=train_df['label'], random_state=42
    )
    
    val_data, test_data = train_test_split(
        temp_data, test_size=CONFIG['test_split'] / (CONFIG['validation_split'] + CONFIG['test_split']),
        stratify=temp_data['label'], random_state=42
    )
    
    print(f"📊 Data splits:")
    print(f"   • Training: {len(train_data)} images")
    print(f"   • Validation: {len(val_data)} images") 
    print(f"   • Testing: {len(test_data)} images")
    
    return train_data, val_data, test_data

def create_data_generators(train_data, val_data):
    """Create optimized data generators for knee images"""
    print("\n🔄 Creating optimized data generators...")
    
    # Advanced augmentation for medical images
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )
    
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    # Convert labels to strings for Keras compatibility
    train_data_copy = train_data.copy()
    val_data_copy = val_data.copy()
    train_data_copy['label'] = train_data_copy['label'].astype(str)
    val_data_copy['label'] = val_data_copy['label'].astype(str)
    
    # Create generators
    train_generator = train_datagen.flow_from_dataframe(
        train_data_copy,
        x_col='filepath',
        y_col='label',
        target_size=CONFIG['image_size'],
        batch_size=CONFIG['batch_size'],
        class_mode='sparse',
        shuffle=True,
        seed=42
    )
    
    val_generator = val_datagen.flow_from_dataframe(
        val_data_copy,
        x_col='filepath',
        y_col='label',
        target_size=CONFIG['image_size'],
        batch_size=CONFIG['batch_size'],
        class_mode='sparse',
        shuffle=False
    )
    
    return train_generator, val_generator

def create_fast_knee_model():
    """Create optimized EfficientNet model for knee osteoarthritis classification"""
    print("\n🏗️  Building fast knee osteoarthritis model...")
    
    # Load pre-trained EfficientNetB0 (lightweight but powerful)
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(*CONFIG['image_size'], 3)
    )
    
    # Freeze most layers for faster training, unfreeze top layers
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    
    # Add custom classification head
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    # Dense layers for knee-specific features
    x = layers.Dense(256, activation='relu', name='knee_features_1')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Dense(128, activation='relu', name='knee_features_2')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    
    # Output layer for 3 classes
    predictions = layers.Dense(CONFIG['num_classes'], activation='softmax', name='knee_classification')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    print(f"✅ Model created successfully!")
    total_params = model.count_params()
    trainable_params = sum([tf.size(var).numpy() for var in model.trainable_variables])
    print(f"   • Total parameters: {total_params:,}")
    print(f"   • Trainable parameters: {trainable_params:,}")
    
    return model

def compile_model(model, train_data):
    """Compile model with optimized settings for medical classification"""
    print("\n⚙️  Compiling model for knee osteoarthritis classification...")
    
    # Calculate class weights for balanced training
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(train_data['label']),
        y=train_data['label']
    )
    class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
    
    print(f"📊 Class weights for balanced training:")
    class_names = {0: 'Normal', 1: 'Doubtful', 2: 'Minimal', 3: 'Moderate', 4: 'Severe'}
    for class_idx, weight in class_weight_dict.items():
        class_name = class_names.get(class_idx, f'Class_{class_idx}')
        print(f"   • Class {class_idx} ({class_name}): {weight:.3f}")
    
    # Compile with optimized settings
    model.compile(
        optimizer=Adam(learning_rate=CONFIG['learning_rate']),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy', 'sparse_top_k_categorical_accuracy']
    )
    
    return model, class_weight_dict

def create_callbacks():
    """Create training callbacks for optimal performance"""
    print("\n🔔 Setting up training callbacks...")
    
    callbacks = [
        ModelCheckpoint(
            CONFIG['model_name'],
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1,
            save_weights_only=False
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True,
            verbose=1,
            mode='max',
            min_delta=0.001
        )
    ]
    
    return callbacks

def train_model(model, train_generator, val_generator, class_weight_dict):
    """Train the knee osteoarthritis model"""
    print("\n🚀 Starting knee osteoarthritis training...")
    print(f"🎯 Target: 95%+ validation accuracy")
    
    start_time = time.time()
    
    callbacks = create_callbacks()
    
    # Train the model
    history = model.fit(
        train_generator,
        epochs=CONFIG['epochs'],
        validation_data=val_generator,
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1
    )
    
    training_time = time.time() - start_time
    
    print(f"\n✅ Training completed!")
    print(f"⏱️  Total training time: {training_time/60:.1f} minutes")
    
    # Get best metrics
    best_accuracy = max(history.history['val_accuracy'])
    best_epoch = history.history['val_accuracy'].index(best_accuracy) + 1
    
    print(f"🏆 Best Results:")
    print(f"   • Best validation accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
    print(f"   • Achieved at epoch: {best_epoch}")
    
    if best_accuracy >= 0.95:
        print("🎉 SUCCESS: Target 95%+ accuracy achieved!")
    else:
        print("📈 Training completed - consider fine-tuning for higher accuracy")
    
    return history, training_time

def evaluate_model(model, test_data):
    """Evaluate the trained model on test data"""
    print("\n📊 Evaluating model on test data...")
    
    # Create test generator  
    test_data_copy = test_data.copy()
    test_data_copy['label'] = test_data_copy['label'].astype(str)
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_dataframe(
        test_data_copy,
        x_col='filepath',
        y_col='label',
        target_size=CONFIG['image_size'],
        batch_size=CONFIG['batch_size'],
        class_mode='sparse',
        shuffle=False
    )
    
    # Evaluate
    test_loss, test_accuracy, test_top_k = model.evaluate(test_generator, verbose=1)
    
    print(f"🎯 Test Results:")
    print(f"   • Test accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"   • Test loss: {test_loss:.4f}")
    print(f"   • Top-k accuracy: {test_top_k:.4f}")
    
    # Detailed classification report
    predictions = model.predict(test_generator, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)
    
    class_names = ['Normal', 'Doubtful', 'Minimal', 'Moderate', 'Severe']
    
    print(f"\n📋 Classification Report:")
    print(classification_report(
        test_generator.labels, 
        predicted_classes,
        target_names=class_names,
        digits=4
    ))
    
    return test_accuracy, predicted_classes

def save_results(history, training_time, test_accuracy):
    """Save training results and metrics"""
    print("\n💾 Saving training results...")
    
    results = {
        'model_name': CONFIG['model_name'],
        'training_time_minutes': training_time / 60,
        'best_val_accuracy': max(history.history['val_accuracy']),
        'test_accuracy': test_accuracy,
        'epochs_trained': len(history.history['loss']),
        'config': CONFIG,
        'timestamp': datetime.now().isoformat()
    }
    
    # Save results
    results_file = 'knee_arthritis_training_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Results saved to {results_file}")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy - Knee Osteoarthritis')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss - Knee Osteoarthritis')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('knee_arthritis_training_history.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Training plots saved to knee_arthritis_training_history.png")

def main():
    """Main training pipeline for knee osteoarthritis classification"""
    print("\n🦴 STARTING KNEE OSTEOARTHRITIS TRAINING PIPELINE")
    print(f"⏰ Start time: {datetime.now().strftime('%H:%M:%S')}")
    
    try:
        # Load and prepare data
        train_data, val_data, test_data = load_and_prepare_data()
        
        # Create data generators
        train_generator, val_generator = create_data_generators(train_data, val_data)
        
        # Create and compile model
        model = create_fast_knee_model()
        model, class_weight_dict = compile_model(model, train_data)
        
        # Train model
        history, training_time = train_model(model, train_generator, val_generator, class_weight_dict)
        
        # Evaluate model
        test_accuracy, predictions = evaluate_model(model, test_data)
        
        # Save results
        save_results(history, training_time, test_accuracy)
        
        print(f"\n🎉 KNEE OSTEOARTHRITIS TRAINING COMPLETE!")
        print(f"⏰ Finish time: {datetime.now().strftime('%H:%M:%S')}")
        print(f"📁 Model saved as: {CONFIG['model_name']}")
        
        # Summary
        best_val_acc = max(history.history['val_accuracy'])
        print(f"\n📊 FINAL SUMMARY:")
        print(f"   🏆 Best validation accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
        print(f"   🎯 Test accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        print(f"   ⏱️  Training time: {training_time/60:.1f} minutes")
        
        if best_val_acc >= 0.95:
            print(f"   ✅ SUCCESS: Target 95%+ accuracy achieved!")
        else:
            print(f"   📈 Consider fine-tuning for higher accuracy")
            
        print(f"\n🦴 Ready for deployment in medical imaging system!")
        
    except Exception as e:
        print(f"\n❌ Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    main()