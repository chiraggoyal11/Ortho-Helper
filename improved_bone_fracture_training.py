#!/usr/bin/env python3
"""
Improved Advanced Bone Fracture Detection Training
Addresses low accuracy issues with optimized architecture and training strategy
"""

import os
import sys
import json
import numpy as np
import tensorflow as tf
from datetime import datetime
from tensorflow.keras.applications import DenseNet201, EfficientNetB3
from tensorflow.keras.layers import (
    Dense, GlobalAveragePooling2D, Dropout, BatchNormalization,
    Conv2D, Multiply, GlobalMaxPooling2D, concatenate, Input
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Configuration
CONFIG = {
    'base_dir': 'Dataset/ARM/MURA_Organized/Forearm',
    'img_height': 320,  # Increased resolution
    'img_width': 320,
    'batch_size': 16,   # Smaller batch for better gradient updates
    'epochs': 50,
    'learning_rate': 1e-4,  # Lower learning rate
    'model_dir': 'models/bone_fracture',
    'validation_split': 0.2,
    'patience': 8,      # Increased patience
    'min_delta': 0.001,
    'factor': 0.3,      # More aggressive LR reduction
    'backbone': 'efficientnet'  # Switch to EfficientNet
}

def setup_gpu():
    """Configure GPU settings"""
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("✓ GPU memory growth enabled")
    else:
        print("⚠ No GPU found, using CPU")

def count_dataset_samples(base_dir):
    """Count samples in each class"""
    negative_dir = os.path.join(base_dir, 'Negative')
    positive_dir = os.path.join(base_dir, 'Positive')
    
    neg_count = len([f for f in os.listdir(negative_dir) if f.endswith('.png')])
    pos_count = len([f for f in os.listdir(positive_dir) if f.endswith('.png')])
    
    print(f"📊 Dataset Distribution:")
    print(f"   Normal (Negative): {neg_count} images")
    print(f"   Fracture (Positive): {pos_count} images")
    print(f"   Class Imbalance Ratio: {neg_count/pos_count:.2f}:1")
    
    return neg_count, pos_count

def create_advanced_data_generators():
    """Create optimized data generators with heavy augmentation"""
    
    # Training data generator with heavy augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=CONFIG['validation_split'],
        
        # Geometric transformations
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.3,
        horizontal_flip=True,
        vertical_flip=False,
        
        # Intensity transformations
        brightness_range=[0.7, 1.3],
        channel_shift_range=0.2,
        
        # Advanced augmentations
        fill_mode='reflect',
        interpolation_order=1
    )
    
    # Validation generator (only rescaling)
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=CONFIG['validation_split']
    )
    
    # Training generator
    train_generator = train_datagen.flow_from_directory(
        CONFIG['base_dir'],
        target_size=(CONFIG['img_height'], CONFIG['img_width']),
        batch_size=CONFIG['batch_size'],
        class_mode='binary',
        subset='training',
        shuffle=True,
        seed=42
    )
    
    # Validation generator
    val_generator = val_datagen.flow_from_directory(
        CONFIG['base_dir'],
        target_size=(CONFIG['img_height'], CONFIG['img_width']),
        batch_size=CONFIG['batch_size'],
        class_mode='binary',
        subset='validation',
        shuffle=False,
        seed=42
    )
    
    return train_generator, val_generator

def attention_block(inputs, ratio=8):
    """Channel attention mechanism"""
    channel_axis = -1
    channel = inputs.shape[channel_axis]
    
    # Squeeze
    squeeze = GlobalAveragePooling2D()(inputs)
    
    # Excitation
    excitation = Dense(units=channel // ratio, activation='relu')(squeeze)
    excitation = Dense(units=channel, activation='sigmoid')(excitation)
    excitation = tf.expand_dims(excitation, 1)
    excitation = tf.expand_dims(excitation, 1)
    
    # Scale
    scale = Multiply()([inputs, excitation])
    return scale

def create_improved_model():
    """Create an improved hybrid model with attention"""
    
    # Input layer
    inputs = Input(shape=(CONFIG['img_height'], CONFIG['img_width'], 3))
    
    if CONFIG['backbone'] == 'efficientnet':
        # EfficientNetB3 base model
        base_model = EfficientNetB3(
            weights='imagenet',
            include_top=False,
            input_tensor=inputs
        )
        
        # Fine-tune from layer 200 onwards (more layers)
        for layer in base_model.layers[:200]:
            layer.trainable = False
        for layer in base_model.layers[200:]:
            layer.trainable = True
            
    else:
        # DenseNet201 fallback
        base_model = DenseNet201(
            weights='imagenet',
            include_top=False,
            input_tensor=inputs
        )
        
        # Fine-tune last 100 layers
        for layer in base_model.layers[:-100]:
            layer.trainable = False
        for layer in base_model.layers[-100:]:
            layer.trainable = True
    
    # Get base model output
    x = base_model.output
    
    # Apply attention mechanism
    x = attention_block(x, ratio=16)
    
    # Multi-scale feature extraction
    gap = GlobalAveragePooling2D()(x)
    gmp = GlobalMaxPooling2D()(x)
    
    # Concatenate different pooling strategies
    combined = concatenate([gap, gmp])
    
    # Dense layers with heavy dropout
    x = Dense(512, activation='relu')(combined)
    x = BatchNormalization()(x)
    x = Dropout(0.6)(x)
    
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    
    # Output layer
    predictions = Dense(1, activation='sigmoid', name='fracture_prediction')(x)
    
    model = Model(inputs=inputs, outputs=predictions)
    return model

def create_callbacks(model_name):
    """Create optimized callbacks"""
    
    # Model checkpoint
    checkpoint_path = os.path.join(CONFIG['model_dir'], f'{model_name}_best.h5')
    checkpoint = ModelCheckpoint(
        checkpoint_path,
        monitor='val_accuracy',
        save_best_only=True,
        save_weights_only=False,
        mode='max',
        verbose=1
    )
    
    # Early stopping with patience
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=CONFIG['patience'],
        min_delta=CONFIG['min_delta'],
        mode='max',
        verbose=1,
        restore_best_weights=True
    )
    
    # Learning rate reduction
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=CONFIG['factor'],
        patience=4,  # Reduce LR earlier
        min_lr=1e-7,
        verbose=1,
        mode='min'
    )
    
    return [checkpoint, early_stopping, reduce_lr]

def compute_class_weights(train_generator):
    """Compute class weights to handle imbalance"""
    
    # Get all labels
    labels = []
    for i in range(len(train_generator)):
        _, batch_labels = train_generator[i]
        labels.extend(batch_labels.flatten().tolist())
    
    # Compute class weights
    unique_labels = np.unique(labels)
    class_weights = compute_class_weight(
        'balanced',
        classes=unique_labels,
        y=labels
    )
    
    class_weight_dict = dict(zip(unique_labels, class_weights))
    
    print(f"📊 Computed Class Weights: {class_weight_dict}")
    return class_weight_dict

def plot_training_history(history, model_name):
    """Plot and save training history"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Accuracy plot
    ax1.plot(history.history['accuracy'], 'b-', label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], 'r-', label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Loss plot
    ax2.plot(history.history['loss'], 'b-', label='Training Loss')
    ax2.plot(history.history['val_loss'], 'r-', label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    # Learning rate plot (if available)
    if 'lr' in history.history:
        ax3.plot(history.history['lr'], 'g-', label='Learning Rate')
        ax3.set_title('Learning Rate Schedule')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.set_yscale('log')
        ax3.legend()
        ax3.grid(True)
    
    # Best metrics summary
    best_val_acc = max(history.history['val_accuracy'])
    best_epoch = history.history['val_accuracy'].index(best_val_acc) + 1
    
    ax4.text(0.1, 0.8, f'Best Validation Accuracy: {best_val_acc:.4f}', 
             fontsize=12, transform=ax4.transAxes)
    ax4.text(0.1, 0.7, f'Best Epoch: {best_epoch}', 
             fontsize=12, transform=ax4.transAxes)
    ax4.text(0.1, 0.6, f'Total Epochs: {len(history.history["accuracy"])}', 
             fontsize=12, transform=ax4.transAxes)
    ax4.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{CONFIG["model_dir"]}/{model_name}_training_history.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return best_val_acc, best_epoch

def evaluate_model(model, val_generator):
    """Comprehensive model evaluation"""
    print("\n🔍 Evaluating model performance...")
    
    # Reset generator
    val_generator.reset()
    
    # Predictions
    predictions = model.predict(val_generator, verbose=1)
    y_pred = (predictions > 0.5).astype(int).flatten()
    
    # True labels
    y_true = val_generator.classes
    
    # Classification report
    report = classification_report(y_true, y_pred, 
                                 target_names=['Normal', 'Fracture'], 
                                 output_dict=True)
    
    print("\n📊 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=['Normal', 'Fracture']))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n🎯 Confusion Matrix:")
    print(f"         Normal  Fracture")
    print(f"Normal     {cm[0,0]}      {cm[0,1]}")
    print(f"Fracture   {cm[1,0]}      {cm[1,1]}")
    
    # Calculate metrics
    accuracy = report['accuracy']
    precision = report['weighted avg']['precision']
    recall = report['weighted avg']['recall']
    f1_score = report['weighted avg']['f1-score']
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'confusion_matrix': cm.tolist(),
        'classification_report': report
    }

def update_model_registry(model_name, model_path, metrics):
    """Update the model registry with new model"""
    
    registry_path = 'model_registry.json'
    
    # Load existing registry
    if os.path.exists(registry_path):
        with open(registry_path, 'r') as f:
            registry = json.load(f)
    else:
        registry = {'models': {}}
    
    # Create model entry
    model_info = {
        'id': f"bone_fracture_{CONFIG['backbone']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        'name': f'Improved Bone Fracture Detection ({CONFIG["backbone"].upper()})',
        'type': 'bone_fracture',
        'architecture': CONFIG['backbone'],
        'version': '2.0',
        'path': model_path,
        'created_at': datetime.now().isoformat(),
        'metrics': metrics,
        'config': CONFIG,
        'status': 'trained',
        'active': False
    }
    
    # Add to registry
    registry['models'][model_info['id']] = model_info
    
    # Save registry
    with open(registry_path, 'w') as f:
        json.dump(registry, f, indent=2)
    
    print(f"✅ Model registered: {model_info['id']}")
    return model_info['id']

def main():
    """Main training function"""
    print("🚀 Starting Improved Bone Fracture Detection Training")
    print("=" * 60)
    
    # Setup
    setup_gpu()
    os.makedirs(CONFIG['model_dir'], exist_ok=True)
    
    # Dataset analysis
    neg_count, pos_count = count_dataset_samples(CONFIG['base_dir'])
    
    # Data generators
    print("\n📁 Creating data generators...")
    train_generator, val_generator = create_advanced_data_generators()
    
    print(f"   Training samples: {train_generator.samples}")
    print(f"   Validation samples: {val_generator.samples}")
    print(f"   Classes found: {list(train_generator.class_indices.keys())}")
    
    # Compute class weights
    class_weights = compute_class_weights(train_generator)
    
    # Create model
    print(f"\n🏗️ Creating improved model with {CONFIG['backbone']} backbone...")
    model = create_improved_model()
    
    # Compile model with optimized settings
    optimizer = AdamW(
        learning_rate=CONFIG['learning_rate'],
        weight_decay=1e-5,  # L2 regularization
        clipnorm=1.0        # Gradient clipping
    )
    
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"   Total parameters: {model.count_params():,}")
    print(f"   Trainable parameters: {sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]):,}")
    
    # Training callbacks
    model_name = f"{CONFIG['backbone']}_bone_fracture_improved_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    callbacks = create_callbacks(model_name)
    
    # Train model
    print(f"\n🎯 Starting training for {CONFIG['epochs']} epochs...")
    print(f"   Using class weights: {class_weights}")
    
    history = model.fit(
        train_generator,
        epochs=CONFIG['epochs'],
        validation_data=val_generator,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    # Save final model
    final_model_path = os.path.join(CONFIG['model_dir'], f'{model_name}_final.h5')
    model.save(final_model_path)
    
    # Plot training history
    print("\n📈 Generating training plots...")
    best_val_acc, best_epoch = plot_training_history(history, model_name)
    
    # Evaluate model
    metrics = evaluate_model(model, val_generator)
    
    # Update registry
    model_id = update_model_registry(model_name, final_model_path, metrics)
    
    # Results summary
    print("\n" + "="*60)
    print("🎉 TRAINING COMPLETED!")
    print("="*60)
    print(f"📊 Final Results:")
    print(f"   Best Validation Accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    print(f"   Final Test Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"   Precision: {metrics['precision']:.4f}")
    print(f"   Recall: {metrics['recall']:.4f}")
    print(f"   F1-Score: {metrics['f1_score']:.4f}")
    print(f"\n💾 Model saved: {final_model_path}")
    print(f"🆔 Model ID: {model_id}")
    print(f"📈 Training plots: {CONFIG['model_dir']}/{model_name}_training_history.png")
    
    # Performance assessment
    target_accuracy = 0.90
    if metrics['accuracy'] >= target_accuracy:
        print(f"🎯 SUCCESS: Target accuracy ({target_accuracy*100:.0f}%) achieved!")
    else:
        print(f"⚠️  Target accuracy ({target_accuracy*100:.0f}%) not yet reached.")
        print("   Consider: 1) More data, 2) Different architecture, 3) Longer training")
    
    return model, history, metrics

if __name__ == "__main__":
    try:
        model, history, metrics = main()
        sys.exit(0)
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)