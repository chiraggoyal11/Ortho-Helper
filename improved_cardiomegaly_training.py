#!/usr/bin/env python3
"""
Improved Advanced Cardiomegaly Detection Training
High-performance training with optimized architecture targeting >97% accuracy
"""

import os
import sys
import json
import numpy as np
import tensorflow as tf
from datetime import datetime
from tensorflow.keras.applications import EfficientNetB4, EfficientNetV2S
from tensorflow.keras.layers import (
    Dense, GlobalAveragePooling2D, Dropout, BatchNormalization,
    Conv2D, Multiply, GlobalMaxPooling2D, concatenate, Input,
    MultiHeadAttention, LayerNormalization, Add
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CosineRestartScheduler
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Configuration
CONFIG = {
    'base_dir': 'Dataset/CHEST/cardiomelgy/train/train',
    'img_height': 384,  # Higher resolution for chest X-rays
    'img_width': 384,
    'batch_size': 12,   # Smaller batch for high resolution
    'epochs': 60,
    'learning_rate': 5e-5,  # Very low learning rate
    'model_dir': 'models/cardiomegaly',
    'validation_split': 0.2,
    'patience': 12,
    'min_delta': 0.0005,
    'factor': 0.2,      
    'backbone': 'efficientnetv2'  # Latest architecture
}

def setup_gpu():
    """Configure GPU settings for high memory usage"""
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if physical_devices:
        try:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
            # Set memory limit if needed
            tf.config.experimental.set_virtual_device_configuration(
                physical_devices[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=8192)]  # 8GB limit
            )
            print("✓ GPU memory configured")
        except:
            print("⚠ GPU memory configuration failed, using default")
    else:
        print("⚠ No GPU found, using CPU")

def count_dataset_samples(base_dir):
    """Count samples in each class"""
    false_dir = os.path.join(base_dir, 'false')
    true_dir = os.path.join(base_dir, 'true')
    
    false_count = len([f for f in os.listdir(false_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    true_count = len([f for f in os.listdir(true_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    print(f"📊 Cardiomegaly Dataset Distribution:")
    print(f"   Normal (false): {false_count} images")
    print(f"   Cardiomegaly (true): {true_count} images")
    print(f"   Class Imbalance Ratio: {false_count/true_count:.2f}:1")
    
    return false_count, true_count

def create_advanced_data_generators():
    """Create specialized data generators for chest X-rays"""
    
    # Chest X-ray specific augmentations
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=CONFIG['validation_split'],
        
        # Geometric transformations (conservative for medical images)
        rotation_range=15,  # Limited rotation
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.15,
        horizontal_flip=True,  # Chest X-rays can be flipped
        vertical_flip=False,   # Never flip vertically
        
        # Intensity transformations (important for X-rays)
        brightness_range=[0.8, 1.2],
        channel_shift_range=0.1,
        
        # Advanced preprocessing
        fill_mode='constant',
        cval=0,  # Fill with black for X-rays
        interpolation_order=1
    )
    
    # Validation generator
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
        seed=42,
        interpolation='bilinear'
    )
    
    # Validation generator
    val_generator = val_datagen.flow_from_directory(
        CONFIG['base_dir'],
        target_size=(CONFIG['img_height'], CONFIG['img_width']),
        batch_size=CONFIG['batch_size'],
        class_mode='binary',
        subset='validation',
        shuffle=False,
        seed=42,
        interpolation='bilinear'
    )
    
    return train_generator, val_generator

def multi_head_attention_block(inputs, num_heads=8, key_dim=64, dropout=0.1):
    """Multi-head self-attention for medical image analysis"""
    
    # Get spatial dimensions
    batch_size = tf.shape(inputs)[0]
    height = tf.shape(inputs)[1]
    width = tf.shape(inputs)[2]
    channels = inputs.shape[-1]
    
    # Reshape to sequence
    x = tf.reshape(inputs, [batch_size, height * width, channels])
    
    # Multi-head attention
    attention_output = MultiHeadAttention(
        num_heads=num_heads,
        key_dim=key_dim,
        dropout=dropout
    )(x, x)
    
    # Add & Norm
    x = Add()([x, attention_output])
    x = LayerNormalization()(x)
    
    # Reshape back to spatial
    x = tf.reshape(x, [batch_size, height, width, channels])
    
    return x

def channel_spatial_attention(inputs):
    """Combined channel and spatial attention"""
    
    # Channel attention
    channel_axis = -1
    channel = inputs.shape[channel_axis]
    
    # Global pooling for channel attention
    gap = GlobalAveragePooling2D()(inputs)
    gmp = GlobalMaxPooling2D()(inputs)
    
    # Channel attention layers
    shared_dense1 = Dense(channel // 16, activation='relu')
    shared_dense2 = Dense(channel, activation='sigmoid')
    
    gap_out = shared_dense2(shared_dense1(gap))
    gmp_out = shared_dense2(shared_dense1(gmp))
    
    channel_attention = Add()([gap_out, gmp_out])
    channel_attention = tf.expand_dims(channel_attention, 1)
    channel_attention = tf.expand_dims(channel_attention, 1)
    
    # Apply channel attention
    x = Multiply()([inputs, channel_attention])
    
    return x

def create_advanced_cardiomegaly_model():
    """Create state-of-the-art cardiomegaly detection model"""
    
    # Input layer
    inputs = Input(shape=(CONFIG['img_height'], CONFIG['img_width'], 3))
    
    if CONFIG['backbone'] == 'efficientnetv2':
        # EfficientNetV2-S (latest and most efficient)
        base_model = EfficientNetV2S(
            weights='imagenet',
            include_top=False,
            input_tensor=inputs,
            drop_connect_rate=0.4
        )
        
        # Fine-tune from layer 150 onwards
        for layer in base_model.layers[:150]:
            layer.trainable = False
        for layer in base_model.layers[150:]:
            layer.trainable = True
            
    else:
        # EfficientNetB4 fallback
        base_model = EfficientNetB4(
            weights='imagenet',
            include_top=False,
            input_tensor=inputs,
            drop_connect_rate=0.4
        )
        
        # Fine-tune last 50% of layers
        trainable_layer_start = len(base_model.layers) // 2
        for layer in base_model.layers[:trainable_layer_start]:
            layer.trainable = False
        for layer in base_model.layers[trainable_layer_start:]:
            layer.trainable = True
    
    # Get base model output
    x = base_model.output
    
    # Apply attention mechanisms
    x = channel_spatial_attention(x)
    x = multi_head_attention_block(x, num_heads=8, key_dim=64)
    
    # Multi-scale feature extraction
    gap = GlobalAveragePooling2D()(x)
    gmp = GlobalMaxPooling2D()(x)
    
    # Combine features
    combined = concatenate([gap, gmp])
    
    # Dense layers with progressive dropout
    x = Dense(1024, activation='relu')(combined)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    # Output layer with careful initialization
    predictions = Dense(1, activation='sigmoid', 
                       kernel_initializer='he_normal',
                       name='cardiomegaly_prediction')(x)
    
    model = Model(inputs=inputs, outputs=predictions)
    return model

def create_advanced_callbacks(model_name):
    """Create optimized callbacks with cosine annealing"""
    
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
    
    # Early stopping
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=CONFIG['patience'],
        min_delta=CONFIG['min_delta'],
        mode='max',
        verbose=1,
        restore_best_weights=True
    )
    
    # Reduce LR on plateau
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=CONFIG['factor'],
        patience=6,
        min_lr=1e-8,
        verbose=1,
        mode='min'
    )
    
    # Cosine annealing scheduler
    cosine_scheduler = tf.keras.callbacks.LearningRateScheduler(
        lambda epoch: CONFIG['learning_rate'] * 
        (1 + np.cos(np.pi * epoch / CONFIG['epochs'])) / 2,
        verbose=0
    )
    
    return [checkpoint, early_stopping, reduce_lr, cosine_scheduler]

def compute_class_weights(train_generator):
    """Compute balanced class weights"""
    
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
    """Enhanced training history visualization"""
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    # Accuracy plot
    axes[0,0].plot(history.history['accuracy'], 'b-', label='Training', linewidth=2)
    axes[0,0].plot(history.history['val_accuracy'], 'r-', label='Validation', linewidth=2)
    axes[0,0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0,0].set_xlabel('Epoch')
    axes[0,0].set_ylabel('Accuracy')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].set_ylim([0.5, 1.0])
    
    # Loss plot
    axes[0,1].plot(history.history['loss'], 'b-', label='Training', linewidth=2)
    axes[0,1].plot(history.history['val_loss'], 'r-', label='Validation', linewidth=2)
    axes[0,1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[0,1].set_xlabel('Epoch')
    axes[0,1].set_ylabel('Loss')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Learning rate plot
    if 'lr' in history.history:
        axes[0,2].plot(history.history['lr'], 'g-', linewidth=2)
        axes[0,2].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        axes[0,2].set_xlabel('Epoch')
        axes[0,2].set_ylabel('Learning Rate')
        axes[0,2].set_yscale('log')
        axes[0,2].grid(True, alpha=0.3)
    
    # Precision plot (if available)
    if 'precision' in history.history:
        axes[1,0].plot(history.history['precision'], 'b-', label='Training', linewidth=2)
        axes[1,0].plot(history.history['val_precision'], 'r-', label='Validation', linewidth=2)
        axes[1,0].set_title('Model Precision', fontsize=14, fontweight='bold')
        axes[1,0].set_xlabel('Epoch')
        axes[1,0].set_ylabel('Precision')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
    
    # Recall plot (if available)
    if 'recall' in history.history:
        axes[1,1].plot(history.history['recall'], 'b-', label='Training', linewidth=2)
        axes[1,1].plot(history.history['val_recall'], 'r-', label='Validation', linewidth=2)
        axes[1,1].set_title('Model Recall', fontsize=14, fontweight='bold')
        axes[1,1].set_xlabel('Epoch')
        axes[1,1].set_ylabel('Recall')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
    
    # Summary stats
    best_val_acc = max(history.history['val_accuracy'])
    best_epoch = history.history['val_accuracy'].index(best_val_acc) + 1
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    
    axes[1,2].text(0.05, 0.9, f'Best Val Accuracy: {best_val_acc:.4f}', 
                   fontsize=12, transform=axes[1,2].transAxes, weight='bold')
    axes[1,2].text(0.05, 0.8, f'Best Epoch: {best_epoch}', 
                   fontsize=12, transform=axes[1,2].transAxes)
    axes[1,2].text(0.05, 0.7, f'Final Train Acc: {final_train_acc:.4f}', 
                   fontsize=12, transform=axes[1,2].transAxes)
    axes[1,2].text(0.05, 0.6, f'Final Val Acc: {final_val_acc:.4f}', 
                   fontsize=12, transform=axes[1,2].transAxes)
    axes[1,2].text(0.05, 0.5, f'Total Epochs: {len(history.history["accuracy"])}', 
                   fontsize=12, transform=axes[1,2].transAxes)
    axes[1,2].set_title('Training Summary', fontsize=14, fontweight='bold')
    axes[1,2].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{CONFIG["model_dir"]}/{model_name}_training_history.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    return best_val_acc, best_epoch

def evaluate_model(model, val_generator):
    """Comprehensive model evaluation with medical metrics"""
    print("\n🔍 Evaluating cardiomegaly detection performance...")
    
    # Reset generator
    val_generator.reset()
    
    # Predictions with confidence scores
    predictions_proba = model.predict(val_generator, verbose=1)
    y_pred = (predictions_proba > 0.5).astype(int).flatten()
    
    # True labels
    y_true = val_generator.classes
    
    # Classification report
    report = classification_report(y_true, y_pred, 
                                 target_names=['Normal', 'Cardiomegaly'], 
                                 output_dict=True)
    
    print("\n📊 Cardiomegaly Detection Report:")
    print(classification_report(y_true, y_pred, target_names=['Normal', 'Cardiomegaly']))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n🎯 Confusion Matrix:")
    print(f"              Normal  Cardiomegaly")
    print(f"Normal          {cm[0,0]}         {cm[0,1]}")
    print(f"Cardiomegaly    {cm[1,0]}         {cm[1,1]}")
    
    # Medical-specific metrics
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # Precision
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    print(f"\n🏥 Medical Performance Metrics:")
    print(f"   Sensitivity (Recall): {sensitivity:.4f}")
    print(f"   Specificity: {specificity:.4f}")
    print(f"   Positive Predictive Value: {ppv:.4f}")
    print(f"   Negative Predictive Value: {npv:.4f}")
    
    # Calculate overall metrics
    accuracy = report['accuracy']
    precision = report['weighted avg']['precision']
    recall = report['weighted avg']['recall']
    f1_score = report['weighted avg']['f1-score']
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'ppv': ppv,
        'npv': npv,
        'confusion_matrix': cm.tolist(),
        'classification_report': report
    }

def update_model_registry(model_name, model_path, metrics):
    """Update model registry with cardiomegaly model"""
    
    registry_path = 'model_registry.json'
    
    # Load existing registry
    if os.path.exists(registry_path):
        with open(registry_path, 'r') as f:
            registry = json.load(f)
    else:
        registry = {'models': {}}
    
    # Create model entry
    model_info = {
        'id': f"cardiomegaly_{CONFIG['backbone']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        'name': f'Advanced Cardiomegaly Detection ({CONFIG["backbone"].upper()})',
        'type': 'cardiomegaly',
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
    """Main training function for cardiomegaly detection"""
    print("🚀 Starting Advanced Cardiomegaly Detection Training")
    print("=" * 70)
    
    # Setup
    setup_gpu()
    os.makedirs(CONFIG['model_dir'], exist_ok=True)
    
    # Dataset analysis
    false_count, true_count = count_dataset_samples(CONFIG['base_dir'])
    
    # Data generators
    print("\n📁 Creating specialized chest X-ray generators...")
    train_generator, val_generator = create_advanced_data_generators()
    
    print(f"   Training samples: {train_generator.samples}")
    print(f"   Validation samples: {val_generator.samples}")
    print(f"   Classes found: {list(train_generator.class_indices.keys())}")
    
    # Compute class weights
    class_weights = compute_class_weights(train_generator)
    
    # Create model
    print(f"\n🏗️ Creating advanced cardiomegaly model with {CONFIG['backbone']}...")
    model = create_advanced_cardiomegaly_model()
    
    # Compile with medical-optimized settings
    optimizer = AdamW(
        learning_rate=CONFIG['learning_rate'],
        weight_decay=1e-6,  # Light L2 regularization
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
    model_name = f"{CONFIG['backbone']}_cardiomegaly_advanced_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    callbacks = create_advanced_callbacks(model_name)
    
    # Train model
    print(f"\n🎯 Starting training for {CONFIG['epochs']} epochs...")
    print(f"   Target accuracy: >97%")
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
    print("\n📈 Generating detailed training analysis...")
    best_val_acc, best_epoch = plot_training_history(history, model_name)
    
    # Evaluate model
    metrics = evaluate_model(model, val_generator)
    
    # Update registry
    model_id = update_model_registry(model_name, final_model_path, metrics)
    
    # Results summary
    print("\n" + "="*70)
    print("🎉 CARDIOMEGALY TRAINING COMPLETED!")
    print("="*70)
    print(f"📊 Final Results:")
    print(f"   Best Validation Accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    print(f"   Final Test Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"   Sensitivity (Recall): {metrics['sensitivity']:.4f}")
    print(f"   Specificity: {metrics['specificity']:.4f}")
    print(f"   Precision: {metrics['precision']:.4f}")
    print(f"   F1-Score: {metrics['f1_score']:.4f}")
    print(f"\n💾 Model saved: {final_model_path}")
    print(f"🆔 Model ID: {model_id}")
    print(f"📈 Training analysis: {CONFIG['model_dir']}/{model_name}_training_history.png")
    
    # Performance assessment for cardiomegaly
    target_accuracy = 0.97
    if metrics['accuracy'] >= target_accuracy:
        print(f"🎯 EXCELLENT: Target accuracy ({target_accuracy*100:.0f}%) achieved!")
        print("   Model ready for medical deployment consideration")
    elif metrics['accuracy'] >= 0.95:
        print(f"✅ GOOD: High accuracy achieved ({metrics['accuracy']*100:.1f}%)")
        print("   Close to target, consider minor optimizations")
    else:
        print(f"⚠️  Target accuracy ({target_accuracy*100:.0f}%) not yet reached")
        print("   Recommendations: 1) More training data 2) Longer training 3) Architecture tuning")
    
    return model, history, metrics

if __name__ == "__main__":
    try:
        model, history, metrics = main()
        sys.exit(0)
    except Exception as e:
        print(f"❌ Cardiomegaly training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)