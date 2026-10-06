#!/usr/bin/env python3
"""
Standalone 95%+ Accuracy Cardiomegaly Training
Handles all data formats and edge cases properly
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, applications
import time
from pathlib import Path
from PIL import Image
import cv2

# Add utils to path
sys.path.append(str(Path(__file__).parent))

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

def load_cardiomegaly_data():
    """Load cardiomegaly dataset manually"""
    print("📁 Loading cardiomegaly dataset manually...")
    
    dataset_path = Path("Dataset/cardiomelgy")
    
    if not dataset_path.exists():
        print(f"❌ Dataset path not found: {dataset_path}")
        return None
    
    # Find all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    
    # Look for train/test structure (handle nested folders)
    train_path = dataset_path / "train"
    test_path = dataset_path / "test"
    
    # Check for nested structure first
    if (train_path / "train").exists():
        train_path = train_path / "train"
    if (test_path / "test").exists():
        test_path = test_path / "test"
    
    if not train_path.exists():
        print(f"❌ Train path not found: {train_path}")
        return None
    
    print(f"✅ Found dataset structure:")
    print(f"   Train: {train_path}")
    print(f"   Test: {test_path}")
    
    # Check what's in train directory
    print(f"   Train contents: {list(train_path.iterdir())}")
    
    # Load training data
    train_images = []
    train_labels = []
    
    for class_folder in train_path.iterdir():
        if class_folder.is_dir():
            class_name = class_folder.name.lower()
            
            # Map class names
            if class_name in ['false', 'normal', '0']:
                label = 0  # Normal
                print(f"   Class 0 (Normal): {class_name}")
            elif class_name in ['true', 'cardiomegaly', '1']:
                label = 1  # Cardiomegaly
                print(f"   Class 1 (Cardiomegaly): {class_name}")
            else:
                print(f"   Unknown class: {class_name}, skipping...")
                continue
            
            # Load images from this class
            class_count = 0
            for img_file in class_folder.iterdir():
                if img_file.suffix.lower() in image_extensions:
                    train_images.append(str(img_file))
                    train_labels.append(label)
                    class_count += 1
            
            print(f"     Loaded {class_count} images")
    
    # Load test data
    test_images = []
    test_labels = []
    
    if test_path.exists():
        for class_folder in test_path.iterdir():
            if class_folder.is_dir():
                class_name = class_folder.name.lower()
                
                # Map class names
                if class_name in ['false', 'normal', '0']:
                    label = 0  # Normal
                elif class_name in ['true', 'cardiomegaly', '1']:
                    label = 1  # Cardiomegaly
                else:
                    continue
                
                # Load images from this class
                class_count = 0
                for img_file in class_folder.iterdir():
                    if img_file.suffix.lower() in image_extensions:
                        test_images.append(str(img_file))
                        test_labels.append(label)
                        class_count += 1
    
    # Convert to numpy arrays
    train_images = np.array(train_images)
    train_labels = np.array(train_labels)
    test_images = np.array(test_images)
    test_labels = np.array(test_labels)
    
    # Split train into train/val
    from sklearn.model_selection import train_test_split
    
    if len(test_images) > 0:
        X_train, X_val, y_train, y_val = train_test_split(
            train_images, train_labels, test_size=0.15, random_state=42, stratify=train_labels
        )
        X_test, y_test = test_images, test_labels
    else:
        # No separate test set, split train into train/val/test
        X_train_temp, X_test, y_train_temp, y_test = train_test_split(
            train_images, train_labels, test_size=0.1, random_state=42, stratify=train_labels
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_temp, y_train_temp, test_size=0.15, random_state=42, stratify=y_train_temp
        )
    
    print(f"✅ Dataset split:")
    print(f"   Training: {len(X_train)} samples")
    print(f"   Validation: {len(X_val)} samples")
    print(f"   Test: {len(X_test)} samples")
    
    # Check class distribution
    train_class_0 = np.sum(y_train == 0)
    train_class_1 = np.sum(y_train == 1)
    print(f"   Training distribution: {train_class_0} Normal, {train_class_1} Cardiomegaly")
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test,
        'num_classes': 2
    }

def load_and_preprocess_image(img_path, target_size=(384, 384)):
    """Load and preprocess a single image"""
    try:
        # Load image
        if isinstance(img_path, str):
            pil_img = Image.open(img_path).convert('RGB')
        else:
            # Already an array
            if hasattr(img_path, 'shape'):
                if len(img_path.shape) == 3:
                    pil_img = Image.fromarray((img_path * 255).astype(np.uint8))
                else:
                    pil_img = Image.fromarray((img_path.squeeze() * 255).astype(np.uint8)).convert('RGB')
            else:
                raise ValueError("Unknown image format")
        
        # Resize
        pil_img = pil_img.resize(target_size, Image.LANCZOS)
        
        # Convert to array and normalize
        img_array = np.array(pil_img) / 255.0
        
        return img_array
        
    except Exception as e:
        print(f"Error loading image {img_path}: {e}")
        # Return blank image
        return np.ones((*target_size, 3)) * 0.5

def create_data_generator(image_paths, labels, batch_size=8, target_size=(384, 384), shuffle=True, augment=False):
    """Create data generator that loads images on demand"""
    
    def generator():
        indices = np.arange(len(image_paths))
        
        while True:
            if shuffle:
                np.random.shuffle(indices)
            
            for i in range(0, len(indices), batch_size):
                batch_indices = indices[i:i + batch_size]
                
                # Load batch images
                batch_images = []
                batch_labels = []
                
                for idx in batch_indices:
                    img = load_and_preprocess_image(image_paths[idx], target_size)
                    
                    # Apply augmentation if requested
                    if augment and np.random.random() > 0.5:
                        # Random horizontal flip
                        if np.random.random() > 0.5:
                            img = np.fliplr(img)
                        
                        # Random brightness
                        brightness_factor = np.random.uniform(0.8, 1.2)
                        img = np.clip(img * brightness_factor, 0, 1)
                        
                        # Random rotation (small)
                        if np.random.random() > 0.7:
                            angle = np.random.uniform(-10, 10)
                            pil_img = Image.fromarray((img * 255).astype(np.uint8))
                            rotated = pil_img.rotate(angle, fillcolor=(128, 128, 128))
                            img = np.array(rotated) / 255.0
                    
                    batch_images.append(img)
                    batch_labels.append(labels[idx])
                
                batch_images = np.array(batch_images)
                batch_labels = keras.utils.to_categorical(batch_labels, num_classes=2)
                
                yield batch_images, batch_labels
    
    return generator

def create_medical_model(input_shape=(384, 384, 3), num_classes=2):
    """Create EfficientNet-B4 based medical model"""
    
    print("🏗️ Building EfficientNet-B4 medical model...")
    
    # Base model
    base_model = applications.EfficientNetB4(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # Fine-tuning
    total_layers = len(base_model.layers)
    trainable_layers = int(total_layers * 0.7)
    
    for layer in base_model.layers[:-trainable_layers]:
        layer.trainable = False
    for layer in base_model.layers[-trainable_layers:]:
        layer.trainable = True
    
    print(f"📊 {total_layers} layers, {trainable_layers} trainable")
    
    # Build model
    inputs = keras.Input(shape=input_shape)
    x = applications.efficientnet.preprocess_input(inputs)
    
    # Base features
    features = base_model(x, training=True)
    
    # Pooling
    x = layers.GlobalAveragePooling2D()(features)
    
    # Attention
    attention = layers.Dense(x.shape[-1], activation='sigmoid')(x)
    x = layers.multiply([x, attention])
    
    # Dense layers
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    
    # Grad-CAM layer
    x = layers.Dense(256, activation='relu', name='gradcam_target_layer')(x)
    x = layers.Dropout(0.3)(x)
    
    # Output
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs, name='cardiomegaly_medical_model')
    
    return model

def focal_loss(alpha=0.75, gamma=2.0):
    """Focal loss implementation"""
    def loss_fn(y_true, y_pred):
        epsilon = 1e-7
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        
        ce_loss = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)
        p_t = tf.reduce_sum(y_true * y_pred, axis=-1)
        
        alpha_factor = alpha
        modulating_factor = tf.pow((1.0 - p_t), gamma)
        
        return tf.reduce_mean(alpha_factor * modulating_factor * ce_loss)
    
    return loss_fn

def cosine_schedule(epoch):
    """Cosine learning rate schedule"""
    if epoch < 5:
        return (epoch + 1) * 0.001 / 5
    else:
        progress = (epoch - 5) / 45
        return 1e-6 + (0.001 - 1e-6) * 0.5 * (1 + np.cos(np.pi * progress))

def train_cardiomegaly_95():
    """Train cardiomegaly model for 95%+ accuracy"""
    
    print("🎯 Medical-Grade Cardiomegaly Training for 95%+ Accuracy")
    print("=" * 60)
    
    # Load data
    dataset_info = load_cardiomegaly_data()
    if not dataset_info:
        print("❌ Failed to load dataset")
        return False
    
    # Create model
    model = create_medical_model()
    print(f"📊 Model parameters: {model.count_params():,}")
    
    # Compile
    optimizer = keras.optimizers.AdamW(
        learning_rate=0.001,
        weight_decay=0.01,
        clipnorm=1.0
    )
    
    model.compile(
        optimizer=optimizer,
        loss=focal_loss(alpha=0.75, gamma=2.0),
        metrics=['accuracy', 'precision', 'recall', keras.metrics.AUC(name='auc')]
    )
    
    # Setup callbacks
    model_path = "models/cardiomegaly_95_medical.h5"
    os.makedirs("models", exist_ok=True)
    
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=15,
            restore_best_weights=True,
            min_delta=0.001,
            mode='max'
        ),
        
        keras.callbacks.LearningRateScheduler(cosine_schedule),
        
        keras.callbacks.ModelCheckpoint(
            model_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        ),
        
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.3,
            patience=8,
            min_lr=1e-7
        )
    ]
    
    # Create generators
    print("📊 Creating data generators...")
    
    train_gen = create_data_generator(
        dataset_info['X_train'], 
        dataset_info['y_train'],
        batch_size=8,
        shuffle=True,
        augment=True
    )
    
    val_gen = create_data_generator(
        dataset_info['X_val'],
        dataset_info['y_val'], 
        batch_size=8,
        shuffle=False,
        augment=False
    )
    
    # Calculate steps
    steps_per_epoch = len(dataset_info['X_train']) // 8
    validation_steps = len(dataset_info['X_val']) // 8
    
    print(f"📋 Training configuration:")
    print(f"   Steps per epoch: {steps_per_epoch}")
    print(f"   Validation steps: {validation_steps}")
    print(f"   Batch size: 8")
    print(f"   Max epochs: 50")
    
    # Train
    print("\n🚂 Starting training...")
    start_time = time.time()
    
    history = model.fit(
        train_gen(),
        steps_per_epoch=steps_per_epoch,
        validation_data=val_gen(),
        validation_steps=validation_steps,
        epochs=50,
        callbacks=callbacks,
        verbose=1
    )
    
    training_time = time.time() - start_time
    
    # Evaluate on test set
    print("\n📊 Final evaluation...")
    
    # Load test images
    test_images = []
    for img_path in dataset_info['X_test']:
        img = load_and_preprocess_image(img_path)
        test_images.append(img)
    
    test_images = np.array(test_images)
    test_labels_cat = keras.utils.to_categorical(dataset_info['y_test'], num_classes=2)
    
    test_results = model.evaluate(test_images, test_labels_cat, batch_size=8, verbose=0)
    
    test_loss = test_results[0]
    test_accuracy = test_results[1]
    test_precision = test_results[2] if len(test_results) > 2 else 0
    test_recall = test_results[3] if len(test_results) > 3 else 0
    test_auc = test_results[4] if len(test_results) > 4 else 0
    
    # Save model
    print("💾 Saving model...")
    model.save(model_path)
    
    # Results
    print("\n" + "=" * 60)
    print("🎉 TRAINING COMPLETED")
    print("=" * 60)
    
    print(f"🎯 Test Accuracy: {test_accuracy:.1%}")
    print(f"📊 AUC Score: {test_auc:.3f}")
    print(f"⚖️ Precision: {test_precision:.1%}")
    print(f"🔍 Recall: {test_recall:.1%}")
    print(f"⏱️ Training Time: {training_time/3600:.1f} hours")
    
    if test_accuracy >= 0.95:
        print("\n🎉 SUCCESS: 95%+ ACCURACY ACHIEVED!")
        print("🏆 Medical-grade performance reached")
        
        if test_auc >= 0.98:
            print("🌟 Excellent discrimination (AUC ≥ 0.98)")
        if test_precision >= 0.92 and test_recall >= 0.92:
            print("🎯 Balanced performance (P&R ≥ 92%)")
        
        print("\n🚀 Ready for clinical validation!")
        
    elif test_accuracy >= 0.90:
        print("\n📈 EXCELLENT: 90%+ accuracy achieved")
        remaining = 0.95 - test_accuracy
        print(f"   Only {remaining:.1%} more needed for 95%")
        
    else:
        print(f"\n⚠️ BELOW TARGET: {test_accuracy:.1%}")
        print("Consider longer training or ensemble approach")
    
    # Training summary
    if hasattr(history, 'history'):
        best_epoch = np.argmax(history.history['val_accuracy']) + 1
        best_val_acc = max(history.history['val_accuracy'])
        
        print(f"\n📈 Training Summary:")
        print(f"   Best epoch: {best_epoch}")
        print(f"   Best val accuracy: {best_val_acc:.1%}")
        print(f"   Total epochs: {len(history.history['accuracy'])}")
    
    return test_accuracy >= 0.95

if __name__ == "__main__":
    print("🏥 Standalone Medical-Grade Cardiomegaly Training")
    print("🎯 Target: 95%+ Accuracy")
    print("🚀 Starting...\n")
    
    success = train_cardiomegaly_95()
    
    if success:
        print("\n✅ MISSION ACCOMPLISHED: 95%+ accuracy achieved!")
    else:
        print("\n📊 Training completed. Check results above.")