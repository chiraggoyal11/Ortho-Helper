#!/usr/bin/env python3
"""
Cardiomegaly Anti-Overfitting Training
Robust 95%+ accuracy without overfitting
"""

import os
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2
import cv2
from PIL import Image

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def load_cardiomegaly_data():
    """Load cardiomegaly data with proper validation split"""
    print("📊 Loading cardiomegaly dataset...")
    
    data_path = "Dataset/cardiomelgy"
    
    if not os.path.exists(data_path):
        print(f"❌ Data path not found: {data_path}")
        return None
    
    # Load image paths from the actual structure
    train_path = os.path.join(data_path, "train", "train")
    normal_path = os.path.join(train_path, "false")  # false = normal
    cardiomegaly_path = os.path.join(train_path, "true")  # true = cardiomegaly
    
    normal_images = []
    cardiomegaly_images = []
    
    if os.path.exists(normal_path):
        for img_file in os.listdir(normal_path):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                normal_images.append(os.path.join(normal_path, img_file))
    
    if os.path.exists(cardiomegaly_path):
        for img_file in os.listdir(cardiomegaly_path):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                cardiomegaly_images.append(os.path.join(cardiomegaly_path, img_file))
    
    print(f"📈 Found {len(normal_images)} normal, {len(cardiomegaly_images)} cardiomegaly images")
    
    # Create labels
    all_images = normal_images + cardiomegaly_images
    all_labels = [0] * len(normal_images) + [1] * len(cardiomegaly_images)
    
    # Shuffle data
    indices = np.random.permutation(len(all_images))
    all_images = [all_images[i] for i in indices]
    all_labels = [all_labels[i] for i in indices]
    
    # Split data with proper validation
    train_split = 0.7  # 70% train
    val_split = 0.15   # 15% validation
    # 15% test
    
    n_total = len(all_images)
    n_train = int(n_total * train_split)
    n_val = int(n_total * val_split)
    
    X_train = all_images[:n_train]
    y_train = all_labels[:n_train]
    
    X_val = all_images[n_train:n_train + n_val]
    y_val = all_labels[n_train:n_train + n_val]
    
    X_test = all_images[n_train + n_val:]
    y_test = all_labels[n_train + n_val:]
    
    print(f"📊 Data split: {len(X_train)} train, {len(X_val)} val, {len(X_test)} test")
    
    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test
    }

def robust_preprocess_image(img_path, target_size=(224, 224)):
    """Robust image preprocessing with normalization"""
    try:
        img = Image.open(img_path).convert('RGB')
        img = img.resize(target_size, Image.Resampling.LANCZOS)
        img_array = np.array(img) / 255.0  # Normalize to [0,1]
        return img_array
    except Exception as e:
        print(f"Error loading {img_path}: {e}")
        return np.zeros((*target_size, 3))

def create_robust_model():
    """Create model with anti-overfitting measures"""
    print("🏗️  Creating robust anti-overfitting model...")
    
    # Base model with lower trainable layers
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    
    # Freeze most layers to prevent overfitting
    for layer in base_model.layers[:-20]:  # Only train last 20 layers
        layer.trainable = False
    
    # Add regularized top layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)  # Strong dropout
    x = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(x)  # L2 regularization
    x = Dropout(0.3)(x)
    predictions = Dense(2, activation='softmax', kernel_regularizer=l2(0.01))(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Compile with lower learning rate
    model.compile(
        optimizer=Adam(learning_rate=0.0001),  # Much lower LR
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"✅ Robust model created with {model.count_params():,} parameters")
    print(f"🔒 Trainable parameters: {sum([tf.size(var).numpy() for var in model.trainable_variables]):,}")
    
    return model

def create_robust_generator(image_paths, labels, batch_size=8, augment=True):
    """Data generator with strong augmentation to prevent overfitting"""
    def generator():
        indices = np.arange(len(image_paths))
        
        while True:
            np.random.shuffle(indices)
            
            for i in range(0, len(indices), batch_size):
                batch_indices = indices[i:i + batch_size]
                
                batch_images = []
                batch_labels = []
                
                for idx in batch_indices:
                    img = robust_preprocess_image(image_paths[idx])
                    
                    # Strong augmentation to prevent overfitting
                    if augment:
                        # Random rotation
                        if np.random.random() > 0.5:
                            angle = np.random.uniform(-15, 15)
                            img = tf.keras.preprocessing.image.apply_affine_transform(
                                img, theta=angle, fill_mode='nearest'
                            )
                        
                        # Random horizontal flip
                        if np.random.random() > 0.5:
                            img = np.fliplr(img)
                        
                        # Random brightness
                        if np.random.random() > 0.5:
                            brightness = np.random.uniform(0.8, 1.2)
                            img = np.clip(img * brightness, 0, 1)
                        
                        # Random zoom
                        if np.random.random() > 0.5:
                            zoom = np.random.uniform(0.9, 1.1)
                            img = tf.keras.preprocessing.image.apply_affine_transform(
                                img, zx=zoom, zy=zoom, fill_mode='nearest'
                            )
                    
                    batch_images.append(img)
                    batch_labels.append(labels[idx])
                
                yield np.array(batch_images), keras.utils.to_categorical(batch_labels, 2)
    
    return generator

def train_robust_cardiomegaly():
    """Train robust cardiomegaly model without overfitting"""
    
    print("🛡️  ROBUST CARDIOMEGALY TRAINING (ANTI-OVERFITTING)")
    print("=" * 60)
    print("🎯 Target: 95%+ accuracy WITHOUT overfitting")
    print("🛡️  Anti-overfitting measures: Dropout, L2, Low LR, Strong Aug")
    print("=" * 60)
    
    # Load data
    dataset = load_cardiomegaly_data()
    if not dataset:
        return None, None
    
    # Create robust model
    model = create_robust_model()
    
    # Robust callbacks with validation monitoring
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',  # Monitor validation, not training
            patience=8,  # More patience
            restore_best_weights=True,
            min_delta=0.001,
            verbose=1
        ),
        
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.3,  # More aggressive LR reduction
            patience=4,
            min_lr=1e-7,
            verbose=1
        ),
        
        ModelCheckpoint(
            'models/robust_cardiomegaly_95.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Create generators with smaller batch size
    train_gen = create_robust_generator(
        dataset['X_train'], 
        dataset['y_train'],
        batch_size=8,  # Smaller batch size
        augment=True
    )
    
    val_gen = create_robust_generator(
        dataset['X_val'],
        dataset['y_val'], 
        batch_size=8,
        augment=False  # No augmentation for validation
    )
    
    # Calculate steps
    steps_per_epoch = len(dataset['X_train']) // 8
    validation_steps = len(dataset['X_val']) // 8
    
    print(f"\n🛡️  Robust training configuration:")
    print(f"   📚 Training samples: {len(dataset['X_train'])}")
    print(f"   🔍 Validation samples: {len(dataset['X_val'])}")
    print(f"   📊 Batch size: 8 (smaller for stability)")
    print(f"   📈 Steps per epoch: {steps_per_epoch}")
    print(f"   🔍 Validation steps: {validation_steps}")
    print(f"   🎯 Max epochs: 30 (with early stopping)")
    
    # Start training
    print("\n🚀 Starting robust training...")
    start_time = time.time()
    
    history = model.fit(
        train_gen(),
        steps_per_epoch=steps_per_epoch,
        validation_data=val_gen(),
        validation_steps=validation_steps,
        epochs=30,  # More epochs with early stopping
        callbacks=callbacks,
        verbose=1
    )
    
    training_time = time.time() - start_time
    
    # Evaluate on test set
    print("\n📊 Final evaluation on test set...")
    
    test_images = []
    test_labels = []
    
    for i, (img_path, label) in enumerate(zip(dataset['X_test'], dataset['y_test'])):
        if i >= 200:  # Evaluate on 200 test images
            break
        img = robust_preprocess_image(img_path)
        test_images.append(img)
        test_labels.append(label)
    
    test_images = np.array(test_images)
    test_labels = keras.utils.to_categorical(test_labels, 2)
    
    results = model.evaluate(test_images, test_labels, verbose=0)
    test_accuracy = results[1]
    test_loss = results[0]
    
    # Get training history for overfitting analysis
    train_acc = max(history.history['accuracy'])
    val_acc = max(history.history['val_accuracy'])
    overfitting_gap = train_acc - val_acc
    
    # Save final model
    model_path = "models/robust_cardiomegaly_95_final.h5"
    os.makedirs("models", exist_ok=True)
    model.save(model_path)
    
    # Results
    print("\n" + "=" * 50)
    print("🛡️  ROBUST TRAINING RESULTS")
    print("=" * 50)
    
    print(f"🎯 Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"📉 Test Loss: {test_loss:.4f}")
    print(f"📚 Max Training Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"🔍 Max Validation Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
    print(f"⚖️  Overfitting Gap: {overfitting_gap:.4f} ({overfitting_gap*100:.2f}%)")
    print(f"⏱️  Training Time: {training_time/60:.1f} minutes")
    print(f"💾 Model saved: {model_path}")
    
    # Analysis
    if overfitting_gap < 0.05:  # Less than 5% gap
        print("\n🎉 EXCELLENT: No overfitting detected!")
        if test_accuracy >= 0.95:
            print("🏆 SUCCESS: 95%+ achieved WITHOUT overfitting!")
        else:
            print(f"📈 Good progress: {test_accuracy*100:.1f}% without overfitting")
    elif overfitting_gap < 0.10:
        print("\n✅ GOOD: Minimal overfitting (acceptable)")
        if test_accuracy >= 0.90:
            print("📊 Strong performance with controlled overfitting")
    else:
        print(f"\n⚠️  WARNING: Significant overfitting detected ({overfitting_gap*100:.1f}% gap)")
        print("🛡️  Consider more regularization")
    
    return model, history

def main():
    """Main training function"""
    try:
        result = train_robust_cardiomegaly()
        
        if result is not None and len(result) == 2:
            model, history = result
            if model is not None:
                print("\n🎉 Robust cardiomegaly training completed!")
                print("🛡️  Anti-overfitting measures successfully applied")
            else:
                print("\n❌ Training failed")
        else:
            print("\n❌ Training failed - dataset not found")
            
    except Exception as e:
        print(f"\n❌ Training error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()