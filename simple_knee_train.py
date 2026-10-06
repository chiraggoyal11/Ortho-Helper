#!/usr/bin/env python3
"""
Simple Knee Arthritis Training - Minimal Dependencies
Fast 95%+ accuracy training for knee osteoarthritis classification
"""

import os
import sys
import numpy as np
import time
from datetime import datetime

# Import TensorFlow with error handling
try:
    import tensorflow as tf
    from tensorflow.keras.applications import EfficientNetB0
    from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
    print("✅ TensorFlow loaded successfully")
except Exception as e:
    print(f"❌ TensorFlow import failed: {e}")
    sys.exit(1)

# Simple configuration
CONFIG = {
    'data_path': r'Dataset/Osteoarthritis Knee X-ray',
    'image_size': (224, 224),
    'batch_size': 16,  # Smaller batch to avoid memory issues
    'num_classes': 5,
    'epochs': 15,
    'learning_rate': 0.0001
}

def simple_knee_model():
    """Create a simple but effective knee model"""
    print("🏗️  Creating simple knee model...")
    
    # Use EfficientNet as base
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(*CONFIG['image_size'], 3)
    )
    
    # Freeze base model initially
    base_model.trainable = False
    
    # Add custom top
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(CONFIG['num_classes'], activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Compile
    model.compile(
        optimizer=Adam(learning_rate=CONFIG['learning_rate']),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"✅ Model created with {model.count_params():,} parameters")
    return model

def setup_data_generators():
    """Setup data generators with simple augmentation"""
    print("📊 Setting up data generators...")
    
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        validation_split=0.2
    )
    
    # Load training data
    train_generator = train_datagen.flow_from_directory(
        CONFIG['data_path'],
        target_size=CONFIG['image_size'],
        batch_size=CONFIG['batch_size'],
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    # Load validation data
    val_generator = train_datagen.flow_from_directory(
        CONFIG['data_path'],
        target_size=CONFIG['image_size'],
        batch_size=CONFIG['batch_size'],
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    print(f"✅ Data loaded: {train_generator.samples} training, {val_generator.samples} validation")
    return train_generator, val_generator

def train_simple_knee_model():
    """Train the knee model with simple approach"""
    print("🦴 SIMPLE KNEE ARTHRITIS TRAINING")
    print("=" * 50)
    
    start_time = time.time()
    
    try:
        # Create model
        model = simple_knee_model()
        
        # Setup data
        train_gen, val_gen = setup_data_generators()
        
        # Setup callbacks
        callbacks = [
            ModelCheckpoint(
                'models/simple_knee_95.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            EarlyStopping(
                monitor='val_accuracy',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_accuracy',
                factor=0.5,
                patience=3,
                verbose=1
            )
        ]
        
        print("🚀 Starting training...")
        
        # Train model
        history = model.fit(
            train_gen,
            epochs=CONFIG['epochs'],
            validation_data=val_gen,
            callbacks=callbacks,
            verbose=1
        )
        
        # Get best accuracy
        best_val_acc = max(history.history['val_accuracy'])
        training_time = time.time() - start_time
        
        print(f"\n🎉 TRAINING COMPLETED!")
        print(f"⏱️  Training time: {training_time/60:.1f} minutes")
        print(f"🎯 Best validation accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
        
        # Save final model if good accuracy
        if best_val_acc >= 0.95:
            final_name = f'models/knee_arthritis_95_SUCCESS_{best_val_acc:.3f}.h5'
            model.save(final_name)
            print(f"✅ SUCCESS! Model saved as: {final_name}")
        else:
            final_name = f'models/knee_arthritis_attempt_{best_val_acc:.3f}.h5'
            model.save(final_name)
            print(f"📊 Model saved as: {final_name}")
            print(f"💡 Target 95% not reached, but good progress made!")
        
        return model, history
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return None, None

def main():
    """Main training function"""
    print("🦴 KNEE OSTEOARTHRITIS SIMPLE TRAINING")
    print("🎯 Target: 95%+ accuracy")
    print("⚡ Parallel with cardiomegaly training")
    print("=" * 60)
    
    # Check if data exists
    if not os.path.exists(CONFIG['data_path']):
        print(f"❌ Data path not found: {CONFIG['data_path']}")
        return
    
    # Ensure models directory exists
    os.makedirs('models', exist_ok=True)
    
    # Start training
    model, history = train_simple_knee_model()
    
    if model is not None:
        print("\n🎉 Knee arthritis training completed successfully!")
    else:
        print("\n❌ Training failed - check logs above")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()