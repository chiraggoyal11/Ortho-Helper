#!/usr/bin/env python3
"""
MAXIMUM Performance DenseNet Bone Fracture Training
Ultimate optimization for highest possible accuracy
"""
import tensorflow as tf
from tensorflow.keras.applications import DenseNet121, EfficientNetB4
from tensorflow.keras.layers import (
    GlobalAveragePooling2D, Dense, Dropout, BatchNormalization, 
    LayerNormalization, Multiply, Add, Concatenate, GlobalMaxPooling2D
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import os
import numpy as np
from datetime import datetime
import json
from sklearn.utils.class_weight import compute_class_weight

class MaximumBoneFractureTrainer:
    def __init__(self, input_shape=(256, 256, 3), num_classes=2):
        """Initialize with maximum input resolution for best performance"""
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        
    def create_maximum_model(self):
        """Create ultimate DenseNet model with maximum optimizations"""
        # Use EfficientNetB4 for better performance than DenseNet121
        base_model = EfficientNetB4(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        # Progressive unfreezing - start with more trainable layers
        for layer in base_model.layers[:-80]:
            layer.trainable = False
        
        # Advanced dual-pooling feature extraction
        x = base_model.output
        
        # Dual pooling for better feature capture
        avg_pool = GlobalAveragePooling2D()(x)
        max_pool = GlobalMaxPooling2D()(x)
        
        # Concatenate pooling outputs
        x = Concatenate()([avg_pool, max_pool])
        
        # Advanced squeeze-and-excitation-like attention
        se_shape = int(x.shape[-1])
        se = Dense(se_shape // 4, activation='relu')(x)
        se = Dense(se_shape, activation='sigmoid')(se)
        x = Multiply()([x, se])
        
        # Multiple residual dense blocks for maximum capacity
        x1 = Dense(1024, activation='gelu', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
        x1 = BatchNormalization()(x1)
        x1 = Dropout(0.4)(x1)
        
        x2 = Dense(512, activation='gelu', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x1)
        x2 = BatchNormalization()(x2)
        x2 = Dropout(0.3)(x2)
        
        # Residual connection with dimension matching
        x1_reduced = Dense(512, activation='linear')(x1)
        x2_residual = Add()([x2, x1_reduced])
        
        x3 = Dense(256, activation='gelu', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x2_residual)
        x3 = BatchNormalization()(x3)
        x3 = Dropout(0.2)(x3)
        
        x4 = Dense(128, activation='gelu', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x3)
        x4 = BatchNormalization()(x4)
        x4 = Dropout(0.1)(x4)
        
        # Final classification layer
        predictions = Dense(self.num_classes, activation='softmax', name='predictions')(x4)
        
        self.model = Model(inputs=base_model.input, outputs=predictions)
        
        # Maximum performance optimizer
        optimizer = Adam(
            learning_rate=5e-5,  # Lower for stability
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-8,
            amsgrad=True
        )
        
        # Fixed metrics compilation
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']  # Simplified metrics to avoid string error
        )
        
        return self.model
    
    def create_maximum_data_generators(self, train_dir, val_dir, batch_size=16):
        """Create maximum quality data generators"""
        
        # Extensive augmentation for maximum generalization
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=25,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.15,
            zoom_range=0.2,
            horizontal_flip=True,
            brightness_range=[0.7, 1.3],
            fill_mode='reflect',
            channel_shift_range=0.15
        )
        
        # Clean validation data
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=self.input_shape[:2],
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=True,
            seed=42
        )
        
        validation_generator = val_datagen.flow_from_directory(
            val_dir,
            target_size=self.input_shape[:2],
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        return train_generator, validation_generator
    
    def create_maximum_callbacks(self, model_save_path):
        """Create maximum performance callbacks"""
        callbacks = [
            ModelCheckpoint(
                model_save_path,
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                mode='max',
                verbose=1
            ),
            EarlyStopping(
                monitor='val_accuracy',
                patience=20,  # Extended patience for maximum training
                restore_best_weights=True,
                mode='max',
                verbose=1,
                min_delta=0.0001
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.1,  # More aggressive LR reduction
                patience=10,
                min_lr=1e-8,
                verbose=1,
                cooldown=5
            ),
            # Custom learning rate schedule for maximum performance
            tf.keras.callbacks.LearningRateScheduler(
                lambda epoch: 5e-5 * 0.97 ** epoch,
                verbose=0
            )
        ]
        
        return callbacks
    
    def calculate_class_weights(self, train_generator):
        """Calculate balanced class weights"""
        labels = train_generator.labels
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(labels),
            y=labels
        )
        
        class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
        print(f"📊 Class weights: {class_weight_dict}")
        
        return class_weight_dict
    
    def train(self, train_dir, val_dir, epochs=100, batch_size=16):
        """Maximum performance training protocol"""
        print("🚀 Starting MAXIMUM Performance DenseNet Bone Fracture Training")
        print(f"Input shape: {self.input_shape}")
        print(f"Batch size: {batch_size}")
        print(f"Max epochs: {epochs}")
        
        # Create maximum model
        model = self.create_maximum_model()
        print(f"✅ Maximum model created with {model.count_params():,} parameters")
        
        # Create maximum data generators
        train_gen, val_gen = self.create_maximum_data_generators(train_dir, val_dir, batch_size)
        
        # Calculate class weights
        class_weights = self.calculate_class_weights(train_gen)
        
        print(f"📊 Training samples: {train_gen.samples}")
        print(f"📊 Validation samples: {val_gen.samples}")
        print(f"📊 Classes: {train_gen.class_indices}")
        
        # Create model save path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_save_path = f"models/EfficientNetB4_bone_fracture_maximum_{timestamp}.h5"
        os.makedirs("models", exist_ok=True)
        
        # Create maximum callbacks
        callbacks = self.create_maximum_callbacks(model_save_path)
        
        # Calculate steps
        steps_per_epoch = train_gen.samples // batch_size
        validation_steps = val_gen.samples // batch_size
        
        print(f"🔄 Steps per epoch: {steps_per_epoch}")
        print(f"🔄 Validation steps: {validation_steps}")
        
        # Multi-stage training for absolute maximum performance
        print("\n🎯 Stage 1: Foundation Training (60 epochs max)")
        history1 = model.fit(
            train_gen,
            steps_per_epoch=steps_per_epoch,
            epochs=min(60, epochs),
            validation_data=val_gen,
            validation_steps=validation_steps,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )
        
        # Stage 2: Progressive unfreezing for maximum fine-tuning
        print("\n🎯 Stage 2: Progressive Fine-tuning")
        for layer in model.layers[-150:]:  # Unfreeze more layers
            layer.trainable = True
        
        # Recompile with ultra-low learning rate
        model.compile(
            optimizer=Adam(learning_rate=1e-6, amsgrad=True),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        remaining_epochs = epochs - min(60, epochs)
        if remaining_epochs > 0:
            history2 = model.fit(
                train_gen,
                steps_per_epoch=steps_per_epoch,
                epochs=remaining_epochs,
                validation_data=val_gen,
                validation_steps=validation_steps,
                callbacks=callbacks,
                class_weight=class_weights,
                verbose=1,
                initial_epoch=min(60, epochs)
            )
        
        # Load absolute best model
        best_model = tf.keras.models.load_model(model_save_path)
        
        # Comprehensive evaluation
        print("\n📈 MAXIMUM Performance Model Evaluation:")
        val_loss, val_accuracy = best_model.evaluate(val_gen, verbose=0)
        
        print(f"✅ Validation Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
        print(f"✅ Validation Loss: {val_loss:.4f}")
        
        # Save final maximum model
        final_model_path = "models/EfficientNetB4_bone_fracture_maximum.h5"
        best_model.save(final_model_path)
        print(f"🎉 MAXIMUM model saved as: {final_model_path}")
        
        return best_model, val_accuracy

def prepare_maximum_dataset():
    """Prepare maximum quality dataset"""
    import shutil
    import random
    
    # Create maximum dataset directory
    max_dir = "Dataset/ARM_Maximum"
    train_dir = os.path.join(max_dir, "train")
    val_dir = os.path.join(max_dir, "val")
    
    # Clean up old directory
    if os.path.exists(max_dir):
        shutil.rmtree(max_dir)
    
    # Create directories
    for split in ['train', 'val']:
        for class_name in ['Normal', 'Fracture']:
            os.makedirs(os.path.join(max_dir, split, class_name), exist_ok=True)
    
    # Source directories
    forearm_neg = "Dataset/ARM/MURA_Organized/Forearm/Negative"
    forearm_pos = "Dataset/ARM/MURA_Organized/Forearm/Positive"
    humerus_neg = "Dataset/ARM/MURA_Organized/Humerus/Negative"
    humerus_pos = "Dataset/ARM/MURA_Organized/Humerus/Positive"
    
    print("🔄 Preparing MAXIMUM quality dataset...")
    
    # Collect all files
    all_negative = []
    all_positive = []
    
    for source_dir in [forearm_neg, humerus_neg]:
        if os.path.exists(source_dir):
            files = [os.path.join(source_dir, f) for f in os.listdir(source_dir) 
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            all_negative.extend(files)
    
    for source_dir in [forearm_pos, humerus_pos]:
        if os.path.exists(source_dir):
            files = [os.path.join(source_dir, f) for f in os.listdir(source_dir) 
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            all_positive.extend(files)
    
    print(f"📊 Found {len(all_negative)} normal images")
    print(f"📊 Found {len(all_positive)} fracture images")
    
    # Use maximum available data for highest quality
    max_samples = min(len(all_negative), len(all_positive), 1200)  # Increased dataset
    
    random.seed(42)
    selected_negative = random.sample(all_negative, max_samples)
    selected_positive = random.sample(all_positive, max_samples)
    
    # Split data (85/15 for maximum training data)
    train_split = int(max_samples * 0.85)
    
    # Copy training files
    for i, file_path in enumerate(selected_negative[:train_split]):
        dest_path = os.path.join(train_dir, "Normal", f"normal_{i}.jpg")
        shutil.copy2(file_path, dest_path)
    
    for i, file_path in enumerate(selected_positive[:train_split]):
        dest_path = os.path.join(train_dir, "Fracture", f"fracture_{i}.jpg")
        shutil.copy2(file_path, dest_path)
    
    # Copy validation files
    for i, file_path in enumerate(selected_negative[train_split:]):
        dest_path = os.path.join(val_dir, "Normal", f"normal_{i}.jpg")
        shutil.copy2(file_path, dest_path)
    
    for i, file_path in enumerate(selected_positive[train_split:]):
        dest_path = os.path.join(val_dir, "Fracture", f"fracture_{i}.jpg")
        shutil.copy2(file_path, dest_path)
    
    print(f"✅ MAXIMUM dataset prepared: {max_dir}")
    print(f"📊 Training: {train_split * 2} images ({train_split} per class)")
    print(f"📊 Validation: {(max_samples - train_split) * 2} images ({max_samples - train_split} per class)")
    
    return train_dir, val_dir

def main():
    """Main MAXIMUM performance training function"""
    print("=" * 75)
    print("🦴 MAXIMUM PERFORMANCE EfficientNetB4 Bone Fracture Training")
    print("=" * 75)
    
    # Prepare maximum dataset
    train_dir, val_dir = prepare_maximum_dataset()
    
    if not train_dir:
        print("❌ Could not prepare dataset")
        return
    
    # Initialize maximum trainer
    trainer = MaximumBoneFractureTrainer(
        input_shape=(256, 256, 3),  # Maximum resolution
        num_classes=2
    )
    
    # Start maximum performance training
    try:
        model, accuracy = trainer.train(
            train_dir=train_dir,
            val_dir=val_dir,
            epochs=100,  # Maximum epochs
            batch_size=16   # Optimal batch size for quality
        )
        
        print(f"\n🎯 MAXIMUM Performance Training Completed!")
        print(f"✅ Final accuracy: {accuracy*100:.2f}%")
        
        if accuracy >= 0.92:
            print(f"🎉 OUTSTANDING! Medical-grade excellence: {accuracy*100:.2f}%")
        elif accuracy >= 0.88:
            print(f"🎉 EXCELLENT! High-performance achieved: {accuracy*100:.2f}%")
        elif accuracy >= 0.85:
            print(f"✅ VERY GOOD performance: {accuracy*100:.2f}%")
        elif accuracy >= 0.82:
            print(f"✅ GOOD improvement: {accuracy*100:.2f}%")
        else:
            print(f"📈 Progress made, targeting >85%: {accuracy*100:.2f}%")
            
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()