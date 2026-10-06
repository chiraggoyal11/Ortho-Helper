#!/usr/bin/env python3
"""
Enhanced DenseNet Bone Fracture Training
Fixed overfitting issues and improved data quality
"""
import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import os
import numpy as np
from datetime import datetime
import json

class EnhancedBoneFractureTrainer:
    def __init__(self, input_shape=(192, 192, 3), num_classes=2):
        """Initialize with optimized input size"""
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        
    def create_model(self):
        """Create enhanced DenseNet121 model with better regularization"""
        # Load pre-trained DenseNet121
        base_model = DenseNet121(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        # Gradually unfreeze layers for better fine-tuning
        for layer in base_model.layers[:-40]:
            layer.trainable = False
        
        # Enhanced classification head with better regularization
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = BatchNormalization()(x)
        
        # More aggressive regularization to prevent overfitting
        x = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
        x = Dropout(0.5)(x)
        x = BatchNormalization()(x)
        
        x = Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
        x = Dropout(0.4)(x)
        x = BatchNormalization()(x)
        
        # Final classification layer
        predictions = Dense(self.num_classes, activation='softmax', name='predictions')(x)
        
        self.model = Model(inputs=base_model.input, outputs=predictions)
        
        # Compile with lower learning rate and better optimization
        self.model.compile(
            optimizer=Adam(learning_rate=5e-4, beta_1=0.9, beta_2=0.999),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model
    
    def create_balanced_data_generators(self, train_dir, val_dir, batch_size=32):
        """Create balanced data generators with better augmentation"""
        
        # More conservative augmentation to prevent overfitting
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            brightness_range=[0.9, 1.1],
            fill_mode='nearest'
        )
        
        # Validation data generator
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
    
    def create_callbacks(self, model_save_path):
        """Create enhanced callbacks"""
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
                patience=12,
                restore_best_weights=True,
                mode='max',
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.3,
                patience=6,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        return callbacks
    
    def train(self, train_dir, val_dir, epochs=50, batch_size=32):
        """Enhanced training with better monitoring"""
        print("🚀 Starting Enhanced DenseNet Bone Fracture Training")
        print(f"Input shape: {self.input_shape}")
        print(f"Batch size: {batch_size}")
        print(f"Max epochs: {epochs}")
        
        # Create model
        model = self.create_model()
        print(f"✅ Model created with {model.count_params():,} parameters")
        
        # Create data generators
        train_gen, val_gen = self.create_balanced_data_generators(train_dir, val_dir, batch_size)
        
        print(f"📊 Training samples: {train_gen.samples}")
        print(f"📊 Validation samples: {val_gen.samples}")
        print(f"📊 Classes: {train_gen.class_indices}")
        
        # Create model save path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_save_path = f"models/DenseNet121_bone_fracture_enhanced_{timestamp}.h5"
        os.makedirs("models", exist_ok=True)
        
        # Create callbacks
        callbacks = self.create_callbacks(model_save_path)
        
        # Calculate steps
        steps_per_epoch = train_gen.samples // batch_size
        validation_steps = val_gen.samples // batch_size
        
        print(f"🔄 Steps per epoch: {steps_per_epoch}")
        print(f"🔄 Validation steps: {validation_steps}")
        
        # Two-phase training for better convergence
        print("\n🎯 Phase 1: Conservative Training")
        history1 = model.fit(
            train_gen,
            steps_per_epoch=steps_per_epoch,
            epochs=min(25, epochs),
            validation_data=val_gen,
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=1
        )
        
        # Phase 2: Fine-tuning with more layers
        print("\n🎯 Phase 2: Fine-tuning")
        for layer in model.layers:
            layer.trainable = True
        
        # Recompile with lower learning rate
        model.compile(
            optimizer=Adam(learning_rate=1e-5, beta_1=0.9, beta_2=0.999),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        remaining_epochs = epochs - min(25, epochs)
        if remaining_epochs > 0:
            history2 = model.fit(
                train_gen,
                steps_per_epoch=steps_per_epoch,
                epochs=remaining_epochs,
                validation_data=val_gen,
                validation_steps=validation_steps,
                callbacks=callbacks,
                verbose=1,
                initial_epoch=min(25, epochs)
            )
        
        # Load best model
        best_model = tf.keras.models.load_model(model_save_path)
        
        # Final evaluation
        print("\n📈 Final Model Evaluation:")
        val_loss, val_accuracy = best_model.evaluate(val_gen, verbose=0)
        
        print(f"✅ Validation Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
        print(f"✅ Validation Loss: {val_loss:.4f}")
        
        # Save as final bone fracture model
        final_model_path = "models/DenseNet121_bone_fracture_enhanced.h5"
        best_model.save(final_model_path)
        print(f"🎉 Model saved as: {final_model_path}")
        
        return best_model, val_accuracy

def prepare_better_bone_fracture_data():
    """Prepare enhanced bone fracture dataset with better balance"""
    import shutil
    import random
    
    # Create enhanced dataset directory
    enhanced_dir = "Dataset/ARM_Enhanced"
    train_dir = os.path.join(enhanced_dir, "train")
    val_dir = os.path.join(enhanced_dir, "val")
    
    # Clean up old directory
    if os.path.exists(enhanced_dir):
        shutil.rmtree(enhanced_dir)
    
    # Create directories
    for split in ['train', 'val']:
        for class_name in ['Normal', 'Fracture']:
            os.makedirs(os.path.join(enhanced_dir, split, class_name), exist_ok=True)
    
    # Source directories
    forearm_neg = "Dataset/ARM/MURA_Organized/Forearm/Negative"
    forearm_pos = "Dataset/ARM/MURA_Organized/Forearm/Positive"
    humerus_neg = "Dataset/ARM/MURA_Organized/Humerus/Negative"
    humerus_pos = "Dataset/ARM/MURA_Organized/Humerus/Positive"
    
    print("🔄 Preparing enhanced bone fracture dataset...")
    
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
    
    # Balance the dataset
    min_samples = min(len(all_negative), len(all_positive), 800)
    
    random.seed(42)
    selected_negative = random.sample(all_negative, min_samples)
    selected_positive = random.sample(all_positive, min_samples)
    
    # Split data (80/20)
    train_split = int(min_samples * 0.8)
    
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
    
    print(f"✅ Balanced dataset prepared: {enhanced_dir}")
    print(f"📊 Training: {train_split * 2} images ({train_split} per class)")
    print(f"📊 Validation: {(min_samples - train_split) * 2} images ({min_samples - train_split} per class)")
    
    return train_dir, val_dir

def main():
    """Main training function"""
    print("=" * 65)
    print("🦴 Enhanced DenseNet Bone Fracture Detection Training")
    print("=" * 65)
    
    # Prepare better dataset
    train_dir, val_dir = prepare_better_bone_fracture_data()
    
    if not train_dir:
        print("❌ Could not prepare dataset")
        return
    
    # Initialize enhanced trainer
    trainer = EnhancedBoneFractureTrainer(
        input_shape=(192, 192, 3),
        num_classes=2
    )
    
    # Start training
    try:
        model, accuracy = trainer.train(
            train_dir=train_dir,
            val_dir=val_dir,
            epochs=50,
            batch_size=32
        )
        
        print(f"\n🎯 Training completed!")
        print(f"✅ Final accuracy: {accuracy*100:.2f}%")
        
        if accuracy >= 0.85:
            print(f"🎉 SUCCESS! Achieved excellent accuracy: {accuracy*100:.2f}%")
        elif accuracy >= 0.80:
            print(f"✅ Good performance achieved: {accuracy*100:.2f}%")
        else:
            print(f"⚠️  Accuracy {accuracy*100:.2f}% needs improvement")
            
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()