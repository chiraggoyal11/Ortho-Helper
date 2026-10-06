#!/usr/bin/env python3
"""
Fast DenseNet Bone Fracture Training
Optimized for reduced training time while maintaining performance
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

class FastDenseNetBoneFractureTrainer:
    def __init__(self, input_shape=(160, 160, 3), num_classes=2):
        """Initialize with smaller input size for faster training"""
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        
    def create_model(self):
        """Create optimized DenseNet121 model for fast training"""
        # Load pre-trained DenseNet121 with smaller input
        base_model = DenseNet121(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        # Freeze most layers, only train last few for speed
        for layer in base_model.layers[:-20]:
            layer.trainable = False
        
        # Simplified classification head for speed
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = BatchNormalization()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.3)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.2)(x)
        
        # Final classification layer
        predictions = Dense(self.num_classes, activation='softmax', name='predictions')(x)
        
        self.model = Model(inputs=base_model.input, outputs=predictions)
        
        # Compile with optimized settings for speed
        self.model.compile(
            optimizer=Adam(learning_rate=1e-3, beta_1=0.9, beta_2=0.999),  # Higher LR for faster convergence
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model
    
    def create_data_generators(self, train_dir, val_dir, batch_size=64):
        """Create optimized data generators for faster training"""
        
        # Simplified augmentation for speed
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
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
        """Create callbacks optimized for fast training"""
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
                patience=8,  # Reduced patience for faster stopping
                restore_best_weights=True,
                mode='max',
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=4,  # Faster LR reduction
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        return callbacks
    
    def train(self, train_dir, val_dir, epochs=40, batch_size=64):
        """Fast training with early stopping"""
        print("🚀 Starting Fast DenseNet Bone Fracture Training")
        print(f"Input shape: {self.input_shape}")
        print(f"Batch size: {batch_size}")
        print(f"Max epochs: {epochs}")
        
        # Create model
        model = self.create_model()
        print(f"✅ Model created with {model.count_params():,} parameters")
        
        # Create data generators
        train_gen, val_gen = self.create_data_generators(train_dir, val_dir, batch_size)
        
        print(f"📊 Training samples: {train_gen.samples}")
        print(f"📊 Validation samples: {val_gen.samples}")
        print(f"📊 Classes: {train_gen.class_indices}")
        
        # Create model save path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_save_path = f"models/DenseNet121_bone_fracture_fast_{timestamp}.h5"
        os.makedirs("models", exist_ok=True)
        
        # Create callbacks
        callbacks = self.create_callbacks(model_save_path)
        
        # Calculate steps (smaller for faster epochs)
        steps_per_epoch = min(train_gen.samples // batch_size, 50)  # Limit steps for speed
        validation_steps = min(val_gen.samples // batch_size, 20)
        
        print(f"🔄 Steps per epoch: {steps_per_epoch}")
        print(f"🔄 Validation steps: {validation_steps}")
        
        # Fast training phase
        print("\n🎯 Fast Training Phase")
        history = model.fit(
            train_gen,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=val_gen,
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=1
        )
        
        # Load best model
        best_model = tf.keras.models.load_model(model_save_path)
        
        # Final evaluation
        print("\n📈 Final Model Evaluation:")
        val_loss, val_accuracy = best_model.evaluate(val_gen, verbose=0)
        
        print(f"✅ Validation Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
        print(f"✅ Validation Loss: {val_loss:.4f}")
        
        # Save as final bone fracture model
        final_model_path = "models/DenseNet121_bone_fracture_fast.h5"
        best_model.save(final_model_path)
        print(f"🎉 Model saved as: {final_model_path}")
        
        # Save training details
        training_details = {
            "model_name": "DenseNet121_bone_fracture_fast",
            "architecture": "DenseNet121",
            "input_shape": self.input_shape,
            "training_time": timestamp,
            "final_accuracy": float(val_accuracy),
            "final_loss": float(val_loss),
            "epochs_trained": len(history.history['accuracy']),
            "total_parameters": model.count_params(),
            "training_samples": train_gen.samples,
            "validation_samples": val_gen.samples
        }
        
        with open("models/bone_fracture_training_details.json", "w") as f:
            json.dump(training_details, f, indent=2)
        
        return best_model, val_accuracy

def prepare_bone_fracture_data():
    """Prepare combined bone fracture dataset from ARM data"""
    import shutil
    
    # Create combined dataset directory
    combined_dir = "Dataset/ARM_Combined"
    train_dir = os.path.join(combined_dir, "train")
    val_dir = os.path.join(combined_dir, "val")
    
    # Create directories
    for split in ['train', 'val']:
        for class_name in ['Normal', 'Fracture']:
            os.makedirs(os.path.join(combined_dir, split, class_name), exist_ok=True)
    
    # Source directories
    forearm_neg = "Dataset/ARM/MURA_Organized/Forearm/Negative"
    forearm_pos = "Dataset/ARM/MURA_Organized/Forearm/Positive"
    humerus_neg = "Dataset/ARM/MURA_Organized/Humerus/Negative"
    humerus_pos = "Dataset/ARM/MURA_Organized/Humerus/Positive"
    
    print("🔄 Preparing combined bone fracture dataset...")
    
    # Copy files if they exist
    if os.path.exists(forearm_neg) and os.path.exists(forearm_pos):
        # Get file lists
        neg_files = [f for f in os.listdir(forearm_neg) if f.lower().endswith(('.png', '.jpg', '.jpeg'))][:1000]  # Limit for speed
        pos_files = [f for f in os.listdir(forearm_pos) if f.lower().endswith(('.png', '.jpg', '.jpeg'))][:1000]
        
        # Split data
        train_split = 0.8
        
        # Normal (negative) files
        train_neg = neg_files[:int(len(neg_files) * train_split)]
        val_neg = neg_files[int(len(neg_files) * train_split):]
        
        # Fracture (positive) files  
        train_pos = pos_files[:int(len(pos_files) * train_split)]
        val_pos = pos_files[int(len(pos_files) * train_split):]
        
        # Copy training files
        for file in train_neg[:500]:  # Limit for faster training
            if os.path.exists(os.path.join(forearm_neg, file)):
                shutil.copy2(os.path.join(forearm_neg, file), os.path.join(train_dir, "Normal", file))
        
        for file in train_pos[:500]:
            if os.path.exists(os.path.join(forearm_pos, file)):
                shutil.copy2(os.path.join(forearm_pos, file), os.path.join(train_dir, "Fracture", file))
        
        # Copy validation files
        for file in val_neg[:100]:
            if os.path.exists(os.path.join(forearm_neg, file)):
                shutil.copy2(os.path.join(forearm_neg, file), os.path.join(val_dir, "Normal", file))
        
        for file in val_pos[:100]:
            if os.path.exists(os.path.join(forearm_pos, file)):
                shutil.copy2(os.path.join(forearm_pos, file), os.path.join(val_dir, "Fracture", file))
        
        print(f"✅ Dataset prepared: {combined_dir}")
        return train_dir, val_dir
    else:
        print("❌ ARM dataset not found, using default structure")
        return None, None

def main():
    """Main training function"""
    print("=" * 60)
    print("🦴 Fast DenseNet Bone Fracture Detection Training")
    print("=" * 60)
    
    # Prepare dataset
    train_dir, val_dir = prepare_bone_fracture_data()
    
    if not train_dir:
        print("❌ Could not prepare dataset")
        return
    
    # Verify dataset
    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        print(f"❌ Dataset directories not found")
        return
    
    # Check if data exists
    train_files = sum([len(os.listdir(os.path.join(train_dir, cls))) for cls in os.listdir(train_dir)])
    val_files = sum([len(os.listdir(os.path.join(val_dir, cls))) for cls in os.listdir(val_dir)])
    
    print(f"📊 Training files: {train_files}")
    print(f"📊 Validation files: {val_files}")
    
    if train_files == 0 or val_files == 0:
        print("❌ No training data found")
        return
    
    # Initialize trainer with smaller input for speed
    trainer = FastDenseNetBoneFractureTrainer(
        input_shape=(160, 160, 3),  # Smaller than 224x224 for speed
        num_classes=2
    )
    
    # Start training
    try:
        model, accuracy = trainer.train(
            train_dir=train_dir,
            val_dir=val_dir,
            epochs=30,  # Reduced epochs
            batch_size=64  # Larger batch size for efficiency
        )
        
        print(f"\n🎯 Training completed!")
        print(f"✅ Final accuracy: {accuracy*100:.2f}%")
        print("✅ Model ready for integration!")
        
        if accuracy >= 0.80:
            print(f"🎉 SUCCESS! Achieved good accuracy: {accuracy*100:.2f}%")
        else:
            print(f"⚠️  Accuracy {accuracy*100:.2f}% could be improved with more training")
            
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()