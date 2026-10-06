#!/usr/bin/env python3
"""
🚀 FAST DenseNet Bone Fracture Training
Optimized for speed while maintaining good accuracy
Quick results with efficient training strategies
"""
import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import os
from datetime import datetime

class FastDenseNetTrainer:
    def __init__(self, input_shape=(224, 224, 3), num_classes=2):
        """Initialize fast DenseNet trainer with speed optimizations"""
        self.input_shape = input_shape
        self.num_classes = num_classes
        
    def create_fast_model(self):
        """Create fast DenseNet121 model with minimal layers"""
        # Base DenseNet121 model
        base_model = DenseNet121(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        # Keep base frozen for speed
        base_model.trainable = False
        
        # Minimal but effective classification head
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.3)(x)
        predictions = Dense(self.num_classes, activation='softmax')(x)
        
        # Create model
        model = Model(inputs=base_model.input, outputs=predictions)
        
        # Compile with higher learning rate for speed
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def create_fast_data_generators(self, train_dir, val_dir, batch_size=64):
        """Create fast data generators with minimal augmentation"""
        
        # Light augmentation for speed
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        # Validation generator
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        # Create generators with larger batch size
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=self.input_shape[:2],
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=True
        )
        
        validation_generator = val_datagen.flow_from_directory(
            val_dir,
            target_size=self.input_shape[:2],
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        return train_generator, validation_generator
    
    def create_fast_callbacks(self, model_save_path):
        """Create fast callbacks with aggressive early stopping"""
        callbacks = [
            ModelCheckpoint(
                model_save_path,
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            EarlyStopping(
                monitor='val_accuracy',
                patience=8,  # Aggressive early stopping
                restore_best_weights=True,
                mode='max',
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.3,
                patience=4,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        return callbacks
    
    def fast_train(self, dataset_path, epochs=30, batch_size=64):
        """Fast training protocol"""
        print("🚀 Starting FAST DenseNet Bone Fracture Training")
        print(f"⚡ Optimized for SPEED with good accuracy")
        print(f"Dataset: {dataset_path}")
        print(f"Input shape: {self.input_shape}")
        print(f"Batch size: {batch_size} (large for speed)")
        print(f"Max epochs: {epochs}")
        
        # Check dataset structure
        train_dir = os.path.join(dataset_path, "train")
        val_dir = os.path.join(dataset_path, "val")
        
        if not os.path.exists(train_dir) or not os.path.exists(val_dir):
            print(f"❌ Dataset directories not found")
            return None, 0
        
        print(f"✅ Training directory: {train_dir}")
        print(f"✅ Validation directory: {val_dir}")
        
        # Create fast model
        model = self.create_fast_model()
        print(f"✅ Fast model created with {model.count_params():,} parameters")
        
        # Create fast data generators
        train_gen, val_gen = self.create_fast_data_generators(train_dir, val_dir, batch_size)
        
        print(f"📊 Training samples: {train_gen.samples}")
        print(f"📊 Validation samples: {val_gen.samples}")
        print(f"📊 Classes: {train_gen.class_indices}")
        
        # Create model save path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_save_path = f"models/FastDenseNet_ARM_{timestamp}.h5"
        os.makedirs("models", exist_ok=True)
        
        # Create fast callbacks
        callbacks = self.create_fast_callbacks(model_save_path)
        
        # Calculate steps (fewer steps for speed)
        steps_per_epoch = max(1, train_gen.samples // batch_size)
        validation_steps = max(1, val_gen.samples // batch_size)
        
        print(f"🔄 Steps per epoch: {steps_per_epoch}")
        print(f"🔄 Validation steps: {validation_steps}")
        
        # Single phase fast training
        print(f"\n⚡ Fast Training Phase")
        start_time = datetime.now()
        
        history = model.fit(
            train_gen,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=val_gen,
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=1
        )
        
        end_time = datetime.now()
        training_duration = end_time - start_time
        
        # Load best model
        try:
            best_model = tf.keras.models.load_model(model_save_path)
        except:
            best_model = model
        
        # Quick evaluation
        print("\n📈 Fast Training Results:")
        val_loss, val_accuracy = best_model.evaluate(val_gen, verbose=0)
        
        print(f"✅ Validation Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
        print(f"✅ Training Time: {training_duration}")
        print(f"⚡ Speed: {training_duration.total_seconds()/60:.1f} minutes")
        
        # Save final model
        final_model_path = "models/FastDenseNet_ARM_final.h5"
        best_model.save(final_model_path)
        print(f"🎉 Fast model saved as: {final_model_path}")
        
        return best_model, val_accuracy

def main():
    """Main fast training function"""
    print("=" * 70)
    print("⚡ FAST DenseNet ARM Bone Fracture Training")
    print("🚀 Speed-Optimized Training for Quick Results")
    print("=" * 70)
    
    # Dataset path
    dataset_path = r"D:\Capstone\mynew\capstoneortho\Dataset\ARM_Combined"
    
    # Check if dataset exists
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found at: {dataset_path}")
        return
    
    print(f"✅ Dataset found: {dataset_path}")
    
    # Initialize fast trainer
    trainer = FastDenseNetTrainer(
        input_shape=(224, 224, 3),  # Standard resolution for speed
        num_classes=2
    )
    
    # Start fast training
    try:
        model, accuracy = trainer.fast_train(
            dataset_path=dataset_path,
            epochs=30,      # Fewer epochs for speed
            batch_size=64   # Large batch size for efficiency
        )
        
        if model:
            print(f"\n🎯 Fast Training Completed!")
            print(f"✅ Final Accuracy: {accuracy*100:.2f}%")
            
            if accuracy >= 0.85:
                print(f"🎉 EXCELLENT! High speed + high accuracy: {accuracy*100:.2f}%")
            elif accuracy >= 0.80:
                print(f"🎉 VERY GOOD! Great speed-accuracy balance: {accuracy*100:.2f}%")
            elif accuracy >= 0.75:
                print(f"✅ GOOD! Fast results: {accuracy*100:.2f}%")
            elif accuracy >= 0.70:
                print(f"✅ DECENT! Quick baseline: {accuracy*100:.2f}%")
            else:
                print(f"⚡ FAST baseline established: {accuracy*100:.2f}%")
            
            print(f"\n💡 Speed Optimizations Used:")
            print(f"   ⚡ Large batch size (64) for GPU efficiency")
            print(f"   ⚡ Minimal augmentation for faster data loading")
            print(f"   ⚡ Frozen backbone for reduced computation")
            print(f"   ⚡ Aggressive early stopping (8 patience)")
            print(f"   ⚡ Simplified architecture")
            print(f"   ⚡ Higher learning rate for faster convergence")
            
        else:
            print("❌ Fast training failed")
            
    except Exception as e:
        print(f"❌ Fast training error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()