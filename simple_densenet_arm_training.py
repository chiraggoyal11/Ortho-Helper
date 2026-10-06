#!/usr/bin/env python3
"""
🦴 Simple DenseNet Bone Fracture Training
Using ARM_Combined dataset without any modifications
Clean and straightforward approach
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

class SimpleDenseNetTrainer:
    def __init__(self, input_shape=(224, 224, 3), num_classes=2):
        """Initialize simple DenseNet trainer"""
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        
    def create_simple_model(self):
        """Create simple DenseNet121 model"""
        # Base DenseNet121 model
        base_model = DenseNet121(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        # Freeze base model initially
        base_model.trainable = False
        
        # Add classification head
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.3)(x)
        predictions = Dense(self.num_classes, activation='softmax')(x)
        
        # Create model
        model = Model(inputs=base_model.input, outputs=predictions)
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def create_data_generators(self, train_dir, val_dir, batch_size=32):
        """Create simple data generators"""
        # Simple training augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            zoom_range=0.1,
            fill_mode='nearest'
        )
        
        # Validation generator (no augmentation)
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        # Create generators
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
    
    def create_callbacks(self, model_save_path):
        """Create training callbacks"""
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
                patience=15,
                restore_best_weights=True,
                mode='max',
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=8,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        return callbacks
    
    def train(self, dataset_path, epochs=50, batch_size=32):
        """Train the model"""
        print("🦴 Starting Simple DenseNet Bone Fracture Training")
        print(f"Dataset: {dataset_path}")
        print(f"Input shape: {self.input_shape}")
        print(f"Batch size: {batch_size}")
        print(f"Max epochs: {epochs}")
        
        # Check dataset structure
        train_dir = os.path.join(dataset_path, "train")
        val_dir = os.path.join(dataset_path, "val")
        
        if not os.path.exists(train_dir):
            print(f"❌ Training directory not found: {train_dir}")
            return None, 0
        
        if not os.path.exists(val_dir):
            print(f"❌ Validation directory not found: {val_dir}")
            return None, 0
        
        print(f"✅ Training directory: {train_dir}")
        print(f"✅ Validation directory: {val_dir}")
        
        # Create model
        model = self.create_simple_model()
        print(f"✅ Model created with {model.count_params():,} parameters")
        
        # Create data generators
        train_gen, val_gen = self.create_data_generators(train_dir, val_dir, batch_size)
        
        print(f"📊 Training samples: {train_gen.samples}")
        print(f"📊 Validation samples: {val_gen.samples}")
        print(f"📊 Classes found: {train_gen.class_indices}")
        
        # Create model save path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_save_path = f"models/SimpleDenseNet_ARM_{timestamp}.h5"
        os.makedirs("models", exist_ok=True)
        
        # Create callbacks
        callbacks = self.create_callbacks(model_save_path)
        
        # Calculate steps
        steps_per_epoch = train_gen.samples // batch_size
        validation_steps = val_gen.samples // batch_size
        
        print(f"🔄 Steps per epoch: {steps_per_epoch}")
        print(f"🔄 Validation steps: {validation_steps}")
        
        # Phase 1: Train frozen base model
        print(f"\n🎯 Phase 1: Training with frozen base model")
        history1 = model.fit(
            train_gen,
            steps_per_epoch=steps_per_epoch,
            epochs=min(25, epochs),
            validation_data=val_gen,
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=1
        )
        
        # Phase 2: Unfreeze and fine-tune
        if epochs > 25:
            print(f"\n🎯 Phase 2: Fine-tuning with unfrozen model")
            
            # Unfreeze the base model
            model.layers[0].trainable = True
            
            # Recompile with lower learning rate
            model.compile(
                optimizer=Adam(learning_rate=0.0001),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            remaining_epochs = epochs - 25
            history2 = model.fit(
                train_gen,
                steps_per_epoch=steps_per_epoch,
                epochs=remaining_epochs,
                validation_data=val_gen,
                validation_steps=validation_steps,
                callbacks=callbacks,
                verbose=1,
                initial_epoch=25
            )
        
        # Load best model
        best_model = tf.keras.models.load_model(model_save_path)
        
        # Final evaluation
        print("\n📈 Final Model Evaluation:")
        val_loss, val_accuracy = best_model.evaluate(val_gen, verbose=0)
        
        print(f"✅ Validation Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
        print(f"✅ Validation Loss: {val_loss:.4f}")
        
        # Save final model
        final_model_path = "models/SimpleDenseNet_ARM_final.h5"
        best_model.save(final_model_path)
        print(f"🎉 Model saved as: {final_model_path}")
        
        return best_model, val_accuracy

def main():
    """Main training function"""
    print("=" * 70)
    print("🦴 Simple DenseNet ARM Bone Fracture Training")
    print("=" * 70)
    
    # Dataset path
    dataset_path = r"D:\Capstone\mynew\capstoneortho\Dataset\ARM_Combined"
    
    # Check if dataset exists
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found at: {dataset_path}")
        print("Please make sure the ARM_Combined dataset exists.")
        return
    
    print(f"✅ Dataset found: {dataset_path}")
    
    # List dataset structure
    print(f"\n📁 Dataset structure:")
    for root, dirs, files in os.walk(dataset_path):
        level = root.replace(dataset_path, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files[:3]:  # Show first 3 files
            print(f"{subindent}{file}")
        if len(files) > 3:
            print(f"{subindent}... and {len(files)-3} more files")
    
    # Initialize trainer
    trainer = SimpleDenseNetTrainer(
        input_shape=(224, 224, 3),
        num_classes=2
    )
    
    # Start training
    try:
        model, accuracy = trainer.train(
            dataset_path=dataset_path,
            epochs=50,
            batch_size=32
        )
        
        if model:
            print(f"\n🎯 Training Completed Successfully!")
            print(f"✅ Final Accuracy: {accuracy*100:.2f}%")
            
            if accuracy >= 0.90:
                print(f"🎉 EXCELLENT! Outstanding performance: {accuracy*100:.2f}%")
            elif accuracy >= 0.85:
                print(f"🎉 VERY GOOD! High performance: {accuracy*100:.2f}%")
            elif accuracy >= 0.80:
                print(f"✅ GOOD! Solid performance: {accuracy*100:.2f}%")
            elif accuracy >= 0.75:
                print(f"✅ DECENT! Room for improvement: {accuracy*100:.2f}%")
            else:
                print(f"📈 BASELINE established: {accuracy*100:.2f}%")
        else:
            print("❌ Training failed")
            
    except Exception as e:
        print(f"❌ Training error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()