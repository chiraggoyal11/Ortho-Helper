#!/usr/bin/env python3
"""
🦴 DenseNet Bone Fracture Training
Optimized DenseNet121 for bone fracture detection
ARM dataset with balanced approach for good accuracy
"""
import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import os
from datetime import datetime
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

class BoneFractureDenseNetTrainer:
    def __init__(self, input_shape=(224, 224, 3), num_classes=2):
        """Initialize bone fracture DenseNet trainer"""
        self.input_shape = input_shape
        self.num_classes = num_classes
        
    def create_bone_fracture_model(self):
        """Create optimized DenseNet121 for bone fracture detection"""
        # Base DenseNet121 model
        base_model = DenseNet121(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        # Start with frozen base model
        base_model.trainable = False
        
        # Optimized classification head for bone fractures
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        
        # First dense layer
        x = Dense(512, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        
        # Second dense layer
        x = Dense(256, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        # Final classification layer
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
    
    def create_bone_data_generators(self, train_dir, val_dir, batch_size=32):
        """Create data generators optimized for bone fracture images"""
        
        # Training augmentation for bone X-rays
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=25,
            width_shift_range=0.15,
            height_shift_range=0.15,
            shear_range=0.1,
            zoom_range=0.15,
            horizontal_flip=True,
            brightness_range=[0.8, 1.2],
            fill_mode='nearest'
        )
        
        # Validation generator
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
    
    def create_bone_callbacks(self, model_save_path):
        """Create callbacks optimized for bone fracture training"""
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
                verbose=1,
                min_delta=0.001
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
    
    def calculate_class_weights(self, train_generator):
        """Calculate balanced class weights"""
        labels = train_generator.labels
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(labels),
            y=labels
        )
        class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
        return class_weight_dict
    
    def train_bone_fracture(self, dataset_path, epochs=50, batch_size=32):
        """Train DenseNet for bone fracture detection"""
        print("🦴 Starting DenseNet Bone Fracture Training")
        print(f"Dataset: {dataset_path}")
        print(f"Input shape: {self.input_shape}")
        print(f"Batch size: {batch_size}")
        print(f"Max epochs: {epochs}")
        
        # Check dataset structure
        train_dir = os.path.join(dataset_path, "train")
        val_dir = os.path.join(dataset_path, "val")
        
        if not os.path.exists(train_dir) or not os.path.exists(val_dir):
            print(f"❌ Dataset directories not found")
            return None, 0
        
        print(f"✅ Training directory: {train_dir}")
        print(f"✅ Validation directory: {val_dir}")
        
        # Create model
        model = self.create_bone_fracture_model()
        print(f"✅ DenseNet model created with {model.count_params():,} parameters")
        
        # Create data generators
        train_gen, val_gen = self.create_bone_data_generators(train_dir, val_dir, batch_size)
        
        # Calculate class weights
        class_weights = self.calculate_class_weights(train_gen)
        print(f"📊 Class weights: {class_weights}")
        
        print(f"📊 Training samples: {train_gen.samples}")
        print(f"📊 Validation samples: {val_gen.samples}")
        print(f"📊 Classes: {train_gen.class_indices}")
        
        # Create model save path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_save_path = f"models/DenseNet_BoneFracture_{timestamp}.h5"
        os.makedirs("models", exist_ok=True)
        
        # Create callbacks
        callbacks = self.create_bone_callbacks(model_save_path)
        
        # Calculate steps
        steps_per_epoch = max(1, train_gen.samples // batch_size)
        validation_steps = max(1, val_gen.samples // batch_size)
        
        print(f"🔄 Steps per epoch: {steps_per_epoch}")
        print(f"🔄 Validation steps: {validation_steps}")
        
        # Phase 1: Train with frozen backbone
        print(f"\n🎯 Phase 1: Training with frozen backbone")
        start_time = datetime.now()
        
        history1 = model.fit(
            train_gen,
            steps_per_epoch=steps_per_epoch,
            epochs=min(30, epochs),
            validation_data=val_gen,
            validation_steps=validation_steps,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )
        
        # Phase 2: Fine-tune with unfrozen backbone
        if epochs > 30:
            print(f"\n🎯 Phase 2: Fine-tuning with unfrozen backbone")
            
            # Unfreeze the base model
            model.layers[0].trainable = True
            
            # Recompile with lower learning rate
            model.compile(
                optimizer=Adam(learning_rate=0.0001),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            remaining_epochs = epochs - 30
            history2 = model.fit(
                train_gen,
                steps_per_epoch=steps_per_epoch,
                epochs=remaining_epochs,
                validation_data=val_gen,
                validation_steps=validation_steps,
                callbacks=callbacks,
                class_weight=class_weights,
                verbose=1,
                initial_epoch=30
            )
        
        end_time = datetime.now()
        training_duration = end_time - start_time
        
        # Load best model
        try:
            best_model = tf.keras.models.load_model(model_save_path)
        except:
            best_model = model
        
        # Final evaluation
        print("\n📈 Bone Fracture Detection Results:")
        val_loss, val_accuracy = best_model.evaluate(val_gen, verbose=0)
        
        print(f"✅ Validation Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
        print(f"✅ Validation Loss: {val_loss:.4f}")
        print(f"⏱️ Training Duration: {training_duration}")
        
        # Save final model
        final_model_path = "models/DenseNet_BoneFracture_final.h5"
        best_model.save(final_model_path)
        print(f"🎉 Model saved as: {final_model_path}")
        
        # Performance analysis
        print(f"\n📊 Bone Fracture Detection Analysis:")
        if val_accuracy >= 0.90:
            print(f"🎉 EXCELLENT! Medical-grade accuracy: {val_accuracy*100:.2f}%")
        elif val_accuracy >= 0.85:
            print(f"🎉 VERY GOOD! High clinical accuracy: {val_accuracy*100:.2f}%")
        elif val_accuracy >= 0.80:
            print(f"✅ GOOD! Solid diagnostic performance: {val_accuracy*100:.2f}%")
        elif val_accuracy >= 0.75:
            print(f"✅ DECENT! Useful for screening: {val_accuracy*100:.2f}%")
        else:
            print(f"📈 BASELINE! Room for improvement: {val_accuracy*100:.2f}%")
        
        return best_model, val_accuracy

def main():
    """Main bone fracture training function"""
    print("=" * 70)
    print("🦴 DenseNet Bone Fracture Detection Training")
    print("🩻 Specialized for ARM X-ray Analysis")
    print("=" * 70)
    
    # Dataset path
    dataset_path = r"D:\Capstone\mynew\capstoneortho\Dataset\ARM_Combined"
    
    # Check if dataset exists
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found at: {dataset_path}")
        return
    
    print(f"✅ ARM Dataset found: {dataset_path}")
    
    # Initialize trainer
    trainer = BoneFractureDenseNetTrainer(
        input_shape=(224, 224, 3),
        num_classes=2
    )
    
    # Start training
    try:
        model, accuracy = trainer.train_bone_fracture(
            dataset_path=dataset_path,
            epochs=50,
            batch_size=32
        )
        
        if model:
            print(f"\n🎯 Bone Fracture Training Completed!")
            print(f"✅ Final Accuracy: {accuracy*100:.2f}%")
            
            print(f"\n💡 Key Features of This Model:")
            print(f"   🦴 Specialized for bone fracture detection")
            print(f"   🩻 Optimized for ARM X-ray images")
            print(f"   ⚖️ Balanced class weighting")
            print(f"   🔄 Two-phase training (frozen → fine-tuned)")
            print(f"   📊 Medical-grade evaluation metrics")
            
        else:
            print("❌ Bone fracture training failed")
            
    except Exception as e:
        print(f"❌ Training error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()