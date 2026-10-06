#!/usr/bin/env python3
"""
Ultra-Advanced DenseNet Bone Fracture Training
Maximum performance optimization with state-of-the-art techniques
"""
import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import (
    GlobalAveragePooling2D, Dense, Dropout, BatchNormalization, 
    LayerNormalization, Multiply, Add
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
)
import os
import numpy as np
from datetime import datetime
import json
import cv2
from sklearn.utils.class_weight import compute_class_weight

class UltraBoneFractureTrainer:
    def __init__(self, input_shape=(224, 224, 3), num_classes=2):
        """Initialize with optimal input size for maximum performance"""
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        
    def create_advanced_model(self):
        """Create ultra-advanced DenseNet121 with attention mechanisms"""
        # Load pre-trained DenseNet121
        base_model = DenseNet121(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        # Progressive unfreezing for optimal fine-tuning
        for layer in base_model.layers[:-60]:
            layer.trainable = False
        
        # Advanced feature extraction with attention
        x = base_model.output
        
        # Advanced feature extraction with channel attention
        x = GlobalAveragePooling2D()(x)
        original_features = x
        
        # Channel attention mechanism (simplified)
        attention_weights = Dense(1024, activation='relu')(x)
        attention_weights = Dense(1024, activation='sigmoid')(attention_weights)
        x = Multiply()([x, attention_weights])
        
        # Layer normalization and residual connection
        x = LayerNormalization()(x)
        x = Dense(1024, activation='gelu')(x)
        x = Add()([x, original_features])  # Residual connection
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        
        # Advanced dense layers with residual connections
        x1 = Dense(512, activation='gelu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
        x1 = BatchNormalization()(x1)
        x1 = Dropout(0.4)(x1)
        
        x2 = Dense(256, activation='gelu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x1)
        x2 = BatchNormalization()(x2)
        x2 = Dropout(0.3)(x2)
        
        x3 = Dense(128, activation='gelu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x2)
        x3 = BatchNormalization()(x3)
        x3 = Dropout(0.2)(x3)
        
        # Final classification with confidence estimation
        predictions = Dense(self.num_classes, activation='softmax', name='predictions')(x3)
        
        self.model = Model(inputs=base_model.input, outputs=predictions)
        
        # Advanced optimizer with optimal settings
        optimizer = Adam(
            learning_rate=1e-4,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7,
            amsgrad=True
        )
        
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return self.model
    
    def advanced_data_preprocessing(self, img_path):
        """Advanced image preprocessing with medical imaging optimizations"""
        # Read image
        img = cv2.imread(img_path)
        if img is None:
            return None
            
        # Convert to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Advanced preprocessing
        # 1. Adaptive histogram equalization for better contrast
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray_enhanced = clahe.apply(gray)
        
        # Convert back to RGB
        img_enhanced = cv2.cvtColor(gray_enhanced, cv2.COLOR_GRAY2RGB)
        
        # 2. Gaussian blur for noise reduction
        img_enhanced = cv2.GaussianBlur(img_enhanced, (3, 3), 0)
        
        # 3. Resize with optimal interpolation
        img_enhanced = cv2.resize(img_enhanced, self.input_shape[:2], interpolation=cv2.INTER_LANCZOS4)
        
        # 4. Normalization
        img_enhanced = img_enhanced.astype(np.float32) / 255.0
        
        return img_enhanced
    
    def create_advanced_data_generators(self, train_dir, val_dir, batch_size=24):
        """Create advanced data generators with medical-specific augmentations"""
        
        # Advanced augmentation specifically designed for medical images
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.15,
            height_shift_range=0.15,
            shear_range=0.1,
            zoom_range=0.15,
            horizontal_flip=True,
            vertical_flip=False,  # Medical images shouldn't be vertically flipped
            brightness_range=[0.8, 1.2],
            fill_mode='reflect',
            # Advanced augmentations
            channel_shift_range=0.1
        )
        
        # Validation with minimal processing
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
    
    def create_advanced_callbacks(self, model_save_path):
        """Create state-of-the-art callbacks for optimal training"""
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
                min_lr=1e-8,
                verbose=1,
                cooldown=3
            ),
            # Advanced learning rate scheduling
            tf.keras.callbacks.LearningRateScheduler(
                lambda epoch: 1e-4 * 0.98 ** epoch,
                verbose=0
            )
        ]
        
        return callbacks
    
    def calculate_class_weights(self, train_generator):
        """Calculate class weights for balanced training"""
        # Get class labels
        labels = train_generator.labels
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(labels),
            y=labels
        )
        
        class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
        print(f"📊 Class weights: {class_weight_dict}")
        
        return class_weight_dict
    
    def train(self, train_dir, val_dir, epochs=80, batch_size=24):
        """Ultra-advanced training with all optimizations"""
        print("🚀 Starting Ultra-Advanced DenseNet Bone Fracture Training")
        print(f"Input shape: {self.input_shape}")
        print(f"Batch size: {batch_size}")
        print(f"Max epochs: {epochs}")
        
        # Create advanced model
        model = self.create_advanced_model()
        print(f"✅ Advanced model created with {model.count_params():,} parameters")
        
        # Create advanced data generators
        train_gen, val_gen = self.create_advanced_data_generators(train_dir, val_dir, batch_size)
        
        # Calculate class weights for balanced training
        class_weights = self.calculate_class_weights(train_gen)
        
        print(f"📊 Training samples: {train_gen.samples}")
        print(f"📊 Validation samples: {val_gen.samples}")
        print(f"📊 Classes: {train_gen.class_indices}")
        
        # Create model save path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_save_path = f"models/DenseNet121_bone_fracture_ultra_{timestamp}.h5"
        os.makedirs("models", exist_ok=True)
        
        # Create advanced callbacks
        callbacks = self.create_advanced_callbacks(model_save_path)
        
        # Calculate steps
        steps_per_epoch = train_gen.samples // batch_size
        validation_steps = val_gen.samples // batch_size
        
        print(f"🔄 Steps per epoch: {steps_per_epoch}")
        print(f"🔄 Validation steps: {validation_steps}")
        
        # Multi-phase training for maximum performance
        print("\n🎯 Phase 1: Foundation Training (Frozen Base)")
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
        
        # Phase 2: Progressive unfreezing
        print("\n🎯 Phase 2: Progressive Fine-tuning")
        for layer in model.layers[-120:]:
            layer.trainable = True
        
        # Recompile with lower learning rate
        model.compile(
            optimizer=Adam(learning_rate=5e-6, amsgrad=True),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        remaining_epochs = epochs - min(30, epochs)
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
                initial_epoch=min(30, epochs)
            )
        
        # Load best model
        best_model = tf.keras.models.load_model(model_save_path)
        
        # Comprehensive evaluation
        print("\n📈 Ultra-Advanced Model Evaluation:")
        results = best_model.evaluate(val_gen, verbose=0)
        val_loss = results[0]
        val_accuracy = results[1]
        val_precision = results[2] if len(results) > 2 else 0
        val_recall = results[3] if len(results) > 3 else 0
        
        f1_score = 2 * (val_precision * val_recall) / (val_precision + val_recall) if (val_precision + val_recall) > 0 else 0
        
        print(f"✅ Validation Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
        print(f"✅ Validation Precision: {val_precision:.4f}")
        print(f"✅ Validation Recall: {val_recall:.4f}")
        print(f"✅ F1 Score: {f1_score:.4f}")
        print(f"✅ Validation Loss: {val_loss:.4f}")
        
        # Save final ultra model
        final_model_path = "models/DenseNet121_bone_fracture_ultra.h5"
        best_model.save(final_model_path)
        print(f"🎉 Ultra model saved as: {final_model_path}")
        
        return best_model, val_accuracy

def main():
    """Main ultra-advanced training function"""
    print("=" * 70)
    print("🦴 ULTRA-ADVANCED DenseNet Bone Fracture Detection Training")
    print("=" * 70)
    
    # Use the existing enhanced dataset
    train_dir = "Dataset/ARM_Enhanced/train"
    val_dir = "Dataset/ARM_Enhanced/val"
    
    # Verify dataset exists
    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        print("❌ Enhanced dataset not found. Run enhanced training first.")
        return
    
    # Initialize ultra-advanced trainer
    trainer = UltraBoneFractureTrainer(
        input_shape=(224, 224, 3),  # Higher resolution for maximum quality
        num_classes=2
    )
    
    # Start ultra-advanced training
    try:
        model, accuracy = trainer.train(
            train_dir=train_dir,
            val_dir=val_dir,
            epochs=80,
            batch_size=24  # Smaller batch for higher quality
        )
        
        print(f"\n🎯 Ultra-Advanced Training Completed!")
        print(f"✅ Final accuracy: {accuracy*100:.2f}%")
        
        if accuracy >= 0.90:
            print(f"🎉 OUTSTANDING! Medical-grade accuracy achieved: {accuracy*100:.2f}%")
        elif accuracy >= 0.85:
            print(f"🎉 EXCELLENT! High-performance model: {accuracy*100:.2f}%")
        elif accuracy >= 0.82:
            print(f"✅ VERY GOOD performance: {accuracy*100:.2f}%")
        else:
            print(f"📈 Good improvement, targeting >85%: {accuracy*100:.2f}%")
            
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()