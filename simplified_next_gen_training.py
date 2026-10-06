#!/usr/bin/env python3
"""
🚀 SIMPLIFIED NEXT-GEN Bone Fracture Detection
Ultra-high performance without compatibility issues
EfficientNetV2 + Vision Transformer concepts + Advanced optimization
"""
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetV2B3
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import *
import os
import numpy as np
from datetime import datetime
import json
from sklearn.utils.class_weight import compute_class_weight

class SimplifiedNextGenTrainer:
    def __init__(self, input_shape=(320, 320, 3), num_classes=2):
        """Initialize with high resolution for maximum detail"""
        self.input_shape = input_shape
        self.num_classes = num_classes
        
    def create_attention_block(self, x, filters):
        """Simplified attention mechanism"""
        # Channel attention
        avg_pool = GlobalAveragePooling2D()(x)
        max_pool = GlobalMaxPooling2D()(x)
        
        # Shared MLP
        shared_layer_one = Dense(filters // 8, activation='relu')
        shared_layer_two = Dense(filters, activation='sigmoid')
        
        avg_pool = shared_layer_one(avg_pool)
        avg_pool = shared_layer_two(avg_pool)
        
        max_pool = shared_layer_one(max_pool)
        max_pool = shared_layer_two(max_pool)
        
        # Channel attention
        channel_attention = Add()([avg_pool, max_pool])
        channel_attention = Reshape((1, 1, filters))(channel_attention)
        
        # Apply attention
        x = Multiply()([x, channel_attention])
        
        # Spatial attention
        avg_pool_spatial = tf.reduce_mean(x, axis=-1, keepdims=True)
        max_pool_spatial = tf.reduce_max(x, axis=-1, keepdims=True)
        spatial_concat = Concatenate(axis=-1)([avg_pool_spatial, max_pool_spatial])
        spatial_attention = Conv2D(1, 7, padding='same', activation='sigmoid')(spatial_concat)
        
        # Apply spatial attention
        x = Multiply()([x, spatial_attention])
        
        return x
    
    def create_advanced_hybrid_model(self):
        """Create advanced hybrid model with attention mechanisms"""
        # Input
        inputs = Input(shape=self.input_shape)
        
        # Advanced preprocessing (built into model)
        x = inputs
        x = tf.cast(x, tf.float32) / 255.0
        
        # Data augmentation layers
        x = RandomFlip("horizontal")(x)
        x = RandomRotation(0.1)(x)
        x = RandomZoom(0.1)(x)
        x = RandomBrightness(0.1)(x)
        x = RandomContrast(0.1)(x)
        
        # Primary backbone: EfficientNetV2B3
        backbone = EfficientNetV2B3(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        # Progressive unfreezing
        for layer in backbone.layers[:-120]:
            layer.trainable = False
        
        # Extract features at multiple scales
        x = backbone(x)
        
        # Apply attention mechanism
        x = self.create_attention_block(x, x.shape[-1])
        
        # Multi-scale feature extraction
        x1 = GlobalAveragePooling2D()(x)
        x2 = GlobalMaxPooling2D()(x)
        # Standard deviation pooling using math operations
        x_mean = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
        x_variance = tf.reduce_mean(tf.square(x - x_mean), axis=[1, 2])
        x3 = tf.sqrt(x_variance + 1e-8)  # Standard deviation
        
        # Combine features
        combined_features = Concatenate()([x1, x2, x3])
        
        # Advanced feature processing with residual connections
        x = Dense(2048, activation='swish', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(combined_features)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        
        # Residual dense blocks
        x1 = Dense(1024, activation='swish', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
        x1 = BatchNormalization()(x1)
        x1 = Dropout(0.4)(x1)
        
        x2 = Dense(512, activation='swish', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x1)
        x2 = BatchNormalization()(x2)
        x2 = Dropout(0.3)(x2)
        
        # Residual connection
        x1_proj = Dense(512, activation='linear')(x1)
        x2_residual = Add()([x2, x1_proj])
        x2_residual = Activation('swish')(x2_residual)
        
        x3 = Dense(256, activation='swish', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x2_residual)
        x3 = BatchNormalization()(x3)
        x3 = Dropout(0.2)(x3)
        
        x4 = Dense(128, activation='swish', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x3)
        x4 = BatchNormalization()(x4)
        x4 = Dropout(0.1)(x4)
        
        # Final classification with ensemble approach
        predictions = Dense(self.num_classes, activation='softmax', name='predictions')(x4)
        
        model = Model(inputs=inputs, outputs=predictions, name="NextGenSimplified")
        return model
    
    def create_advanced_data_generators(self, train_dir, val_dir, batch_size=12):
        """Create advanced data generators with custom augmentation"""
        
        # Advanced training augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=30,
            width_shift_range=0.25,
            height_shift_range=0.25,
            shear_range=0.2,
            zoom_range=0.25,
            horizontal_flip=True,
            brightness_range=[0.6, 1.4],
            channel_shift_range=0.2,
            fill_mode='reflect',
            # Advanced preprocessing
            preprocessing_function=self.advanced_preprocessing
        )
        
        # Clean validation data
        val_datagen = ImageDataGenerator(
            rescale=1./255,
            preprocessing_function=self.simple_preprocessing
        )
        
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=self.input_shape[:2],
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=True,
            seed=42,
            interpolation='bilinear'
        )
        
        validation_generator = val_datagen.flow_from_directory(
            val_dir,
            target_size=self.input_shape[:2],
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False,
            interpolation='bilinear'
        )
        
        return train_generator, validation_generator
    
    def advanced_preprocessing(self, image):
        """Advanced preprocessing for training images"""
        # Convert to numpy if tensor
        if hasattr(image, 'numpy'):
            image = image.numpy()
        
        # Random noise injection (data augmentation)
        if np.random.random() < 0.2:
            noise = np.random.normal(0, 0.02, image.shape)
            image = np.clip(image + noise, 0, 1)
        
        # Histogram equalization occasionally
        if np.random.random() < 0.3:
            image = tf.image.adjust_contrast(image, contrast_factor=1.2)
        
        return image
    
    def simple_preprocessing(self, image):
        """Simple preprocessing for validation"""
        return image
    
    def create_ultra_callbacks(self, model_save_path):
        """Advanced callbacks for maximum performance"""
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
                patience=30,  # Extended patience
                restore_best_weights=True,
                mode='max',
                verbose=1,
                min_delta=0.0005
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,  # More conservative LR reduction
                patience=12,
                min_lr=1e-8,
                verbose=1,
                cooldown=5
            ),
            # Cosine annealing restart
            tf.keras.callbacks.LearningRateScheduler(
                lambda epoch: 1e-4 * 0.5 * (1 + np.cos(np.pi * (epoch % 30) / 30)),
                verbose=0
            )
        ]
        
        return callbacks
    
    def calculate_advanced_class_weights(self, train_generator):
        """Calculate advanced class weights with smoothing"""
        labels = train_generator.labels
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(labels),
            y=labels
        )
        
        # Apply smoothing to prevent extreme weights
        class_weights = np.clip(class_weights, 0.5, 2.0)
        
        class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
        print(f"📊 Smoothed class weights: {class_weight_dict}")
        
        return class_weight_dict
    
    def train_next_gen_model(self, train_dir, val_dir, epochs=120, batch_size=12):
        """Next-generation training protocol"""
        print("🚀 SIMPLIFIED NEXT-GEN Training Starting...")
        print(f"Input resolution: {self.input_shape}")
        print(f"Max epochs: {epochs}")
        print(f"Batch size: {batch_size}")
        
        # Create model
        model = self.create_advanced_hybrid_model()
        print(f"✅ Advanced model created with {model.count_params():,} parameters")
        
        # Advanced optimizer
        optimizer = Adam(
            learning_rate=1e-4,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-8,
            amsgrad=True
        )
        
        # Compile model
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Create data generators
        train_gen, val_gen = self.create_advanced_data_generators(train_dir, val_dir, batch_size)
        
        # Calculate advanced class weights
        class_weights = self.calculate_advanced_class_weights(train_gen)
        
        print(f"📊 Training samples: {train_gen.samples}")
        print(f"📊 Validation samples: {val_gen.samples}")
        print(f"📊 Classes: {train_gen.class_indices}")
        
        # Model save path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_save_path = f"models/NextGenSimplified_bone_fracture_{timestamp}.h5"
        os.makedirs("models", exist_ok=True)
        
        # Create callbacks
        callbacks = self.create_ultra_callbacks(model_save_path)
        
        # Calculate steps
        steps_per_epoch = max(1, train_gen.samples // batch_size)
        validation_steps = max(1, val_gen.samples // batch_size)
        
        print(f"🔄 Steps per epoch: {steps_per_epoch}")
        print(f"🔄 Validation steps: {validation_steps}")
        
        # Multi-phase training
        print("\n🎯 Phase 1: Foundation Training (70 epochs)")
        history1 = model.fit(
            train_gen,
            steps_per_epoch=steps_per_epoch,
            epochs=min(70, epochs),
            validation_data=val_gen,
            validation_steps=validation_steps,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )
        
        # Phase 2: Fine-tuning with more layers
        print("\n🎯 Phase 2: Deep Fine-tuning")
        # Unfreeze more layers
        for layer in model.layers:
            if hasattr(layer, 'layers'):  # If it's a nested model (backbone)
                for sublayer in layer.layers[-200:]:
                    sublayer.trainable = True
            else:
                layer.trainable = True
        
        # Recompile with lower learning rate
        model.compile(
            optimizer=Adam(learning_rate=5e-6, amsgrad=True),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        remaining_epochs = epochs - min(70, epochs)
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
                initial_epoch=min(70, epochs)
            )
        
        # Load best model
        try:
            best_model = tf.keras.models.load_model(model_save_path)
        except:
            best_model = model
        
        # Final evaluation
        print("\n📈 NEXT-GEN Model Evaluation:")
        val_loss, val_accuracy = best_model.evaluate(val_gen, verbose=0)
        
        print(f"✅ Validation Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
        print(f"✅ Validation Loss: {val_loss:.4f}")
        
        # Save final model
        final_model_path = "models/NextGenSimplified_bone_fracture_final.h5"
        best_model.save(final_model_path)
        print(f"🎉 NEXT-GEN model saved: {final_model_path}")
        
        return best_model, val_accuracy

def main():
    """Main next-generation training function"""
    print("=" * 80)
    print("🦴 SIMPLIFIED NEXT-GENERATION Bone Fracture Detection")
    print("🧠 EfficientNetV2 + Advanced Attention + Ultra Optimization")
    print("=" * 80)
    
    # Use existing maximum dataset
    train_dir = "Dataset/ARM_Maximum/train"
    val_dir = "Dataset/ARM_Maximum/val"
    
    if not os.path.exists(train_dir):
        print("❌ Dataset not found. Please ensure ARM_Maximum dataset exists.")
        return
    
    # Initialize trainer
    trainer = SimplifiedNextGenTrainer(
        input_shape=(320, 320, 3),  # High resolution but manageable
        num_classes=2
    )
    
    try:
        model, accuracy = trainer.train_next_gen_model(
            train_dir=train_dir,
            val_dir=val_dir,
            epochs=120,
            batch_size=12
        )
        
        print(f"\n🎯 NEXT-GEN Training Completed!")
        print(f"✅ Final accuracy: {accuracy*100:.2f}%")
        
        if accuracy >= 0.95:
            print(f"🎉 BREAKTHROUGH! Medical Excellence: {accuracy*100:.2f}%")
        elif accuracy >= 0.92:
            print(f"🎉 OUTSTANDING! Research-grade: {accuracy*100:.2f}%")
        elif accuracy >= 0.88:
            print(f"🎉 EXCELLENT! High performance: {accuracy*100:.2f}%")
        elif accuracy >= 0.85:
            print(f"✅ VERY GOOD performance: {accuracy*100:.2f}%")
        elif accuracy >= 0.82:
            print(f"✅ SOLID improvement: {accuracy*100:.2f}%")
        else:
            print(f"📈 Building foundation: {accuracy*100:.2f}%")
            
        # Performance analysis
        print(f"\n📊 Performance Analysis:")
        if accuracy < 0.85:
            print("🔍 Future optimization opportunities:")
            print("   • Ensemble methods with multiple models")
            print("   • Advanced pseudo-labeling techniques") 
            print("   • Test-time augmentation")
            print("   • Model distillation from larger models")
            print("   • Advanced loss functions (focal loss, label smoothing)")
            
    except Exception as e:
        print(f"❌ Next-gen training failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()