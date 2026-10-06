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

# Enable mixed precision for better performance (disabled for CPU)
# tf.keras.mixed_precision.set_global_policy('mixed_float16')

class DenseNet121CardiomegalyTrainer:
    def __init__(self, input_shape=(224, 224, 3), num_classes=2):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        
    def create_model(self):
        """Create DenseNet121 model with custom top layers for cardiomegaly detection"""
        # Load pre-trained DenseNet121
        base_model = DenseNet121(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        # Freeze initial layers, unfreeze later layers for fine-tuning
        for layer in base_model.layers[:-50]:
            layer.trainable = False
        
        # Add custom classification head
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = BatchNormalization()(x)
        x = Dense(1024, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
        x = Dropout(0.5)(x)
        x = BatchNormalization()(x)
        x = Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
        x = Dropout(0.3)(x)
        x = BatchNormalization()(x)
        x = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
        x = Dropout(0.2)(x)
        
        # Final classification layer
        predictions = Dense(self.num_classes, activation='softmax', name='predictions')(x)
        
        self.model = Model(inputs=base_model.input, outputs=predictions)
        
        # Compile with optimized settings
        self.model.compile(
            optimizer=Adam(learning_rate=1e-4, beta_1=0.9, beta_2=0.999),
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        
        return self.model
    
    def create_data_generators(self, train_dir, val_dir, batch_size=32):
        """Create optimized data generators with medical imaging specific augmentations"""
        
        # Training data generator with extensive augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.15,
            horizontal_flip=True,
            brightness_range=[0.8, 1.2],
            fill_mode='reflect',
            validation_split=0.0
        )
        
        # Validation data generator (only rescaling)
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
        """Create training callbacks for optimal performance"""
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
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.3,
                patience=7,
                min_lr=1e-7,
                verbose=1
            ),
            tf.keras.callbacks.LearningRateScheduler(
                lambda epoch: 1e-4 * 0.95 ** epoch
            )
        ]
        
        return callbacks
    
    def train(self, train_dir, val_dir, epochs=100, batch_size=32):
        """Train the DenseNet121 model"""
        print("🚀 Starting DenseNet121 Cardiomegaly Training")
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
        model_save_path = f"models/DenseNet121_cardiomegaly_{timestamp}.h5"
        os.makedirs("models", exist_ok=True)
        
        # Create callbacks
        callbacks = self.create_callbacks(model_save_path)
        
        # Calculate steps
        steps_per_epoch = train_gen.samples // batch_size
        validation_steps = val_gen.samples // batch_size
        
        print(f"🔄 Steps per epoch: {steps_per_epoch}")
        print(f"🔄 Validation steps: {validation_steps}")
        
        # Phase 1: Initial training with frozen base
        print("\n🎯 Phase 1: Training with frozen DenseNet121 base")
        history1 = model.fit(
            train_gen,
            steps_per_epoch=steps_per_epoch,
            epochs=min(20, epochs),
            validation_data=val_gen,
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=1
        )
        
        # Phase 2: Fine-tuning with unfrozen layers
        print("\n🎯 Phase 2: Fine-tuning all layers")
        for layer in model.layers:
            layer.trainable = True
        
        # Recompile with lower learning rate
        model.compile(
            optimizer=Adam(learning_rate=1e-5, beta_1=0.9, beta_2=0.999),
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        
        # Continue training
        remaining_epochs = epochs - min(20, epochs)
        if remaining_epochs > 0:
            history2 = model.fit(
                train_gen,
                steps_per_epoch=steps_per_epoch,
                epochs=remaining_epochs,
                validation_data=val_gen,
                validation_steps=validation_steps,
                callbacks=callbacks,
                verbose=1,
                initial_epoch=min(20, epochs)
            )
        
        # Load best model
        best_model = tf.keras.models.load_model(model_save_path)
        
        # Final evaluation
        print("\n📈 Final Model Evaluation:")
        val_loss, val_accuracy, val_precision, val_recall = best_model.evaluate(val_gen, verbose=0)
        f1_score = 2 * (val_precision * val_recall) / (val_precision + val_recall)
        
        print(f"✅ Validation Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
        print(f"✅ Validation Precision: {val_precision:.4f}")
        print(f"✅ Validation Recall: {val_recall:.4f}")
        print(f"✅ F1 Score: {f1_score:.4f}")
        
        # Save as DenseNet121 as requested
        final_model_path = "models/DenseNet121_cardiomegaly.h5"
        best_model.save(final_model_path)
        print(f"🎉 Model saved as: {final_model_path}")
        
        return best_model, val_accuracy

def main():
    """Main training function"""
    print("=" * 60)
    print("🏥 DenseNet121 Cardiomegaly Detection Training")
    print("=" * 60)
    
    # Dataset paths
    train_dir = "Dataset/CHEST/cardiomelgy/train/train"
    val_dir = "Dataset/CHEST/cardiomelgy/test/test"
    
    # Verify paths exist
    if not os.path.exists(train_dir):
        print(f"❌ Training directory not found: {train_dir}")
        return
    if not os.path.exists(val_dir):
        print(f"❌ Validation directory not found: {val_dir}")
        return
    
    # Initialize trainer
    trainer = DenseNet121CardiomegalyTrainer(
        input_shape=(224, 224, 3),
        num_classes=2
    )
    
    # Start training
    try:
        model, accuracy = trainer.train(
            train_dir=train_dir,
            val_dir=val_dir,
            epochs=80,
            batch_size=32
        )
        
        if accuracy >= 0.90:
            print(f"🎯 SUCCESS! Achieved target accuracy: {accuracy*100:.2f}%")
            print("✅ Model ready for integration as 'DenseNet121'")
        else:
            print(f"⚠️  Accuracy {accuracy*100:.2f}% below 90% target")
            print("🔄 Consider additional training or hyperparameter tuning")
            
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()