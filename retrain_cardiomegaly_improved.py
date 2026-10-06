"""
Enhanced Cardiomegaly Retraining Script
Goal: Improve from 75.82% to 90%+ accuracy
Strategy: Larger image size, more aggressive augmentation, progressive training
"""

import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, Callback
import os
import numpy as np
from datetime import datetime
import json

print("="*80)
print("🫀 ENHANCED CARDIOMEGALY RETRAINING")
print("="*80)
print("Target: 90%+ Validation Accuracy")
print("Current Baseline: 75.82%")
print("Strategy: Higher resolution + Aggressive augmentation + Progressive training")
print("="*80)

class TargetAccuracyCallback(Callback):
    """Stop training when target accuracy is reached"""
    def __init__(self, target=0.90):
        super().__init__()
        self.target = target
        self.best_acc = 0.0
        
    def on_epoch_end(self, epoch, logs=None):
        val_acc = logs.get('val_accuracy', 0)
        if val_acc > self.best_acc:
            self.best_acc = val_acc
            
        print(f"\n📊 Current: {val_acc:.2%} | Best: {self.best_acc:.2%} | Target: {self.target:.0%}")
        
        if val_acc >= self.target:
            print(f"\n🎉 TARGET ACHIEVED! Validation accuracy: {val_acc:.1%}")
            self.model.stop_training = True

class EnhancedCardiomegalyTrainer:
    def __init__(self, input_shape=(320, 320, 3), num_classes=2):
        """Increased image size from 224 to 320 for better feature capture"""
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        
    def create_model(self, unfreeze_layers=100):
        """Create DenseNet121 model with more trainable layers"""
        base_model = DenseNet121(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        # More aggressive unfreezing for better fine-tuning
        total_layers = len(base_model.layers)
        freeze_until = total_layers - unfreeze_layers
        
        for i, layer in enumerate(base_model.layers):
            layer.trainable = (i >= freeze_until)
        
        trainable = sum([1 for layer in base_model.layers if layer.trainable])
        print(f"📊 Base model: {total_layers} layers, {trainable} trainable")
        
        # Enhanced classification head
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = BatchNormalization()(x)
        x = Dense(1024, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
        x = Dropout(0.5)(x)
        x = BatchNormalization()(x)
        x = Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
        x = Dropout(0.4)(x)
        x = BatchNormalization()(x)
        x = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
        x = Dropout(0.3)(x)
        predictions = Dense(self.num_classes, activation='softmax', name='predictions')(x)
        
        self.model = Model(inputs=base_model.input, outputs=predictions)
        return self.model
    
    def compile_model(self, learning_rate=0.0001):
        """Compile with lower learning rate for fine-tuning"""
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy',
                     tf.keras.metrics.Precision(name='precision'),
                     tf.keras.metrics.Recall(name='recall'),
                     tf.keras.metrics.AUC(name='auc')]
        )
        
    def prepare_data(self, data_dir, batch_size=16, validation_split=0.2):
        """Enhanced data augmentation for better generalization"""
        
        # More aggressive augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=validation_split,
            rotation_range=20,  # Increased
            width_shift_range=0.15,  # Increased
            height_shift_range=0.15,  # Increased
            shear_range=0.1,  # Added
            zoom_range=0.2,  # Increased
            horizontal_flip=True,
            brightness_range=[0.7, 1.3],  # More aggressive
            fill_mode='nearest'
        )
        
        # Minimal augmentation for validation
        val_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=validation_split
        )
        
        train_generator = train_datagen.flow_from_directory(
            data_dir,
            target_size=self.input_shape[:2],
            batch_size=batch_size,
            class_mode='categorical',
            subset='training',
            shuffle=True,
            seed=42
        )
        
        val_generator = val_datagen.flow_from_directory(
            data_dir,
            target_size=self.input_shape[:2],
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=False,
            seed=42
        )
        
        return train_generator, val_generator
    
    def train_progressive(self, train_gen, val_gen, epochs_per_phase=15):
        """Progressive training in 3 phases"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs('models/cardiomegaly', exist_ok=True)
        
        # Calculate class weights
        class_counts = np.bincount(train_gen.classes)
        total = sum(class_counts)
        class_weights = {i: total / (len(class_counts) * count) 
                        for i, count in enumerate(class_counts)}
        
        print(f"\n📊 Dataset Info:")
        print(f"   Training samples: {train_gen.n}")
        print(f"   Validation samples: {val_gen.n}")
        print(f"   Classes: {list(train_gen.class_indices.keys())}")
        print(f"   Class weights: {class_weights}")
        
        all_history = {'accuracy': [], 'val_accuracy': [], 'loss': [], 'val_loss': []}
        
        # PHASE 1: Train with fewer trainable layers
        print("\n" + "="*80)
        print("PHASE 1: Initial Training (50 trainable layers)")
        print("="*80)
        
        self.create_model(unfreeze_layers=50)
        self.compile_model(learning_rate=0.001)
        
        callbacks_phase1 = [
            TargetAccuracyCallback(target=0.90),
            ModelCheckpoint(
                f'models/cardiomegaly/improved_phase1_{timestamp}.h5',
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            EarlyStopping(
                monitor='val_accuracy',
                patience=5,
                restore_best_weights=True,
                mode='max',
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        history1 = self.model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs_per_phase,
            class_weight=class_weights,
            callbacks=callbacks_phase1,
            verbose=1
        )
        
        for key in all_history:
            all_history[key].extend(history1.history[key])
        
        phase1_best = max(history1.history['val_accuracy'])
        print(f"\n✅ Phase 1 Complete - Best Val Accuracy: {phase1_best:.2%}")
        
        if phase1_best >= 0.90:
            print("🎉 Target achieved in Phase 1!")
            return self.model, all_history
        
        # PHASE 2: Unfreeze more layers
        print("\n" + "="*80)
        print("PHASE 2: Medium Fine-tuning (100 trainable layers)")
        print("="*80)
        
        self.create_model(unfreeze_layers=100)
        # Load phase 1 weights
        self.model.load_weights(f'models/cardiomegaly/improved_phase1_{timestamp}.h5')
        self.compile_model(learning_rate=0.0001)
        
        callbacks_phase2 = [
            TargetAccuracyCallback(target=0.90),
            ModelCheckpoint(
                f'models/cardiomegaly/improved_phase2_{timestamp}.h5',
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            EarlyStopping(
                monitor='val_accuracy',
                patience=5,
                restore_best_weights=True,
                mode='max',
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        history2 = self.model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs_per_phase,
            class_weight=class_weights,
            callbacks=callbacks_phase2,
            verbose=1
        )
        
        for key in all_history:
            all_history[key].extend(history2.history[key])
        
        phase2_best = max(history2.history['val_accuracy'])
        print(f"\n✅ Phase 2 Complete - Best Val Accuracy: {phase2_best:.2%}")
        
        if phase2_best >= 0.90:
            print("🎉 Target achieved in Phase 2!")
            return self.model, all_history
        
        # PHASE 3: Full fine-tuning
        print("\n" + "="*80)
        print("PHASE 3: Deep Fine-tuning (150 trainable layers)")
        print("="*80)
        
        self.create_model(unfreeze_layers=150)
        # Load phase 2 weights
        self.model.load_weights(f'models/cardiomegaly/improved_phase2_{timestamp}.h5')
        self.compile_model(learning_rate=0.00005)
        
        callbacks_phase3 = [
            TargetAccuracyCallback(target=0.90),
            ModelCheckpoint(
                f'models/cardiomegaly/improved_phase3_{timestamp}.h5',
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            EarlyStopping(
                monitor='val_accuracy',
                patience=5,
                restore_best_weights=True,
                mode='max',
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        history3 = self.model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs_per_phase,
            class_weight=class_weights,
            callbacks=callbacks_phase3,
            verbose=1
        )
        
        for key in all_history:
            all_history[key].extend(history3.history[key])
        
        phase3_best = max(history3.history['val_accuracy'])
        print(f"\n✅ Phase 3 Complete - Best Val Accuracy: {phase3_best:.2%}")
        
        # Save final model
        final_model_path = f'models/cardiomegaly/DenseNet121_cardiomegaly_improved_{timestamp}.h5'
        self.model.save(final_model_path)
        print(f"\n💾 Final model saved: {final_model_path}")
        
        return self.model, all_history

# Main execution
if __name__ == "__main__":
    
    # Configuration
    DATA_DIR = 'Dataset/CHEST/cardiomelgy/train/train'
    IMAGE_SIZE = (320, 320, 3)  # Larger than baseline 224x224
    BATCH_SIZE = 12  # Smaller for better generalization
    EPOCHS_PER_PHASE = 15
    
    print(f"\n📋 Configuration:")
    print(f"   Image Size: {IMAGE_SIZE[:2]}")
    print(f"   Batch Size: {BATCH_SIZE}")
    print(f"   Epochs per Phase: {EPOCHS_PER_PHASE}")
    print(f"   Total Max Epochs: {EPOCHS_PER_PHASE * 3}")
    
    # Initialize trainer
    trainer = EnhancedCardiomegalyTrainer(input_shape=IMAGE_SIZE)
    
    # Prepare data
    print("\n📂 Loading and preparing data...")
    train_gen, val_gen = trainer.prepare_data(
        DATA_DIR,
        batch_size=BATCH_SIZE,
        validation_split=0.2
    )
    
    # Train with progressive strategy
    print("\n🚀 Starting Progressive Training...")
    model, history = trainer.train_progressive(
        train_gen,
        val_gen,
        epochs_per_phase=EPOCHS_PER_PHASE
    )
    
    # Final evaluation
    print("\n" + "="*80)
    print("📊 FINAL RESULTS")
    print("="*80)
    
    best_val_acc = max(history['val_accuracy'])
    final_val_acc = history['val_accuracy'][-1]
    baseline_acc = 0.7582
    improvement = best_val_acc - baseline_acc
    
    print(f"\nBaseline Accuracy:      75.82%")
    print(f"Best Validation Acc:    {best_val_acc:.2%}")
    print(f"Final Validation Acc:   {final_val_acc:.2%}")
    print(f"Improvement:            +{improvement*100:.2f}%")
    
    if best_val_acc >= 0.90:
        print("\n🎉 SUCCESS! Target accuracy (90%) ACHIEVED!")
    elif best_val_acc >= 0.85:
        print("\n📈 Excellent progress! Close to target.")
    else:
        print(f"\n📊 Progress made. Gap to target: {(0.90 - best_val_acc)*100:.2f}%")
    
    # Save history
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    history_path = f'models/cardiomegaly/improved_history_{timestamp}.json'
    with open(history_path, 'w') as f:
        json.dump({
            'baseline_accuracy': baseline_acc,
            'best_val_accuracy': float(best_val_acc),
            'final_val_accuracy': float(final_val_acc),
            'improvement': float(improvement),
            'history': {k: [float(v) for v in vals] for k, vals in history.items()}
        }, f, indent=2)
    
    print(f"\n✅ Training history saved: {history_path}")
    print("\n" + "="*80)
    print("🎊 ENHANCED CARDIOMEGALY RETRAINING COMPLETE")
    print("="*80)
