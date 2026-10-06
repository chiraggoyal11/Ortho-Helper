"""
Advanced DenseNet121 Training for Cardiomegaly - Target 90%+ Accuracy
Previous Best: 76.75% -> Target: 90%+
Strategy: Larger image size, more layers unfrozen, optimized augmentation
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
print("🚀 ADVANCED CARDIOMEGALY TRAINING - TARGET 90%")
print("="*80)
print("Previous Best: 76.75%")
print("Strategy: Enhanced resolution + More trainable layers + Better augmentation")
print("="*80)

class AdvancedTrainingCallback(Callback):
    """Monitor and report training progress"""
    def __init__(self, baseline=0.7675, target=0.90):
        super().__init__()
        self.baseline = baseline
        self.target = target
        self.best_acc = 0.0
        
    def on_epoch_end(self, epoch, logs=None):
        val_acc = logs.get('val_accuracy', 0)
        train_acc = logs.get('accuracy', 0)
        
        if val_acc > self.best_acc:
            self.best_acc = val_acc
            improvement = val_acc - self.baseline
            print(f"\n🎯 NEW BEST! Val: {val_acc:.2%} | Train: {train_acc:.2%} | Improvement: +{improvement*100:.2f}%")
        
        gap_to_target = self.target - val_acc
        if gap_to_target > 0:
            print(f"📊 Gap to 90% target: {gap_to_target*100:.2f}%")
        
        if val_acc >= self.target:
            print(f"\n🎉🎉🎉 TARGET 90% ACHIEVED! 🎉🎉🎉")
            self.model.stop_training = True

# Configuration - ENHANCED for better performance
DATA_DIR = 'Dataset/CHEST/cardiomelgy'
IMG_SIZE = (299, 299)  # Larger than previous 224x224
BATCH_SIZE = 24  # Smaller batch for better generalization
EPOCHS = 40
LEARNING_RATE = 0.00008  # Slightly lower for fine-tuning

print(f"\n📋 Enhanced Configuration:")
print(f"   Image Size: {IMG_SIZE} (was 224x224)")
print(f"   Batch Size: {BATCH_SIZE} (was 32)")
print(f"   Learning Rate: {LEARNING_RATE}")
print(f"   Max Epochs: {EPOCHS}")

# Enhanced data augmentation
print("\n📂 Preparing enhanced data generators...")

train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=18,  # Increased
    width_shift_range=0.12,  # Increased
    height_shift_range=0.12,  # Increased
    shear_range=0.12,  # Increased
    zoom_range=0.18,  # Increased
    horizontal_flip=True,
    vertical_flip=False,
    brightness_range=[0.75, 1.25],  # More aggressive
    channel_shift_range=20.0,  # Added
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True,
    seed=42
)

val_generator = val_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False,
    seed=42
)

print(f"✅ Training samples: {train_generator.n}")
print(f"✅ Validation samples: {val_generator.n}")
print(f"✅ Classes: {list(train_generator.class_indices.keys())}")

# Calculate class weights
class_counts = np.bincount(train_generator.classes)
total = sum(class_counts)
class_weights = {i: total / (len(class_counts) * count) 
                for i, count in enumerate(class_counts)}
print(f"📊 Class weights: {class_weights}")

# Build model with MORE trainable layers
print("\n🔨 Building advanced DenseNet121 model...")

base_model = DenseNet121(
    weights='imagenet',
    include_top=False,
    input_shape=(*IMG_SIZE, 3)
)

# Unfreeze MORE layers (150 instead of 100)
for layer in base_model.layers[:-150]:
    layer.trainable = False

trainable_layers = sum([1 for layer in base_model.layers if layer.trainable])
print(f"📊 Base model: {len(base_model.layers)} total layers, {trainable_layers} trainable (was 100)")

# Enhanced classification head with more capacity
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dense(768, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)  # Increased from 512
x = Dropout(0.5)(x)
x = BatchNormalization()(x)
x = Dense(384, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)  # Increased from 256
x = Dropout(0.45)(x)
x = BatchNormalization()(x)
x = Dense(192, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)  # Added layer
x = Dropout(0.35)(x)
predictions = Dense(2, activation='softmax', name='predictions')(x)

model = Model(inputs=base_model.input, outputs=predictions)

total_params = model.count_params()
trainable_params = sum([tf.size(var).numpy() for var in model.trainable_variables])
print(f"📊 Total parameters: {total_params:,}")
print(f"📊 Trainable parameters: {trainable_params:,}")

# Compile with optimized settings
print("\n⚙️  Compiling model...")
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy',
             tf.keras.metrics.Precision(name='precision'),
             tf.keras.metrics.Recall(name='recall'),
             tf.keras.metrics.AUC(name='auc')]
)
print("✅ Model compiled")

# Setup callbacks
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs('models/cardiomegaly', exist_ok=True)

model_path = f'models/cardiomegaly/DenseNet121_advanced_{timestamp}.h5'

callbacks = [
    AdvancedTrainingCallback(baseline=0.7675, target=0.90),
    ModelCheckpoint(
        model_path,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    EarlyStopping(
        monitor='val_accuracy',
        patience=10,  # Increased patience
        restore_best_weights=True,
        mode='max',
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.4,  # More aggressive reduction
        patience=4,
        min_lr=1e-8,
        verbose=1
    )
]

# Train
print("\n" + "="*80)
print("🚀 STARTING ADVANCED TRAINING TO REACH 90%")
print("="*80 + "\n")

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1
)

# Results
print("\n" + "="*80)
print("📊 TRAINING COMPLETE - FINAL RESULTS")
print("="*80)

best_val_acc = max(history.history['val_accuracy'])
final_val_acc = history.history['val_accuracy'][-1]
best_train_acc = max(history.history['accuracy'])
epochs_trained = len(history.history['val_accuracy'])
baseline = 0.7675
improvement = best_val_acc - baseline

print(f"\n📈 Performance Summary:")
print(f"   Baseline (Previous):    76.75%")
print(f"   Best Validation:        {best_val_acc:.2%}")
print(f"   Final Validation:       {final_val_acc:.2%}")
print(f"   Best Training:          {best_train_acc:.2%}")
print(f"   Improvement:            +{improvement*100:.2f}%")
print(f"   Epochs Trained:         {epochs_trained}/{EPOCHS}")

if best_val_acc >= 0.90:
    print(f"\n🎉🎉🎉 SUCCESS! TARGET 90% ACHIEVED! 🎉🎉🎉")
    print(f"✅ Reached {best_val_acc:.2%}")
elif best_val_acc >= 0.85:
    print(f"\n🎯 Excellent Progress!")
    print(f"✅ Achieved {best_val_acc:.2%}")
    print(f"📊 Only {(0.90 - best_val_acc)*100:.2f}% away from 90% target")
elif best_val_acc >= 0.80:
    print(f"\n👍 Good Progress!")
    print(f"✅ Achieved {best_val_acc:.2%}")
    print(f"📊 Gap to 90%: {(0.90 - best_val_acc)*100:.2f}%")
else:
    print(f"\n📈 Progress Made")
    print(f"✅ Achieved {best_val_acc:.2%}")
    print(f"📊 Improvement: +{improvement*100:.2f}%")

# Save final model
final_model_path = f'models/DenseNet121_cardiomegaly_advanced.h5'
model.save(final_model_path)
print(f"\n💾 Models saved:")
print(f"   Best: {model_path}")
print(f"   Final: {final_model_path}")

# Save detailed history
history_data = {
    'dataset': 'Dataset/CHEST/cardiomelgy',
    'classes': list(train_generator.class_indices.keys()),
    'training_samples': int(train_generator.n),
    'validation_samples': int(val_generator.n),
    'baseline_accuracy': float(baseline),
    'best_val_accuracy': float(best_val_acc),
    'final_val_accuracy': float(final_val_acc),
    'best_train_accuracy': float(best_train_acc),
    'improvement': float(improvement),
    'epochs_trained': int(epochs_trained),
    'target_achieved': bool(best_val_acc >= 0.90),
    'timestamp': timestamp,
    'config': {
        'image_size': IMG_SIZE,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'max_epochs': EPOCHS,
        'trainable_layers': trainable_layers,
        'total_layers': len(base_model.layers)
    },
    'history': {
        'accuracy': [float(x) for x in history.history['accuracy']],
        'val_accuracy': [float(x) for x in history.history['val_accuracy']],
        'loss': [float(x) for x in history.history['loss']],
        'val_loss': [float(x) for x in history.history['val_loss']],
        'precision': [float(x) for x in history.history['precision']],
        'val_precision': [float(x) for x in history.history['val_precision']],
        'recall': [float(x) for x in history.history['recall']],
        'val_recall': [float(x) for x in history.history['val_recall']],
        'auc': [float(x) for x in history.history['auc']],
        'val_auc': [float(x) for x in history.history['val_auc']]
    }
}

history_path = f'models/cardiomegaly/history_advanced_{timestamp}.json'
with open(history_path, 'w') as f:
    json.dump(history_data, f, indent=2)

print(f"📊 History saved: {history_path}")

print("\n" + "="*80)
print("✅ TRAINING COMPLETE!")
print("="*80)
