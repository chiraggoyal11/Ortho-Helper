"""
Ultra-Fast Cardiomegaly Accuracy Booster
Strategy: Start from existing baseline, fine-tune aggressively
Target: 90%+ accuracy in minimal epochs
"""

import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, Callback
import os
import numpy as np
from datetime import datetime
import json
from pathlib import Path

print("="*80)
print("🚀 ULTRA-FAST CARDIOMEGALY ACCURACY BOOSTER")
print("="*80)
print("Strategy: Load best existing model and boost it to 90%+")
print("="*80)

class AccuracyBoostCallback(Callback):
    """Monitor and report accuracy improvements"""
    def __init__(self, baseline=0.7582, target=0.90):
        super().__init__()
        self.baseline = baseline
        self.target = target
        self.best_acc = baseline
        
    def on_epoch_end(self, epoch, logs=None):
        val_acc = logs.get('val_accuracy', 0)
        improvement = val_acc - self.baseline
        
        if val_acc > self.best_acc:
            self.best_acc = val_acc
            print(f"\n🎯 NEW BEST! {val_acc:.2%} (↑{improvement*100:+.2f}% from baseline)")
        
        if val_acc >= self.target:
            print(f"\n🎉 TARGET ACHIEVED! {val_acc:.2%} >= {self.target:.0%}")
            self.model.stop_training = True
        else:
            gap = self.target - val_acc
            print(f"📊 Current: {val_acc:.2%} | Gap to target: {gap*100:.2f}%")

# Check for existing model
existing_model_path = 'models/DenseNet121_cardiomegaly.h5'
use_existing = Path(existing_model_path).exists()

if use_existing:
    print(f"\n✅ Found existing model: {existing_model_path}")
    print("📦 Loading pre-trained model...")
else:
    print("\n⚠️  No existing model found, will train from scratch")

# Data preparation with focused augmentation
print("\n📂 Preparing dataset...")

# Balanced augmentation - not too aggressive
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=12,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.12,
    horizontal_flip=True,
    brightness_range=[0.85, 1.15],
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

DATA_DIR = 'Dataset/CHEST/cardiomelgy/train/train'
IMG_SIZE = (256, 256)  # Balanced size
BATCH_SIZE = 20  # Larger batch for stability

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

# Model setup
if use_existing:
    print("\n🔄 Loading and enhancing existing model...")
    try:
        # Load existing model
        model = load_model(existing_model_path)
        print("✅ Model loaded successfully")
        
        # Make all layers trainable for fine-tuning
        for layer in model.layers:
            layer.trainable = True
        
        print(f"📊 Total layers: {len(model.layers)}")
        print(f"📊 All layers set to trainable")
        
    except Exception as e:
        print(f"⚠️  Error loading model: {e}")
        print("🔨 Building new model instead...")
        use_existing = False

if not use_existing:
    print("\n🔨 Building new optimized model...")
    
    base_model = DenseNet121(
        weights='imagenet',
        include_top=False,
        input_shape=(*IMG_SIZE, 3)
    )
    
    # Unfreeze last 120 layers for effective fine-tuning
    for layer in base_model.layers[:-120]:
        layer.trainable = False
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = Dropout(0.4)(x)
    predictions = Dense(2, activation='softmax', name='predictions')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    print(f"✅ Model built with {len(model.layers)} layers")

# Compile with optimized settings
print("\n⚙️  Compiling model...")
model.compile(
    optimizer=Adam(learning_rate=0.00005),  # Very low learning rate for fine-tuning
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

callbacks = [
    AccuracyBoostCallback(baseline=0.7582, target=0.90),
    ModelCheckpoint(
        f'models/cardiomegaly/boosted_{timestamp}.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    EarlyStopping(
        monitor='val_accuracy',
        patience=8,
        restore_best_weights=True,
        mode='max',
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.3,
        patience=3,
        min_lr=1e-8,
        verbose=1
    )
]

# Train
print("\n" + "="*80)
print("🚀 STARTING AGGRESSIVE FINE-TUNING")
print("="*80)
print(f"Learning Rate: 0.00005")
print(f"Max Epochs: 30")
print(f"Early Stopping: Yes (patience=8)")
print("="*80 + "\n")

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=30,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1
)

# Evaluation
print("\n" + "="*80)
print("📊 FINAL RESULTS")
print("="*80)

best_val_acc = max(history.history['val_accuracy'])
final_val_acc = history.history['val_accuracy'][-1]
baseline = 0.7582
improvement = best_val_acc - baseline

print(f"\n📈 Performance Summary:")
print(f"   Baseline:           75.82%")
print(f"   Best Achieved:      {best_val_acc:.2%}")
print(f"   Final:              {final_val_acc:.2%}")
print(f"   Improvement:        {improvement*100:+.2f}%")
print(f"   Epochs Trained:     {len(history.history['val_accuracy'])}")

if best_val_acc >= 0.90:
    print(f"\n🎉🎉🎉 SUCCESS! TARGET ACHIEVED! 🎉🎉🎉")
    print(f"✅ Reached {best_val_acc:.2%} (target: 90%)")
elif best_val_acc >= 0.85:
    print(f"\n📈 Excellent Progress!")
    print(f"✅ Achieved {best_val_acc:.2%}")
    print(f"📊 Only {(0.90 - best_val_acc)*100:.2f}% away from target")
elif best_val_acc >= 0.80:
    print(f"\n👍 Good Progress!")
    print(f"✅ Achieved {best_val_acc:.2%}")
    print(f"📊 Gap to target: {(0.90 - best_val_acc)*100:.2f}%")
else:
    print(f"\n📊 Progress Made")
    print(f"✅ Achieved {best_val_acc:.2%}")
    print(f"📈 Improvement: {improvement*100:+.2f}%")

# Save final model
final_path = f'models/DenseNet121_cardiomegaly_boosted_{timestamp}.h5'
model.save(final_path)
print(f"\n💾 Model saved: {final_path}")

# Save history
history_data = {
    'baseline': baseline,
    'best_val_accuracy': float(best_val_acc),
    'final_val_accuracy': float(final_val_acc),
    'improvement': float(improvement),
    'epochs_trained': len(history.history['val_accuracy']),
    'timestamp': timestamp,
    'history': {
        'accuracy': [float(x) for x in history.history['accuracy']],
        'val_accuracy': [float(x) for x in history.history['val_accuracy']],
        'loss': [float(x) for x in history.history['loss']],
        'val_loss': [float(x) for x in history.history['val_loss']]
    }
}

history_path = f'models/cardiomegaly/boosted_history_{timestamp}.json'
with open(history_path, 'w') as f:
    json.dump(history_data, f, indent=2)

print(f"📊 History saved: {history_path}")

print("\n" + "="*80)
print("✅ TRAINING COMPLETE")
print("="*80)
