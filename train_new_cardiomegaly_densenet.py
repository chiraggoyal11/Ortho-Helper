"""
DenseNet121 Training for New Cardiomegaly Dataset
Dataset: Dataset/CHEST/cardiomelgy (2,219 images per class)
Target: High accuracy cardiomegaly detection
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
print("🫀 DENSENET121 CARDIOMEGALY TRAINING - NEW DATASET")
print("="*80)
print("Dataset: Dataset/CHEST/cardiomelgy")
print("Classes: Cardiomelgy (2,219) | Normal (2,219)")
print("Total: 4,438 images (perfectly balanced)")
print("="*80)

class TrainingCallback(Callback):
    """Monitor training progress"""
    def __init__(self, target=0.90):
        super().__init__()
        self.target = target
        self.best_acc = 0.0
        
    def on_epoch_end(self, epoch, logs=None):
        val_acc = logs.get('val_accuracy', 0)
        train_acc = logs.get('accuracy', 0)
        
        if val_acc > self.best_acc:
            self.best_acc = val_acc
            print(f"\n🎯 NEW BEST! Val Acc: {val_acc:.2%} | Train Acc: {train_acc:.2%}")
        
        if val_acc >= self.target:
            print(f"\n🎉 TARGET {self.target:.0%} ACHIEVED!")
            self.model.stop_training = True

# Configuration
DATA_DIR = 'Dataset/CHEST/cardiomelgy'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 0.0001

print(f"\n📋 Configuration:")
print(f"   Image Size: {IMG_SIZE}")
print(f"   Batch Size: {BATCH_SIZE}")
print(f"   Learning Rate: {LEARNING_RATE}")
print(f"   Max Epochs: {EPOCHS}")

# Data augmentation
print("\n📂 Preparing data generators...")

train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.15,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
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
print(f"✅ Class distribution: {dict(zip(*np.unique(train_generator.classes, return_counts=True)))}")

# Build DenseNet121 model
print("\n🔨 Building DenseNet121 model...")

base_model = DenseNet121(
    weights='imagenet',
    include_top=False,
    input_shape=(*IMG_SIZE, 3)
)

# Unfreeze last 100 layers for fine-tuning
for layer in base_model.layers[:-100]:
    layer.trainable = False

trainable_layers = sum([1 for layer in base_model.layers if layer.trainable])
print(f"📊 Base model: {len(base_model.layers)} total layers, {trainable_layers} trainable")

# Add custom classification head
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

total_params = model.count_params()
trainable_params = sum([tf.size(var).numpy() for var in model.trainable_variables])
print(f"📊 Total parameters: {total_params:,}")
print(f"📊 Trainable parameters: {trainable_params:,}")

# Compile model
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

model_path = f'models/cardiomegaly/DenseNet121_new_dataset_{timestamp}.h5'

callbacks = [
    TrainingCallback(target=0.90),
    ModelCheckpoint(
        model_path,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    EarlyStopping(
        monitor='val_accuracy',
        patience=7,
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

# Train
print("\n" + "="*80)
print("🚀 STARTING TRAINING")
print("="*80 + "\n")

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

# Results
print("\n" + "="*80)
print("📊 TRAINING COMPLETE - RESULTS")
print("="*80)

best_val_acc = max(history.history['val_accuracy'])
final_val_acc = history.history['val_accuracy'][-1]
best_train_acc = max(history.history['accuracy'])
epochs_trained = len(history.history['val_accuracy'])

print(f"\n✅ Best Validation Accuracy:  {best_val_acc:.2%}")
print(f"✅ Final Validation Accuracy: {final_val_acc:.2%}")
print(f"✅ Best Training Accuracy:    {best_train_acc:.2%}")
print(f"✅ Epochs Trained:            {epochs_trained}")

if best_val_acc >= 0.90:
    print(f"\n🎉🎉🎉 EXCELLENT! Achieved 90%+ accuracy! 🎉🎉🎉")
elif best_val_acc >= 0.85:
    print(f"\n🎯 Very Good! {best_val_acc:.2%} accuracy achieved!")
elif best_val_acc >= 0.80:
    print(f"\n👍 Good Progress! {best_val_acc:.2%} accuracy achieved!")
else:
    print(f"\n📈 Training completed with {best_val_acc:.2%} accuracy")

# Save final model and history
final_model_path = f'models/DenseNet121_cardiomegaly_new.h5'
model.save(final_model_path)
print(f"\n💾 Model saved to:")
print(f"   Best: {model_path}")
print(f"   Final: {final_model_path}")

# Save training history
history_data = {
    'dataset': 'Dataset/CHEST/cardiomelgy',
    'classes': list(train_generator.class_indices.keys()),
    'training_samples': int(train_generator.n),
    'validation_samples': int(val_generator.n),
    'best_val_accuracy': float(best_val_acc),
    'final_val_accuracy': float(final_val_acc),
    'best_train_accuracy': float(best_train_acc),
    'epochs_trained': int(epochs_trained),
    'timestamp': timestamp,
    'config': {
        'image_size': IMG_SIZE,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'max_epochs': EPOCHS
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

history_path = f'models/cardiomegaly/history_new_dataset_{timestamp}.json'
with open(history_path, 'w') as f:
    json.dump(history_data, f, indent=2)

print(f"📊 History saved: {history_path}")

print("\n" + "="*80)
print("✅ ALL DONE!")
print("="*80)
