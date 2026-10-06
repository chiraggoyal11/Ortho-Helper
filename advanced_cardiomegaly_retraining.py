"""
Advanced Cardiomegaly Retraining Script
Target: 90%+ Validation Accuracy
Features: Ensemble methods, advanced augmentation, progressive unfreezing
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import DenseNet121, EfficientNetB4, ResNet152V2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras import regularizers, metrics
import numpy as np
from datetime import datetime
import json
from pathlib import Path
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Configuration
CONFIG = {
    'dataset_path': 'Dataset/CHEST/cardiomelgy/train/train',
    'image_size': (256, 256),  # Higher resolution
    'batch_size': 16,  # Smaller batch for better generalization
    'epochs': 30,
    'learning_rate': 0.0001,
    'target_accuracy': 0.90,
    'ensemble_models': ['DenseNet121', 'EfficientNetB4', 'ResNet152V2']
}

print("=" * 80)
print("🫀 ADVANCED CARDIOMEGALY RETRAINING")
print("=" * 80)
print(f"Target Accuracy: {CONFIG['target_accuracy']*100}%")
print(f"Current Baseline: 75.82%")
print(f"Image Size: {CONFIG['image_size']}")
print(f"Ensemble Models: {len(CONFIG['ensemble_models'])}")
print("=" * 80)

# Advanced Data Augmentation Pipeline
def create_advanced_augmentation():
    """Create comprehensive augmentation for medical imaging"""
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.15),
        layers.RandomTranslation(0.1, 0.1),
        layers.RandomContrast(0.2),
        layers.GaussianNoise(0.01),
    ], name="advanced_augmentation")

# Enhanced Preprocessing
def preprocess_image(image, label):
    """Advanced preprocessing for chest X-rays"""
    image = tf.cast(image, tf.float32) / 255.0
    # Normalize using ImageNet statistics
    mean = tf.constant([0.485, 0.456, 0.406])
    std = tf.constant([0.229, 0.224, 0.225])
    image = (image - mean) / std
    return image, label

# Load and prepare dataset
print("\n📂 Loading Dataset...")
dataset_path = Path(CONFIG['dataset_path'])

# Create datasets with advanced augmentation
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    horizontal_flip=True,
    rotation_range=15,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=CONFIG['image_size'],
    batch_size=CONFIG['batch_size'],
    class_mode='categorical',
    subset='training',
    shuffle=True,
    seed=42
)

val_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=CONFIG['image_size'],
    batch_size=CONFIG['batch_size'],
    class_mode='categorical',
    subset='validation',
    shuffle=False,
    seed=42
)

print(f"✅ Training samples: {train_generator.n}")
print(f"✅ Validation samples: {val_generator.n}")
print(f"✅ Classes: {list(train_generator.class_indices.keys())}")

# Calculate class weights for imbalanced data
class_counts = np.bincount(train_generator.classes)
total_samples = np.sum(class_counts)
class_weights = {i: total_samples / (len(class_counts) * count) 
                 for i, count in enumerate(class_counts)}
print(f"📊 Class weights: {class_weights}")

# Custom callback for target accuracy
class TargetAccuracyCallback(tf.keras.callbacks.Callback):
    def __init__(self, target_accuracy=0.90):
        super().__init__()
        self.target_accuracy = target_accuracy
        self.best_val_acc = 0.0
        
    def on_epoch_end(self, epoch, logs=None):
        val_acc = logs.get('val_accuracy', 0)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            
        if val_acc >= self.target_accuracy:
            print(f"\n🎉 TARGET ACHIEVED! Validation accuracy: {val_acc:.1%}")
            self.model.stop_training = True
        elif val_acc > self.best_val_acc - 0.02:
            print(f"📈 Progress: {val_acc:.1%} (Best: {self.best_val_acc:.1%})")

# Build ensemble model
def build_advanced_model(architecture='DenseNet121'):
    """Build advanced model with attention mechanism"""
    
    input_shape = (*CONFIG['image_size'], 3)
    inputs = layers.Input(shape=input_shape)
    
    # Select base model
    if architecture == 'DenseNet121':
        base_model = DenseNet121(
            include_top=False,
            weights='imagenet',
            input_tensor=inputs,
            pooling='avg'
        )
        freeze_until = 280  # Freeze most layers initially
    elif architecture == 'EfficientNetB4':
        base_model = EfficientNetB4(
            include_top=False,
            weights='imagenet',
            input_tensor=inputs,
            pooling='avg'
        )
        freeze_until = 300
    else:  # ResNet152V2
        base_model = ResNet152V2(
            include_top=False,
            weights='imagenet',
            input_tensor=inputs,
            pooling='avg'
        )
        freeze_until = 400
    
    # Freeze base model layers initially
    base_model.trainable = True
    for layer in base_model.layers[:freeze_until]:
        layer.trainable = False
    
    # Advanced head with regularization
    x = base_model.output
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(2, activation='softmax', name='predictions')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name=f'{architecture}_Cardiomegaly')
    
    return model, base_model

# Progressive training strategy
def progressive_training(model, base_model, phase_name, epochs, initial_lr):
    """Progressive unfreezing training strategy"""
    
    print(f"\n{'='*60}")
    print(f"🚀 {phase_name}")
    print(f"{'='*60}")
    
    # Compile with appropriate optimizer
    model.compile(
        optimizer=optimizers.Adam(learning_rate=initial_lr),
        loss='categorical_crossentropy',
        metrics=['accuracy', 
                 metrics.Precision(name='precision'),
                 metrics.Recall(name='recall'),
                 metrics.AUC(name='auc')]
    )
    
    # Callbacks
    callbacks = [
        TargetAccuracyCallback(target_accuracy=CONFIG['target_accuracy']),
        EarlyStopping(
            monitor='val_accuracy',
            patience=7,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            f'models/cardiomegaly/checkpoint_{phase_name}.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )
    
    return history

# Train ensemble models
print("\n" + "="*80)
print("🎯 TRAINING ENSEMBLE MODELS")
print("="*80)

ensemble_models = []
ensemble_histories = []

for arch in CONFIG['ensemble_models']:
    print(f"\n{'#'*80}")
    print(f"# Training {arch}")
    print(f"{'#'*80}")
    
    # Build model
    model, base_model = build_advanced_model(arch)
    
    print(f"\n📊 Model: {arch}")
    print(f"   Total parameters: {model.count_params():,}")
    trainable_params = sum([tf.size(var).numpy() for var in model.trainable_variables])
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # Phase 1: Train only top layers
    print("\n" + "="*60)
    print("PHASE 1: Training Top Layers Only")
    print("="*60)
    history1 = progressive_training(
        model, base_model, 
        f"{arch}_Phase1", 
        epochs=10, 
        initial_lr=0.001
    )
    
    # Phase 2: Unfreeze more layers
    print("\n" + "="*60)
    print("PHASE 2: Fine-tuning with Partial Unfreezing")
    print("="*60)
    
    # Unfreeze top 50% of base model
    unfreeze_from = len(base_model.layers) // 2
    for layer in base_model.layers[unfreeze_from:]:
        layer.trainable = True
    
    trainable_params = sum([tf.size(var).numpy() for var in model.trainable_variables])
    print(f"   Updated trainable parameters: {trainable_params:,}")
    
    history2 = progressive_training(
        model, base_model,
        f"{arch}_Phase2",
        epochs=15,
        initial_lr=0.0001
    )
    
    # Phase 3: Full fine-tuning with very low learning rate
    print("\n" + "="*60)
    print("PHASE 3: Full Model Fine-tuning")
    print("="*60)
    
    # Unfreeze all layers
    for layer in base_model.layers:
        layer.trainable = True
    
    trainable_params = sum([tf.size(var).numpy() for var in model.trainable_variables])
    print(f"   Final trainable parameters: {trainable_params:,}")
    
    history3 = progressive_training(
        model, base_model,
        f"{arch}_Phase3",
        epochs=10,
        initial_lr=0.00001
    )
    
    # Combine histories
    combined_history = {
        'accuracy': history1.history['accuracy'] + history2.history['accuracy'] + history3.history['accuracy'],
        'val_accuracy': history1.history['val_accuracy'] + history2.history['val_accuracy'] + history3.history['val_accuracy'],
        'loss': history1.history['loss'] + history2.history['loss'] + history3.history['loss'],
        'val_loss': history1.history['val_loss'] + history2.history['val_loss'] + history3.history['val_loss']
    }
    
    ensemble_models.append(model)
    ensemble_histories.append(combined_history)
    
    # Get final metrics
    final_val_acc = combined_history['val_accuracy'][-1]
    best_val_acc = max(combined_history['val_accuracy'])
    
    print(f"\n{'='*60}")
    print(f"✅ {arch} Training Complete")
    print(f"{'='*60}")
    print(f"   Final Validation Accuracy: {final_val_acc:.1%}")
    print(f"   Best Validation Accuracy: {best_val_acc:.1%}")
    print(f"   Improvement from baseline: +{(best_val_acc - 0.7582)*100:.2f}%")
    
    # Save individual model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f'models/cardiomegaly/{arch}_advanced_{timestamp}.h5'
    model.save(model_path)
    print(f"   Model saved: {model_path}")

# Evaluate ensemble performance
print("\n" + "="*80)
print("🎯 ENSEMBLE EVALUATION")
print("="*80)

print("\nEvaluating ensemble predictions...")

# Get validation predictions from all models
ensemble_predictions = []
for model in ensemble_models:
    preds = model.predict(val_generator, verbose=0)
    ensemble_predictions.append(preds)

# Average ensemble predictions
ensemble_pred = np.mean(ensemble_predictions, axis=0)
ensemble_classes = np.argmax(ensemble_pred, axis=1)
true_classes = val_generator.classes

# Calculate ensemble accuracy
ensemble_accuracy = np.mean(ensemble_classes == true_classes)

print(f"\n{'='*60}")
print(f"📊 ENSEMBLE RESULTS")
print(f"{'='*60}")
print(f"Individual Model Accuracies:")
for i, (arch, history) in enumerate(zip(CONFIG['ensemble_models'], ensemble_histories)):
    best_acc = max(history['val_accuracy'])
    print(f"   {arch}: {best_acc:.1%}")

print(f"\n🎯 Ensemble Accuracy: {ensemble_accuracy:.1%}")
print(f"📈 Improvement from baseline (75.82%): +{(ensemble_accuracy - 0.7582)*100:.2f}%")

if ensemble_accuracy >= CONFIG['target_accuracy']:
    print(f"\n🎉 SUCCESS! Target accuracy ({CONFIG['target_accuracy']*100}%) ACHIEVED!")
else:
    print(f"\n📈 Progress made, but target ({CONFIG['target_accuracy']*100}%) not yet reached")
    print(f"   Gap to target: {(CONFIG['target_accuracy'] - ensemble_accuracy)*100:.2f}%")

# Save ensemble results
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results = {
    'timestamp': timestamp,
    'ensemble_accuracy': float(ensemble_accuracy),
    'baseline_accuracy': 0.7582,
    'improvement': float(ensemble_accuracy - 0.7582),
    'individual_models': {
        arch: {
            'best_val_accuracy': float(max(history['val_accuracy'])),
            'final_val_accuracy': float(history['val_accuracy'][-1])
        }
        for arch, history in zip(CONFIG['ensemble_models'], ensemble_histories)
    },
    'config': CONFIG
}

results_path = f'models/cardiomegaly/ensemble_results_{timestamp}.json'
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✅ Results saved: {results_path}")

# Create visualization
print("\n📊 Creating training visualization...")

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Advanced Cardiomegaly Training Results', fontsize=16, fontweight='bold')

# Plot 1: Individual model accuracies
ax = axes[0, 0]
for arch, history in zip(CONFIG['ensemble_models'], ensemble_histories):
    epochs = range(1, len(history['val_accuracy']) + 1)
    ax.plot(epochs, history['val_accuracy'], marker='o', label=arch, linewidth=2)
ax.axhline(y=0.7582, color='red', linestyle='--', label='Baseline (75.82%)', linewidth=2)
ax.axhline(y=CONFIG['target_accuracy'], color='green', linestyle='--', label='Target (90%)', linewidth=2)
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Validation Accuracy', fontsize=12)
ax.set_title('Individual Model Performance', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Loss curves
ax = axes[0, 1]
for arch, history in zip(CONFIG['ensemble_models'], ensemble_histories):
    epochs = range(1, len(history['val_loss']) + 1)
    ax.plot(epochs, history['val_loss'], marker='o', label=arch, linewidth=2)
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Validation Loss', fontsize=12)
ax.set_title('Validation Loss Curves', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Comparison bar chart
ax = axes[1, 0]
model_names = CONFIG['ensemble_models'] + ['Ensemble']
accuracies = [max(h['val_accuracy']) for h in ensemble_histories] + [ensemble_accuracy]
colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold']
bars = ax.bar(model_names, accuracies, color=colors, edgecolor='black', linewidth=2)
ax.axhline(y=0.7582, color='red', linestyle='--', label='Baseline', linewidth=2)
ax.axhline(y=CONFIG['target_accuracy'], color='green', linestyle='--', label='Target', linewidth=2)
ax.set_ylabel('Accuracy', fontsize=12)
ax.set_title('Model Comparison', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1%}',
            ha='center', va='bottom', fontweight='bold')

# Plot 4: Improvement summary
ax = axes[1, 1]
ax.axis('off')
summary_text = f"""
TRAINING SUMMARY
{'='*40}

Baseline Accuracy:     75.82%
Ensemble Accuracy:     {ensemble_accuracy:.2%}
Improvement:           +{(ensemble_accuracy - 0.7582)*100:.2f}%

Individual Models:
"""
for arch, history in zip(CONFIG['ensemble_models'], ensemble_histories):
    best_acc = max(history['val_accuracy'])
    summary_text += f"  • {arch:20s}: {best_acc:.2%}\n"

summary_text += f"""
Target Status:         {'✅ ACHIEVED' if ensemble_accuracy >= CONFIG['target_accuracy'] else f"📈 {(CONFIG['target_accuracy'] - ensemble_accuracy)*100:.1f}% to go"}

Training Configuration:
  • Image Size:          {CONFIG['image_size']}
  • Batch Size:          {CONFIG['batch_size']}
  • Max Epochs:          {CONFIG['epochs']}
  • Progressive Training: 3 phases
  • Ensemble Models:     {len(CONFIG['ensemble_models'])}
"""

ax.text(0.1, 0.9, summary_text, fontsize=11, family='monospace',
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
viz_path = f'models/cardiomegaly/training_results_{timestamp}.png'
plt.savefig(viz_path, dpi=300, bbox_inches='tight')
print(f"✅ Visualization saved: {viz_path}")

# Final summary
print("\n" + "="*80)
print("🎊 ADVANCED CARDIOMEGALY RETRAINING COMPLETE")
print("="*80)
print(f"\n📊 FINAL RESULTS:")
print(f"   Baseline:          75.82%")
print(f"   Ensemble:          {ensemble_accuracy:.2%}")
print(f"   Improvement:       +{(ensemble_accuracy - 0.7582)*100:.2f}%")
print(f"   Target Status:     {'✅ ACHIEVED!' if ensemble_accuracy >= CONFIG['target_accuracy'] else '📈 In Progress'}")
print(f"\n📁 Saved Files:")
print(f"   Results:           {results_path}")
print(f"   Visualization:     {viz_path}")
print(f"   Models:            models/cardiomegaly/*_advanced_{timestamp}.h5")
print("\n" + "="*80)
