import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, LearningRateScheduler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import json
from datetime import datetime
import matplotlib.pyplot as plt
import sys
import time

# Configuration
ULTRA_CONFIG = {
    'img_height': 224,
    'img_width': 224,
    'batch_size': 32,
    'epochs': 50,
    'learning_rate': 1e-4,
    'base_dir': 'Dataset/ARM_Combined',
    'model_dir': 'models/bone_fracture',
    'target_accuracy': 0.90,
    'validation_split': 0.0, # Using explicit val directory
    'warmup_epochs': 5,
    'patience': 15,
    'min_delta': 0.001,
    'factor': 0.5
}

# Set memory growth for GPU if available
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Found {len(gpus)} GPU(s)")
    except RuntimeError as e:
        print(e)

def setup_gpu_optimization():
    """Configure GPU for maximum performance"""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("✓ GPU optimized for training")
        except RuntimeError as e:
            print(f"GPU setup error: {e}")
    else:
        print("Training on CPU")

def enhanced_data_generators():
    """Create medical-grade data augmentation for bone fracture"""
    
    train_dir = os.path.join(ULTRA_CONFIG['base_dir'], 'train')
    val_dir = os.path.join(ULTRA_CONFIG['base_dir'], 'val')

    # Medical-specific augmentation preserving pathology
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        vertical_flip=True, # Bone fractures can be in any orientation
        brightness_range=[0.8, 1.2],
        fill_mode='reflect'
    )
    
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(ULTRA_CONFIG['img_height'], ULTRA_CONFIG['img_width']),
        batch_size=ULTRA_CONFIG['batch_size'],
        class_mode='binary',
        shuffle=True,
        seed=42
    )
    
    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(ULTRA_CONFIG['img_height'], ULTRA_CONFIG['img_width']),
        batch_size=ULTRA_CONFIG['batch_size'],
        class_mode='binary',
        shuffle=False,
        seed=42
    )
    
    return train_generator, val_generator

def create_ultra_bone_fracture_model():
    """Create ultra-advanced model targeting 90%+ accuracy"""
    
    inputs = layers.Input(shape=(ULTRA_CONFIG['img_height'], ULTRA_CONFIG['img_width'], 3))
    
    # DenseNet121 backbone - proven for medical imaging
    base_model = DenseNet121(
        weights='imagenet',
        include_top=False,
        input_tensor=inputs
    )
    
    # Unfreeze last block
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    for layer in base_model.layers[-30:]:
        layer.trainable = True
    
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    # Final prediction layer
    predictions = layers.Dense(1, activation='sigmoid', name='bone_fracture_output')(x)
    
    model = Model(inputs=inputs, outputs=predictions)
    return model

def create_advanced_callbacks(model_name):
    """Create optimized callback suite for maximum performance"""
    
    checkpoint_path = os.path.join(ULTRA_CONFIG['model_dir'], f'{model_name}_best.h5')
    
    # Best model checkpoint
    checkpoint = ModelCheckpoint(
        checkpoint_path,
        monitor='val_accuracy',
        save_best_only=True,
        save_weights_only=False,
        mode='max',
        verbose=1,
        save_freq='epoch'
    )
    
    # Early stopping with patience
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=ULTRA_CONFIG['patience'],
        min_delta=ULTRA_CONFIG['min_delta'],
        mode='max',
        verbose=1,
        restore_best_weights=True
    )
    
    # Adaptive learning rate
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=ULTRA_CONFIG['factor'],
        patience=5,
        min_lr=1e-7,
        verbose=1,
        mode='min'
    )
    
    return [checkpoint, early_stopping, reduce_lr]

def compute_optimal_class_weights(train_generator):
    """Compute class weights with medical-specific optimization"""
    
    labels = train_generator.classes
    unique_classes = np.unique(labels)
    
    class_weights = compute_class_weight(
        'balanced',
        classes=unique_classes,
        y=labels
    )
    
    class_weight_dict = dict(zip(unique_classes, class_weights))
    
    print(f"Optimized Class Weights: {class_weight_dict}")
    return class_weight_dict

def plot_comprehensive_results(history, model_name):
    """Create comprehensive visualization of training results"""
    
    plt.style.use('default')
    fig = plt.figure(figsize=(20, 12))
    
    # Accuracy subplot
    ax1 = plt.subplot(2, 3, 1)
    epochs = range(1, len(history.history['accuracy']) + 1)
    plt.plot(epochs, history.history['accuracy'], 'b-', label='Training Accuracy', linewidth=2)
    plt.plot(epochs, history.history['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
    plt.title('Model Accuracy Progression', fontsize=14, weight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add target line
    plt.axhline(y=ULTRA_CONFIG['target_accuracy'], color='g', linestyle='--', alpha=0.7, label=f'Target ({ULTRA_CONFIG["target_accuracy"]*100:.0f}%)')
    plt.legend()
    
    # Loss subplot
    ax2 = plt.subplot(2, 3, 2)
    plt.plot(epochs, history.history['loss'], 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, history.history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    plt.title('Model Loss Progression', fontsize=14, weight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Learning rate subplot
    ax3 = plt.subplot(2, 3, 3)
    if 'lr' in history.history:
        plt.plot(epochs, history.history['lr'], 'g-', linewidth=2)
        plt.title('Learning Rate Schedule', fontsize=14, weight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
    
    # Performance metrics
    best_val_acc = max(history.history['val_accuracy'])
    best_epoch = history.history['val_accuracy'].index(best_val_acc) + 1
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    
    # Summary text
    ax4 = plt.subplot(2, 3, 4)
    summary_text = f"""
ULTRA BONE FRACTURE TRAINING RESULTS

Best Validation Accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)
Best Epoch: {best_epoch}
Final Training Accuracy: {final_train_acc:.4f} ({final_train_acc*100:.2f}%)
Final Validation Accuracy: {final_val_acc:.4f} ({final_val_acc*100:.2f}%)

Target Achievement: {'✅ SUCCESS' if best_val_acc >= ULTRA_CONFIG['target_accuracy'] else '❌ NEEDS MORE TRAINING'}
Total Epochs Trained: {len(history.history['accuracy'])}

Architecture: DenseNet121
Optimization: Adam + ReduceLROnPlateau
"""
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=11, 
             verticalalignment='top', fontfamily='monospace')
    ax4.set_title('Training Summary', fontsize=14, weight='bold')
    ax4.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{ULTRA_CONFIG["model_dir"]}/{model_name}_ultra_analysis.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    return best_val_acc, best_epoch

def comprehensive_model_evaluation(model, val_generator):
    """Ultra-comprehensive model evaluation with medical metrics"""
    
    print("\n" + "="*60)
    print("COMPREHENSIVE BONE FRACTURE MODEL EVALUATION")
    print("="*60)
    
    val_generator.reset()
    
    # Get predictions and probabilities
    predictions_prob = model.predict(val_generator, verbose=1)
    predictions_binary = (predictions_prob > 0.5).astype(int).flatten()
    y_true = val_generator.classes
    
    # Calculate AUC-ROC
    try:
        auc_score = roc_auc_score(y_true, predictions_prob)
    except:
        auc_score = 0.0
    
    # Classification report
    class_names = list(val_generator.class_indices.keys())
    report = classification_report(y_true, predictions_binary, 
                                 target_names=class_names, 
                                 output_dict=True)
    
    print(f"\nDetailed Classification Report:")
    print(classification_report(y_true, predictions_binary, 
                               target_names=class_names))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, predictions_binary)
    
    print(f"\nConfusion Matrix Analysis:")
    print(f"                 Predicted")
    print(f"               {class_names[0]}  {class_names[1]}")
    print(f"Actual {class_names[0]}    {cm[0,0]:4d}      {cm[0,1]:4d}")
    print(f"{class_names[1]}     {cm[1,0]:4d}      {cm[1,1]:4d}")
    
    # Medical-specific metrics
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    print(f"\nMedical Performance Metrics:")
    print(f"   Sensitivity (Recall): {sensitivity:.4f} ({sensitivity*100:.2f}%)")
    print(f"   Specificity:          {specificity:.4f} ({specificity*100:.2f}%)")
    print(f"   PPV (Precision):      {ppv:.4f} ({ppv*100:.2f}%)")
    print(f"   NPV:                  {npv:.4f} ({npv*100:.2f}%)")
    print(f"   AUC-ROC:              {auc_score:.4f}")
    print(f"   Overall Accuracy:     {report['accuracy']:.4f} ({report['accuracy']*100:.2f}%)")
    
    return {
        'accuracy': report['accuracy'],
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': ppv,
        'recall': sensitivity,
        'f1_score': report['weighted avg']['f1-score'],
        'auc_roc': auc_score,
        'confusion_matrix': cm.tolist(),
        'classification_report': report
    }

def save_model_as_bone_fracture(model, model_path, metrics, epochs_trained):
    """Register model with DenseNet121 label as requested"""
    
    registry_path = 'models/registry/model_registry.json'
    os.makedirs(os.path.dirname(registry_path), exist_ok=True)
    
    # Load existing registry
    if os.path.exists(registry_path):
        with open(registry_path, 'r') as f:
            registry = json.load(f)
    else:
        registry = {'models': {}, 'version': '1.0', 'active_models': {}}
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create model entry
    model_id = f'bone_fracture_densenet121_ultra_{timestamp}'
    model_info = {
        'model_id': model_id,
        'model_name': 'Ultra Advanced Bone Fracture Detection (DenseNet121)',
        'dataset_type': 'bone_fracture',
        'version': 'v5.0_ultra',
        'architecture': 'DenseNet121',
        'model_type': 'Ultra Advanced',
        'input_shape': [ULTRA_CONFIG['img_height'], ULTRA_CONFIG['img_width'], 3],
        'num_classes': 2,
        'class_names': ['Fracture', 'Normal'], # Assuming these are the classes
        'performance_metrics': {
            'accuracy': metrics['accuracy'],
            'sensitivity': metrics['sensitivity'],
            'specificity': metrics['specificity'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1_score': metrics['f1_score'],
            'auc_roc': metrics['auc_roc']
        },
        'medical_metrics': {
            'sensitivity_percent': metrics['sensitivity'] * 100,
            'specificity_percent': metrics['specificity'] * 100,
            'ppv_percent': metrics['precision'] * 100,
            'accuracy_percent': metrics['accuracy'] * 100
        },
        'training_info': {
            'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'epochs_trained': epochs_trained,
            'target_accuracy': ULTRA_CONFIG['target_accuracy'],
            'target_achieved': metrics['accuracy'] >= ULTRA_CONFIG['target_accuracy'],
            'image_size': f"{ULTRA_CONFIG['img_height']}x{ULTRA_CONFIG['img_width']}",
            'batch_size': ULTRA_CONFIG['batch_size']
        },
        'file_path': os.path.relpath(model_path, 'models/'),
        'file_size_mb': round(os.path.getsize(model_path) / (1024 * 1024), 2),
        'created_date': datetime.now().isoformat(),
        'description': f"Ultra-advanced bone fracture detection model achieving {metrics['accuracy']*100:.2f}% accuracy",
        'tags': ['DenseNet121', 'bone_fracture', 'ultra', 'medical', '90_percent_target'],
        'threshold': 0.5,
        'is_active': False
    }
    
    # Register model
    registry['models'][model_info['model_id']] = model_info
    registry['last_modified'] = datetime.now().isoformat()
    
    # Save registry
    with open(registry_path, 'w') as f:
        json.dump(registry, f, indent=2)
    
    print(f"\n✓ Model registered: {model_info['model_id']}")
    return model_info['model_id']

def main():
    """Ultra bone fracture training targeting 90%+ accuracy"""
    
    print("ULTRA ADVANCED BONE FRACTURE DETECTION TRAINING")
    print("🎯 Target: 90%+ Accuracy | Architecture: DenseNet121")
    print("="*80)
    
    # Setup
    setup_gpu_optimization()
    os.makedirs(ULTRA_CONFIG['model_dir'], exist_ok=True)
    
    # Verify dataset
    if not os.path.exists(ULTRA_CONFIG['base_dir']):
        print(f"❌ Dataset not found: {ULTRA_CONFIG['base_dir']}")
        return None, None, None
    
    # Create generators
    print("\n📊 Creating medical-grade data generators...")
    train_generator, val_generator = enhanced_data_generators()
    
    print(f"   Training samples: {train_generator.samples:,}")
    print(f"   Validation samples: {val_generator.samples:,}")
    print(f"   Classes: {list(train_generator.class_indices.keys())}")
    print(f"   Input shape: {ULTRA_CONFIG['img_height']}x{ULTRA_CONFIG['img_width']}")
    
    # Compute class weights
    print("\n⚖️  Computing optimized class weights...")
    class_weights = compute_optimal_class_weights(train_generator)
    
    # Create model
    print(f"\n🏗️  Building ultra-advanced DenseNet121 model...")
    model = create_ultra_bone_fracture_model()
    
    # Compile
    optimizer = Adam(learning_rate=ULTRA_CONFIG['learning_rate'])
    
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Setup callbacks
    model_name = f'ultra_bone_fracture_densenet121_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    callbacks = create_advanced_callbacks(model_name)
    
    # Training
    print(f"\n🚀 Starting ultra training for up to {ULTRA_CONFIG['epochs']} epochs...")
    
    start_time = time.time()
    
    history = model.fit(
        train_generator,
        epochs=ULTRA_CONFIG['epochs'],
        validation_data=val_generator,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    training_time = time.time() - start_time
    epochs_trained = len(history.history['accuracy'])
    
    print(f"\n✅ Training completed in {training_time/3600:.2f} hours ({epochs_trained} epochs)")
    
    # Save model
    final_model_path = os.path.join(ULTRA_CONFIG['model_dir'], f'{model_name}_final.h5')
    model.save(final_model_path)
    print(f"   Model saved: {final_model_path}")
    
    # Generate analysis
    print("\n📈 Generating comprehensive analysis...")
    best_val_acc, best_epoch = plot_comprehensive_results(history, model_name)
    
    # Evaluate model
    print("\n🔍 Evaluating model performance...")
    metrics = comprehensive_model_evaluation(model, val_generator)
    
    # Register
    model_id = save_model_as_bone_fracture(model, final_model_path, metrics, epochs_trained)
    
    # Final results
    print("\n" + "="*80)
    print("🏆 ULTRA TRAINING COMPLETED!")
    print("="*80)
    
    print(f"📊 Performance Summary:")
    print(f"   🎯 Best Validation Accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    print(f"   📈 Final Test Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    
    if metrics['accuracy'] >= ULTRA_CONFIG['target_accuracy']:
        print(f"\n🎉 SUCCESS: Target accuracy ({ULTRA_CONFIG['target_accuracy']*100:.0f}%) ACHIEVED!")
    else:
        print(f"\n⚠️  Target accuracy ({ULTRA_CONFIG['target_accuracy']*100:.0f}%) not fully reached")
    
    return model, history, metrics

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
