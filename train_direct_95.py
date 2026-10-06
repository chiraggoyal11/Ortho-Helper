#!/usr/bin/env python3
"""
Direct 95%+ Accuracy Training Script
Run this directly in Python without Streamlit interface
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, applications
import time
from pathlib import Path

# Add utils to path
sys.path.append(str(Path(__file__).parent))
from utils.data_loader import MedicalDataLoader
from utils.model_manager import ModelManager

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

def create_medical_grade_model(input_shape=(384, 384, 3), num_classes=2):
    """Create medical-grade cardiomegaly model for 95%+ accuracy"""
    
    print("🏗️ Building EfficientNet-B4 medical-grade model...")
    
    # Use EfficientNet-B4
    base_model = applications.EfficientNetB4(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # Fine-tuning: unfreeze 70% of layers
    total_layers = len(base_model.layers)
    trainable_layers = int(total_layers * 0.7)
    
    for layer in base_model.layers[:-trainable_layers]:
        layer.trainable = False
    for layer in base_model.layers[-trainable_layers:]:
        layer.trainable = True
    
    print(f"📊 Model info: {total_layers} layers, {trainable_layers} trainable")
    
    # Build architecture
    inputs = keras.Input(shape=input_shape)
    
    # Preprocessing
    x = applications.efficientnet.preprocess_input(inputs)
    
    # Base model
    features = base_model(x, training=True)
    
    # Global pooling
    x = layers.GlobalAveragePooling2D()(features)
    
    # Attention mechanisms
    attention_dim = x.shape[-1]
    
    # Self-attention
    self_att = layers.Dense(attention_dim, activation='tanh', name='self_attention_tanh')(x)
    self_att = layers.Dense(attention_dim, activation='sigmoid', name='self_attention_gate')(self_att)
    x = layers.multiply([x, self_att], name='self_attention_output')
    
    # Channel attention (SE block)
    se_ratio = 16
    se = layers.Dense(attention_dim // se_ratio, activation='relu', name='se_reduce')(x)
    se = layers.Dense(attention_dim, activation='sigmoid', name='se_expand')(se)
    x = layers.multiply([x, se], name='se_output')
    
    # Dense layers with batch norm and dropout
    x = layers.Dense(1024, activation='relu', name='dense_1024')(x)
    x = layers.BatchNormalization(name='bn_1024')(x)
    x = layers.Dropout(0.5, name='dropout_1024')(x)
    
    x = layers.Dense(512, activation='relu', name='dense_512')(x)
    x = layers.BatchNormalization(name='bn_512')(x)
    x = layers.Dropout(0.4, name='dropout_512')(x)
    
    x = layers.Dense(256, activation='relu', name='dense_256')(x)
    x = layers.BatchNormalization(name='bn_256')(x)
    x = layers.Dropout(0.3, name='dropout_256')(x)
    
    # Grad-CAM layer
    x = layers.Dense(128, activation='relu', name='gradcam_target_layer')(x)
    x = layers.Dropout(0.2, name='dropout_gradcam')(x)
    
    # Output layer
    outputs = layers.Dense(num_classes, activation='softmax', name='predictions')(x)
    
    model = keras.Model(inputs, outputs, name='cardiomegaly_medical_grade')
    
    return model

def focal_loss_medical(alpha=0.75, gamma=2.0):
    """Medical-grade focal loss"""
    def loss_fn(y_true, y_pred):
        # Convert to categorical if needed
        if len(tf.shape(y_true)) == 1:
            y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=tf.shape(y_pred)[-1])
        
        # Clip predictions
        epsilon = 1e-7
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        
        # Cross entropy
        ce_loss = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)
        
        # Focal components
        p_t = tf.reduce_sum(y_true * y_pred, axis=-1)
        alpha_factor = alpha
        modulating_factor = tf.pow((1.0 - p_t), gamma)
        
        focal_loss = alpha_factor * modulating_factor * ce_loss
        return tf.reduce_mean(focal_loss)
    
    return loss_fn

def cosine_decay_schedule(epoch, total_epochs=50):
    """Cosine decay with warmup"""
    warmup_epochs = 5
    max_lr = 0.001
    min_lr = 1e-6
    
    if epoch < warmup_epochs:
        return (epoch + 1) * max_lr / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return min_lr + (max_lr - min_lr) * 0.5 * (1 + np.cos(np.pi * progress))

def resize_dataset(images, target_size=(384, 384)):
    """Safely resize images using PIL - handles both arrays and file paths"""
    print(f"🖼️ Resizing {len(images)} images to {target_size}...")
    
    from PIL import Image
    import cv2
    resized = np.zeros((len(images), target_size[0], target_size[1], 3), dtype=np.float32)
    
    for i, img in enumerate(images):
        if i % 1000 == 0:
            print(f"   Processed {i}/{len(images)} images")
        
        try:
            # Check if img is a file path (string)
            if isinstance(img, str):
                # Load image from file path
                if img.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    pil_img = Image.open(img).convert('RGB')
                else:
                    # Try OpenCV for other formats
                    cv_img = cv2.imread(img)
                    if cv_img is not None:
                        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                        pil_img = Image.fromarray(cv_img)
                    else:
                        print(f"   Warning: Could not load image {img}")
                        # Create a blank image as fallback
                        pil_img = Image.new('RGB', target_size, color=(128, 128, 128))
            else:
                # Handle numpy arrays
                if hasattr(img, 'shape'):
                    if len(img.shape) == 3 and img.shape[-1] == 3:
                        # RGB image array
                        img_uint8 = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
                    elif len(img.shape) == 2:
                        # Grayscale - convert to RGB
                        img_uint8 = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
                        img_uint8 = np.stack([img_uint8] * 3, axis=-1)
                    else:
                        # Squeeze and convert
                        img_squeezed = img.squeeze()
                        img_uint8 = (img_squeezed * 255).astype(np.uint8) if img_squeezed.max() <= 1.0 else img_squeezed.astype(np.uint8)
                        if len(img_uint8.shape) == 2:
                            img_uint8 = np.stack([img_uint8] * 3, axis=-1)
                        
                    pil_img = Image.fromarray(img_uint8)
                else:
                    # Unknown format - create blank image
                    print(f"   Warning: Unknown image format at index {i}")
                    pil_img = Image.new('RGB', target_size, color=(128, 128, 128))
            
            # Resize image
            resized_pil = pil_img.resize(target_size, Image.LANCZOS)
            resized[i] = np.array(resized_pil) / 255.0
            
        except Exception as e:
            print(f"   Error processing image {i}: {e}")
            # Create blank image as fallback
            resized[i] = np.ones((target_size[0], target_size[1], 3)) * 0.5
    
    print(f"✅ Resizing completed: {resized.shape}")
    return resized

def train_medical_grade_cardiomegaly():
    """Train medical-grade cardiomegaly model for 95%+ accuracy"""
    
    print("🎯 Starting Medical-Grade Cardiomegaly Training")
    print("=" * 50)
    
    # Step 1: Load dataset
    print("📁 Loading cardiomegaly dataset...")
    data_loader = MedicalDataLoader()
    dataset_info = data_loader.prepare_dataset('cardiomegaly', test_size=0.1, val_size=0.15)
    
    if not dataset_info:
        print("❌ Failed to load dataset!")
        return False
    
    print(f"✅ Dataset loaded:")
    print(f"   Training: {len(dataset_info['X_train'])} samples")
    print(f"   Validation: {len(dataset_info['X_val'])} samples") 
    print(f"   Test: {len(dataset_info['X_test'])} samples")
    print(f"   Classes: {dataset_info['num_classes']}")
    
    # Step 2: Resize to high resolution
    X_train_384 = resize_dataset(dataset_info['X_train'])
    X_val_384 = resize_dataset(dataset_info['X_val'])
    X_test_384 = resize_dataset(dataset_info['X_test'])
    
    # Step 3: Create model
    model = create_medical_grade_model()
    
    print(f"📊 Model parameters: {model.count_params():,}")
    
    # Step 4: Compile model
    print("⚙️ Compiling model with focal loss...")
    
    optimizer = keras.optimizers.AdamW(
        learning_rate=0.001,
        weight_decay=0.01,
        clipnorm=1.0
    )
    
    model.compile(
        optimizer=optimizer,
        loss=focal_loss_medical(alpha=0.75, gamma=2.0),
        metrics=[
            'accuracy',
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc')
        ]
    )
    
    # Step 5: Setup training
    print("📋 Setting up training configuration...")
    
    model_path = "models/cardiomegaly_medical_grade.h5"
    os.makedirs("models", exist_ok=True)
    
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=15,
            restore_best_weights=True,
            verbose=1,
            min_delta=0.001,
            mode='max'
        ),
        
        keras.callbacks.LearningRateScheduler(
            cosine_decay_schedule,
            verbose=1
        ),
        
        keras.callbacks.ModelCheckpoint(
            model_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.3,
            patience=7,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Step 6: Create data generators
    print("📊 Creating data generators...")
    
    # Training augmentation
    train_datagen = keras.preprocessing.image.ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        zoom_range=0.15,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )
    
    val_datagen = keras.preprocessing.image.ImageDataGenerator()
    
    # Convert labels to categorical
    y_train_cat = keras.utils.to_categorical(dataset_info['y_train'], num_classes=2)
    y_val_cat = keras.utils.to_categorical(dataset_info['y_val'], num_classes=2)
    
    train_generator = train_datagen.flow(
        X_train_384, y_train_cat,
        batch_size=8,
        shuffle=True
    )
    
    val_generator = val_datagen.flow(
        X_val_384, y_val_cat,
        batch_size=8,
        shuffle=False
    )
    
    # Step 7: Train model
    print("🚂 Starting training for 95%+ accuracy...")
    print(f"   Target: 95%+ validation accuracy")
    print(f"   Architecture: EfficientNet-B4 with dual attention")
    print(f"   Resolution: 384×384 pixels")
    print(f"   Batch size: 8")
    print(f"   Max epochs: 50")
    
    start_time = time.time()
    
    steps_per_epoch = len(X_train_384) // 8
    validation_steps = len(X_val_384) // 8
    
    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_generator,
        validation_steps=validation_steps,
        epochs=50,
        callbacks=callbacks,
        verbose=1
    )
    
    training_time = time.time() - start_time
    
    # Step 8: Evaluate model
    print("📊 Evaluating final performance...")
    
    test_results = model.evaluate(
        X_test_384,
        keras.utils.to_categorical(dataset_info['y_test'], num_classes=2),
        batch_size=8,
        verbose=0
    )
    
    test_loss = test_results[0]
    test_accuracy = test_results[1]
    test_precision = test_results[2] if len(test_results) > 2 else 0
    test_recall = test_results[3] if len(test_results) > 3 else 0  
    test_auc = test_results[4] if len(test_results) > 4 else 0
    
    # Step 9: Save and register model
    print("💾 Saving model...")
    
    try:
        model.save(model_path)
        print(f"✅ Model saved to: {model_path}")
        
        # Register model
        model_manager = ModelManager()
        model_manager.register_model(
            model_name="cardiomegaly_medical_grade",
            model_path="cardiomegaly_medical_grade.h5",
            accuracy=test_accuracy,
            dataset="cardiomegaly", 
            architecture="EfficientNet-B4-Medical",
            training_samples=len(X_train_384),
            validation_accuracy=max(history.history.get('val_accuracy', [0])),
            notes=f"Medical-grade model: {test_accuracy:.1%} accuracy, {training_time/3600:.1f}h training"
        )
        print("✅ Model registered successfully")
        
    except Exception as e:
        print(f"⚠️ Registration failed: {e}")
    
    # Step 10: Results summary
    print("\n" + "=" * 50)
    print("🎉 TRAINING COMPLETED")
    print("=" * 50)
    
    print(f"🎯 Test Accuracy: {test_accuracy:.1%}")
    print(f"📊 AUC Score: {test_auc:.3f}")
    print(f"⚖️ Precision: {test_precision:.1%}")
    print(f"🔍 Recall: {test_recall:.1%}")
    print(f"⏱️ Training Time: {training_time/3600:.1f} hours")
    
    # Performance analysis
    if test_accuracy >= 0.95:
        print("\n🎉 SUCCESS: 95%+ ACCURACY ACHIEVED!")
        print("🏆 Medical-grade performance reached")
        
        if test_auc >= 0.98:
            print("🌟 Excellent discrimination (AUC ≥ 0.98)")
        if test_precision >= 0.92 and test_recall >= 0.92:
            print("🎯 Balanced precision/recall (both ≥ 92%)")
            
        print("\n🚀 Ready for:")
        print("   • Clinical validation")
        print("   • Regulatory submission") 
        print("   • Production deployment")
        
    elif test_accuracy >= 0.90:
        print("\n📈 EXCELLENT PROGRESS: 90%+ achieved")
        remaining = 0.95 - test_accuracy
        print(f"   Only {remaining:.1%} more needed for 95% target")
        
        print("\n🔧 Next steps:")
        print("   • Train ensemble models")
        print("   • Increase training epochs")
        print("   • Apply test-time augmentation")
        
    else:
        print(f"\n⚠️ BELOW TARGET: {test_accuracy:.1%} accuracy")
        print("\n🔧 Troubleshooting:")
        print("   • Review data quality")
        print("   • Adjust hyperparameters")
        print("   • Try different architecture")
    
    # Training history summary
    best_epoch = np.argmax(history.history['val_accuracy']) + 1
    best_val_acc = max(history.history['val_accuracy'])
    final_val_acc = history.history['val_accuracy'][-1]
    
    print(f"\n📈 Training Summary:")
    print(f"   Best epoch: {best_epoch}")
    print(f"   Best val accuracy: {best_val_acc:.1%}")
    print(f"   Final val accuracy: {final_val_acc:.1%}")
    print(f"   Total epochs: {len(history.history['accuracy'])}")
    
    return test_accuracy >= 0.95

if __name__ == "__main__":
    print("🏥 Medical-Grade Cardiomegaly AI Training")
    print("🎯 Target: 95%+ Accuracy")
    print("🚀 Starting training...\n")
    
    success = train_medical_grade_cardiomegaly()
    
    if success:
        print("\n✅ MISSION ACCOMPLISHED: 95%+ accuracy achieved!")
    else:
        print("\n📊 Training completed. Review results above for next steps.")