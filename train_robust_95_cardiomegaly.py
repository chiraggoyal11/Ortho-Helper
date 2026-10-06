#!/usr/bin/env python3
"""
Robust 95%+ Accuracy Cardiomegaly Training
Simplified approach without complex augmentations that cause dimension issues
"""

import streamlit as st
import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, applications
import numpy as np
import time
from datetime import datetime
from utils.data_loader import MedicalDataLoader
from utils.model_manager import ModelManager
import matplotlib.pyplot as plt

def create_robust_95_model(input_shape=(384, 384, 3), num_classes=2):
    """
    Create robust model architecture for 95%+ accuracy
    Avoids complex augmentations that cause dimension issues
    """
    # Use EfficientNet-B4 for superior performance
    base_model = applications.EfficientNetB4(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # Unfreeze 75% of layers for fine-tuning
    total_layers = len(base_model.layers)
    trainable_layers = int(total_layers * 0.75)
    
    for layer in base_model.layers[:-trainable_layers]:
        layer.trainable = False
    for layer in base_model.layers[-trainable_layers:]:
        layer.trainable = True
    
    # Build model without complex augmentations in the model
    inputs = keras.Input(shape=input_shape)
    
    # Simple preprocessing only
    x = applications.efficientnet.preprocess_input(inputs)
    
    # Extract features
    features = base_model(x, training=True)
    
    # Global pooling
    x = layers.GlobalAveragePooling2D()(features)
    
    # Attention mechanism
    attention_dim = x.shape[-1]
    
    # Self-attention
    self_att = layers.Dense(attention_dim, activation='tanh')(x)
    self_att = layers.Dense(attention_dim, activation='sigmoid')(self_att)
    x = layers.multiply([x, self_att])
    
    # Channel attention (Squeeze-and-Excitation)
    se_ratio = 16
    se = layers.Dense(attention_dim // se_ratio, activation='relu')(x)
    se = layers.Dense(attention_dim, activation='sigmoid')(se)
    x = layers.multiply([x, se])
    
    # Advanced dense layers
    x1 = layers.Dense(1024, activation='relu')(x)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.Dropout(0.5)(x1)
    
    x2 = layers.Dense(512, activation='relu')(x1)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.Dropout(0.4)(x2)
    
    x3 = layers.Dense(256, activation='relu')(x2)
    x3 = layers.BatchNormalization()(x3)
    x3 = layers.Dropout(0.3)(x3)
    
    # Grad-CAM layer
    gradcam_layer = layers.Dense(128, activation='relu', name='gradcam_target_layer')(x3)
    gradcam_layer = layers.Dropout(0.2)(gradcam_layer)
    
    # Output layer
    outputs = layers.Dense(num_classes, activation='softmax')(gradcam_layer)
    
    model = keras.Model(inputs, outputs, name='robust_cardiomegaly_95_model')
    return model

def focal_loss_robust(alpha=0.75, gamma=2.0):
    """Robust focal loss implementation"""
    def focal_loss_fn(y_true, y_pred):
        # Ensure y_true is in correct format
        if len(tf.shape(y_true)) == 1:
            y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=tf.shape(y_pred)[-1])
        
        # Clip predictions to prevent log(0)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        
        # Calculate cross entropy
        ce_loss = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)
        
        # Calculate p_t
        p_t = tf.reduce_sum(y_true * y_pred, axis=-1)
        
        # Calculate focal loss
        alpha_factor = alpha
        modulating_factor = tf.pow((1.0 - p_t), gamma)
        
        return tf.reduce_mean(alpha_factor * modulating_factor * ce_loss)
    
    return focal_loss_fn

def create_robust_data_generators(X_train, y_train, X_val, y_val, batch_size=8):
    """Create robust data generators with ImageDataGenerator"""
    
    # Data augmentation for training
    train_datagen = keras.preprocessing.image.ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )
    
    # No augmentation for validation
    val_datagen = keras.preprocessing.image.ImageDataGenerator()
    
    # Create generators
    train_generator = train_datagen.flow(
        X_train, 
        keras.utils.to_categorical(y_train, num_classes=2),
        batch_size=batch_size,
        shuffle=True
    )
    
    val_generator = val_datagen.flow(
        X_val,
        keras.utils.to_categorical(y_val, num_classes=2),
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_generator, val_generator

def cosine_schedule(epoch, total_epochs=60):
    """Cosine annealing schedule"""
    if epoch < 5:  # Warmup
        return (epoch + 1) * 0.001 / 5
    else:
        progress = (epoch - 5) / (total_epochs - 5)
        return 1e-5 + (0.001 - 1e-5) * 0.5 * (1 + np.cos(np.pi * progress))

def train_robust_95_model():
    """Main training function with robust implementation"""
    
    st.title("🎯 Robust Cardiomegaly 95%+ Training")
    
    st.info("""
    🛡️ **Robust Training Configuration:**
    • **Architecture**: EfficientNet-B4 with dual attention
    • **Input Resolution**: 384×384 pixels
    • **Stable Augmentation**: Standard ImageDataGenerator
    • **Focal Loss**: For hard example mining
    • **Target**: 95%+ validation accuracy
    """)
    
    # Configuration display
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("🏗️ Base Model", "EfficientNet-B4")
        st.metric("📏 Resolution", "384×384")
    
    with col2:
        st.metric("🎯 Target", "95%+")
        st.metric("⏱️ Epochs", "60")
    
    with col3:
        st.metric("🔬 Loss", "Focal Loss")
        st.metric("💾 Batch", "8")
    
    if st.button("🚀 Start Robust 95%+ Training", type="primary"):
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Step 1: Load data
            status_text.text("📁 Loading cardiomegaly dataset...")
            progress_bar.progress(15)
            
            data_loader = MedicalDataLoader()
            dataset_info = data_loader.prepare_dataset('cardiomegaly', test_size=0.1, val_size=0.15)
            
            if not dataset_info:
                st.error("❌ Failed to load dataset!")
                return
            
            st.success(f"✅ Loaded {len(dataset_info['X_train'])} training samples")
            
            # Step 2: Resize images safely
            status_text.text("🖼️ Resizing images to 384×384...")
            progress_bar.progress(25)
            
            # Resize using numpy/PIL to avoid TF dimension issues
            def resize_images(images, target_size=(384, 384)):
                resized = np.zeros((len(images), target_size[0], target_size[1], 3), dtype=np.float32)
                for i, img in enumerate(images):
                    # Convert to PIL and resize
                    from PIL import Image
                    if img.shape[-1] == 3:
                        pil_img = Image.fromarray((img * 255).astype(np.uint8))
                    else:
                        pil_img = Image.fromarray((img.squeeze() * 255).astype(np.uint8))
                        pil_img = pil_img.convert('RGB')
                    
                    resized_pil = pil_img.resize(target_size)
                    resized[i] = np.array(resized_pil) / 255.0
                return resized
            
            X_train_384 = resize_images(dataset_info['X_train'])
            X_val_384 = resize_images(dataset_info['X_val'])
            X_test_384 = resize_images(dataset_info['X_test'])
            
            # Step 3: Create model
            status_text.text("🏗️ Building EfficientNet-B4 model...")
            progress_bar.progress(35)
            
            model = create_robust_95_model(input_shape=(384, 384, 3), num_classes=2)
            
            # Step 4: Compile model
            status_text.text("⚙️ Compiling with focal loss...")
            progress_bar.progress(45)
            
            optimizer = keras.optimizers.AdamW(
                learning_rate=0.001,
                weight_decay=0.01,
                clipnorm=1.0
            )
            
            model.compile(
                optimizer=optimizer,
                loss=focal_loss_robust(alpha=0.75, gamma=2.0),
                metrics=[
                    'accuracy',
                    keras.metrics.Precision(name='precision'),
                    keras.metrics.Recall(name='recall'),
                    keras.metrics.AUC(name='auc')
                ]
            )
            
            # Step 5: Setup training
            status_text.text("📋 Setting up training pipeline...")
            progress_bar.progress(55)
            
            # Create callbacks
            model_path = "models/robust_cardiomegaly_95_model.h5"
            
            callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor='val_accuracy',
                    patience=20,
                    restore_best_weights=True,
                    verbose=1,
                    min_delta=0.001,
                    mode='max'
                ),
                
                keras.callbacks.LearningRateScheduler(
                    cosine_schedule,
                    verbose=0
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
                    factor=0.5,
                    patience=8,
                    min_lr=1e-7,
                    verbose=1
                )
            ]
            
            # Create data generators
            train_gen, val_gen = create_robust_data_generators(
                X_train_384, dataset_info['y_train'],
                X_val_384, dataset_info['y_val'],
                batch_size=8
            )
            
            # Step 6: Train model
            status_text.text("🚂 Training for 95%+ accuracy...")
            progress_bar.progress(65)
            
            st.write("**🚂 Starting robust training...**")
            
            start_time = time.time()
            
            # Calculate steps
            steps_per_epoch = len(X_train_384) // 8
            validation_steps = len(X_val_384) // 8
            
            history = model.fit(
                train_gen,
                steps_per_epoch=steps_per_epoch,
                validation_data=val_gen,
                validation_steps=validation_steps,
                epochs=60,
                callbacks=callbacks,
                verbose=1
            )
            
            training_time = time.time() - start_time
            
            # Step 7: Evaluate
            status_text.text("📊 Evaluating performance...")
            progress_bar.progress(85)
            
            # Test evaluation
            test_results = model.evaluate(
                X_test_384, 
                keras.utils.to_categorical(dataset_info['y_test'], num_classes=2),
                batch_size=8,
                verbose=0
            )
            
            test_accuracy = test_results[1]  # accuracy is second metric
            test_precision = test_results[2] if len(test_results) > 2 else 0
            test_recall = test_results[3] if len(test_results) > 3 else 0
            test_auc = test_results[4] if len(test_results) > 4 else 0
            
            # Step 8: Save model
            status_text.text("💾 Saving model...")
            progress_bar.progress(95)
            
            try:
                model.save(model_path)
                
                # Update registry
                model_manager = ModelManager()
                model_manager.register_model(
                    model_name="robust_cardiomegaly_95",
                    model_path="robust_cardiomegaly_95_model.h5",
                    accuracy=test_accuracy,
                    dataset="cardiomegaly",
                    architecture="EfficientNet-B4-Robust",
                    training_samples=len(X_train_384),
                    validation_accuracy=max(history.history.get('val_accuracy', [0])),
                    notes=f"Robust 95%+ accuracy model - {test_accuracy:.1%} test accuracy"
                )
            except Exception as e:
                st.warning(f"Registry update failed: {e}")
            
            # Complete
            progress_bar.progress(100)
            status_text.text("✅ Training completed!")
            
            # Display results
            st.subheader("🎉 Training Results")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "🎯 Test Accuracy", 
                    f"{test_accuracy:.1%}",
                    delta=f"+{(test_accuracy - 0.758):.1%}" if test_accuracy > 0.758 else None
                )
            
            with col2:
                st.metric("📊 AUC", f"{test_auc:.3f}")
            
            with col3:
                st.metric("⚖️ Precision", f"{test_precision:.1%}")
            
            with col4:
                st.metric("🔍 Recall", f"{test_recall:.1%}")
            
            # Success analysis
            if test_accuracy >= 0.95:
                st.success("🎉 **95%+ ACCURACY ACHIEVED!** Medical-grade performance!")
                st.balloons()
                
                st.write("**🏆 Success Metrics:**")
                st.write(f"• ✅ Test accuracy: {test_accuracy:.1%}")
                st.write(f"• ⏱️ Training time: {training_time/3600:.1f} hours")
                st.write(f"• 🎯 AUC: {test_auc:.3f}")
                st.write(f"• 💾 Model: `{model_path}`")
                
                # Confidence analysis
                if test_auc >= 0.98:
                    st.success("🌟 **Excellent discrimination capability (AUC ≥ 0.98)**")
                if test_precision >= 0.92 and test_recall >= 0.92:
                    st.success("🎯 **Balanced precision and recall (both ≥ 92%)**")
                    
            elif test_accuracy >= 0.90:
                st.info("📈 **Excellent progress! 90%+ achieved.**")
                remaining = 0.95 - test_accuracy
                st.write(f"**Only {remaining:.1%} more to reach 95% target**")
                
                st.write("**🚀 Quick improvements:**")
                st.write("• Train for more epochs (increase patience)")
                st.write("• Use ensemble of multiple models")
                st.write("• Apply test-time augmentation")
                
            else:
                st.warning(f"⚠️ **{test_accuracy:.1%} accuracy - Below 90%**")
                st.write("**🔧 Troubleshooting steps:**")
                st.write("• Check data quality and balance")
                st.write("• Reduce learning rate")
                st.write("• Increase regularization")
                st.write("• Try different architecture")
            
            # Training progress visualization
            if len(history.history['accuracy']) > 1:
                st.subheader("📈 Training Progress")
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                
                epochs_range = range(1, len(history.history['accuracy']) + 1)
                
                # Accuracy plot
                ax1.plot(epochs_range, history.history['accuracy'], 'bo-', label='Training', alpha=0.7)
                ax1.plot(epochs_range, history.history['val_accuracy'], 'ro-', label='Validation', alpha=0.7)
                ax1.set_title('Model Accuracy')
                ax1.set_xlabel('Epochs')
                ax1.set_ylabel('Accuracy')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # Loss plot
                ax2.plot(epochs_range, history.history['loss'], 'bo-', label='Training', alpha=0.7)
                ax2.plot(epochs_range, history.history['val_loss'], 'ro-', label='Validation', alpha=0.7)
                ax2.set_title('Model Loss')
                ax2.set_xlabel('Epochs')
                ax2.set_ylabel('Loss')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Training summary
                best_epoch = np.argmax(history.history['val_accuracy']) + 1
                best_val_acc = max(history.history['val_accuracy'])
                
                st.info(f"📊 **Best epoch**: {best_epoch} with {best_val_acc:.1%} validation accuracy")
            
            # Next steps
            st.subheader("🎯 Next Steps")
            
            if test_accuracy >= 0.95:
                st.write("""
                **🚀 Ready for Production:**
                • Deploy model for clinical testing
                • Implement uncertainty quantification
                • Set up monitoring pipeline
                • Prepare regulatory documentation
                """)
            else:
                st.write("""
                **📈 Further Optimization:**
                • Ensemble training (train 3-5 models)
                • Increase dataset size
                • Try progressive resizing
                • Implement test-time augmentation
                """)
            
        except Exception as e:
            st.error(f"❌ Training failed: {str(e)}")
            st.write("**Debug Information:**")
            st.code(str(e))
            
            # Additional debugging info
            st.write("**Troubleshooting:**")
            st.write("• Check TensorFlow version compatibility")
            st.write("• Ensure sufficient GPU memory")
            st.write("• Verify dataset integrity")
            st.write("• Try reducing batch size to 4")

if __name__ == "__main__":
    train_robust_95_model()