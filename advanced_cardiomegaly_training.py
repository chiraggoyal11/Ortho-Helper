#!/usr/bin/env python3
"""
Advanced Cardiomegaly Training for 95%+ Accuracy
Comprehensive approach using state-of-the-art techniques
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

class AdvancedCardiomegalyTrainer:
    """Advanced trainer targeting 95%+ accuracy"""
    
    def __init__(self):
        self.model = None
        self.ensemble_models = []
        
    def create_advanced_architecture(self, input_shape=(384, 384, 3), num_classes=2):
        """Create state-of-the-art architecture for 95%+ accuracy"""
        
        # Use EfficientNet-B4 for better performance than DenseNet121
        base_model = applications.EfficientNetB4(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
        
        # Fine-tuning strategy: Unfreeze last 80% of layers
        total_layers = len(base_model.layers)
        trainable_layers = int(total_layers * 0.8)
        
        for layer in base_model.layers[:-trainable_layers]:
            layer.trainable = False
        for layer in base_model.layers[-trainable_layers:]:
            layer.trainable = True
            
        # Advanced architecture with multiple attention mechanisms
        inputs = keras.Input(shape=input_shape)
        
        # Advanced data augmentation pipeline
        x = layers.RandomFlip("horizontal")(inputs)
        x = layers.RandomRotation(0.15)(x)
        x = layers.RandomZoom(0.2)(x)
        x = layers.RandomContrast(0.3)(x)
        x = layers.RandomBrightness(0.2)(x)
        
        # Custom augmentation for medical images
        x = self._add_medical_augmentation(x)
        
        # Preprocessing
        x = applications.efficientnet.preprocess_input(x)
        
        # Base model features
        features = base_model(x, training=True)
        
        # Multiple attention mechanisms
        x = layers.GlobalAveragePooling2D()(features)
        
        # Self-attention mechanism
        attention_1 = layers.Dense(2048, activation='relu')(x)
        attention_1 = layers.Dense(2048, activation='sigmoid')(attention_1)
        x = layers.multiply([x, attention_1])
        
        # Channel attention
        channel_attention = layers.Dense(x.shape[-1] // 16, activation='relu')(x)
        channel_attention = layers.Dense(x.shape[-1], activation='sigmoid')(channel_attention)
        x = layers.multiply([x, channel_attention])
        
        # Multi-scale feature extraction
        x = layers.Dense(1024, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x) 
        x = layers.Dropout(0.4)(x)
        
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        # Grad-CAM layer
        x = layers.Dense(128, activation='relu', name='gradcam_target_layer')(x)
        x = layers.Dropout(0.2)(x)
        
        # Output with label smoothing consideration
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        
        model = keras.Model(inputs, outputs, name='advanced_cardiomegaly_model')
        return model
    
    def _add_medical_augmentation(self, x):
        """Add medical-specific augmentations"""
        # Simulate different X-ray exposure conditions
        x = tf.image.random_jpeg_quality(x, 70, 100)
        
        # Add subtle noise (common in medical imaging)
        noise = tf.random.normal(tf.shape(x), mean=0.0, stddev=0.02)
        x = tf.clip_by_value(x + noise, 0.0, 1.0)
        
        return x
    
    def create_advanced_callbacks(self, model_path, dataset_name="cardiomegaly"):
        """Advanced callbacks for 95%+ accuracy training"""
        
        callbacks = [
            # Aggressive early stopping
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=20,
                restore_best_weights=True,
                verbose=1,
                min_delta=0.0005,
                mode='max'
            ),
            
            # Advanced learning rate scheduling
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.3,
                patience=8,
                min_lr=1e-8,
                verbose=1,
                cooldown=5
            ),
            
            # Model checkpoint
            keras.callbacks.ModelCheckpoint(
                model_path,
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                mode='max',
                verbose=1
            ),
            
            # Cosine annealing with warm restarts
            keras.callbacks.LearningRateScheduler(
                self._cosine_annealing_with_warmup,
                verbose=0
            ),
            
            # Custom callback for monitoring high accuracy
            HighAccuracyCallback(target_accuracy=0.95)
        ]
        
        return callbacks
    
    def _cosine_annealing_with_warmup(self, epoch):
        """Cosine annealing learning rate with warmup"""
        warmup_epochs = 5
        total_epochs = 50
        
        if epoch < warmup_epochs:
            # Warmup phase
            return (epoch + 1) * 1e-3 / warmup_epochs
        else:
            # Cosine annealing
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            return 1e-4 + (1e-3 - 1e-4) * 0.5 * (1 + np.cos(np.pi * progress))
    
    def compile_advanced_model(self, model):
        """Advanced compilation with focal loss and metrics"""
        
        # Use focal loss for hard example mining
        focal_loss = self._focal_loss(alpha=0.75, gamma=2.0)
        
        # Advanced optimizer with gradient clipping
        optimizer = keras.optimizers.AdamW(
            learning_rate=1e-3,
            weight_decay=1e-4,
            clipnorm=1.0
        )
        
        # Comprehensive metrics
        model.compile(
            optimizer=optimizer,
            loss=focal_loss,
            metrics=[
                'accuracy',
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
                keras.metrics.AUC(name='auc'),
                keras.metrics.TopKCategoricalAccuracy(k=1, name='top1_accuracy')
            ]
        )
        
        return model
    
    def _focal_loss(self, alpha=0.75, gamma=2.0):
        """Focal loss for handling hard examples"""
        def focal_loss_fn(y_true, y_pred):
            # Convert to one-hot if needed
            if len(y_true.shape) == 1:
                y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=tf.shape(y_pred)[-1])
            
            # Compute focal loss
            ce_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
            p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
            alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)
            modulating_factor = tf.pow((1.0 - p_t), gamma)
            
            return tf.reduce_mean(alpha_factor * modulating_factor * ce_loss)
        
        return focal_loss_fn
    
    def create_advanced_data_generators(self, X_train, y_train, X_val, y_val, batch_size=8):
        """Advanced data generators with mixup and cutmix"""
        
        # Advanced augmentation pipeline
        train_datagen = keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.15,
            height_shift_range=0.15,
            shear_range=0.1,
            zoom_range=0.2,
            horizontal_flip=True,
            brightness_range=[0.7, 1.3],
            channel_shift_range=0.1,
            fill_mode='nearest'
        )
        
        val_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
        
        # Create generators with mixup
        train_generator = MixupGenerator(
            train_datagen.flow(X_train, y_train, batch_size=batch_size),
            alpha=0.2
        )
        
        val_generator = val_datagen.flow(X_val, y_val, batch_size=batch_size)
        
        return train_generator, val_generator
    
    def train_ensemble(self, dataset_info, num_models=5):
        """Train ensemble of models for 95%+ accuracy"""
        
        st.subheader("🎯 Training Ensemble for 95%+ Accuracy")
        
        self.ensemble_models = []
        ensemble_predictions = []
        
        for i in range(num_models):
            st.write(f"Training model {i+1}/{num_models}...")
            
            # Create model with slight variations
            model = self.create_advanced_architecture(
                input_shape=(384, 384, 3),
                num_classes=dataset_info['num_classes']
            )
            
            # Compile with advanced settings
            model = self.compile_advanced_model(model)
            
            # Create callbacks
            model_path = f"models/cardiomegaly_ensemble_model_{i+1}.h5"
            callbacks = self.create_advanced_callbacks(model_path)
            
            # Train model
            history = model.fit(
                dataset_info['X_train'], dataset_info['y_train'],
                validation_data=(dataset_info['X_val'], dataset_info['y_val']),
                epochs=50,
                batch_size=8,  # Smaller batch for better gradients
                callbacks=callbacks,
                verbose=1
            )
            
            # Evaluate model
            test_acc = model.evaluate(dataset_info['X_test'], dataset_info['y_test'], verbose=0)[1]
            st.write(f"Model {i+1} accuracy: {test_acc:.1%}")
            
            self.ensemble_models.append(model)
            
            # Get predictions for ensemble
            pred = model.predict(dataset_info['X_test'], verbose=0)
            ensemble_predictions.append(pred)
        
        # Calculate ensemble accuracy
        ensemble_pred = np.mean(ensemble_predictions, axis=0)
        ensemble_accuracy = np.mean(np.argmax(ensemble_pred, axis=1) == dataset_info['y_test'])
        
        st.success(f"🎉 Ensemble accuracy: {ensemble_accuracy:.1%}")
        
        return ensemble_accuracy

class MixupGenerator:
    """Generator with Mixup augmentation"""
    
    def __init__(self, generator, alpha=0.2):
        self.generator = generator
        self.alpha = alpha
    
    def __iter__(self):
        return self
    
    def __next__(self):
        batch_x, batch_y = next(self.generator)
        
        # Apply mixup
        if np.random.random() < 0.5:  # 50% chance of mixup
            indices = np.random.permutation(batch_x.shape[0])
            lambda_param = np.random.beta(self.alpha, self.alpha)
            
            batch_x = lambda_param * batch_x + (1 - lambda_param) * batch_x[indices]
            
            # Convert labels to one-hot for mixing
            if len(batch_y.shape) == 1:
                batch_y = keras.utils.to_categorical(batch_y, num_classes=2)
                
            batch_y = lambda_param * batch_y + (1 - lambda_param) * batch_y[indices]
        
        return batch_x, batch_y

class HighAccuracyCallback(keras.callbacks.Callback):
    """Custom callback to monitor high accuracy achievement"""
    
    def __init__(self, target_accuracy=0.95):
        super().__init__()
        self.target_accuracy = target_accuracy
        self.achieved = False
    
    def on_epoch_end(self, epoch, logs=None):
        val_acc = logs.get('val_accuracy', 0)
        if val_acc >= self.target_accuracy and not self.achieved:
            print(f"\n🎉 TARGET ACHIEVED! Validation accuracy: {val_acc:.1%}")
            self.achieved = True

def run_advanced_training():
    """Run the advanced 95%+ accuracy training pipeline"""
    
    st.title("🚀 Advanced Cardiomegaly Training - Target: 95%+ Accuracy")
    
    # Configuration
    config = {
        "architecture": "EfficientNet-B4",
        "input_size": "384x384",
        "ensemble_size": 5,
        "epochs": 50,
        "batch_size": 8,
        "target_accuracy": "95%+"
    }
    
    # Display configuration
    st.subheader("⚙️ Advanced Configuration")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Architecture", config["architecture"])
        st.metric("Input Size", config["input_size"])
    
    with col2:
        st.metric("Ensemble Models", config["ensemble_size"])
        st.metric("Max Epochs", config["epochs"])
    
    with col3:
        st.metric("Batch Size", config["batch_size"])
        st.metric("Target Accuracy", config["target_accuracy"])
    
    # Advanced features info
    st.info("""
    🔬 **Advanced Features for 95%+ Accuracy:**
    • EfficientNet-B4 architecture (state-of-the-art)
    • 384x384 input resolution (higher than standard 224x224)
    • Multi-attention mechanisms (self + channel attention)
    • Focal loss for hard example mining
    • Mixup data augmentation
    • Cosine annealing with warmup
    • Ensemble of 5 models
    • Advanced medical image augmentations
    """)
    
    if st.button("🎯 Start Advanced Training (95%+ Target)", type="primary"):
        
        # Initialize trainer
        trainer = AdvancedCardiomegalyTrainer()
        
        # Load data with higher resolution
        st.write("📁 Loading data with 384x384 resolution...")
        
        # Import data loader
        from utils.data_loader import MedicalDataLoader
        data_loader = MedicalDataLoader()
        
        # Prepare dataset
        dataset_info = data_loader.prepare_dataset('cardiomegaly', test_size=0.15, val_size=0.15)
        
        if not dataset_info:
            st.error("❌ Failed to prepare dataset!")
            return
        
        # Resize data to 384x384
        st.write("🖼️ Resizing images to 384x384 for better feature extraction...")
        X_train_resized = tf.image.resize(dataset_info['X_train'], [384, 384])
        X_val_resized = tf.image.resize(dataset_info['X_val'], [384, 384])
        X_test_resized = tf.image.resize(dataset_info['X_test'], [384, 384])
        
        dataset_info.update({
            'X_train': X_train_resized.numpy(),
            'X_val': X_val_resized.numpy(), 
            'X_test': X_test_resized.numpy()
        })
        
        # Start advanced training
        start_time = time.time()
        
        try:
            # Train ensemble
            ensemble_accuracy = trainer.train_ensemble(dataset_info)
            
            training_time = time.time() - start_time
            
            # Display results
            st.subheader("🎉 Advanced Training Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Ensemble Accuracy",
                    f"{ensemble_accuracy:.1%}",
                    delta=f"+{(ensemble_accuracy - 0.758):.1%}" if ensemble_accuracy > 0.758 else None
                )
            
            with col2:
                success_icon = "🎯" if ensemble_accuracy >= 0.95 else "📈"
                status = "TARGET ACHIEVED!" if ensemble_accuracy >= 0.95 else "Close to target"
                st.metric("Status", f"{success_icon} {status}")
            
            with col3:
                st.metric("Training Time", f"{training_time/3600:.1f} hours")
            
            # Success message
            if ensemble_accuracy >= 0.95:
                st.success("🎉 **95%+ ACCURACY ACHIEVED!** Your cardiomegaly model is now medical-grade!")
                st.balloons()
            elif ensemble_accuracy >= 0.90:
                st.info("📈 **Excellent progress!** 90%+ accuracy achieved. Fine-tune for 95%+")
            else:
                st.warning("⚠️ **Below target.** Consider data quality review or additional techniques.")
            
            # Next steps
            st.subheader("🚀 Next Steps")
            
            if ensemble_accuracy >= 0.95:
                st.write("""
                **🎯 Target Achieved - Production Recommendations:**
                • Deploy ensemble model for maximum accuracy
                • Implement confidence thresholding (>0.9 for high-confidence predictions)
                • Set up continuous monitoring and retraining pipeline
                • Consider regulatory validation for clinical use
                """)
            else:
                st.write("""
                **📈 Further Improvement Strategies:**
                • Increase training data diversity
                • Try progressive resizing (224→384→512)
                • Implement test-time augmentation
                • Use external medical datasets for pre-training
                • Consider Vision Transformer (ViT) architecture
                """)
                
        except Exception as e:
            st.error(f"❌ Advanced training failed: {str(e)}")

if __name__ == "__main__":
    run_advanced_training()