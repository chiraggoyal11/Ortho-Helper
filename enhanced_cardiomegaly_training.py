#!/usr/bin/env python3
"""
Enhanced Cardiomegaly Training Strategy
Optimized for better accuracy and confidence in cardiomegaly detection
"""

import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import json
from pathlib import Path
import matplotlib.pyplot as plt

class EnhancedCardiomegalyTrainer:
    """Enhanced training pipeline specifically for cardiomegaly detection"""
    
    def __init__(self):
        self.model = None
        self.history = None
        
    def create_enhanced_model(self, input_shape=(224, 224, 3), num_classes=2):
        """Create an enhanced model with better architecture for cardiomegaly detection"""
        
        # Use DenseNet121 as base but with modifications
        base_model = keras.applications.DenseNet121(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
        
        # Unfreeze more layers for better feature learning
        # Freeze only the first 80 layers instead of all
        for layer in base_model.layers[:80]:
            layer.trainable = False
        for layer in base_model.layers[80:]:
            layer.trainable = True
            
        # Enhanced architecture
        inputs = keras.Input(shape=input_shape)
        
        # Data augmentation layer
        x = layers.RandomFlip("horizontal")(inputs)
        x = layers.RandomRotation(0.1)(x)
        x = layers.RandomZoom(0.1)(x)
        x = layers.RandomContrast(0.2)(x)  # Important for X-ray images
        
        # Preprocessing
        x = keras.applications.densenet.preprocess_input(x)
        
        # Base model
        x = base_model(x, training=True)
        
        # Enhanced feature extraction
        x = layers.GlobalAveragePooling2D()(x)
        
        # Add attention mechanism
        attention_weights = layers.Dense(1024, activation='relu')(x)
        attention_weights = layers.Dense(1024, activation='sigmoid')(attention_weights)
        x = layers.multiply([x, attention_weights])
        
        # Regularized dense layers
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        # Grad-CAM compatible layer
        x = layers.Dense(128, activation='relu', name='gradcam_target_layer')(x)
        x = layers.Dropout(0.1)(x)
        
        # Output layer with better initialization
        outputs = layers.Dense(
            num_classes, 
            activation='softmax',
            kernel_initializer='he_normal',
            name='predictions'
        )(x)
        
        model = keras.Model(inputs, outputs, name='enhanced_cardiomegaly_model')
        return model
    
    def get_enhanced_callbacks(self, model_path):
        """Get enhanced callbacks for better training"""
        
        callbacks = [
            # Early stopping with patience
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=8,  # Increased patience
                restore_best_weights=True,
                verbose=1,
                mode='max'
            ),
            
            # Learning rate reduction
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=4,
                min_lr=1e-7,
                verbose=1
            ),
            
            # Model checkpointing
            keras.callbacks.ModelCheckpoint(
                model_path,
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                mode='max',
                verbose=1
            ),
            
            # Learning rate scheduling
            keras.callbacks.LearningRateScheduler(
                lambda epoch: 1e-3 * 0.9 ** epoch,
                verbose=0
            )
        ]
        
        return callbacks
    
    def compile_enhanced_model(self, model):
        """Compile model with enhanced settings"""
        
        # Use different optimizers based on training phase
        optimizer = keras.optimizers.Adam(
            learning_rate=1e-3,  # Start with higher learning rate
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7
        )
        
        # Weighted loss for imbalanced classes (if needed)
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
                keras.metrics.AUC(name='auc')
            ]
        )
        
        return model
    
    def get_training_recommendations(self):
        """Get training recommendations for better performance"""
        
        recommendations = {
            "data_improvements": [
                "🔍 Ensure high-quality, diverse cardiomegaly images",
                "⚖️ Check class balance - consider weighted loss if imbalanced",
                "🖼️ Review image preprocessing - ensure consistent intensity normalization",
                "📐 Consider different input sizes (256x256 or 384x384)",
                "🔄 Use more aggressive data augmentation for medical images"
            ],
            
            "training_strategies": [
                "📚 Use transfer learning from medical imaging pretrained models",
                "🎯 Implement focal loss for hard example mining",
                "📊 Use stratified sampling to ensure balanced batches",
                "⏰ Train for more epochs with early stopping",
                "🔄 Implement cyclical learning rates"
            ],
            
            "model_enhancements": [
                "🧠 Add attention mechanisms for better feature focus",
                "🔗 Use ensemble methods (multiple models)",
                "📈 Implement progressive resizing training",
                "🎛️ Fine-tune more layers of the backbone",
                "🔍 Add custom regularization for medical images"
            ],
            
            "validation_improvements": [
                "📝 Use stratified K-fold cross-validation",
                "🎯 Monitor additional metrics (AUC, F1-score)",
                "📊 Analyze confusion matrix for class-specific issues",
                "🖼️ Visualize Grad-CAM to ensure proper feature learning",
                "📈 Track confidence distribution across predictions"
            ]
        }
        
        return recommendations

def analyze_current_performance():
    """Analyze current cardiomegaly model performance"""
    
    st.subheader("📊 Current Cardiomegaly Model Analysis")
    
    # Load training history
    history_path = Path("models/cardiomegaly_DenseNet121_history.json")
    if history_path.exists():
        with open(history_path, 'r') as f:
            history = json.load(f)
        
        # Display training progression
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Final Training Accuracy", f"{history['accuracy'][-1]:.1%}")
            st.metric("Final Validation Accuracy", f"{history['val_accuracy'][-1]:.1%}")
            
        with col2:
            st.metric("Final Training Loss", f"{history['loss'][-1]:.4f}")
            st.metric("Final Validation Loss", f"{history['val_loss'][-1]:.4f}")
        
        # Plot training curves
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Accuracy plot
        ax1.plot(history['accuracy'], label='Training Accuracy', marker='o')
        ax1.plot(history['val_accuracy'], label='Validation Accuracy', marker='s')
        ax1.set_title('Model Accuracy')
        ax1.set_ylabel('Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.legend()
        ax1.grid(True)
        
        # Loss plot
        ax2.plot(history['loss'], label='Training Loss', marker='o')
        ax2.plot(history['val_loss'], label='Validation Loss', marker='s')
        ax2.set_title('Model Loss')
        ax2.set_ylabel('Loss')
        ax2.set_xlabel('Epoch')
        ax2.legend()
        ax2.grid(True)
        
        st.pyplot(fig)
        
        # Performance analysis
        st.subheader("🔍 Performance Analysis")
        
        # Check for overfitting
        final_train_acc = history['accuracy'][-1]
        final_val_acc = history['val_accuracy'][-1]
        acc_gap = final_train_acc - final_val_acc
        
        if acc_gap > 0.05:
            st.warning(f"⚠️ Potential overfitting detected: {acc_gap:.2%} accuracy gap between training and validation")
        else:
            st.success(f"✅ Good generalization: {acc_gap:.2%} accuracy gap")
            
        # Check training stability
        val_acc_std = np.std(history['val_accuracy'])
        if val_acc_std > 0.05:
            st.warning(f"⚠️ Unstable training: High validation accuracy variance ({val_acc_std:.3f})")
        else:
            st.success(f"✅ Stable training: Low variance ({val_acc_std:.3f})")

def create_improved_training_plan():
    """Create an improved training plan for cardiomegaly"""
    
    st.subheader("🚀 Improved Training Strategy")
    
    trainer = EnhancedCardiomegalyTrainer()
    recommendations = trainer.get_training_recommendations()
    
    # Display recommendations in tabs
    tabs = st.tabs(["🔧 Data Improvements", "📈 Training Strategies", "🧠 Model Enhancements", "✅ Validation"])
    
    with tabs[0]:
        st.write("**Data Quality Improvements:**")
        for rec in recommendations["data_improvements"]:
            st.write(f"• {rec}")
    
    with tabs[1]:
        st.write("**Training Strategy Improvements:**")
        for rec in recommendations["training_strategies"]:
            st.write(f"• {rec}")
    
    with tabs[2]:
        st.write("**Model Architecture Enhancements:**")
        for rec in recommendations["model_enhancements"]:
            st.write(f"• {rec}")
    
    with tabs[3]:
        st.write("**Validation & Monitoring:**")
        for rec in recommendations["validation_improvements"]:
            st.write(f"• {rec}")

if __name__ == "__main__":
    st.title("🫀 Enhanced Cardiomegaly Training Analysis")
    
    # Analyze current performance
    analyze_current_performance()
    
    # Show improvement recommendations
    create_improved_training_plan()
    
    st.success("""
    🎯 **Key Recommendations for Better Performance:**
    
    1. **Increase Training Data Diversity**: More varied cardiomegaly cases
    2. **Enhanced Data Augmentation**: Medical-specific transformations
    3. **Fine-tune More Layers**: Unfreeze more backbone layers
    4. **Use Attention Mechanisms**: Focus on relevant cardiac regions
    5. **Implement Ensemble Methods**: Combine multiple models
    6. **Extended Training**: More epochs with proper regularization
    """)