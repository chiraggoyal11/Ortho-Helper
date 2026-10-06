import streamlit as st
import os
import sys
import numpy as np
import tensorflow as tf
from PIL import Image
import json
from datetime import datetime
import matplotlib.pyplot as plt
import time

def run_chest_comprehensive_training():
    """Streamlit interface for comprehensive chest model training"""
    
    st.title("🫁 Comprehensive Chest Model Training")
    st.markdown("**DenseNet121 with Dropout, Early Stopping & Advanced Regularization**")
    
    # Check datasets
    pneumonia_path = "Dataset/CHEST/chest_xray Pneumonia"
    cardiomegaly_path = "Dataset/CHEST/cardiomelgy"
    
    col1, col2 = st.columns(2)
    
    with col1:
        if os.path.exists(pneumonia_path):
            st.success("✅ Pneumonia dataset found")
            pneumonia_count = sum(len(files) for _, _, files in os.walk(pneumonia_path) 
                                if any(f.lower().endswith(('.png', '.jpg', '.jpeg')) for f in files))
            st.info(f"📊 ~{pneumonia_count} pneumonia images")
        else:
            st.error("❌ Pneumonia dataset not found")
    
    with col2:
        if os.path.exists(cardiomegaly_path):
            st.success("✅ Cardiomegaly dataset found")
            cardio_count = sum(len(files) for _, _, files in os.walk(cardiomegaly_path) 
                             if any(f.lower().endswith(('.png', '.jpg', '.jpeg')) for f in files))
            st.info(f"💗 ~{cardio_count} cardiomegaly images")
        else:
            st.error("❌ Cardiomegaly dataset not found")
    
    if not (os.path.exists(pneumonia_path) and os.path.exists(cardiomegaly_path)):
        st.warning("⚠️ Both datasets required for comprehensive training")
        return
    
    st.markdown("---")
    
    # Training configuration
    st.subheader("🛠️ Training Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        epochs = st.slider("Max Epochs", min_value=10, max_value=50, value=25)
        
    with col2:
        batch_size = st.selectbox("Batch Size", [8, 16, 24, 32], index=1)
    
    with col3:
        validation_split = st.slider("Validation Split", min_value=0.15, max_value=0.3, value=0.2)
    
    # Advanced settings
    with st.expander("🔧 Advanced Settings"):
        col1, col2 = st.columns(2)
        
        with col1:
            learning_rate = st.selectbox("Learning Rate", [1e-5, 5e-5, 1e-4, 5e-4, 1e-3], index=2)
            early_stopping_patience = st.slider("Early Stopping Patience", min_value=5, max_value=15, value=8)
        
        with col2:
            dropout_rates = st.multiselect("Dropout Rates", [0.2, 0.3, 0.4, 0.5, 0.6], default=[0.5, 0.4, 0.3])
            use_class_weights = st.checkbox("Use Class Weights", value=True)
    
    # Model architecture info
    st.subheader("🏗️ Model Architecture")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **Base Architecture:** DenseNet121
        - Pre-trained on ImageNet
        - Fine-tuned top layers
        - Global Average Pooling
        """)
    
    with col2:
        st.info("""
        **Regularization Features:**
        - Multiple Dropout layers
        - Batch Normalization
        - Early Stopping
        - Learning Rate Scheduling
        """)
    
    # Target classes
    st.subheader("🎯 Target Classes")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.success("**Normal**\nHealthy chest X-rays")
    
    with col2:
        st.error("**Pneumonia**\nLung infection detection")
    
    with col3:
        st.warning("**Cardiomegaly**\nEnlarged heart detection")
    
    # Training button
    if st.button("🚀 Start Comprehensive Training", type="primary"):
        
        progress_container = st.container()
        with progress_container:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Step 1: Dataset loading
                status_text.text("📂 Loading datasets...")
                progress_bar.progress(10)
                
                # Load datasets
                from tensorflow.keras.applications import DenseNet121
                from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
                from tensorflow.keras.optimizers import Adam
                from tensorflow.keras import Model
                from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
                from sklearn.model_selection import train_test_split
                from sklearn.utils.class_weight import compute_class_weight
                
                # Load data function
                def load_datasets():
                    images = []
                    labels = []
                    
                    # Load pneumonia data
                    for root, dirs, files in os.walk(pneumonia_path):
                        for file in files:
                            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                                try:
                                    img = Image.open(os.path.join(root, file)).convert('RGB')
                                    img = img.resize((224, 224))
                                    img_array = np.array(img) / 255.0
                                    
                                    if 'normal' in root.lower() or 'normal' in file.lower():
                                        label = 0  # Normal
                                    else:
                                        label = 1  # Pneumonia
                                    
                                    images.append(img_array)
                                    labels.append(label)
                                except:
                                    continue
                    
                    # Load cardiomegaly data
                    for root, dirs, files in os.walk(cardiomegaly_path):
                        for file in files:
                            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                                try:
                                    img = Image.open(os.path.join(root, file)).convert('RGB')
                                    img = img.resize((224, 224))
                                    img_array = np.array(img) / 255.0
                                    
                                    if 'normal' in root.lower() or 'normal' in file.lower():
                                        label = 0  # Normal
                                    else:
                                        label = 2  # Cardiomegaly
                                    
                                    images.append(img_array)
                                    labels.append(label)
                                except:
                                    continue
                    
                    return np.array(images), np.array(labels)
                
                images, labels = load_datasets()
                
                if len(images) == 0:
                    st.error("❌ No images loaded!")
                    return
                
                progress_bar.progress(25)
                status_text.text("📊 Dataset loaded successfully!")
                
                # Display dataset info
                unique, counts = np.unique(labels, return_counts=True)
                class_names = ['Normal', 'Pneumonia', 'Cardiomegaly']
                
                col1, col2, col3 = st.columns(3)
                for i, (class_idx, count) in enumerate(zip(unique, counts)):
                    with [col1, col2, col3][i]:
                        st.metric(class_names[class_idx], count)
                
                # Step 2: Model creation
                status_text.text("🏗️ Creating DenseNet121 model...")
                progress_bar.progress(40)
                
                # Create model
                base_model = DenseNet121(
                    weights='imagenet',
                    include_top=False,
                    input_shape=(224, 224, 3)
                )
                
                # Freeze base layers
                for layer in base_model.layers[:-30]:
                    layer.trainable = False
                
                # Build model
                inputs = tf.keras.Input(shape=(224, 224, 3))
                x = base_model(inputs, training=False)
                x = GlobalAveragePooling2D()(x)
                
                # Add dropout layers
                x = Dense(256, activation='relu')(x)
                x = BatchNormalization()(x)
                x = Dropout(dropout_rates[0] if dropout_rates else 0.5)(x)
                
                x = Dense(128, activation='relu', name='gradcam_target_layer')(x)
                x = BatchNormalization()(x)
                x = Dropout(dropout_rates[1] if len(dropout_rates) > 1 else 0.4)(x)
                
                x = Dense(64, activation='relu')(x)
                x = Dropout(dropout_rates[2] if len(dropout_rates) > 2 else 0.3)(x)
                
                outputs = Dense(3, activation='softmax')(x)
                
                model = Model(inputs, outputs)
                
                model.compile(
                    optimizer=Adam(learning_rate=learning_rate),
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy']
                )
                
                st.success(f"✅ Model created! Parameters: {model.count_params():,}")
                
                # Step 3: Training preparation
                status_text.text("⚙️ Preparing training...")
                progress_bar.progress(50)
                
                # Split data
                X_train, X_val, y_train, y_val = train_test_split(
                    images, labels,
                    test_size=validation_split,
                    stratify=labels,
                    random_state=42
                )
                
                # Class weights
                class_weights = None
                if use_class_weights:
                    class_weights = compute_class_weight(
                        'balanced',
                        classes=np.unique(y_train),
                        y=y_train
                    )
                    class_weights = dict(enumerate(class_weights))
                
                # Callbacks
                callbacks = [
                    EarlyStopping(
                        monitor='val_loss',
                        patience=early_stopping_patience,
                        restore_best_weights=True,
                        verbose=1
                    ),
                    ReduceLROnPlateau(
                        monitor='val_loss',
                        factor=0.5,
                        patience=5,
                        min_lr=1e-7,
                        verbose=1
                    )
                ]
                
                # Step 4: Training
                status_text.text("🚀 Training model...")
                progress_bar.progress(60)
                
                # Create training display
                training_placeholder = st.empty()
                
                # Train model
                history = model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=callbacks,
                    class_weight=class_weights,
                    verbose=0
                )
                
                progress_bar.progress(85)
                
                # Step 5: Save model
                status_text.text("💾 Saving model...")
                
                # Save model
                os.makedirs('models', exist_ok=True)
                os.makedirs('models/registry', exist_ok=True)
                
                model_path = 'models/comprehensive_chest_model.h5'
                model.save(model_path)
                
                # Update registry
                registry_path = 'models/registry/model_registry.json'
                
                if os.path.exists(registry_path):
                    with open(registry_path, 'r') as f:
                        registry = json.load(f)
                else:
                    registry = {"version": "2.0", "models": {}, "active_models": {}}
                
                final_accuracy = max(history.history['val_accuracy'])
                
                registry["models"]["chest_conditions"] = {
                    "model_path": "comprehensive_chest_model.h5",
                    "file_path": "comprehensive_chest_model.h5",
                    "dataset_type": "chest_conditions",
                    "model_name": "Comprehensive Chest DenseNet121",
                    "architecture": "DenseNet121",
                    "version": "v3.0",
                    "accuracy": float(final_accuracy),
                    "classes": ["Normal", "Pneumonia", "Cardiomegaly"],
                    "input_shape": [224, 224, 3],
                    "trained_date": datetime.now().isoformat(),
                    "dataset": "Comprehensive Pneumonia + Cardiomegaly",
                    "training_method": "DenseNet121 with Advanced Regularization",
                    "gradcam_target_layer": "gradcam_target_layer",
                    "regularization": f"Dropout {dropout_rates} + BatchNorm + EarlyStopping",
                    "file_size": os.path.getsize(model_path)
                }
                
                registry["active_models"]["chest_conditions"] = "chest_conditions"
                registry["last_modified"] = datetime.now().isoformat()
                
                with open(registry_path, 'w') as f:
                    json.dump(registry, f, indent=2)
                
                progress_bar.progress(100)
                status_text.text("✅ Training completed!")
                
                # Display results
                st.success("🎉 Comprehensive Chest Model Training Completed!")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Final Accuracy", f"{final_accuracy:.3f}")
                
                with col2:
                    st.metric("Epochs Trained", len(history.history['accuracy']))
                
                with col3:
                    st.metric("Model Size", f"{os.path.getsize(model_path) / (1024*1024):.1f} MB")
                
                # Plot training history
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                
                ax1.plot(history.history['accuracy'], 'b-', label='Training', linewidth=2)
                ax1.plot(history.history['val_accuracy'], 'r-', label='Validation', linewidth=2)
                ax1.set_title('Model Accuracy')
                ax1.set_xlabel('Epoch')
                ax1.set_ylabel('Accuracy')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                ax2.plot(history.history['loss'], 'b-', label='Training', linewidth=2)
                ax2.plot(history.history['val_loss'], 'r-', label='Validation', linewidth=2)
                ax2.set_title('Model Loss')
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('Loss')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Summary
                st.markdown("---")
                st.success("🎯 **Model Capabilities:**")
                st.write("✅ **Normal chest X-rays** - Healthy lung detection")
                st.write("✅ **Pneumonia** - Lung infection identification")
                st.write("✅ **Cardiomegaly** - Enlarged heart detection")
                
                st.info("🛡️ **Regularization Features:**")
                st.write(f"• Dropout layers: {dropout_rates}")
                st.write(f"• Early stopping (patience: {early_stopping_patience})")
                st.write("• Learning rate scheduling")
                st.write("• Batch normalization")
                st.write("• Class weight balancing" if use_class_weights else "• No class weighting")
                
            except Exception as e:
                st.error(f"❌ Training failed: {str(e)}")
                st.exception(e)

if __name__ == "__main__":
    run_chest_comprehensive_training()