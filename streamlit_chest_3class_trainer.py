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

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def train_chest_3class_model():
    """Streamlit interface for training 3-class chest model"""
    
    st.title("🫁 Chest Conditions 3-Class Model Training")
    st.markdown("**Train a model to detect: Normal, Pneumonia, and Cardiomegaly**")
    
    # Check dataset availability
    pneumonia_path = "Dataset/CHEST/chest_xray Pneumonia"
    cardiomegaly_path = "Dataset/CHEST/cardiomelgy"
    
    col1, col2 = st.columns(2)
    
    with col1:
        if os.path.exists(pneumonia_path):
            st.success("✅ Pneumonia dataset found")
        else:
            st.error("❌ Pneumonia dataset not found")
    
    with col2:
        if os.path.exists(cardiomegaly_path):
            st.success("✅ Cardiomegaly dataset found")
        else:
            st.error("❌ Cardiomegaly dataset not found")
    
    if not (os.path.exists(pneumonia_path) and os.path.exists(cardiomegaly_path)):
        st.warning("⚠️ Both datasets are required for 3-class training")
        return
    
    st.markdown("---")
    
    # Training parameters
    st.subheader("🛠️ Training Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        epochs = st.slider("Epochs", min_value=5, max_value=50, value=25)
        
    with col2:
        batch_size = st.selectbox("Batch Size", [8, 16, 24, 32], index=1)
    
    with col3:
        validation_split = st.slider("Validation Split", min_value=0.1, max_value=0.3, value=0.2)
    
    # Advanced options
    with st.expander("🔧 Advanced Options"):
        learning_rate = st.selectbox("Learning Rate", [1e-5, 5e-5, 1e-4, 5e-4, 1e-3], index=2)
        max_samples = st.slider("Max Samples per Class", min_value=1000, max_value=5000, value=3000)
        use_augmentation = st.checkbox("Use Data Augmentation", value=True)
    
    # Training button
    if st.button("🚀 Start 3-Class Training", type="primary"):
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Step 1: Dataset preparation
            status_text.text("📂 Preparing datasets...")
            progress_bar.progress(10)
            
            # Load data using existing utilities
            from utils.data_loader import MedicalDataLoader
            data_loader = MedicalDataLoader("Dataset")
            
            # Load pneumonia data
            status_text.text("🫁 Loading pneumonia dataset...")
            progress_bar.progress(20)
            
            pneumonia_data = data_loader.load_chest_data()
            pneumonia_images = pneumonia_data['images']
            pneumonia_labels = pneumonia_data['labels']  # 0=Normal, 1=Pneumonia
            
            # Load cardiomegaly data
            status_text.text("💗 Loading cardiomegaly dataset...")
            progress_bar.progress(30)
            
            cardiomegaly_images = []
            cardiomegaly_labels = []
            
            # Process cardiomegaly dataset
            for root, dirs, files in os.walk(cardiomegaly_path):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        file_path = os.path.join(root, file)
                        
                        try:
                            img = Image.open(file_path).convert('RGB')
                            img = img.resize((224, 224))
                            img_array = np.array(img) / 255.0
                            
                            # Determine label (assuming cardiomegaly dataset structure)
                            folder_name = os.path.basename(root).lower()
                            if 'normal' in folder_name:
                                label = 0  # Normal
                            else:
                                label = 2  # Cardiomegaly
                            
                            cardiomegaly_images.append(img_array)
                            cardiomegaly_labels.append(label)
                            
                        except Exception as e:
                            continue
            
            cardiomegaly_images = np.array(cardiomegaly_images)
            cardiomegaly_labels = np.array(cardiomegaly_labels)
            
            # Combine datasets
            status_text.text("🔄 Combining datasets...")
            progress_bar.progress(40)
            
            all_images = np.vstack([pneumonia_images, cardiomegaly_images])
            all_labels = np.hstack([pneumonia_labels, cardiomegaly_labels])
            
            # Display dataset info
            st.success(f"📊 Combined dataset: {all_images.shape[0]} samples")
            
            class_counts = np.bincount(all_labels)
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Normal", class_counts[0] if len(class_counts) > 0 else 0)
            with col2:
                st.metric("Pneumonia", class_counts[1] if len(class_counts) > 1 else 0)
            with col3:
                st.metric("Cardiomegaly", class_counts[2] if len(class_counts) > 2 else 0)
            
            # Step 2: Model creation
            status_text.text("🏗️ Creating 3-class model...")
            progress_bar.progress(50)
            
            # Create model
            from tensorflow.keras.applications import DenseNet121
            from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
            from tensorflow.keras.optimizers import Adam
            from tensorflow.keras import Model
            
            base_model = DenseNet121(
                weights='imagenet',
                include_top=False,
                input_shape=(224, 224, 3)
            )
            
            # Freeze most layers
            for layer in base_model.layers[:-20]:
                layer.trainable = False
            
            # Add custom layers
            inputs = tf.keras.Input(shape=(224, 224, 3))
            x = base_model(inputs, training=False)
            x = GlobalAveragePooling2D()(x)
            x = Dense(256, activation='relu')(x)
            x = BatchNormalization()(x)
            x = Dropout(0.5)(x)
            x = Dense(128, activation='relu', name='gradcam_target_layer')(x)
            x = Dropout(0.3)(x)
            outputs = Dense(3, activation='softmax')(x)
            
            model = Model(inputs, outputs)
            
            model.compile(
                optimizer=Adam(learning_rate=learning_rate),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            st.success("✅ Model created successfully!")
            
            # Step 3: Training
            status_text.text("🚀 Training model...")
            progress_bar.progress(60)
            
            # Split data
            from sklearn.model_selection import train_test_split
            X_train, X_val, y_train, y_val = train_test_split(
                all_images, all_labels, 
                test_size=validation_split, 
                stratify=all_labels, 
                random_state=42
            )
            
            # Callbacks
            from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
            
            callbacks = [
                EarlyStopping(patience=5, restore_best_weights=True),
                ReduceLROnPlateau(patience=3, factor=0.5, min_lr=1e-7)
            ]
            
            # Training
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=0
            )
            
            progress_bar.progress(80)
            
            # Step 4: Evaluation and saving
            status_text.text("📊 Evaluating and saving model...")
            
            # Evaluate
            val_accuracy = max(history.history['val_accuracy'])
            
            # Save model
            os.makedirs('models', exist_ok=True)
            model_path = 'models/chest_conditions_3class_model.h5'
            model.save(model_path)
            
            # Update model registry
            registry_path = 'models/registry/model_registry.json'
            os.makedirs(os.path.dirname(registry_path), exist_ok=True)
            
            # Load or create registry
            if os.path.exists(registry_path):
                with open(registry_path, 'r') as f:
                    registry = json.load(f)
            else:
                registry = {"version": "1.0", "models": {}, "active_models": {}}
            
            # Update registry
            registry["models"]["chest_conditions"] = {
                "model_path": "chest_conditions_3class_model.h5",
                "file_path": "chest_conditions_3class_model.h5",
                "dataset_type": "chest_conditions",
                "model_name": "Chest 3-Class DenseNet121",
                "architecture": "DenseNet121",
                "version": "v2.0",
                "accuracy": float(val_accuracy),
                "classes": ["Normal", "Pneumonia", "Cardiomegaly"],
                "input_shape": [224, 224, 3],
                "trained_date": datetime.now().isoformat(),
                "dataset": "Combined Pneumonia + Cardiomegaly",
                "gradcam_target_layer": "gradcam_target_layer",
                "file_size": os.path.getsize(model_path)
            }
            
            registry["active_models"]["chest_conditions"] = "chest_conditions"
            registry["last_modified"] = datetime.now().isoformat()
            
            with open(registry_path, 'w') as f:
                json.dump(registry, f, indent=2)
            
            progress_bar.progress(100)
            status_text.text("✅ Training completed!")
            
            # Display results
            st.success("🎉 3-Class Chest Model Training Completed!")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Final Accuracy", f"{val_accuracy:.3f}")
            
            with col2:
                st.metric("Total Epochs", len(history.history['accuracy']))
            
            with col3:
                st.metric("Model Size", f"{os.path.getsize(model_path) / (1024*1024):.1f} MB")
            
            # Plot training history
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            ax1.plot(history.history['accuracy'], label='Training')
            ax1.plot(history.history['val_accuracy'], label='Validation')
            ax1.set_title('Model Accuracy')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Accuracy')
            ax1.legend()
            
            ax2.plot(history.history['loss'], label='Training')
            ax2.plot(history.history['val_loss'], label='Validation')
            ax2.set_title('Model Loss')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Loss')
            ax2.legend()
            
            plt.tight_layout()
            st.pyplot(fig)
            
            st.markdown("---")
            st.info("📝 **Next Steps:**\n"
                   "1. The new 3-class model is now active\n"
                   "2. Test it using the main interface\n"
                   "3. It can now detect Normal, Pneumonia, and Cardiomegaly")
            
        except Exception as e:
            st.error(f"❌ Training failed: {str(e)}")
            st.exception(e)

if __name__ == "__main__":
    train_chest_3class_model()