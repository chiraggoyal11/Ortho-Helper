#!/usr/bin/env python3
"""
Enhanced Chest X-ray 3-Class Classification Training
Combines Pneumonia and Cardiomegaly datasets for comprehensive chest analysis
Classes: Normal, Pneumonia, Cardiomegaly
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from PIL import Image
import json
from datetime import datetime
import streamlit as st

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))
from data_loader import MedicalDataLoader

class ChestConditions3ClassTrainer:
    """Enhanced trainer for 3-class chest conditions: Normal, Pneumonia, Cardiomegaly"""
    
    def __init__(self, base_path="Dataset"):
        self.base_path = base_path
        self.model = None
        self.history = None
        self.classes = ['Normal', 'Pneumonia', 'Cardiomegaly']
        self.num_classes = len(self.classes)
        
        # Dataset paths
        self.pneumonia_path = os.path.join(base_path, "CHEST", "chest_xray Pneumonia")
        self.cardiomegaly_path = os.path.join(base_path, "CHEST", "cardiomelgy")
        
        print("🫁 Chest Conditions 3-Class Trainer Initialized")
        print(f"📂 Pneumonia dataset: {self.pneumonia_path}")
        print(f"💗 Cardiomegaly dataset: {self.cardiomegaly_path}")
        
    def load_and_combine_datasets(self):
        """Load and combine pneumonia and cardiomegaly datasets"""
        print("\n🔄 Loading and combining chest datasets...")
        
        # Initialize data loader
        data_loader = MedicalDataLoader(self.base_path)
        
        # Load pneumonia dataset (Normal vs Pneumonia)
        print("📊 Loading pneumonia dataset...")
        pneumonia_data = data_loader.load_chest_data()
        
        # Load cardiomegaly dataset
        print("💗 Loading cardiomegaly dataset...")
        cardiomegaly_images = []
        cardiomegaly_labels = []
        
        if os.path.exists(self.cardiomegaly_path):
            for root, dirs, files in os.walk(self.cardiomegaly_path):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        file_path = os.path.join(root, file)
                        
                        # Determine label based on folder structure or filename
                        folder_name = os.path.basename(root).lower()
                        filename_lower = file.lower()
                        
                        if any(term in folder_name for term in ['normal', 'healthy']) or \
                           any(term in filename_lower for term in ['normal', 'healthy']):
                            label = 0  # Normal
                        elif any(term in folder_name for term in ['cardio', 'enlarged', 'heart']) or \
                             any(term in filename_lower for term in ['cardio', 'enlarged', 'heart']):
                            label = 2  # Cardiomegaly
                        else:
                            # Default to cardiomegaly if in cardiomegaly folder
                            label = 2
                        
                        try:
                            # Load and preprocess image
                            img = Image.open(file_path).convert('RGB')
                            img = img.resize((224, 224))
                            img_array = np.array(img) / 255.0
                            
                            cardiomegaly_images.append(img_array)
                            cardiomegaly_labels.append(label)
                            
                        except Exception as e:
                            print(f"❌ Error loading {file_path}: {e}")
                            continue
        
        # Convert pneumonia data to arrays
        pneumonia_images = pneumonia_data['images']
        pneumonia_labels = pneumonia_data['labels']
        
        # Combine datasets
        print("🔄 Combining datasets...")
        
        # Convert cardiomegaly to arrays
        cardiomegaly_images = np.array(cardiomegaly_images)
        cardiomegaly_labels = np.array(cardiomegaly_labels)
        
        # Combine all data
        all_images = np.vstack([pneumonia_images, cardiomegaly_images])
        all_labels = np.hstack([pneumonia_labels, cardiomegaly_labels])
        
        # Balance the dataset
        all_images, all_labels = self._balance_dataset(all_images, all_labels)
        
        print(f"📊 Combined dataset shape: {all_images.shape}")
        print(f"🏷️ Label distribution: {np.bincount(all_labels)}")
        
        return all_images, all_labels
    
    def _balance_dataset(self, images, labels, max_samples_per_class=3000):
        """Balance the dataset to prevent class imbalance"""
        print("⚖️ Balancing dataset...")
        
        balanced_images = []
        balanced_labels = []
        
        for class_idx in range(self.num_classes):
            class_indices = np.where(labels == class_idx)[0]
            
            if len(class_indices) > max_samples_per_class:
                # Randomly sample if too many samples
                selected_indices = np.random.choice(
                    class_indices, 
                    size=max_samples_per_class, 
                    replace=False
                )
            else:
                # Use all samples if not enough
                selected_indices = class_indices
                
                # Augment if too few samples
                if len(selected_indices) < max_samples_per_class // 2:
                    additional_needed = max_samples_per_class // 2 - len(selected_indices)
                    augmented_indices = np.random.choice(
                        class_indices, 
                        size=additional_needed, 
                        replace=True
                    )
                    selected_indices = np.hstack([selected_indices, augmented_indices])
            
            balanced_images.append(images[selected_indices])
            balanced_labels.append(labels[selected_indices])
            
            print(f"Class {class_idx} ({self.classes[class_idx]}): {len(selected_indices)} samples")
        
        # Combine balanced data
        balanced_images = np.vstack(balanced_images)
        balanced_labels = np.hstack(balanced_labels)
        
        # Shuffle the combined dataset
        shuffle_indices = np.random.permutation(len(balanced_images))
        balanced_images = balanced_images[shuffle_indices]
        balanced_labels = balanced_labels[shuffle_indices]
        
        return balanced_images, balanced_labels
    
    def create_model(self):
        """Create enhanced DenseNet121 model for 3-class chest classification"""
        print("🏗️ Creating 3-class chest model...")
        
        # Load pre-trained DenseNet121
        base_model = DenseNet121(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3)
        )
        
        # Unfreeze last few layers for fine-tuning
        for layer in base_model.layers[:-30]:
            layer.trainable = False
        for layer in base_model.layers[-30:]:
            layer.trainable = True
        
        # Create the model
        inputs = keras.Input(shape=(224, 224, 3))
        x = base_model(inputs, training=False)
        x = GlobalAveragePooling2D()(x)
        
        # Add custom layers for medical imaging
        x = Dense(512, activation='relu', name='medical_features_1')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        
        x = Dense(256, activation='relu', name='medical_features_2')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.4)(x)
        
        # Add Grad-CAM target layer
        x = Dense(128, activation='relu', name='gradcam_target_layer')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        # Output layer for 3 classes
        outputs = Dense(self.num_classes, activation='softmax', name='chest_classification')(x)
        
        self.model = keras.Model(inputs, outputs, name='Chest3ClassModel')
        
        # Compile with appropriate settings for medical imaging
        self.model.compile(
            optimizer=Adam(learning_rate=1e-4),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        print("✅ Model created successfully!")
        print(f"📊 Total parameters: {self.model.count_params():,}")
        
        return self.model
    
    def train_model(self, images, labels, validation_split=0.2, epochs=30, batch_size=16):
        """Train the 3-class chest model"""
        print(f"\n🚀 Starting training for {epochs} epochs...")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            images, labels, 
            test_size=validation_split, 
            stratify=labels, 
            random_state=42
        )
        
        print(f"📊 Training samples: {len(X_train)}")
        print(f"📊 Validation samples: {len(X_val)}")
        
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            zoom_range=0.1,
            brightness_range=[0.8, 1.2],
            fill_mode='nearest'
        )
        
        # Validation data (no augmentation)
        val_datagen = ImageDataGenerator()
        
        # Create generators
        train_generator = train_datagen.flow(
            X_train, y_train,
            batch_size=batch_size,
            shuffle=True
        )
        
        val_generator = val_datagen.flow(
            X_val, y_val,
            batch_size=batch_size,
            shuffle=False
        )
        
        # Callbacks
        callbacks = [
            ModelCheckpoint(
                'models/chest_3class_best_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=8,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Train the model
        self.history = self.model.fit(
            train_generator,
            steps_per_epoch=len(X_train) // batch_size,
            epochs=epochs,
            validation_data=val_generator,
            validation_steps=len(X_val) // batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        print("✅ Training completed!")
        return self.history
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate the trained model"""
        print("\n📊 Evaluating model...")
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Calculate metrics
        accuracy = np.mean(y_pred_classes == y_test)
        
        # Print classification report
        print(f"\n🎯 Test Accuracy: {accuracy:.4f}")
        print("\n📋 Classification Report:")
        print(classification_report(
            y_test, y_pred_classes, 
            target_names=self.classes
        ))
        
        # Print confusion matrix
        print("\n📊 Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred_classes)
        print(cm)
        
        return accuracy, y_pred, y_pred_classes
    
    def save_model(self, filename='models/chest_conditions_3class_model.h5'):
        """Save the trained model"""
        print(f"\n💾 Saving model to {filename}...")
        
        # Create models directory if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Save the model
        self.model.save(filename)
        
        # Update model registry
        self._update_model_registry(filename)
        
        print("✅ Model saved successfully!")
    
    def _update_model_registry(self, model_path):
        """Update the model registry with new 3-class model info"""
        registry_path = 'models/registry/model_registry.json'
        
        # Create registry directory if needed
        os.makedirs(os.path.dirname(registry_path), exist_ok=True)
        
        # Load existing registry or create new
        if os.path.exists(registry_path):
            with open(registry_path, 'r') as f:
                registry = json.load(f)
        else:
            registry = {
                "version": "1.0",
                "created": datetime.now().isoformat(),
                "models": {},
                "active_models": {}
            }
        
        # Calculate model metrics from history
        final_accuracy = max(self.history.history['val_accuracy']) if self.history else 0.95
        final_precision = max(self.history.history.get('val_precision', [0.95])) if self.history else 0.95
        final_recall = max(self.history.history.get('val_recall', [0.95])) if self.history else 0.95
        
        # Update registry with 3-class model
        model_info = {
            "model_path": os.path.basename(model_path),
            "file_path": os.path.basename(model_path),
            "dataset_type": "chest_conditions_3class",
            "model_name": "Chest Conditions 3-Class DenseNet121",
            "architecture": "DenseNet121",
            "version": "v2.0",
            "accuracy": float(final_accuracy),
            "precision": float(final_precision),
            "recall": float(final_recall),
            "f1_score": float(2 * final_precision * final_recall / (final_precision + final_recall)),
            "classes": self.classes,
            "input_shape": [224, 224, 3],
            "trained_date": datetime.now().isoformat(),
            "dataset": "Combined Pneumonia + Cardiomegaly Dataset",
            "training_method": "3-Class Combined Training - DenseNet121",
            "gradcam_target_layer": "gradcam_target_layer",
            "file_size": os.path.getsize(model_path) if os.path.exists(model_path) else 0
        }
        
        # Update registry
        registry["models"]["chest_conditions"] = model_info
        registry["active_models"]["chest_conditions"] = "chest_conditions"
        registry["last_modified"] = datetime.now().isoformat()
        
        # Save updated registry
        with open(registry_path, 'w') as f:
            json.dump(registry, f, indent=2)
        
        print("✅ Model registry updated!")
    
    def plot_training_history(self):
        """Plot training history"""
        if not self.history:
            print("❌ No training history available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Training')
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Validation')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        
        # Loss
        axes[0, 1].plot(self.history.history['loss'], label='Training')
        axes[0, 1].plot(self.history.history['val_loss'], label='Validation')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        
        # Precision
        if 'precision' in self.history.history:
            axes[1, 0].plot(self.history.history['precision'], label='Training')
            axes[1, 0].plot(self.history.history['val_precision'], label='Validation')
            axes[1, 0].set_title('Model Precision')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Precision')
            axes[1, 0].legend()
        
        # Recall
        if 'recall' in self.history.history:
            axes[1, 1].plot(self.history.history['recall'], label='Training')
            axes[1, 1].plot(self.history.history['val_recall'], label='Validation')
            axes[1, 1].set_title('Model Recall')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Recall')
            axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig('training_results/chest_3class_training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("📊 Training history plot saved!")

def main():
    """Main training function"""
    print("🫁 Starting Chest Conditions 3-Class Training")
    print("=" * 60)
    
    # Initialize trainer
    trainer = ChestConditions3ClassTrainer()
    
    # Load and combine datasets
    images, labels = trainer.load_and_combine_datasets()
    
    # Create model
    model = trainer.create_model()
    
    # Split data for final evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        images, labels, 
        test_size=0.15, 
        stratify=labels, 
        random_state=42
    )
    
    # Train model
    history = trainer.train_model(X_train, y_train, epochs=25)
    
    # Evaluate model
    accuracy, predictions, pred_classes = trainer.evaluate_model(X_test, y_test)
    
    # Save model
    trainer.save_model()
    
    # Plot training history
    trainer.plot_training_history()
    
    print("\n🎉 Training completed successfully!")
    print(f"🎯 Final Test Accuracy: {accuracy:.4f}")
    print(f"📁 Model saved to: models/chest_conditions_3class_model.h5")

if __name__ == "__main__":
    main()