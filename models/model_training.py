# Model Training Scripts for Medical X-ray AI System

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os
import pandas as pd
from typing import Tuple, List, Dict, Any
import streamlit as st

class MedicalImageModelTrainer:
    """
    Trainer class for medical image classification models
    Supports bone fracture, chest conditions, and knee conditions
    """
    
    def __init__(self, model_type: str, input_shape: Tuple[int, int, int] = (224, 224, 3)):
        self.model_type = model_type
        self.input_shape = input_shape
        self.model = None
        self.history = None
        
        # Model configurations
        self.config = {
            'bone_fracture': {
                'classes': ['Normal', 'Fracture'],
                'loss': 'binary_crossentropy',
                'metrics': ['accuracy', 'precision', 'recall']
            },
            'chest_conditions': {
                'classes': ['Normal', 'Pneumonia', 'Cardiomegaly'],
                'loss': 'categorical_crossentropy',
                'metrics': ['accuracy', 'precision', 'recall']
            },
            'knee_conditions': {
                'classes': ['Normal', 'Osteoporosis', 'Arthritis'],
                'loss': 'categorical_crossentropy', 
                'metrics': ['accuracy', 'precision', 'recall']
            }
        }
    
    def create_model(self, architecture: str = 'custom_cnn') -> keras.Model:
        """Create model based on specified architecture"""
        
        if architecture == 'custom_cnn':
            model = self._create_custom_cnn()
        elif architecture == 'resnet50':
            model = self._create_resnet50_transfer()
        elif architecture == 'efficientnet':
            model = self._create_efficientnet_transfer()
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
        
        self.model = model
        return model
    
    def _create_custom_cnn(self) -> keras.Model:
        """Create custom CNN architecture for medical images"""
        
        num_classes = len(self.config[self.model_type]['classes'])
        
        model = keras.Sequential([
            # Input layer
            layers.Input(shape=self.input_shape),
            
            # First convolutional block
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second convolutional block
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third convolutional block
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Fourth convolutional block
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Global average pooling
            layers.GlobalAveragePooling2D(),
            
            # Dense layers
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            
            # Output layer
            layers.Dense(
                num_classes if num_classes > 2 else 1,
                activation='softmax' if num_classes > 2 else 'sigmoid'
            )
        ])
        
        return model
    
    def _create_resnet50_transfer(self) -> keras.Model:
        """Create ResNet50-based transfer learning model"""
        
        num_classes = len(self.config[self.model_type]['classes'])
        
        # Load pre-trained ResNet50
        base_model = keras.applications.ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        # Freeze base model layers
        base_model.trainable = False
        
        # Add custom top layers
        model = keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(
                num_classes if num_classes > 2 else 1,
                activation='softmax' if num_classes > 2 else 'sigmoid'
            )
        ])
        
        return model
    
    def _create_efficientnet_transfer(self) -> keras.Model:
        """Create EfficientNet-based transfer learning model"""
        
        num_classes = len(self.config[self.model_type]['classes'])
        
        # Load pre-trained EfficientNetB0
        base_model = keras.applications.EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        # Freeze base model layers
        base_model.trainable = False
        
        # Add custom top layers
        model = keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(
                num_classes if num_classes > 2 else 1,
                activation='softmax' if num_classes > 2 else 'sigmoid'
            )
        ])
        
        return model
    
    def compile_model(self, learning_rate: float = 0.001):
        """Compile the model with appropriate loss function and metrics"""
        
        if self.model is None:
            raise ValueError("Model not created. Call create_model() first.")
        
        config = self.config[self.model_type]
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss=config['loss'],
            metrics=config['metrics']
        )
    
    def train_model(self, 
                   x_train: np.ndarray, 
                   y_train: np.ndarray,
                   x_val: np.ndarray,
                   y_val: np.ndarray,
                   epochs: int = 50,
                   batch_size: int = 32) -> keras.callbacks.History:
        """Train the model"""
        
        if self.model is None:
            raise ValueError("Model not created and compiled.")
        
        # Create callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=0.0001
            ),
            keras.callbacks.ModelCheckpoint(
                f'models/{self.model_type}_best_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                mode='max'
            )
        ]
        
        # Train model
        self.history = self.model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def evaluate_model(self, x_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Evaluate model performance"""
        
        if self.model is None:
            raise ValueError("Model not trained.")
        
        # Get predictions
        y_pred_proba = self.model.predict(x_test)
        
        # Convert probabilities to class predictions
        if len(self.config[self.model_type]['classes']) > 2:
            y_pred = np.argmax(y_pred_proba, axis=1)
            y_true = np.argmax(y_test, axis=1) if y_test.ndim > 1 else y_test
        else:
            y_pred = (y_pred_proba > 0.5).astype(int).flatten()
            y_true = y_test.flatten()
        
        # Calculate metrics
        test_loss, test_accuracy = self.model.evaluate(x_test, y_test, verbose=0)
        
        # Classification report
        class_names = self.config[self.model_type]['classes']
        report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        evaluation_results = {
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        return evaluation_results
    
    def plot_training_history(self):
        """Plot training history"""
        
        if self.history is None:
            raise ValueError("No training history available.")
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Training Accuracy')
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Loss
        axes[0, 1].plot(self.history.history['loss'], label='Training Loss')
        axes[0, 1].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Precision (if available)
        if 'precision' in self.history.history:
            axes[1, 0].plot(self.history.history['precision'], label='Training Precision')
            axes[1, 0].plot(self.history.history['val_precision'], label='Validation Precision')
            axes[1, 0].set_title('Model Precision')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Precision')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # Recall (if available)
        if 'recall' in self.history.history:
            axes[1, 1].plot(self.history.history['recall'], label='Training Recall')
            axes[1, 1].plot(self.history.history['val_recall'], label='Validation Recall')
            axes[1, 1].set_title('Model Recall')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Recall')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        return fig
    
    def plot_confusion_matrix(self, evaluation_results: Dict[str, Any]):
        """Plot confusion matrix"""
        
        cm = evaluation_results['confusion_matrix']
        class_names = self.config[self.model_type]['classes']
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names, ax=ax)
        
        ax.set_title(f'Confusion Matrix - {self.model_type.replace("_", " ").title()}')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        
        return fig
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        
        if self.model is None:
            raise ValueError("No model to save.")
        
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")

def load_dataset_from_directory(data_path: str, 
                               image_size: Tuple[int, int] = (224, 224),
                               batch_size: int = 32,
                               validation_split: float = 0.2) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Load dataset from directory structure using tf.keras.utils.image_dataset_from_directory
    
    Expected directory structure:
    data_path/
        class1/
            image1.jpg
            image2.jpg
        class2/
            image3.jpg
            image4.jpg
    """
    
    # Training dataset
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_path,
        validation_split=validation_split,
        subset="training",
        seed=123,
        image_size=image_size,
        batch_size=batch_size
    )
    
    # Validation dataset
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_path,
        validation_split=validation_split,
        subset="validation",
        seed=123,
        image_size=image_size,
        batch_size=batch_size
    )
    
    return train_ds, val_ds

def preprocess_for_training(dataset: tf.data.Dataset) -> tf.data.Dataset:
    """Preprocess dataset for training"""
    
    # Normalize pixel values to [0, 1]
    normalization_layer = keras.layers.Rescaling(1./255)
    
    dataset = dataset.map(lambda x, y: (normalization_layer(x), y))
    
    # Configure for performance
    AUTOTUNE = tf.data.AUTOTUNE
    dataset = dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    
    return dataset

# Example training script for bone fracture detection
def train_bone_fracture_model(data_path: str):
    """Example training script for bone fracture detection"""
    
    print("Training Bone Fracture Detection Model...")
    
    # Initialize trainer
    trainer = MedicalImageModelTrainer('bone_fracture')
    
    # Load dataset (assuming you have organized your FracAtlas dataset)
    train_ds, val_ds = load_dataset_from_directory(data_path)
    
    # Preprocess datasets
    train_ds = preprocess_for_training(train_ds)
    val_ds = preprocess_for_training(val_ds)
    
    # Create and compile model
    model = trainer.create_model('custom_cnn')
    trainer.compile_model(learning_rate=0.001)
    
    print(f"Model created with {model.count_params()} parameters")
    
    # Convert tf.data.Dataset to numpy arrays for training
    # (This is a simplified example - in practice you'd handle this more efficiently)
    print("Converting datasets to numpy arrays...")
    
    # For demonstration, we'll create dummy data
    x_train = np.random.random((1000, 224, 224, 3))
    y_train = np.random.randint(0, 2, (1000, 1))
    x_val = np.random.random((200, 224, 224, 3))
    y_val = np.random.randint(0, 2, (200, 1))
    
    # Train model
    print("Starting training...")
    history = trainer.train_model(x_train, y_train, x_val, y_val, epochs=10)
    
    # Evaluate model
    x_test = np.random.random((100, 224, 224, 3))
    y_test = np.random.randint(0, 2, (100, 1))
    
    results = trainer.evaluate_model(x_test, y_test)
    print(f"Test Accuracy: {results['test_accuracy']:.4f}")
    
    # Save model
    trainer.save_model('models/bone_fracture_model.h5')
    
    return trainer

# Model placeholder creation (for when actual models aren't available)
def create_model_placeholders():
    """Create placeholder model files for demonstration"""
    
    model_configs = {
        'bone_fracture_model.h5': {
            'input_shape': (224, 224, 3),
            'classes': 2
        },
        'chest_conditions_model.h5': {
            'input_shape': (224, 224, 3), 
            'classes': 3
        },
        'knee_conditions_model.h5': {
            'input_shape': (224, 224, 3),
            'classes': 3
        }
    }
    
    for model_name, config in model_configs.items():
        model_path = f'models/{model_name}'
        
        if not os.path.exists(model_path):
            print(f"Creating placeholder model: {model_name}")
            
            # Create simple model
            model = keras.Sequential([
                keras.layers.Input(shape=config['input_shape']),
                keras.layers.Conv2D(32, 3, activation='relu'),
                keras.layers.GlobalAveragePooling2D(),
                keras.layers.Dense(64, activation='relu'),
                keras.layers.Dense(
                    config['classes'] if config['classes'] > 2 else 1,
                    activation='softmax' if config['classes'] > 2 else 'sigmoid'
                )
            ])
            
            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy' if config['classes'] > 2 else 'binary_crossentropy',
                metrics=['accuracy']
            )
            
            model.save(model_path)
            print(f"Placeholder model saved: {model_path}")

if __name__ == "__main__":
    print("Model training module loaded successfully!")
    
    # Create placeholder models if they don't exist
    create_model_placeholders()
    
    print("Placeholder models created. Ready for training with real data.")