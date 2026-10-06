#!/usr/bin/env python3
"""
Four Binary Classification Models Training System
DenseNet121 Architecture with Early Stopping and Dropout
Medical X-ray AI System - Binary Classification Approach
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
from datetime import datetime
import shutil

# Set memory growth for GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")

class FourBinaryModelsTrainer:
    """
    Trainer for 4 separate binary classification models:
    1. Normal vs Pneumonia (Chest)
    2. Normal vs Cardiomegaly (Chest)  
    3. Normal vs Arthritis (Knee)
    4. Normal vs Osteoporosis (Knee)
    """
    
    def __init__(self, base_path='Dataset'):
        self.base_path = base_path
        self.models = {}
        self.histories = {}
        self.input_shape = (224, 224, 3)
        self.batch_size = 32
        self.epochs = 50
        
        # Define dataset paths for each model
        self.model_configs = {
            'pneumonia': {
                'name': 'Pneumonia Detection',
                'dataset_path': 'Dataset/CHEST/chest_xray Pneumonia',
                'classes': ['NORMAL', 'PNEUMONIA'],
                'model_file': 'models/pneumonia_binary_model.h5'
            },
            'cardiomegaly': {
                'name': 'Cardiomegaly Detection', 
                'dataset_path': 'Dataset/CHEST/cardiomelgy',
                'classes': ['normal', 'Cardiomegaly'],
                'model_file': 'models/cardiomegaly_binary_model.h5'
            },
            'arthritis': {
                'name': 'Arthritis Detection',
                'dataset_path': 'Dataset/KNEE/Osteoarthritis Knee X-ray',
                'classes': ['Normal', 'Osteoarthritis'],  
                'model_file': 'models/arthritis_binary_model.h5'
            },
            'osteoporosis': {
                'name': 'Osteoporosis Detection',
                'dataset_path': 'Dataset/KNEE/Osteoporosis Knee', 
                'classes': ['Normal', 'Osteoporosis'],
                'model_file': 'models/osteoporosis_binary_model.h5'
            }
        }
        
        # Create models directory
        os.makedirs('models', exist_ok=True)
        os.makedirs('models/binary_backups', exist_ok=True)
        
    def create_densenet121_model(self, model_name: str) -> keras.Model:
        """
        Create DenseNet121 model with dropout and early stopping optimization
        """
        print(f"\n🏗️ Creating DenseNet121 model for {model_name}...")
        
        # Input layer
        inputs = keras.Input(shape=self.input_shape, name='input_layer')
        
        # Data augmentation layer (applied during training only)
        data_augmentation = keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
            layers.RandomContrast(0.1),
        ], name="data_augmentation")
        
        x = data_augmentation(inputs)
        
        # Preprocessing
        x = layers.Rescaling(1./255)(x)
        
        # DenseNet121 base model (pre-trained on ImageNet)
        base_model = DenseNet121(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape,
            pooling=None
        )
        
        # Freeze base model initially for transfer learning
        base_model.trainable = False
        
        # Extract features from DenseNet121
        x = base_model(x, training=False)
        
        # Custom top layers with dropout for regularization
        x = layers.GlobalAveragePooling2D(name='global_avg_pooling')(x)
        
        # First dense block with dropout
        x = layers.Dense(512, activation='relu', name='dense_1')(x)
        x = layers.BatchNormalization(name='batch_norm_1')(x)
        x = layers.Dropout(0.5, name='dropout_1')(x)
        
        # Second dense block with dropout  
        x = layers.Dense(256, activation='relu', name='dense_2')(x)
        x = layers.BatchNormalization(name='batch_norm_2')(x)
        x = layers.Dropout(0.4, name='dropout_2')(x)
        
        # Third dense block with dropout
        x = layers.Dense(128, activation='relu', name='dense_3')(x)
        x = layers.BatchNormalization(name='batch_norm_3')(x)
        x = layers.Dropout(0.3, name='dropout_3')(x)
        
        # Grad-CAM target layer for visualization
        x = layers.Activation('relu', name='gradcam_target_layer')(x)
        x = layers.Dropout(0.2, name='dropout_final')(x)
        
        # Output layer for binary classification
        outputs = layers.Dense(2, activation='softmax', name='predictions')(x)
        
        # Create model
        model = keras.Model(inputs, outputs, name=f'densenet121_{model_name}_binary')
        
        return model, base_model
    
    def prepare_dataset_generators(self, model_key: str):
        """
        Prepare data generators for training, validation, and testing
        """
        config = self.model_configs[model_key]
        dataset_path = config['dataset_path']
        
        print(f"\n📁 Preparing dataset for {config['name']}...")
        print(f"Dataset path: {dataset_path}")
        
        # Check if dataset path exists
        if not os.path.exists(dataset_path):
            print(f"❌ Dataset path not found: {dataset_path}")
            return None, None, None
        
        # List available folders
        available_folders = os.listdir(dataset_path)
        print(f"Available folders: {available_folders}")
        
        # Data generators with augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest',
            validation_split=0.2  # 80% train, 20% validation
        )
        
        # No augmentation for validation/test
        val_test_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=0.2
        )
        
        try:
            # Training generator
            train_generator = train_datagen.flow_from_directory(
                dataset_path,
                target_size=(224, 224),
                batch_size=self.batch_size,
                class_mode='categorical',
                subset='training',
                shuffle=True,
                seed=42
            )
            
            # Validation generator  
            val_generator = val_test_datagen.flow_from_directory(
                dataset_path,
                target_size=(224, 224),
                batch_size=self.batch_size,
                class_mode='categorical',
                subset='validation',
                shuffle=False,
                seed=42
            )
            
            # Test generator (using validation split for testing)
            test_generator = val_test_datagen.flow_from_directory(
                dataset_path,
                target_size=(224, 224),
                batch_size=self.batch_size,
                class_mode='categorical',
                subset='validation',
                shuffle=False,
                seed=42
            )
            
            print(f"✅ Dataset loaded successfully!")
            print(f"Training samples: {train_generator.samples}")
            print(f"Validation samples: {val_generator.samples}")
            print(f"Classes found: {list(train_generator.class_indices.keys())}")
            
            return train_generator, val_generator, test_generator
            
        except Exception as e:
            print(f"❌ Error loading dataset: {str(e)}")
            return None, None, None
    
    def calculate_class_weights(self, train_generator):
        """Calculate class weights for handling class imbalance"""
        try:
            # Get class labels
            y_labels = train_generator.classes
            
            # Calculate class weights
            class_weights = compute_class_weight(
                'balanced',
                classes=np.unique(y_labels),
                y=y_labels
            )
            
            class_weight_dict = dict(enumerate(class_weights))
            print(f"Class weights: {class_weight_dict}")
            
            return class_weight_dict
            
        except Exception as e:
            print(f"Warning: Could not calculate class weights: {str(e)}")
            return None
    
    def get_callbacks(self, model_key: str):
        """
        Setup callbacks including early stopping, model checkpoint, and learning rate reduction
        """
        config = self.model_configs[model_key]
        
        callbacks = [
            # Early stopping to prevent overfitting
            EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True,
                verbose=1,
                mode='max'
            ),
            
            # Save best model
            ModelCheckpoint(
                config['model_file'],
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1,
                mode='max'
            ),
            
            # Reduce learning rate when plateauing
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        return callbacks
    
    def train_single_model(self, model_key: str):
        """
        Train a single binary classification model
        """
        config = self.model_configs[model_key]
        print(f"\n🚀 Training {config['name']} Model")
        print("=" * 60)
        
        # Prepare datasets
        train_gen, val_gen, test_gen = self.prepare_dataset_generators(model_key)
        
        if train_gen is None:
            print(f"❌ Failed to load dataset for {model_key}")
            return False
        
        # Create model
        model, base_model = self.create_densenet121_model(model_key)
        
        # Calculate class weights
        class_weights = self.calculate_class_weights(train_gen)
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        print(f"\n📊 Model Architecture for {config['name']}:")
        model.summary()
        
        # Get callbacks
        callbacks = self.get_callbacks(model_key)
        
        # Train model
        print(f"\n🏋️ Training {config['name']} model...")
        
        history = model.fit(
            train_gen,
            epochs=self.epochs,
            validation_data=val_gen,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )
        
        # Fine-tuning: Unfreeze base model and train with lower learning rate
        print(f"\n🔧 Fine-tuning {config['name']} model...")
        
        base_model.trainable = True
        
        # Freeze early layers, fine-tune later layers
        for layer in base_model.layers[:-20]:
            layer.trainable = False
            
        # Recompile with lower learning rate
        model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='categorical_crossentropy', 
            metrics=['accuracy', 'precision', 'recall']
        )
        
        # Continue training for fine-tuning
        history_finetune = model.fit(
            train_gen,
            epochs=20,
            validation_data=val_gen,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )
        
        # Evaluate model
        print(f"\n📊 Evaluating {config['name']} model...")
        
        test_loss, test_accuracy, test_precision, test_recall = model.evaluate(test_gen, verbose=0)
        
        print(f"✅ {config['name']} Results:")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test Precision: {test_precision:.4f}")
        print(f"Test Recall: {test_recall:.4f}")
        
        # Generate predictions for detailed metrics
        test_gen.reset()
        predictions = model.predict(test_gen, verbose=0)
        predicted_classes = np.argmax(predictions, axis=1)
        
        true_classes = test_gen.classes[:len(predicted_classes)]
        class_labels = list(test_gen.class_indices.keys())
        
        # Classification report
        report = classification_report(
            true_classes, 
            predicted_classes,
            target_names=class_labels,
            output_dict=True
        )
        
        print(f"\n📈 Classification Report for {config['name']}:")
        print(classification_report(true_classes, predicted_classes, target_names=class_labels))
        
        # Store model and results
        self.models[model_key] = {
            'model': model,
            'history': history,
            'history_finetune': history_finetune,
            'test_accuracy': test_accuracy,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'classification_report': report,
            'config': config
        }
        
        return True
    
    def train_all_models(self):
        """
        Train all 4 binary classification models
        """
        print("🎯 Starting Training of 4 Binary Classification Models")
        print("Using DenseNet121 with Early Stopping and Dropout")
        print("=" * 80)
        
        successful_models = 0
        
        for model_key in self.model_configs.keys():
            try:
                success = self.train_single_model(model_key)
                if success:
                    successful_models += 1
                    print(f"✅ {self.model_configs[model_key]['name']} trained successfully!")
                else:
                    print(f"❌ Failed to train {self.model_configs[model_key]['name']}")
            except Exception as e:
                print(f"❌ Error training {model_key}: {str(e)}")
        
        print(f"\n🎉 Training Complete! {successful_models}/4 models trained successfully.")
        
        if successful_models > 0:
            self.update_model_registry()
            self.save_training_summary()
            self.plot_training_results()
        
        return successful_models
    
    def update_model_registry(self):
        """
        Update model registry with all 4 binary models
        """
        registry_path = 'models/registry/model_registry.json'
        os.makedirs(os.path.dirname(registry_path), exist_ok=True)
        
        # Load existing registry
        if os.path.exists(registry_path):
            with open(registry_path, 'r') as f:
                registry = json.load(f)
        else:
            registry = {
                "version": "2.0",
                "created": datetime.now().isoformat(),
                "models": {},
                "active_models": {}
            }
        
        # Update registry for each model
        for model_key, model_data in self.models.items():
            config = model_data['config']
            
            model_info = {
                "model_path": config['model_file'],
                "file_path": config['model_file'],
                "dataset_type": model_key,
                "model_name": f"{config['name']} - DenseNet121 Binary",
                "architecture": "DenseNet121_Binary",
                "version": "v1.0", 
                "accuracy": float(model_data['test_accuracy']),
                "precision": float(model_data['test_precision']),
                "recall": float(model_data['test_recall']),
                "f1_score": float(model_data['classification_report']['weighted avg']['f1-score']),
                "classes": config['classes'],
                "input_shape": list(self.input_shape),
                "trained_date": datetime.now().isoformat(),
                "dataset": f"{config['name']} Binary Classification Dataset",
                "training_method": "Binary_Classification_DenseNet121_with_Dropout_EarlyStopping",
                "gradcam_target_layer": "gradcam_target_layer",
                "classification_type": "binary",
                "per_class_metrics": {}
            }
            
            # Add per-class metrics
            for class_name in config['classes']:
                if class_name.lower() in model_data['classification_report']:
                    class_metrics = model_data['classification_report'][class_name.lower()]
                elif class_name in model_data['classification_report']:
                    class_metrics = model_data['classification_report'][class_name]
                else:
                    continue
                    
                model_info["per_class_metrics"][class_name] = {
                    "precision": float(class_metrics['precision']),
                    "recall": float(class_metrics['recall']),
                    "f1_score": float(class_metrics['f1-score'])
                }
            
            # Add file size if model exists
            if os.path.exists(config['model_file']):
                model_info["file_size"] = os.path.getsize(config['model_file'])
            
            # Update registry
            registry["models"][model_key] = model_info
            registry["active_models"][model_key] = model_key
        
        registry["last_modified"] = datetime.now().isoformat()
        
        # Save updated registry
        with open(registry_path, 'w') as f:
            json.dump(registry, f, indent=2)
        
        print(f"✅ Model registry updated: {registry_path}")
    
    def save_training_summary(self):
        """
        Save comprehensive training summary
        """
        summary = {
            "training_date": datetime.now().isoformat(),
            "architecture": "DenseNet121_Binary_Classification",
            "total_models": len(self.models),
            "successful_models": len([m for m in self.models.values() if m is not None]),
            "models": {}
        }
        
        for model_key, model_data in self.models.items():
            if model_data is not None:
                summary["models"][model_key] = {
                    "name": model_data['config']['name'],
                    "test_accuracy": float(model_data['test_accuracy']),
                    "test_precision": float(model_data['test_precision']),
                    "test_recall": float(model_data['test_recall']),
                    "model_file": model_data['config']['model_file']
                }
        
        summary_path = f"models/binary_training_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✅ Training summary saved: {summary_path}")
    
    def plot_training_results(self):
        """
        Plot training results for all models
        """
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            axes = axes.flatten()
            
            for idx, (model_key, model_data) in enumerate(self.models.items()):
                if model_data is None or idx >= 4:
                    continue
                    
                history = model_data['history']
                
                # Plot accuracy
                axes[idx].plot(history.history['accuracy'], label='Training Accuracy', color='blue')
                axes[idx].plot(history.history['val_accuracy'], label='Validation Accuracy', color='red')
                axes[idx].set_title(f"{model_data['config']['name']} - Accuracy")
                axes[idx].set_xlabel('Epochs')
                axes[idx].set_ylabel('Accuracy')
                axes[idx].legend()
                axes[idx].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_path = f"models/binary_training_plots_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.show()
            
            print(f"✅ Training plots saved: {plot_path}")
            
        except Exception as e:
            print(f"⚠️ Could not generate plots: {str(e)}")

def main():
    """
    Main training function
    """
    print("🏥 Medical X-ray AI - 4 Binary Models Training System")
    print("DenseNet121 with Early Stopping and Dropout")
    print("=" * 80)
    
    # Initialize trainer
    trainer = FourBinaryModelsTrainer()
    
    # Train all models
    successful_count = trainer.train_all_models()
    
    if successful_count > 0:
        print(f"\n🎊 SUCCESS! {successful_count}/4 models trained and saved.")
        print("\n📊 Model Performance Summary:")
        print("-" * 50)
        
        for model_key, model_data in trainer.models.items():
            if model_data is not None:
                config = model_data['config']
                print(f"{config['name']}: {model_data['test_accuracy']:.3f} accuracy")
        
        print("\n✅ All models are ready for medical X-ray diagnosis!")
        print("Models saved in 'models/' directory with registry updated.")
        
    else:
        print("\n❌ No models were successfully trained.")
        print("Please check dataset paths and try again.")

if __name__ == "__main__":
    main()