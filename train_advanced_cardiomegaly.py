#!/usr/bin/env python3
"""
Advanced Cardiomegaly Training Script
=====================================
High-performance cardiomegaly detection model training with EfficientNetB4.
Target: >97% accuracy for medical-grade deployment.
"""

import os
import sys
import json
import datetime
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')

class AdvancedCardiomegalyTrainer:
    def __init__(self):
        """Initialize the advanced cardiomegaly trainer"""
        
        # Configuration
        self.input_shape = (224, 224, 3)
        self.batch_size = 16
        self.epochs = 60
        self.learning_rate = 0.001
        
        # Paths
        self.base_dir = Path(__file__).parent
        self.data_dir = self.base_dir / "Dataset"
        self.models_dir = self.base_dir / "models"
        
        # Model ID
        self.model_id = f"cardiomegaly_efficientb4_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Initialize
        self.model = None
        self.history = None
        
        print(f"🏥 Advanced Cardiomegaly Trainer Initialized")
        print(f"🎯 Target: >97% accuracy")
        print(f"📁 Data directory: {self.data_dir}")
        print(f"💾 Models directory: {self.models_dir}")
        print(f"🆔 Model ID: {self.model_id}")
    
    def setup_data_generators(self):
        """Setup advanced data generators with augmentation"""
        
        print("📊 Setting up data generators...")
        
        # Advanced augmentation for medical images
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.15,
            horizontal_flip=True,
            brightness_range=[0.8, 1.2],
            channel_shift_range=0.1,
            validation_split=0.2
        )
        
        val_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=0.2
        )
        
        # Find cardiomegaly dataset - check the actual dataset structure
        chest_paths = [
            self.data_dir / "CHEST" / "cardiomelgy" / "train" / "train",
            self.data_dir / "CHEST" / "cardiomelgy" / "train",
            self.data_dir / "CHEST",
            self.data_dir / "chest", 
            self.data_dir / "Chest",
            self.data_dir / "cardiomegaly",
            self.data_dir / "Cardiomegaly"
        ]
        
        dataset_path = None
        for path in chest_paths:
            if path.exists():
                subdirs = [d.name for d in path.iterdir() if d.is_dir()]
                expected_classes = ['Normal', 'Cardiomegaly', 'normal', 'cardiomegaly', 'false', 'true']
                if any(expected in subdirs for expected in expected_classes):
                    print(f"✅ Found cardiomegaly dataset at: {path}")
                    dataset_path = path
                    break
        
        if not dataset_path:
            print("❌ No cardiomegaly dataset found. Creating structure...")
            dataset_path = self.data_dir / "cardiomegaly"
            dataset_path.mkdir(parents=True, exist_ok=True)
            (dataset_path / "Normal").mkdir(exist_ok=True)
            (dataset_path / "Cardiomegaly").mkdir(exist_ok=True)
            print(f"📁 Created dataset structure at: {dataset_path}")
            print("⚠️ Please add your chest X-ray images to the Normal and Cardiomegaly folders")
        
        # Determine class names - handle the false/true naming convention
        subdirs = [d.name for d in dataset_path.iterdir() if d.is_dir()]
        class_names = []
        for subdir in subdirs:
            if subdir.lower() in ['normal', 'no_cardiomegaly', 'healthy', 'false']:
                class_names.insert(0, subdir)  # Normal first
            elif subdir.lower() in ['cardiomegaly', 'enlarged', 'abnormal', 'true']:
                class_names.append(subdir)
        
        if len(class_names) < 2:
            print(f"⚠️ Using default class names. Found directories: {subdirs}")
            class_names = ['Normal', 'Cardiomegaly']
        
        print(f"📊 Using classes: {class_names}")
        
        # Map class names for consistency
        if 'false' in [c.lower() for c in class_names] and 'true' in [c.lower() for c in class_names]:
            # Use the actual directory names but map them conceptually
            print("📝 Note: Using 'false'=Normal, 'true'=Cardiomegaly mapping")
        
        # Create generators
        try:
            train_generator = train_datagen.flow_from_directory(
                dataset_path,
                target_size=self.input_shape[:2],
                batch_size=self.batch_size,
                class_mode='categorical',
                subset='training',
                classes=class_names,
                shuffle=True
            )
            
            validation_generator = val_datagen.flow_from_directory(
                dataset_path,
                target_size=self.input_shape[:2],
                batch_size=self.batch_size,
                class_mode='categorical',
                subset='validation',
                classes=class_names,
                shuffle=True
            )
        except Exception as e:
            print(f"❌ Error creating data generators: {e}")
            print(f"📁 Dataset path: {dataset_path}")
            print(f"📁 Available directories: {[d.name for d in dataset_path.iterdir() if d.is_dir()]}")
            raise
        
        print(f"📊 Training samples: {train_generator.samples}")
        print(f"📊 Validation samples: {validation_generator.samples}")
        print(f"📊 Class indices: {train_generator.class_indices}")
        
        return train_generator, validation_generator
    
    def create_advanced_model(self):
        """Create state-of-the-art model architecture"""
        
        print("🧠 Building advanced cardiomegaly detection model...")
        
        # Use EfficientNetB4 for optimal performance/efficiency balance
        base_model = keras.applications.EfficientNetB4(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape,
            pooling=None
        )
        
        # Fine-tuning strategy: unfreeze top 40% of layers
        trainable_layers = int(len(base_model.layers) * 0.4)
        for layer in base_model.layers[:-trainable_layers]:
            layer.trainable = False
        for layer in base_model.layers[-trainable_layers:]:
            layer.trainable = True
        
        print(f"🔧 Total layers: {len(base_model.layers)}")
        print(f"🔧 Trainable layers: {sum(1 for layer in base_model.layers if layer.trainable)}")
        
        # Build the model
        inputs = keras.Input(shape=self.input_shape)
        
        # Data augmentation (applied during training)
        x = layers.RandomFlip("horizontal", seed=42)(inputs)
        x = layers.RandomRotation(0.08, seed=42)(x)
        x = layers.RandomZoom(0.1, seed=42)(x)
        x = layers.RandomContrast(0.2, seed=42)(x)
        x = layers.RandomBrightness(0.15, seed=42)(x)
        
        # Preprocessing
        x = keras.applications.efficientnet.preprocess_input(x)
        
        # Base model
        x = base_model(x, training=True)
        
        # Advanced head for cardiomegaly detection
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        
        # Multi-scale feature extraction
        x1 = layers.Dense(1536, activation='relu')(x)
        x1 = layers.Dropout(0.5)(x1)
        x1 = layers.BatchNormalization()(x1)
        
        x2 = layers.Dense(768, activation='relu')(x)
        x2 = layers.Dropout(0.4)(x2)
        x2 = layers.BatchNormalization()(x2)
        
        x3 = layers.Dense(384, activation='relu')(x)
        x3 = layers.Dropout(0.3)(x3)
        x3 = layers.BatchNormalization()(x3)
        
        # Combine features
        combined = layers.concatenate([x1, x2, x3])
        
        # Final classification layers
        x = layers.Dense(512, activation='relu')(combined)
        x = layers.Dropout(0.3)(x)
        x = layers.BatchNormalization()(x)
        
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        
        # Output layer
        outputs = layers.Dense(2, activation='softmax', name='cardiomegaly_predictions')(x)
        
        model = keras.Model(inputs, outputs, name='AdvancedCardiomegalyModel')
        
        # Compile with advanced optimizer
        optimizer = optimizers.AdamW(
            learning_rate=self.learning_rate,
            weight_decay=0.01,
            beta_1=0.9,
            beta_2=0.999
        )
        
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("✅ Advanced cardiomegaly model created successfully")
        model.summary()
        
        self.model = model
        return model
    
    def setup_callbacks(self):
        """Setup advanced training callbacks"""
        
        checkpoint_dir = self.models_dir / "cardiomegaly" / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Progressive learning rate schedule
        def lr_schedule(epoch):
            if epoch < 20:
                return self.learning_rate
            elif epoch < 40:
                return self.learning_rate * 0.3
            elif epoch < 55:
                return self.learning_rate * 0.1
            else:
                return self.learning_rate * 0.03
        
        callbacks = [
            ModelCheckpoint(
                filepath=str(checkpoint_dir / f"best_{self.model_id}.h5"),
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                mode='max',
                verbose=1
            ),
            EarlyStopping(
                monitor='val_accuracy',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=8,
                min_lr=1e-8,
                verbose=1
            ),
            LearningRateScheduler(lr_schedule, verbose=1)
        ]
        
        return callbacks
    
    def train_model(self, train_generator, validation_generator):
        """Execute advanced training process"""
        
        print("🚀 Starting advanced cardiomegaly training...")
        
        callbacks = self.setup_callbacks()
        
        # Calculate steps per epoch
        steps_per_epoch = max(1, train_generator.samples // self.batch_size)
        validation_steps = max(1, validation_generator.samples // self.batch_size)
        
        print(f"📊 Steps per epoch: {steps_per_epoch}")
        print(f"📊 Validation steps: {validation_steps}")
        
        # Train the model
        self.history = self.model.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=self.epochs,
            validation_data=validation_generator,
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=1,
            class_weight={0: 1.0, 1: 1.3}
        )
        
        print("✅ Training completed successfully!")
    
    def evaluate_model(self, validation_generator):
        """Comprehensive model evaluation"""
        
        print("📊 Evaluating cardiomegaly model performance...")
        
        # Reset validation generator
        validation_generator.reset()
        
        # Predict on validation set
        predictions = self.model.predict(
            validation_generator, 
            steps=validation_generator.samples // self.batch_size + 1
        )
        predicted_classes = np.argmax(predictions, axis=1)
        
        # Get true labels
        true_labels = validation_generator.classes[:len(predicted_classes)]
        
        # Calculate metrics
        accuracy = np.mean(predicted_classes == true_labels)
        
        # Classification report
        class_names = ['Normal', 'Cardiomegaly']
        report = classification_report(
            true_labels, 
            predicted_classes, 
            target_names=class_names, 
            output_dict=True
        )
        
        print(f"🎯 Final Validation Accuracy: {accuracy:.4f}")
        print(f"🎯 Precision: {report['macro avg']['precision']:.4f}")
        print(f"🎯 Recall: {report['macro avg']['recall']:.4f}")
        print(f"🎯 F1-Score: {report['macro avg']['f1-score']:.4f}")
        
        # Cardiomegaly-specific metrics (class 1)
        if '1' in report:
            cardio_precision = report['1']['precision']
            cardio_recall = report['1']['recall']
            print(f"❤️ Cardiomegaly Detection Precision: {cardio_precision:.4f}")
            print(f"❤️ Cardiomegaly Detection Recall: {cardio_recall:.4f}")
        
        return {
            'accuracy': accuracy,
            'precision': report['macro avg']['precision'],
            'recall': report['macro avg']['recall'],
            'f1_score': report['macro avg']['f1-score'],
            'classification_report': report
        }
    
    def save_model_and_registry(self, metrics):
        """Save model and update registry with proper metadata"""
        
        print("💾 Saving cardiomegaly model and updating registry...")
        
        # Create model directory
        model_dir = self.models_dir / "cardiomegaly"
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save final model
        model_filename = f"efficientb4_cardiomegaly_advanced_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.h5"
        model_path = model_dir / model_filename
        self.model.save(str(model_path))
        
        print(f"💾 Model saved to: {model_path}")
        
        # Update model registry
        registry_path = self.models_dir / "registry" / "model_registry.json"
        registry_path.parent.mkdir(parents=True, exist_ok=True)
        
        if registry_path.exists():
            with open(registry_path, 'r') as f:
                registry = json.load(f)
        else:
            registry = {"models": {}, "active_models": {}}
        
        # Create model metadata
        model_metadata = {
            "model_id": self.model_id,
            "model_name": "EfficientNetB4 Cardiomegaly Detection (Advanced Training)",
            "dataset_type": "cardiomegaly",
            "version": "4.0_advanced",
            "architecture": "EfficientNetB4",
            "input_shape": list(self.input_shape),
            "num_classes": 2,
            "class_names": ["Normal", "Cardiomegaly"],
            "performance_metrics": {
                "accuracy": round(metrics['accuracy'], 4),
                "precision": round(metrics['precision'], 4),
                "recall": round(metrics['recall'], 4),
                "f1_score": round(metrics['f1_score'], 4)
            },
            "training_info": {
                "training_date": datetime.datetime.now().isoformat(),
                "architecture": "EfficientNetB4",
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "learning_rate": self.learning_rate,
                "performance_level": "Medical Grade" if metrics['accuracy'] > 0.95 else "Research Grade"
            },
            "file_path": f"cardiomegaly/{model_filename}",
            "file_size": round(model_path.stat().st_size / (1024*1024), 2),
            "created_date": datetime.datetime.now().strftime('%Y-%m-%d'),
            "description": f"Advanced cardiomegaly detection model with {metrics['accuracy']:.1%} accuracy using EfficientNetB4",
            "tags": ["EfficientNetB4", "medical", "advanced_training", "cardiomegaly"],
            "is_active": False,
            "source_location": "advanced_training"
        }
        
        # Add to registry
        registry["models"][self.model_id] = model_metadata
        registry["last_modified"] = datetime.datetime.now().isoformat()
        
        # Save updated registry
        with open(registry_path, 'w') as f:
            json.dump(registry, f, indent=2)
        
        print(f"📋 Model registered with ID: {self.model_id}")
        print(f"🎯 Performance: {metrics['accuracy']:.1%} accuracy")
        
        # Save training history
        history_path = model_dir / f"history_{self.model_id}.json"
        history_data = {
            "history": {key: [float(val) for val in values] for key, values in self.history.history.items()},
            "model_id": self.model_id,
            "training_date": datetime.datetime.now().isoformat(),
            "final_metrics": metrics
        }
        
        with open(history_path, 'w') as f:
            json.dump(history_data, f, indent=2)
        
        print(f"📊 Training history saved to: {history_path}")
        
        return model_path, self.model_id
    
    def run_complete_training(self):
        """Execute the complete training pipeline"""
        
        print("🏥 Starting Complete Advanced Cardiomegaly Training Pipeline")
        print("=" * 60)
        
        try:
            # Setup data
            train_gen, val_gen = self.setup_data_generators()
            
            if train_gen.samples == 0 or val_gen.samples == 0:
                print("❌ No training data found! Please check your dataset directory.")
                return None, None
            
            # Create model
            self.create_advanced_model()
            
            # Train model
            self.train_model(train_gen, val_gen)
            
            # Evaluate
            metrics = self.evaluate_model(val_gen)
            
            # Save everything
            model_path, model_id = self.save_model_and_registry(metrics)
            
            print("=" * 60)
            print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
            print(f"🎯 Final Accuracy: {metrics['accuracy']:.1%}")
            print(f"💾 Model saved: {model_path}")
            print(f"📋 Registry ID: {model_id}")
            print("=" * 60)
            
            return model_path, model_id
            
        except Exception as e:
            print(f"❌ Training failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, None

def main():
    """Main training execution"""
    
    print("🏥 Advanced Cardiomegaly Training System")
    print("🎯 Goal: Achieve >97% accuracy")
    print()
    
    # Initialize trainer
    trainer = AdvancedCardiomegalyTrainer()
    
    # Run complete training
    model_path, model_id = trainer.run_complete_training()
    
    if model_path and model_id:
        print("✅ SUCCESS: New cardiomegaly model ready for deployment!")
        print(f"📁 Model file: {model_path}")
        print(f"🆔 Model ID: {model_id}")
        print("📋 To activate this model, update the registry's active_models section")
    else:
        print("❌ Training failed. Please check the error messages above.")

if __name__ == "__main__":
    main()