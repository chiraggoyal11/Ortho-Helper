import os
import json
import datetime
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

class AdvancedBoneFractureTrainer:
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.data_dir = self.base_dir / "data"
        self.models_dir = self.base_dir / "models"
        
        # Advanced configuration for bone fracture detection
        self.input_shape = (320, 320, 3)  # Good resolution for bone details
        self.batch_size = 20  # Moderate batch size for memory efficiency
        self.epochs = 70  # Sufficient epochs for convergence
        self.learning_rate = 1.5e-4  # Conservative learning rate
        
        # Create unique model ID
        self.model_id = f"bone_fracture_densenet201_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.history = None
        self.model = None
        
        # Setup directories
        self.models_dir.mkdir(parents=True, exist_ok=True)
        (self.models_dir / "bone_fracture").mkdir(parents=True, exist_ok=True)
        (self.models_dir / "registry").mkdir(parents=True, exist_ok=True)
        
        print(f"🦴 Advanced Bone Fracture Training System Initialized")
        print(f"🆔 Model ID: {self.model_id}")
        print(f"📁 Base directory: {self.base_dir}")
        
    def setup_data_generators(self):
        """Setup advanced data augmentation for bone fracture detection"""
        
        print("📊 Setting up bone fracture data generators...")
        
        # Advanced augmentation specifically for bone X-rays
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,  # Bones can be at various angles
            width_shift_range=0.12,
            height_shift_range=0.12,
            shear_range=0.08,
            zoom_range=0.15,
            horizontal_flip=True,  # Safe for most bone X-rays
            fill_mode='constant',
            cval=0,
            brightness_range=[0.85, 1.15],  # Account for X-ray variations
            contrast_range=[0.9, 1.1],
            validation_split=0.2
        )
        
        val_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=0.2
        )
        
        # Look for bone fracture data in multiple potential locations
        bone_paths = [
            self.data_dir / "ARM" / "ARM",
            self.data_dir / "ARM",
            self.data_dir / "bone_fracture",
            self.data_dir / "fracture",
            self.data_dir / "BONE",
            Path("data") / "ARM"
        ]
        
        dataset_path = None
        for path in bone_paths:
            if path.exists():
                # Check if it has the right subdirectories
                expected_classes = ['Normal', 'Fracture', 'normal', 'fracture', 'Bone_Fracture', 'bone_fracture']
                subdirs = [d.name for d in path.iterdir() if d.is_dir()]
                if any(expected in subdirs for expected in expected_classes):
                    print(f"✅ Found bone fracture dataset at: {path}")
                    dataset_path = path
                    break
        
        if not dataset_path:
            print("❌ No bone fracture dataset found. Creating structure...")
            dataset_path = self.data_dir / "bone_fracture"
            dataset_path.mkdir(parents=True, exist_ok=True)
            (dataset_path / "Normal").mkdir(exist_ok=True)
            (dataset_path / "Fracture").mkdir(exist_ok=True)
            print(f"📁 Created dataset structure at: {dataset_path}")
            print("⚠️ Please add your bone X-ray images to the Normal and Fracture folders")
            
        # Determine class names from actual directories
        subdirs = [d.name for d in dataset_path.iterdir() if d.is_dir()]
        class_names = []
        for subdir in subdirs:
            if subdir.lower() in ['normal', 'no_fracture', 'healthy']:
                class_names.insert(0, subdir)  # Normal first
            elif subdir.lower() in ['fracture', 'bone_fracture', 'broken']:
                class_names.append(subdir)
        
        if len(class_names) < 2:
            print(f"⚠️ Using default class names. Found directories: {subdirs}")
            class_names = ['Normal', 'Fracture']
            
        print(f"📊 Using classes: {class_names}")
        
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
        """Create state-of-the-art model architecture for bone fracture detection"""
        
        print("🧠 Building advanced bone fracture detection model...")
        
        # Use DenseNet201 for better feature extraction of fractures
        base_model = keras.applications.DenseNet201(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape,
            pooling=None
        )
        
        # Fine-tuning strategy: unfreeze top 30% of layers
        trainable_layers = int(len(base_model.layers) * 0.3)
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
        x = layers.RandomRotation(0.05, seed=42)(x)
        x = layers.RandomZoom(0.08, seed=42)(x)
        x = layers.RandomContrast(0.15, seed=42)(x)
        x = layers.RandomBrightness(0.1, seed=42)(x)
        
        # Preprocessing
        x = keras.applications.densenet.preprocess_input(x)
        
        # Base model
        x = base_model(x, training=True)
        
        # Advanced head architecture for fracture detection
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        
        # Multi-scale feature extraction
        x1 = layers.Dense(1024, activation='relu')(x)
        x1 = layers.Dropout(0.4)(x1)
        x1 = layers.BatchNormalization()(x1)
        
        x2 = layers.Dense(512, activation='relu')(x)
        x2 = layers.Dropout(0.3)(x2)
        x2 = layers.BatchNormalization()(x2)
        
        # Combine features
        combined = layers.concatenate([x1, x2])
        
        # Final classification layers
        x = layers.Dense(256, activation='relu')(combined)
        x = layers.Dropout(0.2)(x)
        x = layers.BatchNormalization()(x)
        
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.1)(x)
        
        # Output layer
        outputs = layers.Dense(2, activation='softmax', name='fracture_predictions')(x)
        
        model = keras.Model(inputs, outputs, name='AdvancedBoneFractureModel')
        
        # Custom learning rate schedule
        def scheduler(epoch, lr):
            if epoch < 10:
                return lr
            elif epoch < 30:
                return lr * 0.9
            else:
                return lr * 0.8
        
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
            metrics=['accuracy', 'precision', 'recall']
        )
        
        print("✅ Advanced bone fracture model created successfully")
        model.summary()
        
        self.model = model
        return model
    
    def setup_callbacks(self):
        """Setup advanced training callbacks"""
        
        checkpoint_dir = self.models_dir / "bone_fracture" / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Learning rate scheduler
        def lr_schedule(epoch):
            if epoch < 15:
                return self.learning_rate
            elif epoch < 35:
                return self.learning_rate * 0.5
            elif epoch < 50:
                return self.learning_rate * 0.1
            else:
                return self.learning_rate * 0.05
        
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
                patience=12,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.3,
                patience=6,
                min_lr=1e-8,
                verbose=1
            ),
            LearningRateScheduler(lr_schedule, verbose=1)
        ]
        
        return callbacks
    
    def train_model(self, train_generator, validation_generator):
        """Execute advanced training process"""
        
        print("🚀 Starting advanced bone fracture training...")
        
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
            class_weight={0: 1.0, 1: 1.2}  # Slightly favor fracture detection
        )
        
        print("✅ Training completed successfully!")
        
    def evaluate_model(self, validation_generator):
        """Comprehensive model evaluation"""
        
        print("📊 Evaluating bone fracture model performance...")
        
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
        class_names = ['Normal', 'Fracture']
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
        
        # Fracture-specific metrics (class 1)
        if '1' in report:
            fracture_precision = report['1']['precision']
            fracture_recall = report['1']['recall']
            print(f"🦴 Fracture Detection Precision: {fracture_precision:.4f}")
            print(f"🦴 Fracture Detection Recall: {fracture_recall:.4f}")
        
        return {
            'accuracy': accuracy,
            'precision': report['macro avg']['precision'],
            'recall': report['macro avg']['recall'],
            'f1_score': report['macro avg']['f1-score'],
            'fracture_precision': report.get('1', {}).get('precision', 0),
            'fracture_recall': report.get('1', {}).get('recall', 0),
            'classification_report': report
        }
    
    def save_model_and_registry(self, metrics):
        """Save model and update registry with proper metadata"""
        
        print("💾 Saving bone fracture model and updating registry...")
        
        # Create model directory
        model_dir = self.models_dir / "bone_fracture"
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save final model
        model_filename = f"densenet201_bone_fracture_advanced_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.h5"
        model_path = model_dir / model_filename
        self.model.save(str(model_path))
        
        print(f"💾 Model saved to: {model_path}")
        
        # Update model registry
        registry_path = self.models_dir / "registry" / "model_registry.json"
        
        if registry_path.exists():
            with open(registry_path, 'r') as f:
                registry = json.load(f)
        else:
            registry = {"models": {}, "active_models": {}}
        
        # Create model metadata
        model_metadata = {
            "model_id": self.model_id,
            "model_name": "DenseNet201 Bone Fracture Detection (Advanced Training)",
            "dataset_type": "bone_fracture",
            "version": "4.0_advanced",
            "architecture": "DenseNet201",
            "input_shape": list(self.input_shape),
            "num_classes": 2,
            "class_names": ["Normal", "Bone_Fracture"],
            "performance_metrics": {
                "accuracy": round(metrics['accuracy'], 4),
                "precision": round(metrics['precision'], 4),
                "recall": round(metrics['recall'], 4),
                "f1_score": round(metrics['f1_score'], 4),
                "fracture_precision": round(metrics['fracture_precision'], 4),
                "fracture_recall": round(metrics['fracture_recall'], 4)
            },
            "training_info": {
                "training_date": datetime.datetime.now().isoformat(),
                "architecture": "DenseNet201",
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "learning_rate": self.learning_rate,
                "performance_level": "Medical Grade" if metrics['accuracy'] > 0.90 else "Research Grade"
            },
            "file_path": f"bone_fracture/{model_filename}",
            "file_size": round(model_path.stat().st_size / (1024*1024), 2),
            "created_date": datetime.datetime.now().strftime('%Y-%m-%d'),
            "description": f"Advanced bone fracture detection model with {metrics['accuracy']:.1%} accuracy using DenseNet201",
            "tags": ["DenseNet201", "medical", "advanced_training", "bone_fracture"],
            "is_active": False,  # Will be activated separately
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
        print(f"🦴 Fracture Detection: {metrics['fracture_precision']:.1%} precision, {metrics['fracture_recall']:.1%} recall")
        
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
    
    def plot_training_history(self):
        """Create training visualization plots"""
        
        if not self.history:
            print("❌ No training history available")
            return
        
        plt.figure(figsize=(15, 10))
        
        # Accuracy plot
        plt.subplot(2, 3, 1)
        plt.plot(self.history.history['accuracy'], label='Training Accuracy')
        plt.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        # Loss plot
        plt.subplot(2, 3, 2)
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Precision plot
        plt.subplot(2, 3, 3)
        if 'precision' in self.history.history:
            plt.plot(self.history.history['precision'], label='Training Precision')
            plt.plot(self.history.history['val_precision'], label='Validation Precision')
            plt.title('Model Precision')
            plt.xlabel('Epoch')
            plt.ylabel('Precision')
            plt.legend()
            plt.grid(True)
        
        # Recall plot
        plt.subplot(2, 3, 4)
        if 'recall' in self.history.history:
            plt.plot(self.history.history['recall'], label='Training Recall')
            plt.plot(self.history.history['val_recall'], label='Validation Recall')
            plt.title('Model Recall')
            plt.xlabel('Epoch')
            plt.ylabel('Recall')
            plt.legend()
            plt.grid(True)
        
        # Learning rate plot
        plt.subplot(2, 3, 5)
        if 'lr' in self.history.history:
            plt.plot(self.history.history['lr'], label='Learning Rate')
            plt.title('Learning Rate Schedule')
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.legend()
            plt.yscale('log')
            plt.grid(True)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.models_dir / "bone_fracture" / f"training_plot_{self.model_id}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📈 Training plots saved to: {plot_path}")
        
        plt.show()
    
    def run_complete_training(self):
        """Execute the complete training pipeline"""
        
        print("🦴 Starting Complete Advanced Bone Fracture Training Pipeline")
        print("=" * 70)
        
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
            
            # Plot results
            self.plot_training_history()
            
            print("\n" + "=" * 70)
            print("🎉 BONE FRACTURE TRAINING COMPLETED SUCCESSFULLY!")
            print(f"🎯 Final Accuracy: {metrics['accuracy']:.1%}")
            print(f"🦴 Fracture Detection: {metrics['fracture_precision']:.1%} precision")
            print(f"💾 Model saved: {model_path}")
            print(f"📋 Registry ID: {model_id}")
            print("=" * 70)
            
            return model_path, model_id
            
        except Exception as e:
            print(f"❌ Training failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, None

def main():
    """Main training execution"""
    
    print("🦴 Advanced Bone Fracture Training System")
    print("🎯 Goal: Achieve >90% accuracy with robust fracture detection")
    print()
    
    # Initialize trainer
    trainer = AdvancedBoneFractureTrainer()
    
    # Run complete training
    model_path, model_id = trainer.run_complete_training()
    
    if model_path and model_id:
        print(f"\n✅ SUCCESS: New bone fracture model ready for deployment!")
        print(f"📁 Model file: {model_path}")
        print(f"🆔 Model ID: {model_id}")
        print("\n📋 To activate this model, update the registry's active_models section")
    else:
        print("\n❌ Training failed. Please check the error messages above.")

if __name__ == "__main__":
    main()