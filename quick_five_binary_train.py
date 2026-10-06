#!/usr/bin/env python3
"""
Quick Training Script for 5 Binary Classification Models
Fast testing version for medical X-ray analysis
Updated with Bone Fracture Detection
"""

import os
import sys
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
from datetime import datetime
import time

# Set memory growth for GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU configured: {len(gpus)} device(s)")
    except RuntimeError as e:
        print(f"⚠️ GPU configuration error: {e}")
else:
    print("💻 Running on CPU")

class QuickFiveBinaryModelsTrainer:
    """
    Quick trainer for 5 binary classification models
    """
    
    def __init__(self):
        self.input_shape = (224, 224, 3)
        self.model_configs = {
            'pneumonia': {
                'name': 'Pneumonia Detection',
                'dataset_path': 'Dataset/CHEST/Pneumonia_Organized',
                'classes': ['Normal', 'Pneumonia'],
                'model_file': 'models/pneumonia_binary_model.h5',
                'description': 'Binary classification: Normal vs Pneumonia chest X-rays'
            },
            'cardiomegaly': {
                'name': 'Cardiomegaly Detection', 
                'dataset_path': 'Dataset/CHEST/cardiomelgy',
                'classes': ['normal', 'Cardiomegaly'],
                'model_file': 'models/cardiomegaly_binary_model.h5',
                'description': 'Binary classification: Normal vs Cardiomegaly chest X-rays'
            },
            'arthritis': {
                'name': 'Arthritis Detection',
                'dataset_path': 'Dataset/KNEE/Osteoarthritis/Combined_Osteoarthritis_Dataset',
                'classes': ['Normal', 'Osteoarthritis'],  
                'model_file': 'models/arthritis_binary_model.h5',
                'description': 'Binary classification: Normal vs Arthritis knee X-rays'
            },
            'osteoporosis': {
                'name': 'Osteoporosis Detection',
                'dataset_path': 'Dataset/KNEE/Osteoporosis/Combined_Osteoporosis_Dataset',
                'classes': ['Normal', 'Osteoporosis'],
                'model_file': 'models/osteoporosis_binary_model.h5',
                'description': 'Binary classification: Normal vs Osteoporosis knee X-rays'
            },
            'bone_fracture': {
                'name': 'Bone Fracture Detection',
                'dataset_path': 'Dataset/ARM/MURA_Organized/Forearm',
                'classes': ['Negative', 'Positive'],
                'model_file': 'models/bone_fracture_binary_model.h5',
                'description': 'Binary classification: Normal vs Fractured bones (hand/leg bones only)',
                'uses_subdirs': True
            }
        }
        
        # Create directories
        os.makedirs('models', exist_ok=True)
        os.makedirs('models/registry', exist_ok=True)
        
        print(f"🏥 Quick 5 Binary Models Trainer Initialized")
        print(f"📊 Models: {list(self.model_configs.keys())}")
    
    def check_dataset_availability(self):
        """Check which datasets are available"""
        print("\n📋 Checking dataset availability...")
        
        available_models = []
        for model_key, config in self.model_configs.items():
            path = config['dataset_path']
            uses_subdirs = config.get('uses_subdirs', False)
            
            if os.path.exists(path):
                try:
                    if uses_subdirs:
                        # Check train/val/test structure
                        subdirs = ['train', 'val', 'test']
                        missing_subdirs = []
                        
                        for subdir in subdirs:
                            subdir_path = os.path.join(path, subdir)
                            if not os.path.exists(subdir_path):
                                missing_subdirs.append(subdir)
                        
                        if not missing_subdirs:
                            available_models.append(model_key)
                            print(f"  ✅ {config['name']}: Available (train/val/test structure)")
                        else:
                            print(f"  ❌ {config['name']}: Missing subdirectories: {missing_subdirs}")
                    else:
                        folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
                        if len(folders) >= 2:
                            available_models.append(model_key)
                            print(f"  ✅ {config['name']}: Available ({len(folders)} classes)")
                        else:
                            print(f"  ❌ {config['name']}: Insufficient class folders ({len(folders)})")
                except Exception as e:
                    print(f"  ❌ {config['name']}: Error - {e}")
            else:
                print(f"  ❌ {config['name']}: Path not found")
        
        print(f"\n✅ Available models: {len(available_models)}/5")
        return available_models
    
    def create_densenet121_model(self, model_name: str) -> keras.Model:
        """Create DenseNet121 model with dropout"""
        
        # Input layer
        inputs = keras.Input(shape=self.input_shape, name='input_layer')
        
        # Data augmentation
        data_augmentation = keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
            layers.RandomContrast(0.1),
        ], name="data_augmentation")
        
        x = data_augmentation(inputs)
        x = layers.Rescaling(1./255)(x)
        
        # DenseNet121 base
        base_model = DenseNet121(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape,
            pooling=None
        )
        
        base_model.trainable = False
        x = base_model(x, training=False)
        
        # Custom top with dropout
        x = layers.GlobalAveragePooling2D(name='global_avg_pooling')(x)
        x = layers.Dense(512, activation='relu', name='dense_1')(x)
        x = layers.BatchNormalization(name='batch_norm_1')(x)
        x = layers.Dropout(0.5, name='dropout_1')(x)
        
        x = layers.Dense(256, activation='relu', name='dense_2')(x)
        x = layers.BatchNormalization(name='batch_norm_2')(x)
        x = layers.Dropout(0.4, name='dropout_2')(x)
        
        x = layers.Dense(128, activation='relu', name='dense_3')(x)
        x = layers.BatchNormalization(name='batch_norm_3')(x)
        x = layers.Dropout(0.3, name='dropout_3')(x)
        
        # Grad-CAM target layer
        x = layers.Activation('relu', name='gradcam_target_layer')(x)
        x = layers.Dropout(0.2, name='dropout_final')(x)
        
        # Binary classification output
        outputs = layers.Dense(2, activation='softmax', name='predictions')(x)
        
        model = keras.Model(inputs, outputs, name=f'densenet121_{model_name}_binary')
        return model, base_model
    
    def prepare_data_generators(self, model_key: str, batch_size: int = 32):
        """Prepare data generators"""
        
        config = self.model_configs[model_key]
        dataset_path = config['dataset_path']
        uses_subdirs = config.get('uses_subdirs', False)
        
        if uses_subdirs:
            # Handle train/val/test structure
            train_path = os.path.join(dataset_path, 'train')
            val_path = os.path.join(dataset_path, 'val')
            
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=10,
                width_shift_range=0.1,
                height_shift_range=0.1,
                horizontal_flip=True,
                fill_mode='nearest'
            )
            
            val_datagen = ImageDataGenerator(rescale=1./255)
            
            train_generator = train_datagen.flow_from_directory(
                train_path,
                target_size=(224, 224),
                batch_size=batch_size,
                class_mode='categorical',
                shuffle=True,
                seed=42
            )
            
            val_generator = val_datagen.flow_from_directory(
                val_path,
                target_size=(224, 224),
                batch_size=batch_size,
                class_mode='categorical',
                shuffle=False,
                seed=42
            )
            
        else:
            # Handle simple structure with validation split
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=10,
                width_shift_range=0.1,
                height_shift_range=0.1,
                horizontal_flip=True,
                fill_mode='nearest',
                validation_split=0.2
            )
            
            val_datagen = ImageDataGenerator(
                rescale=1./255,
                validation_split=0.2
            )
            
            train_generator = train_datagen.flow_from_directory(
                dataset_path,
                target_size=(224, 224),
                batch_size=batch_size,
                class_mode='categorical',
                subset='training',
                shuffle=True,
                seed=42
            )
            
            val_generator = val_datagen.flow_from_directory(
                dataset_path,
                target_size=(224, 224),
                batch_size=batch_size,
                class_mode='categorical',
                subset='validation',
                shuffle=False,
                seed=42
            )
        
        return train_generator, val_generator
    
    def train_single_model(self, model_key: str, epochs: int = 5, batch_size: int = 32):
        """Train a single model quickly"""
        
        config = self.model_configs[model_key]
        print(f"\n🚀 Training {config['name']}...")
        
        start_time = time.time()
        
        try:
            # Prepare data
            train_gen, val_gen = self.prepare_data_generators(model_key, batch_size)
            print(f"  📊 Data loaded: {train_gen.samples} train, {val_gen.samples} val samples")
            
            # Create model
            model, base_model = self.create_densenet121_model(model_key)
            print(f"  🧠 Model created: {model.count_params():,} parameters")
            
            # Calculate class weights
            class_weights = compute_class_weight(
                'balanced',
                classes=np.unique(train_gen.classes),
                y=train_gen.classes
            )
            class_weight_dict = dict(enumerate(class_weights))
            
            # Compile model
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy', 'precision', 'recall']
            )
            
            # Callbacks
            callbacks = [
                EarlyStopping(
                    monitor='val_accuracy',
                    patience=3,
                    restore_best_weights=True,
                    verbose=1,
                    mode='max'
                ),
                ModelCheckpoint(
                    config['model_file'],
                    monitor='val_accuracy',
                    save_best_only=True,
                    verbose=1,
                    mode='max'
                )
            ]
            
            # Train model
            print(f"  🏋️ Training for {epochs} epochs...")
            history = model.fit(
                train_gen,
                epochs=epochs,
                validation_data=val_gen,
                callbacks=callbacks,
                class_weight=class_weight_dict,
                verbose=1
            )
            
            # Quick evaluation
            val_gen.reset()
            test_loss, test_accuracy, test_precision, test_recall = model.evaluate(
                val_gen, verbose=0
            )
            
            # Generate predictions for classification report
            val_gen.reset()
            predictions = model.predict(val_gen, verbose=0)
            predicted_classes = np.argmax(predictions, axis=1)
            true_classes = val_gen.classes[:len(predicted_classes)]
            
            class_labels = list(val_gen.class_indices.keys())
            report = classification_report(
                true_classes, 
                predicted_classes,
                target_names=class_labels,
                output_dict=True
            )
            
            training_time = time.time() - start_time
            
            results = {
                'model_key': model_key,
                'test_accuracy': test_accuracy,
                'test_precision': test_precision,
                'test_recall': test_recall,
                'f1_score': report['weighted avg']['f1-score'],
                'classification_report': report,
                'training_time': training_time,
                'epochs_completed': len(history.history['accuracy']),
                'config': config
            }
            
            print(f"  ✅ {config['name']} completed in {training_time:.1f}s!")
            print(f"     Accuracy: {test_accuracy:.4f}")
            print(f"     Precision: {test_precision:.4f}")
            print(f"     Recall: {test_recall:.4f}")
            print(f"     F1-Score: {report['weighted avg']['f1-score']:.4f}")
            
            return results
            
        except Exception as e:
            print(f"  ❌ Error training {config['name']}: {str(e)}")
            return None
    
    def train_all_available_models(self, epochs: int = 5, batch_size: int = 32):
        """Train all available models"""
        
        # Check availability
        available_models = self.check_dataset_availability()
        
        if not available_models:
            print("❌ No datasets available for training!")
            return
        
        print(f"\n🚀 Starting quick training for {len(available_models)} models...")
        print(f"   Epochs: {epochs}")
        print(f"   Batch Size: {batch_size}")
        
        results = {}
        total_start_time = time.time()
        
        for i, model_key in enumerate(available_models, 1):
            print(f"\n{'='*50}")
            print(f"Model {i}/{len(available_models)}: {self.model_configs[model_key]['name']}")
            print(f"{'='*50}")
            
            result = self.train_single_model(model_key, epochs, batch_size)
            results[model_key] = result
        
        # Training summary
        total_time = time.time() - total_start_time
        successful_models = [k for k, v in results.items() if v is not None]
        
        print(f"\n🎉 TRAINING COMPLETE!")
        print(f"   Total Time: {total_time:.1f}s")
        print(f"   Successful Models: {len(successful_models)}/{len(available_models)}")
        
        if successful_models:
            # Update model registry
            self.update_model_registry(results)
            
            # Results table
            print(f"\n📊 RESULTS SUMMARY:")
            print("-" * 80)
            print(f"{'Model':<25} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
            print("-" * 80)
            
            for model_key in successful_models:
                result = results[model_key]
                config = result['config']
                print(f"{config['name']:<25} "
                      f"{result['test_accuracy']:<10.4f} "
                      f"{result['test_precision']:<10.4f} "
                      f"{result['test_recall']:<10.4f} "
                      f"{result['f1_score']:<10.4f}")
            
            print("-" * 80)
            
            # Next steps
            print(f"\n🔄 NEXT STEPS:")
            print(f"   1. Run full training with more epochs: python train_five_binary_models.py")
            print(f"   2. Use Streamlit interface: streamlit run streamlit_five_binary_trainer.py")
            print(f"   3. Test models with: python app.py")
            
        return results
    
    def update_model_registry(self, results_dict):
        """Update model registry with results"""
        
        registry_path = 'models/registry/model_registry.json'
        
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
        
        # Update registry for each successful model
        for model_key, model_data in results_dict.items():
            if model_data is not None:
                config = model_data['config']
                
                model_info = {
                    "model_path": config['model_file'],
                    "file_path": config['model_file'],
                    "dataset_type": model_key,
                    "model_name": f"{config['name']} - DenseNet121 Binary",
                    "architecture": "DenseNet121_Binary_Quick",
                    "version": "v1.0_quick",
                    "accuracy": float(model_data['test_accuracy']),
                    "precision": float(model_data['test_precision']),
                    "recall": float(model_data['test_recall']),
                    "f1_score": float(model_data['f1_score']),
                    "classes": config['classes'],
                    "input_shape": list(self.input_shape),
                    "trained_date": datetime.now().isoformat(),
                    "dataset": f"{config['name']} Binary Classification Dataset",
                    "training_method": "Quick_Binary_DenseNet121_Dropout",
                    "gradcam_target_layer": "gradcam_target_layer",
                    "classification_type": "binary",
                    "description": config['description'],
                    "training_time": model_data['training_time'],
                    "epochs_completed": model_data['epochs_completed']
                }
                
                # Add file size
                if os.path.exists(config['model_file']):
                    model_info["file_size"] = os.path.getsize(config['model_file'])
                
                registry["models"][model_key] = model_info
                registry["active_models"][model_key] = model_key
        
        registry["last_modified"] = datetime.now().isoformat()
        
        # Save registry
        with open(registry_path, 'w') as f:
            json.dump(registry, f, indent=2)
        
        print(f"✅ Model registry updated: {registry_path}")

def main():
    """Main function"""
    
    print("🏥 Quick 5 Binary Models Trainer")
    print("=" * 50)
    
    # Get parameters
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    
    # Initialize trainer
    trainer = QuickFiveBinaryModelsTrainer()
    
    # Train all available models
    trainer.train_all_available_models(epochs=epochs, batch_size=batch_size)

if __name__ == "__main__":
    main()