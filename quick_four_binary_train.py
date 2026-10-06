#!/usr/bin/env python3
"""
Quick Training Script for 4 Binary Models
Fast testing version with reduced epochs
"""

import os
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
from datetime import datetime

class QuickBinaryTrainer:
    """Quick trainer for testing the 4 binary models setup"""
    
    def __init__(self):
        self.model_configs = {
            'pneumonia': {
                'name': 'Pneumonia Detection',
                'dataset_path': 'Dataset/CHEST/Pneumonia_Organized',
                'classes': ['Normal', 'Pneumonia'],
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
                'dataset_path': 'Dataset/KNEE/Osteoarthritis/Combined_Osteoarthritis_Dataset',
                'classes': ['Normal', 'Osteoarthritis'],  
                'model_file': 'models/arthritis_binary_model.h5'
            },
            'osteoporosis': {
                'name': 'Osteoporosis Detection',
                'dataset_path': 'Dataset/KNEE/Osteoporosis/Combined_Osteoporosis_Dataset',
                'classes': ['Normal', 'Osteoporosis'],
                'model_file': 'models/osteoporosis_binary_model.h5'
            }
        }
        
        os.makedirs('models', exist_ok=True)
        os.makedirs('models/registry', exist_ok=True)
    
    def check_datasets(self):
        """Check which datasets are available"""
        available = {}
        
        print("🔍 Checking dataset availability...")
        
        for model_key, config in self.model_configs.items():
            path = config['dataset_path']
            
            if os.path.exists(path):
                try:
                    folders = os.listdir(path)
                    sample_count = 0
                    
                    for folder in folders:
                        folder_path = os.path.join(path, folder)
                        if os.path.isdir(folder_path):
                            files = [f for f in os.listdir(folder_path) 
                                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                            sample_count += len(files)
                    
                    available[model_key] = {
                        'status': True,
                        'samples': sample_count,
                        'folders': folders
                    }
                    
                    print(f"✅ {config['name']}: {sample_count} samples in {folders}")
                    
                except Exception as e:
                    available[model_key] = {'status': False, 'error': str(e)}
                    print(f"❌ {config['name']}: Error - {str(e)}")
            else:
                available[model_key] = {'status': False, 'error': 'Path not found'}
                print(f"❌ {config['name']}: Path not found - {path}")
        
        return available
    
    def create_quick_model(self, model_name: str):
        """Create a simplified DenseNet121 model for quick training"""
        
        inputs = keras.Input(shape=(224, 224, 3))
        
        # Simple preprocessing
        x = layers.Rescaling(1./255)(inputs)
        
        # DenseNet121 base (frozen for speed)
        base_model = DenseNet121(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3),
            pooling='avg'
        )
        base_model.trainable = False
        
        x = base_model(x, training=False)
        
        # Simple top layers with dropout
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(128, activation='relu', name='gradcam_target_layer')(x)
        x = layers.Dropout(0.3)(x)
        
        # Binary classification
        outputs = layers.Dense(2, activation='softmax')(x)
        
        model = keras.Model(inputs, outputs, name=f'quick_{model_name}')
        return model
    
    def prepare_quick_data(self, dataset_path: str):
        """Prepare data generators with minimal augmentation"""
        
        datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=0.2,
            horizontal_flip=True,
            rotation_range=10
        )
        
        train_gen = datagen.flow_from_directory(
            dataset_path,
            target_size=(224, 224),
            batch_size=16,  # Smaller batch for speed
            class_mode='categorical',
            subset='training',
            shuffle=True
        )
        
        val_gen = datagen.flow_from_directory(
            dataset_path,
            target_size=(224, 224),
            batch_size=16,
            class_mode='categorical',
            subset='validation',
            shuffle=False
        )
        
        return train_gen, val_gen
    
    def quick_train_model(self, model_key: str, epochs=5):
        """Quick train a single model"""
        
        config = self.model_configs[model_key]
        print(f"\n🚀 Quick training {config['name']}...")
        
        # Check if dataset exists
        if not os.path.exists(config['dataset_path']):
            print(f"❌ Dataset not found: {config['dataset_path']}")
            return None
        
        try:
            # Prepare data
            train_gen, val_gen = self.prepare_quick_data(config['dataset_path'])
            print(f"📊 Data: {train_gen.samples} train, {val_gen.samples} validation")
            
            # Create model
            model = self.create_quick_model(model_key)
            
            # Compile
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Callbacks
            callbacks = [
                EarlyStopping(
                    monitor='val_accuracy',
                    patience=3,
                    restore_best_weights=True,
                    verbose=1
                ),
                ModelCheckpoint(
                    config['model_file'],
                    monitor='val_accuracy',
                    save_best_only=True,
                    verbose=1
                )
            ]
            
            # Train
            history = model.fit(
                train_gen,
                epochs=epochs,
                validation_data=val_gen,
                callbacks=callbacks,
                verbose=1
            )
            
            # Evaluate
            val_gen.reset()
            val_loss, val_accuracy = model.evaluate(val_gen, verbose=0)
            
            print(f"✅ {config['name']} - Final Accuracy: {val_accuracy:.4f}")
            
            return {
                'model': model,
                'history': history,
                'accuracy': val_accuracy,
                'config': config
            }
            
        except Exception as e:
            print(f"❌ Error training {model_key}: {str(e)}")
            return None
    
    def train_all_quick(self, epochs=5):
        """Quick train all available models"""
        
        print("🎯 Quick Training 4 Binary Models")
        print("DenseNet121 with Dropout - Fast Version")
        print("=" * 50)
        
        # Check datasets
        available_datasets = self.check_datasets()
        
        results = {}
        successful = 0
        
        for model_key, config in self.model_configs.items():
            if available_datasets[model_key]['status']:
                result = self.quick_train_model(model_key, epochs)
                if result:
                    results[model_key] = result
                    successful += 1
            else:
                print(f"⏭️ Skipping {config['name']} - dataset not available")
        
        print(f"\n🎉 Quick training complete! {successful}/4 models trained.")
        
        if successful > 0:
            self.update_registry(results)
            self.print_summary(results)
        
        return results
    
    def update_registry(self, results):
        """Update model registry"""
        
        registry_path = 'models/registry/model_registry.json'
        
        if os.path.exists(registry_path):
            with open(registry_path, 'r') as f:
                registry = json.load(f)
        else:
            registry = {"version": "2.0", "models": {}, "active_models": {}}
        
        for model_key, result in results.items():
            config = result['config']
            
            registry["models"][model_key] = {
                "model_path": config['model_file'],
                "model_name": f"{config['name']} - Quick Binary",
                "architecture": "DenseNet121_Quick_Binary",
                "accuracy": float(result['accuracy']),
                "classes": config['classes'],
                "trained_date": datetime.now().isoformat(),
                "training_method": "Quick_Binary_Training",
                "classification_type": "binary"
            }
            
            registry["active_models"][model_key] = model_key
        
        registry["last_modified"] = datetime.now().isoformat()
        
        with open(registry_path, 'w') as f:
            json.dump(registry, f, indent=2)
        
        print(f"✅ Registry updated: {registry_path}")
    
    def print_summary(self, results):
        """Print training summary"""
        
        print("\n📊 Training Summary:")
        print("-" * 40)
        
        for model_key, result in results.items():
            config = result['config']
            accuracy = result['accuracy']
            print(f"{config['name']}: {accuracy:.3f} accuracy")
        
        print("\n✅ Models ready for medical diagnosis!")

def main():
    """Main function"""
    
    print("🏥 Quick Binary Models Trainer")
    print("Testing 4 DenseNet121 binary classification models")
    print("=" * 60)
    
    trainer = QuickBinaryTrainer()
    
    # Quick training with 5 epochs for testing
    results = trainer.train_all_quick(epochs=5)
    
    if results:
        print(f"\n🎊 Success! {len(results)} models trained and ready.")
        print("Use the full trainer for production-quality models.")
    else:
        print("\n❌ No models were trained. Check dataset paths.")

if __name__ == "__main__":
    main()