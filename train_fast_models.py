"""
Fast Model Training - Optimized for Speed
Creates lightweight models in minimum time
"""

import os
import json
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from datetime import datetime
from pathlib import Path

class FastModelTrainer:
    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base_dir = Path("models")
        self.results = []
        
        # Dataset configurations
        self.datasets = {
            'pneumonia': {
                'path': 'Dataset/CHEST/Pneumonia_Organized',
                'classes': ['Normal', 'Pneumonia'],
                'display_name': '🫁 Pneumonia Detection',
                'anatomy': 'chest'
            },
            'cardiomegaly': {
                'path': 'Dataset/CHEST/cardiomelgy/train/train',
                'classes': ['false', 'true'],
                'display_name': '❤️ Cardiomegaly Detection',
                'anatomy': 'chest'
            },
            'arthritis': {
                'path': 'Dataset/KNEE/Osteoarthritis/Combined_Osteoarthritis_Dataset',
                'classes': ['Normal', 'Osteoarthritis'],
                'display_name': '🦵 Knee Arthritis Detection',
                'anatomy': 'knee'
            },
            'osteoporosis': {
                'path': 'Dataset/KNEE/Osteoporosis/Combined_Osteoporosis_Dataset',
                'classes': ['Normal', 'Osteoporosis'],
                'display_name': '🦴 Knee Osteoporosis Detection',
                'anatomy': 'knee'
            },
            'bone_fracture': {
                'path': 'Dataset/ARM/MURA_Organized/Forearm',
                'classes': ['Negative', 'Positive'],
                'display_name': '💀 Bone Fracture Detection',
                'anatomy': 'limb'
            }
        }
        
        # FAST training configuration
        self.config = {
            'epochs': 3,  # Reduced from 5 for speed
            'batch_size': 64,  # Larger batches = fewer steps
            'learning_rate': 0.002,  # Higher LR for faster convergence
            'image_size': (128, 128),  # Smaller images = 4x faster
            'validation_split': 0.15,  # Less validation data
            'steps_per_epoch': 50,  # Limit steps per epoch
            'validation_steps': 10,  # Limit validation steps
        }
    
    def create_model(self, num_classes=2):
        """Create MobileNetV2 model (much faster than DenseNet121)"""
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(128, 128, 3),
            alpha=0.75  # Reduced width for speed
        )
        
        # Freeze ALL base layers for maximum speed
        base_model.trainable = False
        
        model = keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(64, activation='relu'),  # Smaller dense layers
            layers.Dropout(0.3),
            layers.Dense(1, activation='sigmoid') if num_classes == 2 else layers.Dense(num_classes, activation='softmax')
        ])
        
        return model
    
    def prepare_dataset(self, dataset_name, config):
        """Prepare dataset with minimal augmentation"""
        print(f"\n📊 Preparing {config['display_name']} dataset...")
        
        dataset_path = Path(config['path'])
        if not dataset_path.exists():
            print(f"❌ Dataset path not found: {config['path']}")
            return None, None
        
        # Minimal data augmentation for speed
        train_datagen = keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            validation_split=self.config['validation_split']
        )
        
        try:
            train_generator = train_datagen.flow_from_directory(
                config['path'],
                target_size=self.config['image_size'],
                batch_size=self.config['batch_size'],
                class_mode='binary' if len(config['classes']) == 2 else 'categorical',
                subset='training',
                shuffle=True
            )
            
            val_generator = train_datagen.flow_from_directory(
                config['path'],
                target_size=self.config['image_size'],
                batch_size=self.config['batch_size'],
                class_mode='binary' if len(config['classes']) == 2 else 'categorical',
                subset='validation',
                shuffle=False
            )
            
            print(f"✅ Training samples: {train_generator.samples}")
            print(f"✅ Validation samples: {val_generator.samples}")
            
            return train_generator, val_generator
            
        except Exception as e:
            print(f"❌ Error preparing dataset: {e}")
            return None, None
    
    def train_model(self, dataset_name, config):
        """Train a single model quickly"""
        print(f"\n{'='*80}")
        print(f"TRAINING: {config['display_name']}")
        print(f"{'='*80}\n")
        
        start_time = time.time()
        
        # Prepare dataset
        train_gen, val_gen = self.prepare_dataset(dataset_name, config)
        if train_gen is None:
            print(f"\n⚠️ {config['display_name']}: SKIPPED\n")
            return None
        
        # Create model
        print(f"\n🏗️ Creating model...")
        num_classes = len(config['classes'])
        model = self.create_model(num_classes)
        
        # Compile with faster optimizer
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.config['learning_rate']),
            loss='binary_crossentropy' if num_classes == 2 else 'categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"✅ Model created with {model.count_params():,} parameters")
        
        # Setup callbacks - minimal for speed
        model_dir = self.base_dir / dataset_name
        model_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = model_dir / f"mobilenet_{dataset_name}_fast_{self.timestamp}_best.h5"
        
        callbacks = [
            ModelCheckpoint(
                str(model_path),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=2,
                restore_best_weights=True,
                verbose=1
            )
        ]
        
        # Train with limited steps
        print(f"\n🚀 Starting FAST training for {self.config['epochs']} epochs...")
        print(f"   (Limited to {self.config['steps_per_epoch']} steps per epoch)\n")
        
        history = model.fit(
            train_gen,
            epochs=self.config['epochs'],
            validation_data=val_gen,
            callbacks=callbacks,
            steps_per_epoch=min(self.config['steps_per_epoch'], len(train_gen)),
            validation_steps=min(self.config['validation_steps'], len(val_gen)),
            verbose=1
        )
        
        training_time = time.time() - start_time
        
        # Save final model
        final_path = model_dir / f"mobilenet_{dataset_name}_fast_{self.timestamp}_final.h5"
        model.save(str(final_path))
        
        # Convert to .keras format
        keras_path = model_dir / f"mobilenet_{dataset_name}_fast_{self.timestamp}_final.keras"
        model.save(str(keras_path))
        
        # Get best metrics
        best_acc = max(history.history['val_accuracy'])
        best_epoch = history.history['val_accuracy'].index(best_acc) + 1
        
        result = {
            'dataset': dataset_name,
            'display_name': config['display_name'],
            'training_time': round(training_time / 60, 2),
            'epochs_trained': len(history.history['loss']),
            'best_epoch': best_epoch,
            'best_accuracy': round(best_acc * 100, 2),
            'final_accuracy': round(history.history['val_accuracy'][-1] * 100, 2),
            'model_path': str(model_path),
            'parameters': model.count_params()
        }
        
        self.results.append(result)
        
        print(f"\n{'='*80}")
        print(f"✅ {config['display_name']}: COMPLETED")
        print(f"   ⏱️  Training Time: {result['training_time']:.2f} minutes")
        print(f"   🎯 Best Accuracy: {result['best_accuracy']:.2f}%")
        print(f"   📊 Final Accuracy: {result['final_accuracy']:.2f}%")
        print(f"   💾 Model saved to: {model_path.name}")
        print(f"{'='*80}\n")
        
        return result
    
    def train_all_models(self):
        """Train all models sequentially"""
        print("\n" + "="*80)
        print("🚀 FAST MODEL TRAINING PIPELINE")
        print("="*80)
        print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"⚙️  Configuration:")
        print(f"   - Epochs: {self.config['epochs']}")
        print(f"   - Batch Size: {self.config['batch_size']}")
        print(f"   - Learning Rate: {self.config['learning_rate']}")
        print(f"   - Image Size: {self.config['image_size']}")
        print(f"   - Steps per Epoch: {self.config['steps_per_epoch']}")
        print(f"   - Architecture: MobileNetV2 (Fast & Lightweight)")
        print("="*80)
        
        start_time = time.time()
        
        # Train each dataset
        for dataset_name, dataset_config in self.datasets.items():
            try:
                self.train_model(dataset_name, dataset_config)
            except KeyboardInterrupt:
                print("\n⚠️ Training interrupted by user")
                break
            except Exception as e:
                print(f"\n❌ Error training {dataset_name}: {e}")
                continue
        
        total_time = time.time() - start_time
        
        # Generate summary report
        self.generate_report(total_time)
    
    def generate_report(self, total_time):
        """Generate training summary report"""
        print("\n" + "="*80)
        print("📊 FAST TRAINING SUMMARY REPORT")
        print("="*80)
        print(f"⏱️  Total Training Time: {total_time/60:.2f} minutes ({total_time/3600:.2f} hours)")
        print(f"✅ Models Trained: {len(self.results)}/5")
        print("\n" + "-"*80)
        print(f"{'Model':<30} {'Time (min)':<12} {'Accuracy':<12} {'Parameters':<15}")
        print("-"*80)
        
        for result in self.results:
            print(f"{result['display_name']:<30} {result['training_time']:<12.2f} "
                  f"{result['best_accuracy']:<12.2f}% {result['parameters']:<15,}")
        
        print("="*80)
        
        # Calculate average training time
        if self.results:
            avg_time = sum(r['training_time'] for r in self.results) / len(self.results)
            avg_acc = sum(r['best_accuracy'] for r in self.results) / len(self.results)
            print(f"\n📈 Average Training Time: {avg_time:.2f} minutes per model")
            print(f"📈 Average Accuracy: {avg_acc:.2f}%")
        
        # Save results to JSON
        results_file = self.base_dir / f"fast_training_results_{self.timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump({
                'timestamp': self.timestamp,
                'total_time_minutes': round(total_time/60, 2),
                'configuration': self.config,
                'results': self.results
            }, f, indent=2)
        
        print(f"\n💾 Results saved to: {results_file}")
        print("\n✅ FAST TRAINING COMPLETE!")
        print("="*80 + "\n")

def main():
    print("\nStarting FAST Model Training...")
    print("This will create lightweight models optimized for speed\n")
    
    trainer = FastModelTrainer()
    trainer.train_all_models()

if __name__ == "__main__":
    main()
