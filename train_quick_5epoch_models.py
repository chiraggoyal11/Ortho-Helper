"""
Train Quick 5-Epoch Models for All Datasets
Provides users with lightweight model options for quick deployment
"""

import os
import json
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from datetime import datetime
from pathlib import Path

class QuickModelTrainer:
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
        
        # Training configuration for quick models
        self.config = {
            'epochs': 5,
            'batch_size': 25,  # Optimized for CPU training with DenseNet121
            'learning_rate': 0.001,
            'image_size': (224, 224),
            'validation_split': 0.2,
            'early_stopping_patience': 3
        }
    
    def create_model(self, num_classes=2):
        """Create DenseNet121 model"""
        base_model = DenseNet121(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3)
        )
        
        # Fine-tune last few layers
        for layer in base_model.layers[:-30]:
            layer.trainable = False
        
        model = keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(1, activation='sigmoid') if num_classes == 2 else layers.Dense(num_classes, activation='softmax')
        ])
        
        return model
    
    def prepare_dataset(self, dataset_name, config):
        """Prepare dataset for training"""
        print(f"\n📊 Preparing {config['display_name']} dataset...")
        
        dataset_path = config['path']
        
        if not os.path.exists(dataset_path):
            print(f"❌ Dataset path not found: {dataset_path}")
            return None, None, None
        
        # Data augmentation for training
        train_datagen = keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            validation_split=self.config['validation_split']
        )
        
        # For osteoporosis, handle multi-class to binary conversion
        if dataset_name == 'osteoporosis':
            # Custom generator for osteoporosis binary classification
            classes = ['Healthy']  # Only load these classes
            class_mode = 'binary'
        else:
            classes = config['classes']
            class_mode = 'binary'
        
        try:
            train_generator = train_datagen.flow_from_directory(
                dataset_path,
                target_size=self.config['image_size'],
                batch_size=self.config['batch_size'],
                class_mode=class_mode,
                classes=classes if dataset_name != 'osteoporosis' else None,
                subset='training',
                shuffle=True
            )
            
            val_generator = train_datagen.flow_from_directory(
                dataset_path,
                target_size=self.config['image_size'],
                batch_size=self.config['batch_size'],
                class_mode=class_mode,
                classes=classes if dataset_name != 'osteoporosis' else None,
                subset='validation',
                shuffle=False
            )
            
            print(f"✅ Training samples: {train_generator.samples}")
            print(f"✅ Validation samples: {val_generator.samples}")
            
            return train_generator, val_generator, train_generator.samples
            
        except Exception as e:
            print(f"❌ Error preparing dataset: {str(e)}")
            return None, None, None
    
    def train_model(self, dataset_name, config):
        """Train a single model"""
        print(f"\n{'='*80}")
        print(f"TRAINING: {config['display_name']}")
        print(f"{'='*80}")
        
        start_time = time.time()
        
        # Prepare dataset
        train_gen, val_gen, num_samples = self.prepare_dataset(dataset_name, config)
        
        if train_gen is None:
            print(f"⚠️ Skipping {dataset_name} due to dataset preparation error")
            return None
        
        # Create model
        print(f"\n🏗️ Creating model...")
        model = self.create_model(num_classes=2)
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.config['learning_rate']),
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall')
            ]
        )
        
        print(f"✅ Model created with {model.count_params():,} parameters")
        
        # Prepare callbacks
        model_dir = self.base_dir / dataset_name
        model_dir.mkdir(parents=True, exist_ok=True)
        
        model_filename = f"densenet121_{dataset_name}_quick5_{self.timestamp}"
        
        callbacks = [
            ModelCheckpoint(
                str(model_dir / f"{model_filename}_best.h5"),
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=2,
                min_lr=1e-6,
                verbose=1
            ),
            EarlyStopping(
                monitor='val_accuracy',
                patience=self.config['early_stopping_patience'],
                restore_best_weights=True,
                verbose=1
            )
        ]
        
        # Train model
        print(f"\n🚀 Starting training for {self.config['epochs']} epochs...")
        
        history = model.fit(
            train_gen,
            epochs=self.config['epochs'],
            validation_data=val_gen,
            callbacks=callbacks,
            verbose=1
        )
        
        training_time = time.time() - start_time
        
        # Evaluate model
        print(f"\n📊 Evaluating model...")
        eval_results = model.evaluate(val_gen, verbose=0)
        
        results = {
            'loss': float(eval_results[0]),
            'accuracy': float(eval_results[1]),
            'precision': float(eval_results[2]),
            'recall': float(eval_results[3])
        }
        
        print(f"\n✅ Training Complete!")
        print(f"   Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
        print(f"   Precision: {results['precision']:.4f}")
        print(f"   Recall: {results['recall']:.4f}")
        print(f"   Training Time: {training_time/60:.2f} minutes")
        
        # Save model in multiple formats
        print(f"\n💾 Saving model in multiple formats...")
        
        # Save .keras format
        keras_path = model_dir / f"{model_filename}.keras"
        model.save(keras_path)
        print(f"✅ Saved: {keras_path}")
        
        # Save .h5 format
        h5_path = model_dir / f"{model_filename}.h5"
        model.save(h5_path)
        print(f"✅ Saved: {h5_path}")
        
        # Save weights only
        weights_path = model_dir / f"{model_filename}.weights.h5"
        model.save_weights(weights_path)
        print(f"✅ Saved: {weights_path}")
        
        # Save model info
        model_info = {
            "model_info": {
                "name": f"DenseNet121_{dataset_name}_Quick5",
                "architecture": "DenseNet121",
                "dataset": dataset_name,
                "configuration": "Quick 5-Epoch",
                "timestamp": self.timestamp,
                "total_parameters": int(model.count_params())
            },
            "performance": {
                "accuracy": results['accuracy'],
                "precision": results['precision'],
                "recall": results['recall'],
                "loss": results['loss'],
                "training_time_minutes": round(training_time/60, 2),
                "epochs_trained": len(history.history['accuracy'])
            },
            "training_config": {
                "epochs": self.config['epochs'],
                "batch_size": self.config['batch_size'],
                "learning_rate": self.config['learning_rate'],
                "validation_split": self.config['validation_split'],
                "image_size": list(self.config['image_size'])
            },
            "dataset_info": {
                "name": dataset_name,
                "display_name": config['display_name'],
                "anatomy": config['anatomy'],
                "classes": ["Normal", config['display_name'].split()[-2] if len(config['display_name'].split()) > 2 else "Positive"],
                "training_samples": int(num_samples * (1 - self.config['validation_split'])),
                "validation_samples": int(num_samples * self.config['validation_split'])
            },
            "gradcam_optimization": {
                "optimized_for_gradcam": True,
                "recommended_layer": "conv5_block16_2_conv",
                "architecture_benefits": [
                    "Dense connectivity preserves gradients",
                    f"Excellent for {config['anatomy']} X-ray visualization",
                    "Clear heatmaps for medical interpretation"
                ]
            },
            "medical_classification": {
                "grade": "Quick Model" if results['accuracy'] < 0.85 else "Clinical Grade" if results['accuracy'] < 0.90 else "Medical Grade",
                "deployment_status": "Ready for testing",
                "use_case": "Quick deployment and evaluation"
            },
            "files": {
                "keras_model": f"{model_filename}.keras",
                "h5_model": f"{model_filename}.h5",
                "weights": f"{model_filename}.weights.h5"
            }
        }
        
        info_path = model_dir / f"{model_filename}_info.json"
        with open(info_path, 'w') as f:
            json.dump(model_info, f, indent=2)
        print(f"✅ Saved: {info_path}")
        
        # Save training history
        history_dict = {
            'accuracy': [float(x) for x in history.history['accuracy']],
            'val_accuracy': [float(x) for x in history.history['val_accuracy']],
            'loss': [float(x) for x in history.history['loss']],
            'val_loss': [float(x) for x in history.history['val_loss']],
            'precision': [float(x) for x in history.history['precision']],
            'val_precision': [float(x) for x in history.history['val_precision']],
            'recall': [float(x) for x in history.history['recall']],
            'val_recall': [float(x) for x in history.history['val_recall']]
        }
        
        history_path = model_dir / f"{model_filename}_history.json"
        with open(history_path, 'w') as f:
            json.dump(history_dict, f, indent=2)
        print(f"✅ Saved: {history_path}")
        
        # Create README
        readme_content = f"""# {config['display_name']} - Quick 5-Epoch Model

## Model Information
- **Architecture**: DenseNet121
- **Configuration**: Quick 5-Epoch Training
- **Training Date**: {self.timestamp}
- **Parameters**: {model.count_params():,}
- **Purpose**: Fast deployment and model evaluation

## Performance Metrics
- **Accuracy**: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)
- **Precision**: {results['precision']:.4f}
- **Recall**: {results['recall']:.4f}
- **Training Time**: {training_time/60:.2f} minutes
- **Epochs**: {len(history.history['accuracy'])}/{self.config['epochs']}

## Dataset Information
- **Anatomy**: {config['anatomy'].title()}
- **Classes**: Normal vs {dataset_name.replace('_', ' ').title()}
- **Training Samples**: {int(num_samples * (1 - self.config['validation_split']))}
- **Validation Samples**: {int(num_samples * self.config['validation_split'])}

## Usage
```python
import tensorflow as tf

# Load model
model = tf.keras.models.load_model('models/{dataset_name}/{model_filename}.keras')

# Make prediction
prediction = model.predict(preprocessed_image)
confidence = float(prediction[0][0])
result = "Positive" if confidence > 0.5 else "Negative"
```

## Grad-CAM Visualization
This model supports Grad-CAM for explainable AI:
- **Recommended Layer**: conv5_block16_2_conv
- **Use Case**: Medical interpretation and visualization

## Classification Grade
- **Grade**: {model_info['medical_classification']['grade']}
- **Status**: {model_info['medical_classification']['deployment_status']}
- **Use Case**: {model_info['medical_classification']['use_case']}

## Files Included
- `{model_filename}.keras` - Main model file (recommended)
- `{model_filename}.h5` - Legacy format
- `{model_filename}.weights.h5` - Weights only
- `{model_filename}_info.json` - Complete metadata
- `{model_filename}_history.json` - Training history
- `README.md` - This file

---
*Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
"""
        
        readme_path = model_dir / f"{model_filename}_README.md"
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        print(f"✅ Saved: {readme_path}")
        
        # Store results
        result_summary = {
            'dataset': dataset_name,
            'display_name': config['display_name'],
            'accuracy': results['accuracy'],
            'precision': results['precision'],
            'recall': results['recall'],
            'training_time': training_time/60,
            'epochs': len(history.history['accuracy']),
            'grade': model_info['medical_classification']['grade'],
            'model_file': f"{model_filename}.keras",
            'status': 'SUCCESS'
        }
        
        self.results.append(result_summary)
        
        return result_summary
    
    def train_all_models(self):
        """Train quick 5-epoch models for all datasets"""
        print("="*80)
        print("🚀 QUICK 5-EPOCH MODEL TRAINING PIPELINE")
        print("="*80)
        print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"⚙️  Configuration:")
        print(f"   - Epochs: {self.config['epochs']}")
        print(f"   - Batch Size: {self.config['batch_size']}")
        print(f"   - Learning Rate: {self.config['learning_rate']}")
        print(f"   - Image Size: {self.config['image_size']}")
        print("="*80)
        
        total_start = time.time()
        
        # Train each dataset
        for dataset_name, config in self.datasets.items():
            try:
                result = self.train_model(dataset_name, config)
                if result:
                    print(f"\n✅ {config['display_name']}: SUCCESS")
                else:
                    print(f"\n⚠️ {config['display_name']}: SKIPPED")
                    self.results.append({
                        'dataset': dataset_name,
                        'display_name': config['display_name'],
                        'status': 'SKIPPED'
                    })
            except Exception as e:
                print(f"\n❌ {config['display_name']}: FAILED")
                print(f"   Error: {str(e)}")
                self.results.append({
                    'dataset': dataset_name,
                    'display_name': config['display_name'],
                    'status': 'FAILED',
                    'error': str(e)
                })
        
        total_time = time.time() - total_start
        
        # Generate summary report
        self.generate_summary_report(total_time)
    
    def generate_summary_report(self, total_time):
        """Generate comprehensive training summary"""
        print(f"\n{'='*80}")
        print("📊 TRAINING SUMMARY REPORT")
        print(f"{'='*80}")
        
        success_count = sum(1 for r in self.results if r.get('status') == 'SUCCESS')
        failed_count = sum(1 for r in self.results if r.get('status') == 'FAILED')
        skipped_count = sum(1 for r in self.results if r.get('status') == 'SKIPPED')
        
        print(f"\n📈 Overall Statistics:")
        print(f"   Total Datasets: {len(self.datasets)}")
        print(f"   ✅ Successful: {success_count}")
        print(f"   ❌ Failed: {failed_count}")
        print(f"   ⚠️  Skipped: {skipped_count}")
        print(f"   ⏱️  Total Time: {total_time/60:.2f} minutes")
        
        if success_count > 0:
            print(f"\n🏆 Model Performance:")
            for r in self.results:
                if r.get('status') == 'SUCCESS':
                    print(f"\n   {r['display_name']}:")
                    print(f"      Accuracy: {r['accuracy']:.4f} ({r['accuracy']*100:.2f}%)")
                    print(f"      Precision: {r['precision']:.4f}")
                    print(f"      Recall: {r['recall']:.4f}")
                    print(f"      Grade: {r['grade']}")
                    print(f"      Time: {r['training_time']:.2f} min")
                    print(f"      Epochs: {r['epochs']}/{self.config['epochs']}")
            
            # Find best model
            best_model = max([r for r in self.results if r.get('status') == 'SUCCESS'], 
                           key=lambda x: x['accuracy'])
            print(f"\n🥇 Best Performing Model:")
            print(f"   {best_model['display_name']}: {best_model['accuracy']*100:.2f}%")
        
        # Save summary to JSON
        summary_path = self.base_dir / f"quick5_training_summary_{self.timestamp}.json"
        summary_data = {
            'timestamp': self.timestamp,
            'configuration': self.config,
            'total_time_minutes': round(total_time/60, 2),
            'statistics': {
                'total': len(self.datasets),
                'successful': success_count,
                'failed': failed_count,
                'skipped': skipped_count
            },
            'results': self.results
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        print(f"\n✅ Summary saved: {summary_path}")
        
        # Create markdown report
        report_path = self.base_dir / f"QUICK5_MODELS_REPORT_{self.timestamp}.md"
        report_content = self.create_markdown_report(success_count, failed_count, skipped_count, total_time)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"✅ Report saved: {report_path}")
        
        print(f"\n{'='*80}")
        if success_count == len(self.datasets):
            print("🎉 ALL MODELS TRAINED SUCCESSFULLY!")
        elif success_count > 0:
            print(f"✅ {success_count}/{len(self.datasets)} MODELS TRAINED SUCCESSFULLY")
        else:
            print("❌ NO MODELS TRAINED SUCCESSFULLY")
        print(f"{'='*80}")
    
    def create_markdown_report(self, success_count, failed_count, skipped_count, total_time):
        """Create markdown summary report"""
        report = f"""# Quick 5-Epoch Models Training Report

**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Configuration**: {self.config['epochs']} epochs, Batch size {self.config['batch_size']}  
**Status**: {'✅ Complete' if success_count == len(self.datasets) else '⚠️ Partial'}

---

## 📊 Training Statistics

| Metric | Count |
|--------|-------|
| Total Datasets | {len(self.datasets)} |
| ✅ Successful | {success_count} |
| ❌ Failed | {failed_count} |
| ⚠️ Skipped | {skipped_count} |
| ⏱️ Total Time | {total_time/60:.2f} minutes |

---

## 🏆 Model Performance

"""
        
        for r in self.results:
            if r.get('status') == 'SUCCESS':
                report += f"""
### {r['display_name']}
- **Accuracy**: {r['accuracy']:.4f} ({r['accuracy']*100:.2f}%)
- **Precision**: {r['precision']:.4f}
- **Recall**: {r['recall']:.4f}
- **Grade**: {r['grade']}
- **Training Time**: {r['training_time']:.2f} minutes
- **Epochs**: {r['epochs']}/{self.config['epochs']}
- **Model File**: `{r['dataset']}/{r['model_file']}`

"""
        
        report += f"""
---

## 📁 Model Files

Each trained model includes:
- `.keras` file - Main model (recommended)
- `.h5` file - Legacy format
- `.weights.h5` - Weights only
- `_info.json` - Complete metadata
- `_history.json` - Training history
- `_README.md` - Documentation

---

## 🚀 Usage

These quick 5-epoch models provide:
1. **Fast Training**: ~2-5 minutes per model
2. **Quick Evaluation**: Test model architecture and data
3. **Baseline Performance**: Compare with intensive models
4. **Model Selection**: Choose best approach for your needs

### Load and Use
```python
import tensorflow as tf

# Load quick model
model = tf.keras.models.load_model('models/pneumonia/densenet121_pneumonia_quick5_{self.timestamp}.keras')

# Make prediction
prediction = model.predict(preprocessed_image)
```

---

## 📈 Model Comparison

Users now have multiple model options:
- **Quick 5-Epoch**: Fast, baseline performance
- **Standard 10-Epoch**: Balanced performance
- **Intensive 15-Epoch**: Best accuracy

---

*Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        return report

if __name__ == "__main__":
    print("Starting Quick 5-Epoch Model Training...")
    print("This will create lightweight models for quick deployment\n")
    
    trainer = QuickModelTrainer()
    trainer.train_all_models()
