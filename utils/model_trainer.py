# Model Training Module for Medical X-ray AI System

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.utils import plot_model
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import pickle
import streamlit as st
from typing import Dict, Any, Tuple, List, Optional
from utils.data_loader import MedicalDataLoader
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class MedicalModelTrainer:
    """
    Advanced model trainer for medical X-ray classification
    Supports multiple architectures and training strategies
    """
    
    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Available model architectures
        self.model_architectures = {
            'DenseNet121': {
                'base_model': DenseNet121,
                'input_shape': (224, 224, 3),
                'preprocess': tf.keras.applications.densenet.preprocess_input,
                'description': 'Dense connections, efficient feature reuse - Optimized for medical imaging'
            }
        }
        
        # Training configurations
        self.training_configs = {
            'quick_test': {
                'epochs': 5,
                'batch_size': 16,
                'learning_rate': 0.001,
                'description': 'Quick test training for validation'
            },
            'standard': {
                'epochs': 25,
                'batch_size': 32,
                'learning_rate': 0.0001,
                'description': 'Standard training for good results'
            },
            'intensive': {
                'epochs': 50,
                'batch_size': 16,
                'learning_rate': 0.00001,
                'description': 'Intensive training for best performance'
            }
        }
    
    def create_model(self, architecture: str, num_classes: int, input_shape: Tuple[int, int, int] = (224, 224, 3)) -> tf.keras.Model:
        """Create a model based on specified architecture"""
        
        if architecture not in self.model_architectures:
            raise ValueError(f"Unsupported architecture: {architecture}")
        
        # Only DenseNet121 is supported now
        return self._create_transfer_learning_model(architecture, num_classes, input_shape)
    
    def _create_transfer_learning_model(self, architecture: str, num_classes: int, input_shape: Tuple[int, int, int]) -> tf.keras.Model:
        """Create transfer learning model using pre-trained base"""
        
        arch_config = self.model_architectures[architecture]
        base_model_class = arch_config['base_model']
        
        # Create base model
        base_model = base_model_class(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
        
        # Enhanced freezing strategy for better performance
        if architecture == 'DenseNet121' and num_classes == 2:  # Cardiomegaly case
            # Unfreeze more layers for cardiomegaly (medical images need more adaptation)
            for layer in base_model.layers[:-60]:  # Freeze only first layers
                layer.trainable = False
            for layer in base_model.layers[-60:]:  # Unfreeze last 60 layers
                layer.trainable = True
        else:
            # Standard freezing for other cases
            base_model.trainable = False
        
        # Enhanced architecture with attention mechanism for medical imaging
        inputs = layers.Input(shape=input_shape)
        
        # Data augmentation for medical images (only during training)
        x = layers.RandomFlip("horizontal")(inputs)
        x = layers.RandomRotation(0.1)(x)
        x = layers.RandomZoom(0.1)(x)
        x = layers.RandomContrast(0.2)(x)  # Important for X-ray contrast
        
        # Base model features
        x = base_model(x, training=True)
        x = layers.GlobalAveragePooling2D()(x)
        
        # Attention mechanism for better feature focus
        attention = layers.Dense(1024, activation='relu')(x)
        attention = layers.Dense(1024, activation='sigmoid')(attention)
        x = layers.multiply([x, attention])
        
        # Enhanced dense layers with better regularization
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        # Grad-CAM compatible layer
        x = layers.Dense(128, activation='relu', name='gradcam_target_layer')(x)
        x = layers.Dropout(0.1)(x)
        
        # Output layer
        outputs = layers.Dense(num_classes, activation='softmax' if num_classes > 2 else 'sigmoid')(x)
        
        model = models.Model(inputs, outputs)
        return model
    
    def compile_model(self, model: tf.keras.Model, num_classes: int, learning_rate: float = 0.001) -> tf.keras.Model:
        """Compile model with appropriate loss function and metrics"""
        
        # Use sparse categorical crossentropy for binary classification since labels are integers
        # All models are now binary (2 classes: Normal vs Condition)
        loss = 'sparse_categorical_crossentropy'
        
        # Use only basic metrics to avoid tensor shape issues
        metrics = ['accuracy']
        
        # Compile model
        model.compile(
            optimizer=optimizers.Adam(learning_rate=learning_rate),
            loss=loss,
            metrics=metrics
        )
        
        return model
    
    def create_callbacks(self, model_name: str, dataset_name: str) -> List[tf.keras.callbacks.Callback]:
        """Create enhanced training callbacks with cardiomegaly-specific improvements"""
        
        # Create directories
        checkpoint_dir = self.model_dir / "checkpoints" / f"{dataset_name}_{model_name}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        logs_dir = self.model_dir / "logs" / f"{dataset_name}_{model_name}"
        logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Enhanced callbacks for better cardiomegaly training
        callbacks_list = [
            # Model checkpoint
            callbacks.ModelCheckpoint(
                filepath=str(checkpoint_dir / "best_model.h5"),
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            ),
            
            # Enhanced early stopping with better patience for cardiomegaly
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15 if dataset_name == 'cardiomegaly' else 10,  # More patience for cardiomegaly
                restore_best_weights=True,
                verbose=1,
                min_delta=0.001
            ),
            
            # Enhanced learning rate reduction
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5 if dataset_name == 'cardiomegaly' else 0.2,  # Gentler LR reduction for cardiomegaly
                patience=8 if dataset_name == 'cardiomegaly' else 5,
                min_lr=1e-7,
                verbose=1,
                cooldown=2
            ),
            
            # TensorBoard logging
            callbacks.TensorBoard(
                log_dir=str(logs_dir),
                histogram_freq=1,
                write_graph=True,
                write_images=True
            ),
            
            # CSV logger
            callbacks.CSVLogger(
                str(logs_dir / "training_log.csv"),
                append=True
            )
        ]
        
        # Add cyclical learning rate for cardiomegaly
        if dataset_name == 'cardiomegaly':
            # Cyclical learning rate callback for better convergence
            def cyclical_lr(epoch):
                base_lr = 1e-4
                max_lr = 1e-3
                cycle_len = 8
                cycle = np.floor(1 + epoch / (2 * cycle_len))
                x = np.abs(epoch / cycle_len - 2 * cycle + 1)
                lr = base_lr + (max_lr - base_lr) * max(0, (1 - x))
                return lr
            
            callbacks_list.append(
                callbacks.LearningRateScheduler(cyclical_lr, verbose=0)
            )
        
        return callbacks_list
    
    def train_model(self, 
                   model: tf.keras.Model,
                   train_ds: tf.data.Dataset,
                   val_ds: tf.data.Dataset,
                   dataset_name: str,
                   architecture: str,
                   config_name: str = 'standard') -> Dict[str, Any]:
        """Train the model with specified configuration"""
        
        if config_name not in self.training_configs:
            raise ValueError(f"Unknown training config: {config_name}")
        
        config = self.training_configs[config_name]
        
        st.info(f"🚀 Starting training: {architecture} on {dataset_name}")
        st.info(f"Configuration: {config['description']}")
        
        # Create callbacks
        callbacks_list = self.create_callbacks(architecture, dataset_name)
        
        # Training progress placeholder
        progress_bar = st.progress(0)
        status_text = st.empty()
        metrics_placeholder = st.empty()
        
        # Custom callback for Streamlit updates
        class StreamlitCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                progress = (epoch + 1) / config['epochs']
                progress_bar.progress(progress)
                status_text.text(f"Epoch {epoch + 1}/{config['epochs']}")
                
                if logs:
                    metrics_text = f"""
                    **Training Metrics:**
                    - Loss: {logs.get('loss', 0):.4f}
                    - Accuracy: {logs.get('accuracy', 0):.4f}
                    - Val Loss: {logs.get('val_loss', 0):.4f}
                    - Val Accuracy: {logs.get('val_accuracy', 0):.4f}
                    """
                    metrics_placeholder.markdown(metrics_text)
        
        callbacks_list.append(StreamlitCallback())
        
        # Start training
        start_time = time.time()
        
        try:
            history = model.fit(
                train_ds,
                epochs=config['epochs'],
                validation_data=val_ds,
                callbacks=callbacks_list,
                verbose=0  # Suppress default output since we have custom callback
            )
            
            training_time = time.time() - start_time
            
            # Save model
            model_path = self.model_dir / f"{dataset_name}_{architecture}_model.h5"
            model.save(str(model_path))
            
            st.success(f"✅ Training completed in {training_time/60:.1f} minutes!")
            st.success(f"💾 Model saved: {model_path}")
            
            # Save training history
            history_path = self.model_dir / f"{dataset_name}_{architecture}_history.json"
            with open(history_path, 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                history_dict = {key: [float(val) for val in values] for key, values in history.history.items()}
                json.dump(history_dict, f, indent=2)
            
            # Create training summary
            training_summary = {
                'dataset_name': dataset_name,
                'architecture': architecture,
                'config_name': config_name,
                'training_time_minutes': training_time / 60,
                'epochs_completed': len(history.history['loss']),
                'final_train_accuracy': float(history.history['accuracy'][-1]),
                'final_val_accuracy': float(history.history['val_accuracy'][-1]),
                'best_val_accuracy': float(max(history.history['val_accuracy'])),
                'model_path': str(model_path),
                'history_path': str(history_path),
                'timestamp': datetime.now().isoformat()
            }
            
            # Save training summary
            summary_path = self.model_dir / f"{dataset_name}_{architecture}_summary.json"
            with open(summary_path, 'w') as f:
                json.dump(training_summary, f, indent=2)
            
            return training_summary
            
        except Exception as e:
            st.error(f"❌ Training failed: {str(e)}")
            return {}
    
    def train_model_with_progress(self, 
                                model: tf.keras.Model,
                                train_ds: tf.data.Dataset,
                                val_ds: tf.data.Dataset,
                                dataset_name: str,
                                architecture: str,
                                config: Dict[str, Any],
                                progress_bar,
                                status_text,
                                metrics_container) -> Dict[str, Any]:
        """Train model with real-time Streamlit progress updates"""
        
        st.info(f"🚀 Starting {config['description']}")
        
        # Create callbacks
        callbacks_list = self.create_callbacks(architecture, dataset_name)
        
        # Custom callback for real-time Streamlit updates
        class LiveProgressCallback(tf.keras.callbacks.Callback):
            def __init__(self, total_epochs, progress_bar, status_text, metrics_container):
                super().__init__()
                self.total_epochs = total_epochs
                self.progress_bar = progress_bar
                self.status_text = status_text
                self.metrics_container = metrics_container
                self.current_epoch = 0
                
            def on_epoch_begin(self, epoch, logs=None):
                self.current_epoch = epoch + 1
                self.status_text.text(f"🔥 Training Epoch {self.current_epoch}/{self.total_epochs}...")
                
            def on_epoch_end(self, epoch, logs=None):
                # Update progress bar
                progress = (epoch + 1) / self.total_epochs
                self.progress_bar.progress(progress)
                
                # Update status
                self.status_text.text(f"✅ Completed Epoch {epoch + 1}/{self.total_epochs}")
                
                # Update metrics
                if logs:
                    with self.metrics_container:
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Training Loss", f"{logs.get('loss', 0):.4f}")
                        with col2:
                            st.metric("Training Accuracy", f"{logs.get('accuracy', 0)*100:.1f}%")
                        with col3:
                            st.metric("Validation Loss", f"{logs.get('val_loss', 0):.4f}")
                        with col4:
                            st.metric("Validation Accuracy", f"{logs.get('val_accuracy', 0)*100:.1f}%")
        
        # Add live progress callback
        live_callback = LiveProgressCallback(
            config['epochs'], progress_bar, status_text, metrics_container
        )
        callbacks_list.append(live_callback)
        
        # Start training
        start_time = time.time()
        
        try:
            history = model.fit(
                train_ds,
                epochs=config['epochs'],
                validation_data=val_ds,
                callbacks=callbacks_list,
                verbose=0  # Suppress default output since we have custom callback
            )
            
            training_time = time.time() - start_time
            
            # Final progress update
            progress_bar.progress(1.0)
            status_text.text(f"🎉 Training Complete! ({training_time/60:.1f} minutes)")
            
            # Save model
            model_path = self.model_dir / f"{dataset_name}_{architecture}_model.h5"
            model.save(str(model_path))
            
            st.success(f"💾 Model saved: {model_path.name}")
            
            # Save training history
            history_path = self.model_dir / f"{dataset_name}_{architecture}_history.json"
            with open(history_path, 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                history_dict = {key: [float(val) for val in values] for key, values in history.history.items()}
                json.dump(history_dict, f, indent=2)
            
            # Create training summary
            training_summary = {
                'dataset_name': dataset_name,
                'architecture': architecture,
                'config_name': 'custom',
                'training_time': training_time / 60,
                'epochs_completed': len(history.history['loss']),
                'final_accuracy': float(history.history['val_accuracy'][-1]),
                'final_loss': float(history.history['val_loss'][-1]),
                'best_accuracy': float(max(history.history['val_accuracy'])),
                'model_path': str(model_path)
            }
            
            # Register model in model registry for management interface
            try:
                from utils.model_manager import ModelManager
                manager = ModelManager()
                
                # Get dataset-specific class information - Binary models only
                if dataset_name == 'bone_fracture':
                    class_names = ['Normal', 'Fracture']
                elif dataset_name == 'pneumonia':
                    class_names = ['Normal', 'Pneumonia']
                elif dataset_name == 'cardiomegaly':
                    class_names = ['Normal', 'Cardiomegaly']
                elif dataset_name == 'arthritis':
                    class_names = ['Normal', 'Arthritis']
                elif dataset_name == 'osteoporosis':
                    class_names = ['Normal', 'Osteoporosis']
                else:
                    class_names = ['Normal', 'Condition']
                
                # All models are binary (2 classes)
                num_classes = 2
                
                # Create model info for registration
                model_info = {
                    'model_name': f"{dataset_name.replace('_', ' ').title()} {architecture}",
                    'dataset_type': dataset_name,
                    'architecture': architecture,
                    'version': 'v1.0',
                    'input_shape': [224, 224, 3],
                    'num_classes': num_classes,
                    'class_names': class_names,
                    'performance_metrics': {
                        'test_accuracy': float(history.history['val_accuracy'][-1]),
                        'test_loss': float(history.history['val_loss'][-1]),
                        'final_accuracy': float(history.history['val_accuracy'][-1])
                    },
                    'training_info': {
                        'epochs': config['epochs'],
                        'batch_size': config['batch_size'],
                        'learning_rate': config['learning_rate'],
                        'architecture': architecture,
                        'training_time': training_time / 60
                    },
                    'file_path': str(model_path),
                    'file_size': model_path.stat().st_size,
                    'file_hash': '',  # Will be calculated by ModelManager
                    'created_date': datetime.now().isoformat(),
                    'description': f"{architecture} model trained on {dataset_name} dataset using Streamlit interface",
                    'tags': ['streamlit', 'medical', 'xray', dataset_name, architecture]
                }
                
                # Register the model
                manager.register_model(str(model_path), model_info)
                st.info("📝 Model registered in management system")
                
            except Exception as e:
                st.warning(f"⚠️ Model training completed but registration failed: {str(e)}")
            
            return training_summary
            
        except Exception as e:
            error_msg = str(e)
            st.error(f"❌ Training failed: {error_msg}")
            status_text.text("❌ Training interrupted")
            
            # Provide specific guidance based on error type
            if "Dimensions must be equal" in error_msg or "shapes" in error_msg.lower():
                st.info("💡 **Batch size issue detected.** Try reducing the batch size to 4 or 8.")
            elif "out of memory" in error_msg.lower() or "oom" in error_msg.lower():
                st.info("💡 **Memory issue detected.** Try reducing batch size or image resolution.")
            elif "no such file" in error_msg.lower() or "dataset" in error_msg.lower():
                st.info("💡 **Dataset issue detected.** Please re-prepare the dataset in Dataset Overview.")
            else:
                st.info("💡 Try re-preparing the dataset or adjusting training parameters.")
            
            # Reset training state
            st.session_state.training_started = False
            st.session_state.training_complete = False
            return {}
    
    def evaluate_model(self, model: tf.keras.Model, test_ds: tf.data.Dataset, class_names: List[str]) -> Dict[str, Any]:
        """Evaluate trained model on test set"""
        
        st.info("📊 Evaluating model performance...")
        
        # Make predictions
        y_pred_probs = model.predict(test_ds)
        y_pred = np.argmax(y_pred_probs, axis=1)
        
        # Get true labels
        y_true = []
        for batch in test_ds:
            y_true.extend(batch[1].numpy())
        y_true = np.array(y_true)
        
        # Calculate metrics
        test_loss, test_accuracy = model.evaluate(test_ds, verbose=0)
        
        # Classification report
        class_report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        evaluation_results = {
            'test_loss': float(test_loss),
            'test_accuracy': float(test_accuracy),
            'classification_report': class_report,
            'confusion_matrix': cm.tolist(),
            'class_names': class_names,
            'predictions': y_pred_probs.tolist()
        }
        
        # Display results
        st.success(f"✅ Test Accuracy: {test_accuracy:.4f}")
        
        # Show classification report
        st.markdown("### 📋 Classification Report")
        report_df = pd.DataFrame(class_report).transpose()
        st.dataframe(report_df)
        
        # Show confusion matrix
        st.markdown("### 🔄 Confusion Matrix")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_title('Confusion Matrix')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        st.pyplot(fig)
        plt.close()
        
        return evaluation_results
    
    def plot_training_history(self, history_path: str):
        """Plot training history from saved JSON file"""
        
        with open(history_path, 'r') as f:
            history = json.load(f)
        
        # Create plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy plot
        ax1.plot(history['accuracy'], label='Training Accuracy')
        ax1.plot(history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Loss plot
        ax2.plot(history['loss'], label='Training Loss')
        ax2.plot(history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        # Learning rate plot (if available)
        if 'lr' in history:
            ax3.plot(history['lr'])
            ax3.set_title('Learning Rate')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Learning Rate')
            ax3.grid(True)
        else:
            ax3.text(0.5, 0.5, 'Learning Rate\nNot Available', 
                    ha='center', va='center', transform=ax3.transAxes)
        
        # Additional metrics (if available)
        if 'val_precision' in history:
            ax4.plot(history['val_precision'], label='Precision')
            ax4.plot(history['val_recall'], label='Recall')
            ax4.set_title('Additional Metrics')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Score')
            ax4.legend()
            ax4.grid(True)
        else:
            ax4.text(0.5, 0.5, 'Additional Metrics\nNot Available', 
                    ha='center', va='center', transform=ax4.transAxes)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    def create_model_card(self, summary: Dict[str, Any], evaluation: Dict[str, Any]) -> str:
        """Create a model card with training and evaluation information"""
        
        model_card = f"""
# Medical X-ray Classification Model

## Model Information
- **Dataset:** {summary['dataset_name']}
- **Architecture:** {summary['architecture']}
- **Training Configuration:** {summary['config_name']}
- **Training Time:** {summary['training_time_minutes']:.1f} minutes
- **Training Date:** {summary['timestamp'][:10]}

## Performance Metrics
- **Final Training Accuracy:** {summary['final_train_accuracy']:.4f}
- **Final Validation Accuracy:** {summary['final_val_accuracy']:.4f}
- **Best Validation Accuracy:** {summary['best_val_accuracy']:.4f}
- **Test Accuracy:** {evaluation['test_accuracy']:.4f}

## Model Files
- **Model File:** `{summary['model_path']}`
- **Training History:** `{summary['history_path']}`

## Class Performance
"""
        
        # Add per-class performance
        for class_name in evaluation['class_names']:
            if class_name in evaluation['classification_report']:
                metrics = evaluation['classification_report'][class_name]
                model_card += f"""
### {class_name}
- Precision: {metrics['precision']:.4f}
- Recall: {metrics['recall']:.4f}
- F1-Score: {metrics['f1-score']:.4f}
"""
        
        model_card += f"""
## Usage
This model can be loaded and used for inference on {summary['dataset_name']} X-ray images.

```python
import tensorflow as tf
model = tf.keras.models.load_model('{summary['model_path']}')
```

## Training Details
- Epochs Completed: {summary['epochs_completed']}
- Architecture: {summary['architecture']}
- Input Shape: (224, 224, 3)
- Number of Classes: {len(evaluation['class_names'])}
"""
        
        return model_card

# Import pandas for dataframe operations
try:
    import pandas as pd
except ImportError:
    st.warning("pandas not installed. Some features may be limited.")
    import numpy as pd  # Fallback

def display_training_interface():
    """Display training interface in Streamlit"""
    
    st.markdown("## 🚀 Model Training Interface")
    
    trainer = MedicalModelTrainer()
    
    # Architecture selection
    st.markdown("### 🏗️ Model Architecture")
    architecture = st.selectbox(
        "Choose model architecture:",
        list(trainer.model_architectures.keys()),
        help="Select the neural network architecture for training"
    )
    
    st.info(f"**{architecture}:** {trainer.model_architectures[architecture]['description']}")
    
    # Training configuration
    st.markdown("### ⚙️ Training Configuration")
    config_name = st.selectbox(
        "Choose training configuration:",
        list(trainer.training_configs.keys()),
        help="Select training intensity and duration"
    )
    
    config = trainer.training_configs[config_name]
    st.info(f"**{config_name}:** {config['description']}")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Epochs", config['epochs'])
    with col2:
        st.metric("Batch Size", config['batch_size'])
    with col3:
        st.metric("Learning Rate", f"{config['learning_rate']:.0e}")
    
    # Dataset selection
    st.markdown("### 📊 Dataset Selection")
    
    # Dataset selection - Binary models only
    dataset_options = ['bone_fracture', 'pneumonia', 'cardiomegaly', 'arthritis', 'osteoporosis']
    dataset_help = "Binary models specialized for detecting one specific condition vs normal"
    
    dataset_name = st.selectbox(
        "Choose dataset to train on:",
        dataset_options,
        help=dataset_help
    )
    
    # Show model description - Binary models only
    model_descriptions = {
        'bone_fracture': '🦴 Bone Fracture Detection (Binary: Normal vs Fracture)',
        'pneumonia': '🫁 Pneumonia Detection (Binary: Normal vs Pneumonia)',
        'cardiomegaly': '❤️ Cardiomegaly Detection (Binary: Normal vs Cardiomegaly)',
        'arthritis': '🦵 Arthritis Detection (Binary: Normal vs Arthritis)',
        'osteoporosis': '🦴 Osteoporosis Detection (Binary: Normal vs Osteoporosis)'
    }
    
    st.info(f"📋 **Selected Model:** {model_descriptions.get(dataset_name, 'Unknown model')}")
    
    # Advanced options
    with st.expander("🔧 Advanced Options"):
        custom_epochs = st.number_input("Custom Epochs", min_value=1, max_value=100, value=config['epochs'])
        custom_batch_size = st.number_input("Custom Batch Size", min_value=8, max_value=64, value=config['batch_size'])
        custom_lr = st.number_input("Custom Learning Rate", min_value=1e-6, max_value=1e-1, value=config['learning_rate'], format="%.6f")
    
    # Initialize training state
    if 'training_started' not in st.session_state:
        st.session_state.training_started = False
    if 'training_complete' not in st.session_state:
        st.session_state.training_complete = False
    
    # Check if dataset is prepared first
    dataset_info_path = Path("models/dataset_info") / f"{dataset_name}_info.json"
    
    if not dataset_info_path.exists():
        st.error(f"❌ Dataset {dataset_name} not prepared. Please prepare it first in the Dataset Overview section.")
        st.info("💡 Go to 📊 Dataset Overview → Find your dataset → Click 🚀 Prepare Dataset")
        return
    
    # Training button
    if not st.session_state.training_started and not st.session_state.training_complete:
        if st.button(f"🚀 Start Training {architecture} on {dataset_name}", type="primary", use_container_width=True):
            st.session_state.training_started = True
            st.rerun()
    
    # Training execution
    if st.session_state.training_started and not st.session_state.training_complete:
        st.markdown("### 🔥 Training in Progress")
        
        # Load dataset info (basic info for validation)
        with open(dataset_info_path, 'r') as f:
            basic_info = json.load(f)
        
        st.success(f"✅ Dataset loaded: {basic_info['total_samples']} samples, {basic_info['num_classes']} classes")
        
        # Create and compile model
        with st.spinner("🏗️ Creating model..."):
            model = trainer.create_model(architecture, basic_info['num_classes'])
            model = trainer.compile_model(model, basic_info['num_classes'], custom_lr)
            st.success("✅ Model created and compiled!")
        
        # Start actual training
        try:
            # Update config with custom values
            updated_config = config.copy()
            updated_config['epochs'] = custom_epochs
            updated_config['batch_size'] = custom_batch_size  
            updated_config['learning_rate'] = custom_lr
            
            # Progress containers
            progress_container = st.container()
            status_container = st.container()
            
            with status_container:
                st.success("🎉 **Training Started!** Your model is now learning from the medical images.")
                st.info(f"📊 Training {custom_epochs} epochs with batch size {custom_batch_size}")
            
            # Prepare the full dataset for training
            data_loader = MedicalDataLoader()
            
            # Re-prepare the dataset to get the full data structure needed for training
            with st.spinner("🔄 Preparing training data..."):
                full_dataset_info = data_loader.prepare_dataset(dataset_name)
                
                # Validate and adjust batch size for stability
                min_samples = min(full_dataset_info['train_samples'], full_dataset_info['val_samples'])
                
                # Ensure batch size is reasonable
                if custom_batch_size > min_samples // 2:
                    # Reduce batch size to ensure at least 2 batches per dataset
                    new_batch_size = max(4, min_samples // 4)  # At least 4, but allow multiple batches
                    st.warning(f"⚠️ Batch size ({custom_batch_size}) too large for dataset ({min_samples} samples). Reducing to {new_batch_size}.")
                    custom_batch_size = new_batch_size
                elif custom_batch_size < 4:
                    # Ensure minimum batch size for stable training
                    custom_batch_size = 4
                    st.info(f"📊 Minimum batch size set to: {custom_batch_size}")
                
                st.info(f"📊 Using batch size: {custom_batch_size} (Train: {full_dataset_info['train_samples']} samples, Val: {full_dataset_info['val_samples']} samples)")
                
                # Create TensorFlow datasets from the full dataset info
                train_ds, val_ds, test_ds = data_loader.create_tf_dataset(
                    full_dataset_info, 
                    batch_size=custom_batch_size,
                    shuffle=True
                )
                
                st.success(f"✅ Datasets created - Train batches: ~{full_dataset_info['train_samples']//custom_batch_size}, Val batches: ~{full_dataset_info['val_samples']//custom_batch_size}")
            
            # Create custom training configuration
            custom_config = {
                'epochs': custom_epochs,
                'batch_size': custom_batch_size,
                'learning_rate': custom_lr,
                'description': f'Custom Training ({custom_epochs} epochs)'
            }
            
            # Add custom config to trainer temporarily
            trainer.training_configs['custom'] = custom_config
            
            # Show training progress setup
            st.markdown("### 📊 Live Training Progress")
            progress_bar = st.progress(0)
            status_text = st.empty()
            metrics_container = st.container()
            
            # Start actual training with live progress
            training_results = trainer.train_model_with_progress(
                model=model,
                train_ds=train_ds,
                val_ds=val_ds,
                dataset_name=dataset_name,
                architecture=architecture,
                config=custom_config,
                progress_bar=progress_bar,
                status_text=status_text,
                metrics_container=metrics_container
            )
            
            # Mark training as complete
            st.session_state.training_complete = True
            st.session_state.training_started = False
            
            # Training completed
            st.balloons()
            st.success("🎉 **Model Training Completed!** Your model has been saved and is ready for use.")
            
            # Show final results
            if training_results:
                st.markdown("### 📊 Final Training Results")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Final Accuracy", f"{training_results.get('final_accuracy', 0)*100:.1f}%")
                with col2:
                    st.metric("Final Loss", f"{training_results.get('final_loss', 0):.4f}")
                with col3:
                    st.metric("Training Time", f"{training_results.get('training_time', 0):.1f} min")
            
        except Exception as e:
            st.error(f"❌ Training failed: {str(e)}")
            st.info("💡 Please check your dataset preparation and try again.")
            st.session_state.training_started = False
            st.session_state.training_complete = False
    
    # Training completed state
    elif st.session_state.training_complete:
        st.success("✅ **Training Completed Successfully!**")
        st.info("Your model has been trained and saved. You can now use it for predictions!")
        
        if st.button("🔄 Train Another Model", type="secondary"):
            st.session_state.training_started = False
            st.session_state.training_complete = False
            st.rerun()
    
    # Default state - show training interface setup
    else:
        st.info("🎯 Configure your training parameters above and click the Start Training button.")
        
        # Add reset button if stuck in bad state
        if st.button("🔄 Reset Training Interface", type="secondary"):
            if 'training_started' in st.session_state:
                del st.session_state.training_started
            if 'training_complete' in st.session_state:
                del st.session_state.training_complete
            st.rerun()

# Example usage
if __name__ == "__main__":
    print("Medical Model Trainer initialized!")
    
    # Test model creation
    trainer = MedicalModelTrainer()
    model = trainer.create_model('DenseNet121', num_classes=2)
    print(f"Created model with {model.count_params()} parameters")