#!/usr/bin/env python3
"""
🚀 ULTRA-HIGH ACCURACY Bone Fracture Detection
Advanced Ensemble + Test-Time Augmentation + Model Fusion
Targeting 90%+ accuracy through multiple optimization techniques
"""
import tensorflow as tf
from tensorflow.keras.applications import DenseNet121, EfficientNetB4, ResNet152V2
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import *
import os
import numpy as np
from datetime import datetime
import json
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import cv2

class UltraHighAccuracyTrainer:
    def __init__(self, input_shape=(384, 384, 3), num_classes=2):
        """Initialize ultra-high accuracy trainer with maximum resolution"""
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.models = []
        
    def create_advanced_densenet(self, name="densenet"):
        """Create advanced DenseNet with attention and regularization"""
        base_model = DenseNet121(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        # Progressive unfreezing
        for layer in base_model.layers[:-80]:
            layer.trainable = False
        
        x = base_model.output
        
        # Multi-scale feature extraction
        x1 = GlobalAveragePooling2D()(x)
        x2 = GlobalMaxPooling2D()(x)
        
        # Attention mechanism
        attention = Dense(x.shape[-1] // 4, activation='relu')(x1)
        attention = Dense(x.shape[-1], activation='sigmoid')(attention)
        x1_attended = Multiply()([x1, attention])
        
        # Combine features
        combined = Concatenate()([x1_attended, x2])
        
        # Advanced dense layers with residual connections
        x = Dense(1024, activation='swish', kernel_regularizer=tf.keras.regularizers.l2(0.001))(combined)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        
        x1 = Dense(512, activation='swish', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
        x1 = BatchNormalization()(x1)
        x1 = Dropout(0.4)(x1)
        
        x2 = Dense(256, activation='swish', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x1)
        x2 = BatchNormalization()(x2)
        x2 = Dropout(0.3)(x2)
        
        # Residual connection
        x1_proj = Dense(256)(x1)
        x2_residual = Add()([x2, x1_proj])
        
        x3 = Dense(128, activation='swish', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x2_residual)
        x3 = BatchNormalization()(x3)
        x3 = Dropout(0.2)(x3)
        
        predictions = Dense(self.num_classes, activation='softmax', name=f'{name}_predictions')(x3)
        
        model = Model(inputs=base_model.input, outputs=predictions, name=f"Advanced_{name}")
        return model
    
    def create_efficientnet_model(self, name="efficientnet"):
        """Create EfficientNet model with advanced features"""
        base_model = EfficientNetB4(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        # Progressive unfreezing
        for layer in base_model.layers[:-100]:
            layer.trainable = False
        
        x = base_model.output
        
        # Squeeze-and-Excitation enhancement
        se_shape = x.shape[-1]
        se = GlobalAveragePooling2D()(x)
        se = Dense(se_shape // 16, activation='swish')(se)
        se = Dense(se_shape, activation='sigmoid')(se)
        se = Reshape((1, 1, se_shape))(se)
        x = Multiply()([x, se])
        
        # Multi-pooling strategy
        avg_pool = GlobalAveragePooling2D()(x)
        max_pool = GlobalMaxPooling2D()(x)
        combined = Concatenate()([avg_pool, max_pool])
        
        # Advanced classification head
        x = Dense(2048, activation='swish', kernel_regularizer=tf.keras.regularizers.l2(0.0005))(combined)
        x = BatchNormalization()(x)
        x = Dropout(0.6)(x)
        
        x = Dense(1024, activation='swish', kernel_regularizer=tf.keras.regularizers.l2(0.0005))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        
        x = Dense(512, activation='swish', kernel_regularizer=tf.keras.regularizers.l2(0.0005))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.4)(x)
        
        predictions = Dense(self.num_classes, activation='softmax', name=f'{name}_predictions')(x)
        
        model = Model(inputs=base_model.input, outputs=predictions, name=f"Advanced_{name}")
        return model
    
    def create_resnet_model(self, name="resnet"):
        """Create ResNet model for ensemble diversity"""
        base_model = ResNet152V2(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        # Progressive unfreezing
        for layer in base_model.layers[:-120]:
            layer.trainable = False
        
        x = base_model.output
        
        # Feature enhancement
        x1 = GlobalAveragePooling2D()(x)
        x2 = GlobalMaxPooling2D()(x)
        
        # Feature fusion with attention
        attention_weights = Dense(x1.shape[-1], activation='softmax')(x1)
        x1_weighted = Multiply()([x1, attention_weights])
        
        combined = Concatenate()([x1_weighted, x2])
        
        # Classification layers
        x = Dense(1024, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(combined)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        
        x = Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.4)(x)
        
        predictions = Dense(self.num_classes, activation='softmax', name=f'{name}_predictions')(x)
        
        model = Model(inputs=base_model.input, outputs=predictions, name=f"Advanced_{name}")
        return model
    
    def create_ultra_data_generators(self, train_dir, val_dir, batch_size=16):
        """Create ultra-advanced data generators with maximum augmentation"""
        
        # Extreme augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=40,
            width_shift_range=0.3,
            height_shift_range=0.3,
            shear_range=0.3,
            zoom_range=0.3,
            horizontal_flip=True,
            vertical_flip=False,
            brightness_range=[0.5, 1.5],
            channel_shift_range=0.3,
            fill_mode='reflect',
            preprocessing_function=self.advanced_preprocessing
        )
        
        # Clean validation
        val_datagen = ImageDataGenerator(
            rescale=1./255,
            preprocessing_function=self.validation_preprocessing
        )
        
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=self.input_shape[:2],
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=True,
            seed=42
        )
        
        validation_generator = val_datagen.flow_from_directory(
            val_dir,
            target_size=self.input_shape[:2],
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        return train_generator, validation_generator
    
    def advanced_preprocessing(self, image):
        """Advanced preprocessing with multiple enhancement techniques"""
        # Convert to numpy if needed
        if hasattr(image, 'numpy'):
            image = image.numpy()
        
        # Random histogram equalization
        if np.random.random() < 0.3:
            # Convert to uint8 for CLAHE
            img_uint8 = (image * 255).astype(np.uint8)
            
            # Apply CLAHE to each channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            for i in range(3):
                img_uint8[:,:,i] = clahe.apply(img_uint8[:,:,i])
            
            image = img_uint8.astype(np.float32) / 255.0
        
        # Random noise injection
        if np.random.random() < 0.2:
            noise = np.random.normal(0, 0.01, image.shape)
            image = np.clip(image + noise, 0, 1)
        
        # Random contrast adjustment
        if np.random.random() < 0.3:
            alpha = np.random.uniform(0.8, 1.2)
            image = np.clip(alpha * image, 0, 1)
        
        return image
    
    def validation_preprocessing(self, image):
        """Clean preprocessing for validation"""
        if hasattr(image, 'numpy'):
            image = image.numpy()
        return image
    
    def create_advanced_callbacks(self, model_save_path, model_name):
        """Create advanced callbacks for maximum performance"""
        callbacks = [
            ModelCheckpoint(
                model_save_path,
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            EarlyStopping(
                monitor='val_accuracy',
                patience=25,
                restore_best_weights=True,
                mode='max',
                verbose=1,
                min_delta=0.001
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.1,
                patience=12,
                min_lr=1e-8,
                verbose=1,
                cooldown=5
            ),
            # Cosine annealing with warm restarts
            tf.keras.callbacks.LearningRateScheduler(
                lambda epoch: self.cosine_annealing_with_restarts(epoch),
                verbose=0
            )
        ]
        
        return callbacks
    
    def cosine_annealing_with_restarts(self, epoch):
        """Cosine annealing learning rate with warm restarts"""
        initial_lr = 1e-4
        restart_period = 30
        t_cur = epoch % restart_period
        lr = initial_lr * 0.5 * (1 + np.cos(np.pi * t_cur / restart_period))
        return max(lr, 1e-8)
    
    def train_ensemble_models(self, train_dir, val_dir, epochs=80, batch_size=16):
        """Train multiple models for ensemble"""
        print("🚀 Starting ULTRA-HIGH ACCURACY Ensemble Training")
        print(f"Target: 90%+ accuracy through advanced ensemble methods")
        
        # Create data generators
        train_gen, val_gen = self.create_ultra_data_generators(train_dir, val_dir, batch_size)
        
        # Calculate advanced class weights
        labels = train_gen.labels
        class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
        class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
        print(f"📊 Class weights: {class_weight_dict}")
        
        print(f"📊 Training samples: {train_gen.samples}")
        print(f"📊 Validation samples: {val_gen.samples}")
        
        # Model configurations
        model_configs = [
            {"model_func": self.create_advanced_densenet, "name": "DenseNet", "lr": 1e-4},
            {"model_func": self.create_efficientnet_model, "name": "EfficientNet", "lr": 8e-5},
            {"model_func": self.create_resnet_model, "name": "ResNet", "lr": 1e-4}
        ]
        
        trained_models = []
        model_accuracies = []
        
        for i, config in enumerate(model_configs):
            print(f"\n🎯 Training Model {i+1}/3: {config['name']}")
            
            # Create model
            model = config["model_func"](config["name"].lower())
            print(f"✅ {config['name']} created with {model.count_params():,} parameters")
            
            # Compile model
            optimizer = Adam(learning_rate=config["lr"], amsgrad=True)
            model.compile(
                optimizer=optimizer,
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Model save path
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_save_path = f"models/Ultra_{config['name']}_{timestamp}.h5"
            
            # Create callbacks
            callbacks = self.create_advanced_callbacks(model_save_path, config["name"])
            
            # Calculate steps
            steps_per_epoch = max(1, train_gen.samples // batch_size)
            validation_steps = max(1, val_gen.samples // batch_size)
            
            # Train model
            history = model.fit(
                train_gen,
                steps_per_epoch=steps_per_epoch,
                epochs=epochs,
                validation_data=val_gen,
                validation_steps=validation_steps,
                callbacks=callbacks,
                class_weight=class_weight_dict,
                verbose=1
            )
            
            # Load best model
            try:
                best_model = tf.keras.models.load_model(model_save_path)
            except:
                best_model = model
            
            # Evaluate model
            val_loss, val_accuracy = best_model.evaluate(val_gen, verbose=0)
            print(f"✅ {config['name']} - Validation Accuracy: {val_accuracy*100:.2f}%")
            
            trained_models.append(best_model)
            model_accuracies.append(val_accuracy)
        
        return trained_models, model_accuracies
    
    def test_time_augmentation(self, model, image, num_augmentations=10):
        """Apply test-time augmentation for better predictions"""
        predictions = []
        
        # Original image
        pred = model.predict(np.expand_dims(image, axis=0), verbose=0)
        predictions.append(pred[0])
        
        # Augmented versions
        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            zoom_range=0.1,
            fill_mode='nearest'
        )
        
        for _ in range(num_augmentations - 1):
            # Generate augmented image
            img_array = np.expand_dims(image, axis=0)
            aug_iter = datagen.flow(img_array, batch_size=1)
            aug_image = next(aug_iter)[0]
            
            # Predict on augmented image
            pred = model.predict(np.expand_dims(aug_image, axis=0), verbose=0)
            predictions.append(pred[0])
        
        # Average predictions
        avg_prediction = np.mean(predictions, axis=0)
        return avg_prediction
    
    def ensemble_predict(self, models, val_gen, use_tta=True):
        """Create ensemble predictions with test-time augmentation"""
        print("\n🔮 Creating Ensemble Predictions with Test-Time Augmentation")
        
        all_predictions = []
        true_labels = []
        
        # Get true labels
        val_gen.reset()
        for i in range(len(val_gen)):
            batch_x, batch_y = val_gen[i]
            true_labels.extend(np.argmax(batch_y, axis=1))
        
        # Reset generator
        val_gen.reset()
        
        for i in range(len(val_gen)):
            batch_x, _ = val_gen[i]
            batch_predictions = []
            
            # Get predictions from each model
            for model in models:
                if use_tta:
                    # Use test-time augmentation
                    batch_pred = []
                    for img in batch_x:
                        tta_pred = self.test_time_augmentation(model, img, num_augmentations=5)
                        batch_pred.append(tta_pred)
                    batch_pred = np.array(batch_pred)
                else:
                    batch_pred = model.predict(batch_x, verbose=0)
                
                batch_predictions.append(batch_pred)
            
            # Ensemble averaging with weights
            weights = [0.4, 0.35, 0.25]  # Weight models by expected performance
            ensemble_pred = np.average(batch_predictions, axis=0, weights=weights)
            all_predictions.extend(ensemble_pred)
        
        # Convert to class predictions
        ensemble_classes = np.argmax(all_predictions, axis=1)
        
        # Calculate ensemble accuracy
        ensemble_accuracy = np.mean(ensemble_classes == true_labels)
        
        return ensemble_accuracy, ensemble_classes, true_labels

def main():
    """Main ultra-high accuracy training function"""
    print("=" * 80)
    print("🦴 ULTRA-HIGH ACCURACY Bone Fracture Detection")
    print("🎯 Target: 90%+ Accuracy through Advanced Ensemble Methods")
    print("=" * 80)
    
    # Use ARM_Combined dataset
    dataset_path = r"D:\Capstone\mynew\capstoneortho\Dataset\ARM_Combined"
    train_dir = os.path.join(dataset_path, "train")
    val_dir = os.path.join(dataset_path, "val")
    
    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        print(f"❌ Dataset not found at: {dataset_path}")
        return
    
    # Initialize trainer
    trainer = UltraHighAccuracyTrainer(
        input_shape=(384, 384, 3),  # High resolution for maximum detail
        num_classes=2
    )
    
    try:
        # Train ensemble models
        models, accuracies = trainer.train_ensemble_models(
            train_dir=train_dir,
            val_dir=val_dir,
            epochs=80,
            batch_size=16
        )
        
        print(f"\n📊 Individual Model Results:")
        model_names = ["DenseNet", "EfficientNet", "ResNet"]
        for name, acc in zip(model_names, accuracies):
            print(f"   {name}: {acc*100:.2f}%")
        
        # Create validation generator for ensemble testing
        val_datagen = ImageDataGenerator(rescale=1./255)
        val_gen = val_datagen.flow_from_directory(
            val_dir,
            target_size=(384, 384),
            batch_size=16,
            class_mode='categorical',
            shuffle=False
        )
        
        # Test ensemble performance
        ensemble_acc, pred_classes, true_classes = trainer.ensemble_predict(models, val_gen, use_tta=True)
        
        print(f"\n🎉 ENSEMBLE RESULTS:")
        print(f"✅ Ensemble Accuracy: {ensemble_acc*100:.2f}%")
        print(f"✅ Best Individual: {max(accuracies)*100:.2f}%")
        print(f"✅ Improvement: +{(ensemble_acc - max(accuracies))*100:.2f}%")
        
        if ensemble_acc >= 0.92:
            print(f"🎉 BREAKTHROUGH! Medical-grade accuracy achieved: {ensemble_acc*100:.2f}%")
        elif ensemble_acc >= 0.88:
            print(f"🎉 OUTSTANDING! Research-level performance: {ensemble_acc*100:.2f}%")
        elif ensemble_acc >= 0.85:
            print(f"✅ EXCELLENT! High-quality results: {ensemble_acc*100:.2f}%")
        else:
            print(f"📈 SOLID foundation established: {ensemble_acc*100:.2f}%")
        
        # Print detailed classification report
        print(f"\n📋 Detailed Classification Report:")
        from sklearn.metrics import classification_report
        report = classification_report(true_classes, pred_classes, target_names=['Fracture', 'Normal'])
        print(report)
        
    except Exception as e:
        print(f"❌ Ultra-high accuracy training failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()